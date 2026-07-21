"""Pareto Frontier（Slice 5 §十七）。

只在 **validation feasible** 的 candidate 上建立 frontier。明確 maximize / minimize，
**不產生任何不透明加權總分**（§17.8）。

不可計算的 objective（§17.2 / §17.3）：

- ``exclude_candidate``：該 candidate 直接排除於 frontier 計算之外（記錄原因）。
- ``ignore_objective``：該 objective 對**所有** candidate 一併移除 —— 必須是全域決定，
  否則不同 candidate 用不同 objective 集合比較，dominance 將失去意義。

**絕不**把不可計算值當 0。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from evaluation.v11.search.constraints import MetricLookup
from evaluation.v11.search.search_config import (
    DIRECTION_MAXIMIZE,
    OBJ_EXCLUDE_CANDIDATE,
    ParetoObjective,
)

PARETO_TOLERANCE_VERSION = "v1"


@dataclass
class ObjectiveValue:
    objective: ParetoObjective
    value: Optional[float]
    computable: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.objective.to_dict(),
            "value": self.value,
            "computable": self.computable,
            "reason": self.reason,
        }


@dataclass
class ParetoPoint:
    """參與 Pareto 計算的一個點。``key`` 為 candidate 的 config_hash。"""

    key: str
    values: list[ObjectiveValue]
    included: bool = True
    exclusion_reason: Optional[str] = None
    pareto_member: bool = False
    pareto_rank: Optional[int] = None
    dominated_by: list[str] = field(default_factory=list)
    dominates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.key,
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
            "pareto_member": self.pareto_member,
            "pareto_rank": self.pareto_rank,
            "dominated_by": list(self.dominated_by),
            "dominates": list(self.dominates),
            "objectives": [v.to_dict() for v in self.values],
        }


@dataclass
class ParetoResult:
    points: list[ParetoPoint]
    active_objectives: list[ParetoObjective]
    ignored_objectives: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def frontier(self) -> list[ParetoPoint]:
        return [p for p in self.points if p.pareto_member]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tolerance_version": PARETO_TOLERANCE_VERSION,
            "active_objectives": [o.to_dict() for o in self.active_objectives],
            "ignored_objectives": list(self.ignored_objectives),
            "excluded_candidates": list(self.excluded),
            "frontier_size": len(self.frontier),
            "points": [p.to_dict() for p in self.points],
        }


# --------------------------------------------------------------------------
def _oriented(value: float, objective: ParetoObjective) -> float:
    """統一轉成「越大越好」，使 dominance 判定只需一種方向。"""
    return value if objective.direction == DIRECTION_MAXIMIZE else -value


def dominates(
    a: Sequence[float], b: Sequence[float], tolerance: float
) -> bool:
    """A 是否 dominate B（§17.4 標準定義）。

    兩者皆為已轉向的「越大越好」向量：**所有** objective 上 A 不差於 B，
    且**至少一項** A 嚴格優於 B。比較使用版本化 tolerance（§17.5）。
    """
    if len(a) != len(b):  # pragma: no cover - 呼叫端保證等長
        raise ValueError("objective 向量長度不一致")
    if not a:
        return False
    not_worse = all(x >= y - tolerance for x, y in zip(a, b))
    strictly_better = any(x > y + tolerance for x, y in zip(a, b))
    return not_worse and strictly_better


def compute_pareto(
    candidates: dict[str, MetricLookup],
    objectives: Sequence[ParetoObjective],
    *,
    tolerance: float = 1e-9,
) -> ParetoResult:
    """建立 Pareto frontier。

    :param candidates: ``{config_hash: MetricLookup}``（僅傳入 feasible candidate）。
    :param objectives: 已驗證的 objective 清單。
    :param tolerance: 浮點比較容差（版本化，見 :data:`PARETO_TOLERANCE_VERSION`）。
    """
    # 順序 deterministic：一律依 config_hash 排序（§17.6）。
    keys = sorted(candidates)

    # -- 1. 取值 --
    raw: dict[str, list[ObjectiveValue]] = {}
    for key in keys:
        lookup = candidates[key]
        vals: list[ObjectiveValue] = []
        for obj in objectives:
            value, reason, _ = lookup.value(obj.ref)
            vals.append(
                ObjectiveValue(
                    objective=obj,
                    value=value,
                    computable=reason is None,
                    reason=reason,
                )
            )
        raw[key] = vals

    # -- 2. 決定全域忽略的 objective（ignore_objective 政策）--
    ignored: list[dict[str, Any]] = []
    active: list[ParetoObjective] = []
    for i, obj in enumerate(objectives):
        non_computable_keys = [k for k in keys if not raw[k][i].computable]
        if non_computable_keys and obj.on_not_computable != OBJ_EXCLUDE_CANDIDATE:
            ignored.append(
                {
                    **obj.to_dict(),
                    "reason": (
                        f"{len(non_computable_keys)} 個 candidate 無法計算此 objective，"
                        "依 on_not_computable=ignore_objective 對所有 candidate 一併移除"
                    ),
                    "affected_candidates": non_computable_keys,
                }
            )
            continue
        active.append(obj)

    active_index = [objectives.index(o) for o in active]

    # -- 3. 決定被排除的 candidate（exclude_candidate 政策）--
    points: list[ParetoPoint] = []
    excluded: list[dict[str, Any]] = []
    for key in keys:
        vals = raw[key]
        bad = [
            vals[i]
            for i in active_index
            if not vals[i].computable
        ]
        point = ParetoPoint(key=key, values=vals)
        if bad:
            point.included = False
            point.exclusion_reason = (
                "objective 不可計算："
                + "；".join(f"{v.objective.ref.describe()}（{v.reason}）" for v in bad)
            )
            excluded.append(
                {"config_hash": key, "reason": point.exclusion_reason}
            )
        points.append(point)

    # -- 4. dominance --
    included = [p for p in points if p.included]
    vectors: dict[str, list[float]] = {
        p.key: [
            _oriented(float(p.values[i].value), objectives[i])  # type: ignore[arg-type]
            for i in active_index
        ]
        for p in included
    }

    for p in included:
        for q in included:
            if p.key == q.key:
                continue
            if dominates(vectors[q.key], vectors[p.key], tolerance):
                p.dominated_by.append(q.key)
            if dominates(vectors[p.key], vectors[q.key], tolerance):
                p.dominates.append(q.key)
        p.dominated_by.sort()
        p.dominates.sort()

    # -- 5. non-dominated sorting → pareto_rank（1 = frontier）--
    remaining = {p.key for p in included}
    rank = 1
    while remaining:
        current = [
            k
            for k in sorted(remaining)
            if not any(d in remaining for d in _point(points, k).dominated_by)
        ]
        if not current:  # pragma: no cover - dominance 為嚴格偏序，不應發生
            break
        for k in current:
            pt = _point(points, k)
            pt.pareto_rank = rank
            pt.pareto_member = rank == 1
        remaining -= set(current)
        rank += 1

    return ParetoResult(
        points=points,
        active_objectives=active,
        ignored_objectives=ignored,
        excluded=excluded,
    )


def _point(points: list[ParetoPoint], key: str) -> ParetoPoint:
    for p in points:
        if p.key == key:
            return p
    raise KeyError(key)  # pragma: no cover
