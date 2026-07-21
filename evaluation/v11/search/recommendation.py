"""Deterministic Recommendation Policy（Slice 5 §十六、§十八）。

以**可配置的 lexicographic 順序**排序，不使用任何不透明加權總分（§16 / §18.4）。
排序鍵可以是 metric 參照，或三個特殊鍵：``constraint_pass_count`` /
``config_complexity`` / ``config_hash``（後者強制為最後一鍵，保證完全 deterministic）。

Recommendation **只**從「validation feasible 且 Pareto member」中選出（§十八首句）。
沒有可行 candidate 時回傳 ``no_feasible_candidate``、``recommended_config_hash=None``，
**絕不**從不合格 candidate 中硬選「最少失敗」者（§18.6）。
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from evaluation.v11.search import complexity as complexity_mod
from evaluation.v11.search.constraints import ConstraintEvaluation, MetricLookup
from evaluation.v11.search.search_config import (
    DIRECTION_MAXIMIZE,
    TIEBREAK_CONFIG_COMPLEXITY,
    TIEBREAK_CONFIG_HASH,
    TIEBREAK_CONSTRAINT_PASS_COUNT,
    SortKey,
)

STATUS_OK = "ok"
STATUS_NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"


@dataclass
class RankableCandidate:
    """排序所需的最小 candidate 視圖。"""

    config_hash: str
    metrics: MetricLookup
    constraint_eval: Optional[ConstraintEvaluation] = None
    chunk_config: dict[str, Any] = field(default_factory=dict)
    retrieval_config: dict[str, Any] = field(default_factory=dict)

    def complexity_vector(self) -> tuple[int, int, int, int]:
        return complexity_mod.complexity_vector(self.chunk_config, self.retrieval_config)


# --------------------------------------------------------------------------
# 排序鍵取值
# --------------------------------------------------------------------------
def resolve_key(candidate: RankableCandidate, key: SortKey) -> tuple[Any, Optional[str]]:
    """回傳 ``(value, not_computable_reason)``；``value=None`` 代表不可比較（一律排最後）。"""
    if key.key == TIEBREAK_CONSTRAINT_PASS_COUNT:
        if candidate.constraint_eval is None:
            return None, "no_constraint_evaluation"
        return candidate.constraint_eval.pass_count, None
    if key.key == TIEBREAK_CONFIG_COMPLEXITY:
        return candidate.complexity_vector(), None
    if key.key == TIEBREAK_CONFIG_HASH:
        return candidate.config_hash, None
    assert key.ref is not None  # schema 保證非特殊鍵一定有 ref
    value, reason, _ = candidate.metrics.value(key.ref)
    return value, reason


def _compare_values(a: Any, b: Any, direction: str, tolerance: float) -> int:
    """回傳 -1 / 0 / 1（-1 表示 a 應排在 b 之前）。

    不可比較（None）一律排在最後，與 direction 無關 —— 不可計算不得因為方向而佔便宜。
    """
    if a is None and b is None:
        return 0
    if a is None:
        return 1
    if b is None:
        return -1

    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if abs(fa - fb) <= tolerance:
            return 0
        better_first = fa > fb if direction == DIRECTION_MAXIMIZE else fa < fb
        return -1 if better_first else 1

    if a == b:
        return 0
    better_first = a > b if direction == DIRECTION_MAXIMIZE else a < b
    return -1 if better_first else 1


@dataclass
class TieBreakStep:
    key: str
    direction: str
    left_value: Any
    right_value: Any
    decided: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "direction": self.direction,
            "left_value": _jsonable(self.left_value),
            "right_value": _jsonable(self.right_value),
            "decided": self.decided,
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return list(v)
    return v


def compare_candidates(
    a: RankableCandidate,
    b: RankableCandidate,
    sort_keys: Sequence[SortKey],
    tolerance: float,
) -> tuple[int, list[TieBreakStep]]:
    """依 lexicographic 順序比較，並回傳每一步的 tie-break 記錄（§18.3）。"""
    steps: list[TieBreakStep] = []
    for key in sort_keys:
        va, _ = resolve_key(a, key)
        vb, _ = resolve_key(b, key)
        cmp = _compare_values(va, vb, key.direction, tolerance)
        steps.append(
            TieBreakStep(
                key=key.key,
                direction=key.direction,
                left_value=va,
                right_value=vb,
                decided=cmp != 0,
            )
        )
        if cmp != 0:
            return cmp, steps
    return 0, steps


@dataclass
class RankedCandidate:
    rank: int
    config_hash: str
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "config_hash": self.config_hash,
            "values": {k: _jsonable(v) for k, v in self.values.items()},
        }


def rank_candidates(
    candidates: Sequence[RankableCandidate],
    sort_keys: Sequence[SortKey],
    *,
    tolerance: float = 1e-9,
) -> list[RankedCandidate]:
    """依政策排序（1 = 最佳）。相同輸入永遠得到相同輸出。"""

    def _cmp(x: RankableCandidate, y: RankableCandidate) -> int:
        cmp, _ = compare_candidates(x, y, sort_keys, tolerance)
        return cmp

    ordered = sorted(candidates, key=functools.cmp_to_key(_cmp))
    out: list[RankedCandidate] = []
    for i, c in enumerate(ordered, start=1):
        values = {}
        for key in sort_keys:
            v, _ = resolve_key(c, key)
            values[key.key] = v
        out.append(RankedCandidate(rank=i, config_hash=c.config_hash, values=values))
    return out


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------
@dataclass
class RecommendationResult:
    status: str
    recommended_config_hash: Optional[str]
    ranking: list[RankedCandidate] = field(default_factory=list)
    tie_break_trace: list[dict[str, Any]] = field(default_factory=list)
    closest_infeasible: list[dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "recommended_config_hash": self.recommended_config_hash,
            "ranking": [r.to_dict() for r in self.ranking],
            "tie_break_trace": list(self.tie_break_trace),
            "closest_infeasible_candidates": list(self.closest_infeasible),
            "reason": self.reason,
        }


def select_recommendation(
    eligible: Sequence[RankableCandidate],
    sort_keys: Sequence[SortKey],
    *,
    tolerance: float = 1e-9,
    infeasible: Sequence[RankableCandidate] = (),
) -> RecommendationResult:
    """從 **feasible ∩ Pareto member** 中選出 recommendation。

    :param eligible: 已同時滿足 validation feasible 與 pareto_member 的 candidate。
    :param infeasible: 未通過限制者；只用於產生 ``closest_infeasible_candidates``
        參考清單，**絕不**會被推薦（§18.7）。
    """
    closest = _closest_infeasible(infeasible)

    if not eligible:
        # §18.5：明確失敗，不硬選。
        return RecommendationResult(
            status=STATUS_NO_FEASIBLE_CANDIDATE,
            recommended_config_hash=None,
            closest_infeasible=closest,
            reason=(
                "沒有任何 candidate 同時滿足 validation hard constraints 與 Pareto membership；"
                "依 §18.5 / §18.6 不得從不合格 candidate 中硬選，recommended_config_hash 為 null。"
            ),
        )

    ranking = rank_candidates(eligible, sort_keys, tolerance=tolerance)
    winner = ranking[0].config_hash

    # 保存冠軍與第二名的逐鍵 tie-break 記錄（§18.3）。
    trace: list[dict[str, Any]] = []
    if len(eligible) > 1:
        by_hash = {c.config_hash: c for c in eligible}
        runner_up = ranking[1].config_hash
        _, steps = compare_candidates(
            by_hash[winner], by_hash[runner_up], sort_keys, tolerance
        )
        trace = [
            {
                "comparison": f"{winner} vs {runner_up}",
                "steps": [s.to_dict() for s in steps],
            }
        ]

    return RecommendationResult(
        status=STATUS_OK,
        recommended_config_hash=winner,
        ranking=ranking,
        tie_break_trace=trace,
        closest_infeasible=closest,
    )


def _closest_infeasible(
    infeasible: Sequence[RankableCandidate],
) -> list[dict[str, Any]]:
    """§18.7：可另列「最接近可行」的 candidate，但**不得**標為 recommended。"""
    rows: list[dict[str, Any]] = []
    for c in infeasible:
        ce = c.constraint_eval
        if ce is None:
            continue
        rows.append(
            {
                "config_hash": c.config_hash,
                "fail_count": ce.fail_count,
                "pass_count": ce.pass_count,
                "failed_constraints": ce.failed_names(),
                "rejection_reasons": ce.rejection_reasons(),
                "recommended": False,
                "note": "未通過 hard constraints，僅供參考，不得作為 recommendation。",
            }
        )
    rows.sort(key=lambda r: (r["fail_count"], -r["pass_count"], r["config_hash"]))
    return rows
