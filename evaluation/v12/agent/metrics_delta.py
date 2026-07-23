"""Parent ↔ Candidate metric delta（Slice 3 §六）。

以 v1.1 :class:`MetricLookup` 取值，計算 Parent 與 Candidate 在指定 metric 上的差值。
供 tune metric delta（回饋給下一輪 LLM）與報告使用。**不**做任何升級判定——升級由
Orchestrator 的 deterministic comparator（重用 v1.1 recommendation ordering）負責。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from evaluation.v11.run_models import TrialMetric
from evaluation.v11.search.constraints import MetricLookup
from evaluation.v11.search.search_config import MetricRef, SortKey


def build_lookup(metrics: Iterable[TrialMetric]) -> MetricLookup:
    return MetricLookup(metrics)


def metric_delta(
    parent: MetricLookup, candidate: MetricLookup, refs: Sequence[MetricRef]
) -> dict[str, Any]:
    """回傳 ``{metric_describe: {parent, candidate, delta}}``；不可計算者 delta=None。"""
    out: dict[str, Any] = {}
    for ref in refs:
        pv, _, _ = parent.value(ref)
        cv, _, _ = candidate.value(ref)
        delta = (cv - pv) if (pv is not None and cv is not None) else None
        out[ref.describe()] = {"parent": pv, "candidate": cv, "delta": delta}
    return out


def comparator_metric_refs(sort_keys: Sequence[SortKey]) -> list[MetricRef]:
    """由 comparator ranking 取出所有 metric 參照（略過 special tie-break 鍵）。"""
    return [k.ref for k in sort_keys if k.ref is not None]
