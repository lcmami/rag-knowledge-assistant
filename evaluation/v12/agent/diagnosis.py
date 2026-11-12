"""Tune failure 分類（Slice 4 §七）。

把 v1.1 的 ``case_status`` 與 identifier/confusable 明細對映到 :mod:`proposal` 的受控
``FAILURE_TYPES`` 詞彙。供 RealTrialExecutor 由 ``trial_case_results`` 萃取 tune failure 時
標註 failure_type，讓 LLM 讀到一致的診斷標籤（LLM 仍只提案，不下最終結論）。
"""
from __future__ import annotations

from typing import Optional, Sequence

from evaluation.v12.agent.proposal import FAILURE_TYPES

# v1.1 case_status（retrieval_metrics.score_case）。
CASE_STATUS_RELEVANT_HIT = "relevant_hit"
CASE_STATUS_MISS = "miss"
CASE_STATUS_UNANSWERABLE_FP = "unanswerable_false_positive"
CASE_STATUS_UNANSWERABLE_OK = "unanswerable_ok"


def classify_failure(
    *,
    case_status: str,
    answerable: bool,
    first_relevant_rank: Optional[int],
    conflicting_identifiers: Sequence[str] = (),
    retrieved_any: bool = True,
) -> Optional[str]:
    """由 evaluator 已知事實決定 failure type；LLM 不重新分類。"""

    if case_status == CASE_STATUS_UNANSWERABLE_FP:
        return "unanswerable_false_positive"

    if case_status == CASE_STATUS_UNANSWERABLE_OK:
        return None

    # answerable 且 relevant 已經 rank 1：沒有 retrieval failure
    if first_relevant_rank == 1:
        return None

    # 有 relevant，但被混淆結果壓在後面
    if conflicting_identifiers:
        return "confusable_confusion"

    # relevant 有找到，但不是第一名：MRR 仍有改善空間
    if first_relevant_rank is not None and first_relevant_rank > 1:
        return "wrong_rank"

    # 完全沒結果，多半是 threshold 過嚴
    if not retrieved_any:
        return "threshold_too_high"

    # 有結果，但 relevant 完全沒進榜
    return "miss"


def is_known_failure_type(value: str) -> bool:
    return value in FAILURE_TYPES
