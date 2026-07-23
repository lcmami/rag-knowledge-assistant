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
    """回傳受控 ``failure_type``；非失敗（relevant_hit / unanswerable_ok）回傳 None。"""
    if case_status == CASE_STATUS_UNANSWERABLE_FP:
        return "unanswerable_false_positive"
    if case_status in (CASE_STATUS_RELEVANT_HIT, CASE_STATUS_UNANSWERABLE_OK):
        return None
    # 以下為 answerable 但未命中的情形。
    if conflicting_identifiers:
        return "confusable_confusion"
    if not retrieved_any:
        return "threshold_too_high"
    if first_relevant_rank is None:
        return "miss"
    if first_relevant_rank > 1:
        return "wrong_rank"
    return "miss"


def is_known_failure_type(value: str) -> bool:
    return value in FAILURE_TYPES
