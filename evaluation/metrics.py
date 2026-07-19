"""Baseline 指標計算（v1.1 Slice 0）—— 純函式，不依賴任何服務。

分兩層：
1. :func:`build_case_result`：把「一個 case + 一串排序後的檢索結果」轉為
   :class:`~evaluation.schemas.CaseResult`（計算 hit / rank / reciprocal rank）。
   Runner（真實檢索）與單元測試（deterministic fake retrieval）共用同一函式。
2. 聚合指標函式（Hit@K / Recall@5 / MRR / Exact-code Top-1 / Confusable Error /
   Unanswerable FP / query_type 分組 / latency 統計 / threshold 掃描）。

Slice 0 僅計算目前資料格式可可靠計算的指標；Evidence Coverage、Graded NDCG、
Evidence-level Precision@5、Canonical Evidence overlap 等留給 Slice 1／2，不預先偽造。
"""
from __future__ import annotations

from typing import Any, Optional

from evaluation.schemas import CaseResult, EvaluationCase


# --------------------------------------------------------------------------
# 純統計小工具（沿用原 evaluate_retrieval.py，行為不變）
# --------------------------------------------------------------------------
def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "p25": round(percentile(values, 0.25), 4),
        "median": round(percentile(values, 0.50), 4),
        "mean": round(sum(values) / len(values), 4),
        "p75": round(percentile(values, 0.75), 4),
        "max": round(max(values), 4),
    }


def rate(numer: int, denom: int) -> float:
    return round(numer / denom, 4) if denom else 0.0


# --------------------------------------------------------------------------
# 核心：一個 case + 排序後的檢索結果 -> CaseResult
# --------------------------------------------------------------------------
def build_case_result(
    case: EvaluationCase,
    ranked: list[dict[str, Any]],
    retrieval_time_ms: int,
    top_k: int = 5,
) -> CaseResult:
    """把排序後的檢索結果（由高分到低分）轉為 :class:`CaseResult`。

    ``ranked`` 每筆為 ``{"document", "section", "page", "score"}``，已依 score 由高到低排序。
    Runner 以真實檢索結果呼叫；單元測試以 deterministic fake ranking 呼叫（同一計算）。
    """
    ranked_documents = [r["document"] for r in ranked]
    top1 = ranked[0] if ranked else None

    expected_doc = case.expected_document
    expected_section = case.expected_section
    expected_page = case.expected_page

    def _rank_of(document: Optional[str]) -> Optional[int]:
        if document is None:
            return None
        for idx, doc in enumerate(ranked_documents, start=1):
            if doc == document:
                return idx
        return None

    expected_rank = _rank_of(expected_doc)
    confusable_rank = _rank_of(case.confusable_document)

    doc_hit_1 = (ranked_documents[:1] == [expected_doc]) if expected_doc else None
    doc_hit_3 = (expected_doc in ranked_documents[:3]) if expected_doc else None
    doc_hit_5 = (expected_doc in ranked_documents[:5]) if expected_doc else None

    section_hit_1 = (
        bool(top1 and top1["document"] == expected_doc and top1.get("section") == expected_section)
        if (expected_doc and expected_section)
        else None
    )
    page_hit_1 = (
        bool(top1 and top1["document"] == expected_doc and top1.get("page") == expected_page)
        if (expected_doc and expected_page is not None)
        else None
    )

    reciprocal_rank = (1.0 / expected_rank) if expected_rank else 0.0

    top5 = [
        {
            "rank": i,
            "document": r["document"],
            "section": r.get("section"),
            "page": r.get("page"),
            "score": round(float(r["score"]), 4) if r.get("score") is not None else None,
        }
        for i, r in enumerate(ranked[: max(5, top_k)], start=1)
    ]

    return CaseResult(
        id=case.id,
        question=case.question,
        query_type=case.query_type,
        should_answer=case.should_answer,
        expected_document=expected_doc,
        expected_section=expected_section,
        expected_page=expected_page,
        confusable_document=case.confusable_document,
        top1_score=round(float(top1["score"]), 4) if top1 and top1.get("score") is not None else None,
        top1_document=top1["document"] if top1 else None,
        top1_section=top1.get("section") if top1 else None,
        top1_page=top1.get("page") if top1 else None,
        top3_documents=ranked_documents[:3],
        ranked_documents=ranked_documents,
        top5=top5,
        doc_hit_at_1=doc_hit_1,
        doc_hit_at_3=doc_hit_3,
        doc_hit_at_5=doc_hit_5,
        section_hit_at_1=section_hit_1,
        page_hit_at_1=page_hit_1,
        expected_rank=expected_rank,
        confusable_rank=confusable_rank,
        reciprocal_rank=round(reciprocal_rank, 4),
        retrieval_time_ms=retrieval_time_ms,
    )


# --------------------------------------------------------------------------
# 聚合指標
# --------------------------------------------------------------------------
def hit_rate(cases: list[CaseResult], attr: str) -> tuple[float, int, int]:
    """回傳 (rate, hits, considered)；considered = 該欄位非 None 的案例數。"""
    considered = [c for c in cases if getattr(c, attr) is not None]
    hits = [c for c in considered if getattr(c, attr)]
    return rate(len(hits), len(considered)), len(hits), len(considered)


def mrr(cases: list[CaseResult]) -> tuple[float, int]:
    """可回答且有 expected_document 的案例平均 reciprocal rank。"""
    considered = [c for c in cases if c.should_answer and c.expected_document]
    if not considered:
        return 0.0, 0
    total = sum(c.reciprocal_rank for c in considered)
    return round(total / len(considered), 4), len(considered)


def exact_code_top1_accuracy(cases: list[CaseResult]) -> dict[str, Any]:
    """query_type == 'exact_code' 案例的 Top-1 命中率。"""
    considered = [c for c in cases if c.query_type == "exact_code" and c.expected_document]
    hits = [c for c in considered if c.doc_hit_at_1]
    return {
        "accuracy": rate(len(hits), len(considered)),
        "hits": len(hits),
        "considered": len(considered),
        "miss_ids": [c.id for c in considered if not c.doc_hit_at_1],
    }


def confusable_error(cases: list[CaseResult]) -> dict[str, Any]:
    """標了 confusable_document 的案例中，confusable 文件搶到 expected 之前的比率。

    error 定義：confusable 文件出現在檢索結果，且（expected 不在結果中，或
    confusable 排名優於 expected）。反映「精確識別碼被相似干擾碼壓過」的失敗。
    """
    considered = [c for c in cases if c.confusable_document]
    errors = []
    for c in considered:
        if c.confusable_rank is None:
            continue
        if c.expected_rank is None or c.confusable_rank < c.expected_rank:
            errors.append(c)
    return {
        "rate": rate(len(errors), len(considered)),
        "errors": len(errors),
        "considered": len(considered),
        "error_ids": [c.id for c in errors],
    }


def unanswerable_false_positive_rate(
    cases: list[CaseResult], threshold: float
) -> dict[str, Any]:
    """should_answer=False 案例中，top1_score >= threshold（會被誤接受）的比率。"""
    considered = [c for c in cases if not c.should_answer]
    fps = [
        c
        for c in considered
        if c.top1_score is not None and c.top1_score >= threshold
    ]
    return {
        "threshold": threshold,
        "rate": rate(len(fps), len(considered)),
        "false_positives": len(fps),
        "considered": len(considered),
        "false_positive_ids": [c.id for c in fps],
    }


def group_by_query_type(cases: list[CaseResult]) -> dict[str, Any]:
    """每種 query_type 的樣本數與主要指標。"""
    groups: dict[str, list[CaseResult]] = {}
    for c in cases:
        key = c.query_type or "unlabeled"
        groups.setdefault(key, []).append(c)

    out: dict[str, Any] = {}
    for key, members in sorted(groups.items()):
        hit1, hit1_n, hit1_d = hit_rate(members, "doc_hit_at_1")
        hit5, hit5_n, hit5_d = hit_rate(members, "doc_hit_at_5")
        group_mrr, mrr_n = mrr(members)
        out[key] = {
            "count": len(members),
            "case_ids": [c.id for c in members],
            "hit_at_1": hit1,
            "hit_at_1_considered": hit1_d,
            "hit_at_5": hit5,
            "hit_at_5_considered": hit5_d,
            "mrr": group_mrr,
            "mrr_considered": mrr_n,
        }
    return out


def threshold_report(cases: list[CaseResult], threshold: float) -> dict[str, Any]:
    """沿用原 evaluate_retrieval.py 的 threshold 校準統計（行為不變）。"""
    answerable = [c for c in cases if c.should_answer]
    unanswerable = [c for c in cases if not c.should_answer]

    def accepted(c: CaseResult) -> bool:
        return c.top1_score is not None and c.top1_score >= threshold

    ans_accepted = [c for c in answerable if accepted(c)]
    ans_rejected = [c for c in answerable if not accepted(c)]
    un_accepted = [c for c in unanswerable if accepted(c)]
    un_rejected = [c for c in unanswerable if not accepted(c)]

    accepted_all = ans_accepted + un_accepted
    true_accept = [c for c in ans_accepted if c.doc_hit_at_1]
    precision = rate(len(true_accept), len(accepted_all))

    return {
        "threshold": threshold,
        "answerable_accept_rate": rate(len(ans_accepted), len(answerable)),
        "answerable_false_reject_rate": rate(len(ans_rejected), len(answerable)),
        "unanswerable_refuse_rate": rate(len(un_rejected), len(unanswerable)),
        "unanswerable_false_accept_rate": rate(len(un_accepted), len(unanswerable)),
        "precision": precision,
        "false_accepted_unanswerable_ids": [c.id for c in un_accepted],
        "false_rejected_answerable_ids": [c.id for c in ans_rejected],
    }


def aggregate_metrics(
    cases: list[CaseResult], configured_threshold: float
) -> dict[str, Any]:
    """組出 Baseline 報告用的完整聚合指標區塊。"""
    answerable = [c for c in cases if c.should_answer]
    unanswerable = [c for c in cases if not c.should_answer]

    hit1, hit1_n, hit1_d = hit_rate(cases, "doc_hit_at_1")
    hit3, hit3_n, hit3_d = hit_rate(cases, "doc_hit_at_3")
    hit5, hit5_n, hit5_d = hit_rate(cases, "doc_hit_at_5")
    sec1, sec1_n, sec1_d = hit_rate(cases, "section_hit_at_1")
    page1, page1_n, page1_d = hit_rate(cases, "page_hit_at_1")
    mrr_value, mrr_n = mrr(cases)

    latencies = [float(c.retrieval_time_ms) for c in cases]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    ans_scores = [c.top1_score for c in answerable if c.top1_score is not None]
    un_scores = [c.top1_score for c in unanswerable if c.top1_score is not None]

    return {
        "total_questions": len(cases),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        # 保留既有已驗證指標
        "hit_at_1": hit1,
        "hit_at_1_considered": hit1_d,
        "hit_at_3": hit3,
        "hit_at_3_considered": hit3_d,
        "section_hit_at_1": sec1,
        "section_hit_considered": sec1_d,
        "page_hit_at_1": page1,
        "page_hit_considered": page1_d,
        "average_latency_ms": avg_latency,
        "p50_latency_ms": round(percentile(latencies, 0.50), 2),
        "p95_latency_ms": round(percentile(latencies, 0.95), 2),
        "answerable_score_summary": summary(ans_scores),
        "unanswerable_score_summary": summary(un_scores),
        # Slice 0 新增最低限度指標
        "hit_at_5": hit5,
        "recall_at_5": hit5,  # 單一相關文件下 Recall@5 == Hit@5
        "hit_at_5_considered": hit5_d,
        "mrr": mrr_value,
        "mrr_considered": mrr_n,
        "exact_code_top1": exact_code_top1_accuracy(cases),
        "confusable_code_error": confusable_error(cases),
        "unanswerable_false_positive": unanswerable_false_positive_rate(
            cases, configured_threshold
        ),
        "by_query_type": group_by_query_type(cases),
    }
