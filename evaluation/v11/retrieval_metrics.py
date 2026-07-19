"""Retrieval Metrics（Slice 2 §十二 / §十三）。

以 Slice 1 :class:`~evaluation.v11.evidence_matcher.EvidenceMatcher` 判定每個 retrieved
chunk 相對 case Gold Evidence 的 relevance level，據此計算版本化指標。純函式、可單元測試，
不依賴 Qdrant / 模型。

核心原則：
- 不可計算的指標**不得**偽裝為 0：以 ``computable=false`` + ``not_computable_reason`` 標示。
- 分母為 0 不得 crash：回傳 ``computable=false``。
- 每個 metric 附 ``numerator`` / ``denominator`` / ``sample_count``。
- 所有可調參數集中於版本化 :class:`~evaluation.v11.metric_config.MetricConfig`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.models.canonical import ProvenanceSpan
from app.services.identifier_extractor import IdentifierExtractor
from evaluation.v11.case_schema import EvaluationCaseV11, EvidenceSpan
from evaluation.v11.evidence_matcher import EvidenceMatcher
from evaluation.v11.metric_config import DEFAULT_METRIC_CONFIG, MetricConfig
from evaluation.v11.run_models import TrialMetric


# --------------------------------------------------------------------------
# 輸入資料結構
# --------------------------------------------------------------------------
@dataclass
class RankedResult:
    """一筆排序後的 retrieved chunk（含 provenance，供 EvidenceMatcher 判定）。"""

    rank: int
    document_id: Optional[str]
    document_sha256: Optional[str]
    filename: Optional[str]
    section: Optional[str]
    page: Optional[int]
    chunk_index: Optional[int]
    score: Optional[float]
    text: str
    provenance_spans: list[dict] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    canonical_text_version: Optional[str] = None


@dataclass
class CaseRetrieval:
    """一個 case 的檢索輸出（排序後結果 + latency）。"""

    case: EvaluationCaseV11
    results: list[RankedResult]
    latency_ms: int


@dataclass
class IndexChunk:
    """Trial Index Universe 中的一個 Chunk（供 trial_index_gold NDCG 的 IDCG 計算）。

    僅保留 EvidenceMatcher 判定 relevance 所需欄位；**不含**文件原文全文（Report 不輸出）。
    """

    document_id: Optional[str]
    document_sha256: Optional[str]
    chunk_index: Optional[int]
    text: str
    page: Optional[int]
    section: Optional[str]
    provenance_spans: list[dict] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    canonical_text_version: Optional[str] = None


def _span_from_payload(d: dict) -> ProvenanceSpan:
    """由 payload / 測試 dict 寬鬆建立 ProvenanceSpan（缺欄位以 None/0 退化）。"""
    return ProvenanceSpan(
        document_sha256=d.get("document_sha256"),
        canonical_text_version=d.get("canonical_text_version"),
        loader_version=d.get("loader_version"),
        page=d.get("page"),
        section=d.get("section"),
        paragraph_index=d.get("paragraph_index", 0),
        offset_scope=d.get("offset_scope"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
    )


# --------------------------------------------------------------------------
# 每個 result 的比對
# --------------------------------------------------------------------------
@dataclass
class ScoredResult:
    result: RankedResult
    relevance_level: int
    match_reasons: list[str]
    coverage: float
    matched_evidence_ids: list[str]
    is_duplicate: bool = False


@dataclass
class CaseEvaluation:
    case_id: str
    query_type: Optional[str]
    answerable: bool
    latency_ms: int
    first_relevant_rank: Optional[int]
    case_status: str
    scored: list[ScoredResult]
    # trial_index_gold NDCG 明細（evaluate_trial 於取得 index_universe 後回填）。
    ndcg: Optional[dict[str, Any]] = None

    # 供 trial_case_results 持久化
    def top_results_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "rank": s.result.rank,
                "document_id": s.result.document_id,
                "filename": s.result.filename,
                "section": s.result.section,
                "page": s.result.page,
                "chunk_index": s.result.chunk_index,
                "score": round(s.result.score, 4) if s.result.score is not None else None,
                "relevance_level": s.relevance_level,
                "identifiers": list(s.result.identifiers),
                "is_duplicate": s.is_duplicate,
            }
            for s in self.scored
        ]

    def evidence_match_payload(self) -> dict[str, Any]:
        return {
            "matches": [
                {
                    "rank": s.result.rank,
                    "relevance_level": s.relevance_level,
                    "match_reasons": s.match_reasons,
                    "coverage": round(s.coverage, 4),
                    "matched_evidence_ids": s.matched_evidence_ids,
                }
                for s in self.scored
            ],
            "ndcg": self.ndcg,
        }


def score_case(
    cr: CaseRetrieval,
    matcher: EvidenceMatcher,
    cfg: MetricConfig,
) -> CaseEvaluation:
    """比對單一 case 的所有 result，標記 relevance / duplicate / first_relevant_rank。"""
    evidence = cr.case.expected_evidence
    scored: list[ScoredResult] = []
    for r in cr.results:
        spans = [_span_from_payload(d) for d in r.provenance_spans]
        mr = matcher.match(evidence, spans, r.text)
        scored.append(
            ScoredResult(
                result=r,
                relevance_level=mr.relevance_level,
                match_reasons=mr.match_reasons,
                coverage=mr.coverage,
                matched_evidence_ids=mr.matched_evidence_ids,
            )
        )

    _mark_duplicates(scored, cfg)

    first_relevant_rank: Optional[int] = None
    for s in scored:
        if s.relevance_level >= 1:
            first_relevant_rank = s.result.rank
            break

    if cr.case.answerable:
        case_status = "relevant_hit" if first_relevant_rank is not None else "miss"
    else:
        case_status = "unanswerable_false_positive" if scored else "unanswerable_ok"

    return CaseEvaluation(
        case_id=cr.case.id,
        query_type=cr.case.query_type,
        answerable=cr.case.answerable,
        latency_ms=cr.latency_ms,
        first_relevant_rank=first_relevant_rank,
        case_status=case_status,
        scored=scored,
    )


# --------------------------------------------------------------------------
# duplicate / near-duplicate（§12.9）
# --------------------------------------------------------------------------
def _char_ngrams(text: str, n: int) -> set[str]:
    t = "".join(text.split()).casefold()
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _mark_duplicates(scored: list[ScoredResult], cfg: MetricConfig) -> None:
    """從 rank 2 起，與任一較早結果 exact/near duplicate 者標記 is_duplicate（只量測，不重排）。"""
    ngram_cache: dict[int, set[str]] = {}

    def ngrams(i: int) -> set[str]:
        if i not in ngram_cache:
            ngram_cache[i] = _char_ngrams(scored[i].result.text or "", cfg.near_duplicate_ngram)
        return ngram_cache[i]

    for i in range(1, len(scored)):
        ri = scored[i].result
        for j in range(i):
            rj = scored[j].result
            exact = (
                ri.document_id is not None
                and ri.document_id == rj.document_id
                and ri.chunk_index is not None
                and ri.chunk_index == rj.chunk_index
            )
            near = (
                ri.document_id is not None
                and ri.document_id == rj.document_id
                and ri.section == rj.section
                and _jaccard(ngrams(i), ngrams(j)) >= cfg.near_duplicate_jaccard_threshold
            )
            if exact or near:
                scored[i].is_duplicate = True
                break


# --------------------------------------------------------------------------
# metric helpers
# --------------------------------------------------------------------------
def _metric(
    name: str,
    numerator: Optional[float],
    denominator: Optional[float],
    *,
    scope: str = "aggregate",
    query_type: Optional[str] = None,
    sample_count: Optional[int] = None,
    computable: Optional[bool] = None,
    reason: Optional[str] = None,
) -> TrialMetric:
    if computable is None:
        computable = bool(denominator)
    value = (numerator / denominator) if (computable and denominator) else None
    if not computable and reason is None:
        reason = "no_computable_sample"
    return TrialMetric(
        trial_id="",  # runner 填入
        metric_name=name,
        scope=scope,
        query_type=query_type,
        value=round(value, 6) if value is not None else None,
        numerator=numerator if computable else None,
        denominator=denominator if computable else None,
        sample_count=sample_count if sample_count is not None else (
            int(denominator) if denominator is not None else None
        ),
        computable=computable,
        not_computable_reason=None if computable else reason,
    )


def _count_metric(
    name: str,
    count: int,
    *,
    sample_count: int,
    computable: bool,
    reason: str,
) -> TrialMetric:
    """count 型指標：value 為絕對計數（非 rate）；不可計算時 value=None。"""
    return TrialMetric(
        trial_id="",
        metric_name=name,
        scope="aggregate",
        query_type=None,
        value=float(count) if computable else None,
        numerator=float(count) if computable else None,
        denominator=float(sample_count) if computable else None,
        sample_count=sample_count,
        computable=computable,
        not_computable_reason=None if computable else reason,
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


# --------------------------------------------------------------------------
# 個別指標計算（皆吃「已 score 的 case 清單」）
# --------------------------------------------------------------------------
def _hit_at_k(cases: list[CaseEvaluation], k: int, level: int) -> tuple[int, int]:
    """answerable case 中 top-k 有 result relevance_level>=level 的比率 → (hits, considered)。"""
    considered = [c for c in cases if c.answerable]
    hits = 0
    for c in considered:
        if any(s.relevance_level >= level for s in c.scored if s.result.rank <= k):
            hits += 1
    return hits, len(considered)


def _mrr(cases: list[CaseEvaluation]) -> tuple[float, int]:
    considered = [c for c in cases if c.answerable]
    if not considered:
        return 0.0, 0
    total = sum(1.0 / c.first_relevant_rank if c.first_relevant_rank else 0.0 for c in considered)
    return total, len(considered)


def _dcg(gains: list[int]) -> float:
    import math

    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _chunk_in_docs(uc: IndexChunk, gold_docs: set[str]) -> bool:
    if uc.document_sha256 in gold_docs:
        return True
    return any(d.get("document_sha256") in gold_docs for d in uc.provenance_spans)


def _chunk_version_matches(uc: IndexChunk, gold_versions: set[str]) -> bool:
    if uc.canonical_text_version in gold_versions:
        return True
    return any(d.get("canonical_text_version") in gold_versions for d in uc.provenance_spans)


def _compute_case_ndcg(
    ce: CaseEvaluation,
    gold: list[EvidenceSpan],
    universe_levels: Optional[list[int]],
    index_universe: Optional[list[IndexChunk]],
    cfg: MetricConfig,
) -> dict[str, Any]:
    """trial_index_gold NDCG@K（§二/§四）。

    IDCG 由 Trial Index Universe 全部 Chunk 的 gain 降序取前 K 計算；DCG 由實際 Retrieved
    Top-K 的 gain（rank 序）計算。不可計算時 value=None + 明確 reason，不以 0 假裝。
    """
    k = cfg.ndcg_k

    def nc(reason: str) -> dict[str, Any]:
        return {
            "computable": False, "value": None, "dcg": None, "idcg": None,
            "relevant_chunk_count": None, "not_computable_reason": reason,
            "variant": cfg.ndcg_variant,
        }

    if not ce.answerable:
        return nc("unanswerable_case")
    if index_universe is None:
        return nc("missing_index_universe")
    if not any(e.has_offset for e in gold):
        # anchor-only：無法可靠產生 graded relevance
        return nc("insufficient_graded_evidence")

    gold_docs = {e.document_sha256 for e in gold}
    gold_versions = {e.canonical_text_version for e in gold}
    doc_chunks = [uc for uc in index_universe if _chunk_in_docs(uc, gold_docs)]
    if doc_chunks and not any(_chunk_version_matches(uc, gold_versions) for uc in doc_chunks):
        return nc("version_mismatch")

    if universe_levels is None:  # 防禦：理論上不會發生（前述 gate 已保證計算）
        return nc("missing_index_universe")

    relevant = sum(1 for lvl in universe_levels if lvl >= 1)
    ideal_gains = sorted((cfg.gain(lvl) for lvl in universe_levels), reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    if relevant == 0 or idcg <= 0:
        return nc("no_relevant_chunks_in_trial_index")

    retrieved_gains = [cfg.gain(s.relevance_level) for s in ce.scored if s.result.rank <= k]
    dcg = _dcg(retrieved_gains)
    return {
        "computable": True,
        "value": round(dcg / idcg, 6),
        "dcg": round(dcg, 6),
        "idcg": round(idcg, 6),
        "relevant_chunk_count": relevant,
        "not_computable_reason": None,
        "variant": cfg.ndcg_variant,
    }


def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _evidence_coverage_at_k(
    cases: list[CaseEvaluation],
    evidence_by_case: dict[str, list[EvidenceSpan]],
    cr_by_case: dict[str, CaseRetrieval],
    cfg: MetricConfig,
):
    """Evidence Coverage@k（§12.4）：Level-2 result provenance 對 gold offset span 的覆蓋。

    回傳 (sum_case_coverage, computable_case_count, not_computable_count)。
    """
    k = cfg.top_k
    total = 0.0
    computable_cases = 0
    not_computable = 0
    for c in cases:
        if not c.answerable:
            continue
        gold = evidence_by_case.get(c.case_id, [])
        offset_gold = [e for e in gold if e.has_offset]
        if not offset_gold:
            not_computable += 1  # anchor-only：不可計算，不假造覆蓋率
            continue
        # 收集 top-k Level-2 result 的 provenance span
        level2_spans: list[ProvenanceSpan] = []
        for s in c.scored:
            if s.result.rank <= k and s.relevance_level == 2:
                level2_spans.extend(_span_from_payload(d) for d in s.result.provenance_spans)

        per_span_cov: list[float] = []
        for ev in offset_gold:
            gold_len = max(1, ev.char_end - ev.char_start)
            intervals: list[tuple[int, int]] = []
            for p in level2_spans:
                if p.document_sha256 != ev.document_sha256:
                    continue
                if p.canonical_text_version != ev.canonical_text_version:
                    continue
                if p.char_start is None or p.char_end is None:
                    continue
                if ev.offset_scope is not None and p.offset_scope != ev.offset_scope:
                    continue
                if ev.offset_scope == "page" and ev.page is not None and p.page != ev.page:
                    continue
                lo = max(ev.char_start, p.char_start)
                hi = min(ev.char_end, p.char_end)
                if hi > lo:
                    intervals.append((lo, hi))
            covered = _merge_intervals(intervals)
            per_span_cov.append(covered / gold_len)
        # 同題先對所有 evidence span macro average（§12.4）
        case_cov = sum(per_span_cov) / len(per_span_cov)
        total += case_cov
        computable_cases += 1
    return total, computable_cases, not_computable


def _relevant_section_precision_at_5(cases: list[CaseEvaluation], cfg: MetricConfig):
    """§12.5：answerable case 的 top-5 中 relevance_level>=1 數 / 5（分母固定 5）。"""
    considered = [c for c in cases if c.answerable]
    if not considered:
        return None, 0
    k = cfg.top_k
    total = 0.0
    for c in considered:
        rel = sum(1 for s in c.scored if s.result.rank <= k and s.relevance_level >= 1)
        total += rel / k
    return total, len(considered)


def _exact_code_top1(
    cases: list[CaseEvaluation],
    cr_by_case: dict[str, CaseRetrieval],
    extractor: IdentifierExtractor,
):
    """§12.6：exact_code + answerable 的 Top-1 正確率（含 identifier 一致性檢查）。"""
    considered = [c for c in cases if c.answerable and c.query_type == "exact_code"]
    hits = 0
    miss_ids: list[str] = []
    for c in considered:
        cr = cr_by_case[c.case_id]
        query_ids = set(extractor.extract_normalized(cr.case.question))
        top1 = next((s for s in c.scored if s.result.rank == 1), None)
        ok = False
        if top1 is not None and top1.relevance_level >= 1:
            if query_ids:
                ok = bool(query_ids & set(top1.result.identifiers))
            else:
                ok = True
        if ok:
            hits += 1
        else:
            miss_ids.append(c.case_id)
    return hits, len(considered), miss_ids


def _confusable_errors(
    cases: list[CaseEvaluation],
    cr_by_case: dict[str, CaseRetrieval],
    extractor: IdentifierExtractor,
):
    """§12.7：錯誤 identifier 的 Level-0 result 排在第一個 relevant 之前。"""
    details: list[dict[str, Any]] = []
    considered_ids: list[str] = []
    for c in cases:
        cr = cr_by_case[c.case_id]
        query_ids = set(extractor.extract_normalized(cr.case.question))
        is_considered = c.query_type == "confusable" or (
            c.query_type == "exact_code" and bool(query_ids)
        )
        if not is_considered:
            continue
        considered_ids.append(c.case_id)
        frr = c.first_relevant_rank
        # 找排名最前、帶「非 query identifier」的 Level-0 result
        for s in c.scored:
            if s.relevance_level != 0:
                continue
            conflicting = [i for i in s.result.identifiers if i not in query_ids]
            if not conflicting:
                continue
            if frr is None or s.result.rank < frr:
                details.append(
                    {
                        "case_id": c.case_id,
                        "query_identifier": sorted(query_ids),
                        "conflicting_identifier": conflicting[0],
                        "wrong_rank": s.result.rank,
                        "first_relevant_rank": frr,
                    }
                )
                break
    return details, considered_ids


def _unanswerable_false_positives(cases: list[CaseEvaluation]):
    considered = [c for c in cases if not c.answerable]
    fps = [c for c in considered if c.scored]
    return len(fps), len(considered), [c.case_id for c in fps]


def _duplicate_ratio(cases: list[CaseEvaluation], cfg: MetricConfig):
    """§12.9：對每個有結果的 case 計 duplicate ratio，聚合 macro average。"""
    considered = [c for c in cases if c.scored]
    if not considered:
        return None, 0
    k = cfg.top_k
    total = 0.0
    denom = max(k - 1, 1)
    for c in considered:
        top = [s for s in c.scored if s.result.rank <= k]
        dup = sum(1 for s in top if s.is_duplicate)
        total += dup / denom
    return total, len(considered)


# --------------------------------------------------------------------------
# 對外主入口
# --------------------------------------------------------------------------
@dataclass
class TrialEvaluation:
    case_evaluations: list[CaseEvaluation]
    metrics: list[TrialMetric]


def evaluate_trial(
    case_retrievals: list[CaseRetrieval],
    *,
    index_universe: Optional[list[IndexChunk]] = None,
    matcher: Optional[EvidenceMatcher] = None,
    metric_config: MetricConfig = DEFAULT_METRIC_CONFIG,
    identifier_extractor: Optional[IdentifierExtractor] = None,
    point_count: Optional[int] = None,
    index_build_time_ms: Optional[int] = None,
) -> TrialEvaluation:
    cfg = metric_config
    matcher = matcher or EvidenceMatcher(offset_coverage_threshold=cfg.offset_coverage_threshold)
    extractor = identifier_extractor or IdentifierExtractor()

    scored_cases = [score_case(cr, matcher, cfg) for cr in case_retrievals]
    cr_by_case = {cr.case.id: cr for cr in case_retrievals}
    evidence_by_case = {cr.case.id: cr.case.expected_evidence for cr in case_retrievals}

    # --- trial_index_gold NDCG：對 Trial Index Universe 全部 Chunk 判 relevance（IDCG）---
    # 預先轉一次 provenance / text，供每題重用（不為每題重新檢索或重新 scroll）。
    universe_prepared: Optional[list[tuple[list, str]]] = None
    if index_universe is not None:
        universe_prepared = [
            ([_span_from_payload(d) for d in uc.provenance_spans], uc.text)
            for uc in index_universe
        ]
    for ce in scored_cases:
        gold = evidence_by_case.get(ce.case_id, [])
        universe_levels: Optional[list[int]] = None
        if universe_prepared is not None and ce.answerable and any(e.has_offset for e in gold):
            universe_levels = [
                matcher.match(gold, provs, text).relevance_level
                for provs, text in universe_prepared
            ]
        ce.ndcg = _compute_case_ndcg(ce, gold, universe_levels, index_universe, cfg)

    metrics: list[TrialMetric] = []

    # --- Evidence Hit / Recall@K + Strong / Relevant context（§12.1）---
    for k in (1, 3, 5):
        strong_hits, ans = _hit_at_k(scored_cases, k, level=2)
        rel_hits, _ = _hit_at_k(scored_cases, k, level=1)
        metrics.append(_metric(f"strong_evidence_hit_at_{k}", strong_hits, ans,
                               computable=bool(ans), reason="no_answerable_cases"))
        metrics.append(_metric(f"relevant_context_hit_at_{k}", rel_hits, ans,
                               computable=bool(ans), reason="no_answerable_cases"))
        # Recall@K == relevant context hit（單一相關來源）
        metrics.append(_metric(f"recall_at_{k}", rel_hits, ans,
                               computable=bool(ans), reason="no_answerable_cases"))

    # --- MRR（§12.2）---
    mrr_sum, mrr_n = _mrr(scored_cases)
    metrics.append(_metric("mrr", mrr_sum, mrr_n, computable=bool(mrr_n),
                           reason="no_answerable_cases"))

    # --- Graded NDCG@5（trial_index_gold；只對 computable case macro average）---
    ndcg_vals = [ce.ndcg["value"] for ce in scored_cases if ce.ndcg and ce.ndcg["computable"]]
    ndcg_computable = len(ndcg_vals)
    ndcg_non_computable = sum(
        1 for ce in scored_cases if ce.ndcg and not ce.ndcg["computable"]
    )
    metrics.append(TrialMetric(
        trial_id="", metric_name=f"graded_ndcg_at_{cfg.ndcg_k}", scope="aggregate",
        query_type=None,
        value=round(sum(ndcg_vals) / ndcg_computable, 6) if ndcg_computable else None,
        numerator=round(sum(ndcg_vals), 6) if ndcg_computable else None,
        denominator=float(ndcg_computable) if ndcg_computable else None,
        sample_count=ndcg_computable,
        computable=bool(ndcg_computable),
        not_computable_reason=None if ndcg_computable else "no_computable_ndcg_cases",
    ))
    metrics.append(TrialMetric(
        trial_id="", metric_name="graded_ndcg_computable_count", scope="aggregate",
        query_type=None, value=float(ndcg_computable), sample_count=len(scored_cases),
        computable=True,
    ))
    metrics.append(TrialMetric(
        trial_id="", metric_name="graded_ndcg_non_computable_count", scope="aggregate",
        query_type=None, value=float(ndcg_non_computable), sample_count=len(scored_cases),
        computable=True,
    ))

    # --- Evidence Coverage@5（§12.4）---
    cov_sum, cov_n, cov_nc = _evidence_coverage_at_k(scored_cases, evidence_by_case, cr_by_case, cfg)
    metrics.append(_metric(f"evidence_coverage_at_{cfg.top_k}", cov_sum, cov_n,
                           sample_count=cov_n, computable=bool(cov_n),
                           reason="no_offset_grounded_gold_evidence"))

    # --- Relevant Section Precision@5（§12.5）---
    prec_sum, prec_n = _relevant_section_precision_at_5(scored_cases, cfg)
    metrics.append(_metric(f"relevant_section_precision_at_{cfg.top_k}", prec_sum, prec_n,
                           computable=bool(prec_n), reason="no_answerable_cases"))

    # --- Exact-code Top-1（§12.6）---
    ec_hits, ec_n, ec_miss = _exact_code_top1(scored_cases, cr_by_case, extractor)
    metrics.append(_metric("exact_code_top1_accuracy", ec_hits, ec_n,
                           computable=bool(ec_n), reason="no_exact_code_cases"))

    # --- Confusable error（§12.7）---
    conf_details, conf_considered = _confusable_errors(scored_cases, cr_by_case, extractor)
    conf_n = len(conf_considered)
    metrics.append(_count_metric("confusable_code_error_count", len(conf_details),
                                 sample_count=conf_n, computable=bool(conf_n),
                                 reason="no_confusable_cases"))
    metrics.append(_metric("confusable_code_error_rate", float(len(conf_details)),
                           float(conf_n) if conf_n else 0.0, sample_count=conf_n,
                           computable=bool(conf_n), reason="no_confusable_cases"))

    # --- Unanswerable false positive（§12.8）---
    fp, un_n, fp_ids = _unanswerable_false_positives(scored_cases)
    metrics.append(_count_metric("unanswerable_false_positive_count", fp,
                                 sample_count=un_n, computable=bool(un_n),
                                 reason="no_unanswerable_cases"))
    metrics.append(_metric("unanswerable_false_positive_rate", float(fp),
                           float(un_n) if un_n else 0.0, sample_count=un_n,
                           computable=bool(un_n), reason="no_unanswerable_cases"))

    # --- Duplicate / near-duplicate ratio（§12.9）---
    dup_sum, dup_n = _duplicate_ratio(scored_cases, cfg)
    metrics.append(_metric("duplicate_ratio", dup_sum, dup_n,
                           computable=bool(dup_n), reason="no_cases_with_results"))

    # --- Latency（§12.10）---
    latencies = [float(c.latency_ms) for c in scored_cases]
    n_cases = len(scored_cases)
    metrics.append(_metric("mean_retrieval_latency_ms", sum(latencies), float(n_cases),
                           computable=bool(n_cases), reason="no_cases"))
    metrics.append(TrialMetric(trial_id="", metric_name="p50_retrieval_latency_ms",
                               scope="aggregate", query_type=None,
                               value=round(_percentile(latencies, 0.50), 4) if latencies else None,
                               sample_count=n_cases, computable=bool(latencies),
                               not_computable_reason=None if latencies else "no_cases"))
    metrics.append(TrialMetric(trial_id="", metric_name="p95_retrieval_latency_ms",
                               scope="aggregate", query_type=None,
                               value=round(_percentile(latencies, 0.95), 4) if latencies else None,
                               sample_count=n_cases, computable=bool(latencies),
                               not_computable_reason=None if latencies else "no_cases"))

    # --- Index point count / build time（trial 層級，由 runner 提供）---
    if point_count is not None:
        metrics.append(TrialMetric(trial_id="", metric_name="index_point_count",
                                   scope="aggregate", query_type=None, value=float(point_count),
                                   sample_count=1, computable=True))
    if index_build_time_ms is not None:
        metrics.append(TrialMetric(trial_id="", metric_name="index_build_time_ms",
                                   scope="aggregate", query_type=None,
                                   value=float(index_build_time_ms), sample_count=1, computable=True))
    if index_universe is not None:
        # 供 §八.10 交叉驗證：Index Universe 數量應等於 Trial Point Count。
        metrics.append(TrialMetric(trial_id="", metric_name="index_universe_size",
                                   scope="aggregate", query_type=None,
                                   value=float(len(index_universe)), sample_count=1, computable=True))

    # --- Per query type（§十三）---
    metrics.extend(_per_query_type_metrics(scored_cases, cfg))

    return TrialEvaluation(case_evaluations=scored_cases, metrics=metrics)


def _per_query_type_metrics(cases: list[CaseEvaluation], cfg: MetricConfig) -> list[TrialMetric]:
    out: list[TrialMetric] = []
    types = sorted({c.query_type or "unlabeled" for c in cases})
    for qt in types:
        members = [c for c in cases if (c.query_type or "unlabeled") == qt]
        answerable = [c for c in members if c.answerable]
        for k in (1, 5):
            hits, ans = _hit_at_k(members, k, level=1)
            out.append(_metric(f"relevant_context_hit_at_{k}", hits, ans, scope="query_type",
                               query_type=qt, computable=bool(ans), reason="no_answerable_cases"))
        mrr_sum, mrr_n = _mrr(members)
        out.append(_metric("mrr", mrr_sum, mrr_n, scope="query_type", query_type=qt,
                           computable=bool(mrr_n), reason="no_answerable_cases"))
        out.append(TrialMetric(trial_id="", metric_name="case_count", scope="query_type",
                               query_type=qt, value=float(len(members)),
                               sample_count=len(members), computable=True))
    return out
