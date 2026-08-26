"""IdentifierAwareRetrievalStrategy（v1.1 Slice 4 §十一–§十六）。

正式流程：

    query → extract/normalize query identifiers（Slice 1 共用 IdentifierExtractor）
          → embed query 一次（由呼叫端提供 query_vector）
          → dense vector search  ∪  identifier-filtered vector search
          → candidate union → dedup（document_id + chunk_index）
          → score components（exact boost / confusable penalty；不改寫 raw vector_score）
          → admission（exact candidate 用 identifier_vector_floor；其餘用 score_threshold）
          → deterministic ranking（ranking_score desc, exact, vector_score, identity）
          → final Top-K

Query 無 identifier 時退化為與 dense **等價**（§十.7 / §21.4）：不做 keyword search、不套
boost/penalty，直接沿用 dense 排序（含其 tie-break），確保 no-identifier 結果與 dense 一致。
"""
from __future__ import annotations

from typing import Optional

from app.services.identifier_extractor import IdentifierExtractor
from evaluation.v11.retrieval.base import (
    ADMISSION_DENSE_THRESHOLD,
    ADMISSION_EXACT_IDENTIFIER_FLOOR,
    ADMISSION_REJECTED_BELOW_IDENTIFIER_FLOOR,
    ADMISSION_REJECTED_BELOW_THRESHOLD,
    SOURCE_DENSE,
    SOURCE_IDENTIFIER,
    Candidate,
    RankedRetrievalResult,
    RetrievalRequest,
    RetrievalStrategy,
    SearchBackend,
    StrategyRetrievalOutcome,
)
from evaluation.v11.retrieval.dense_strategy import DenseRetrievalStrategy
from evaluation.v11.retrieval.rerank_stage import maybe_rerank

# 兩次查詢對同一 point 的 raw vector_score 若有極小浮點差異，超過此值才記 warning（§十二.3）。
_SCORE_EPSILON = 1e-6


class IdentifierAwareRetrievalStrategy(RetrievalStrategy):
    name = "identifier_aware"

    def __init__(self, extractor: Optional[IdentifierExtractor] = None) -> None:
        # Slice 1 共用 normalizer（§十三：不得另寫一套 regex）。
        self._extractor = extractor or IdentifierExtractor()

    def retrieve(
        self, request: RetrievalRequest, backend: SearchBackend
    ) -> StrategyRetrievalOutcome:
        cfg = request.config
        query_ids = self._extractor.extract_normalized(request.query)
        query_id_set = set(query_ids)

        # --- 無 identifier → 退化為 dense 等價（§十.7 / §21.4）---
        if not query_id_set:
            # dense 內部已自行套用 rerank（若啟用），此處只需原樣承接其結果與統計，
            # 不得重複 rerank，也不得把 rerank 統計丟掉。
            dense = DenseRetrievalStrategy().retrieve(request, backend)
            return StrategyRetrievalOutcome(
                results=dense.results,
                query_identifiers=[],
                strategy_name=self.name,
                ranking_version=cfg.ranking_version,
                dense_candidate_count=dense.dense_candidate_count,
                identifier_candidate_count=0,
                union_candidate_count=dense.union_candidate_count,
                deduplicated_count=dense.deduplicated_count,
                exact_match_candidate_count=0,
                confusable_candidate_count=0,
                threshold_rejected_count=dense.threshold_rejected_count,
                identifier_floor_rejected_count=0,
                rerank_strategy=dense.rerank_strategy,
                admitted_count=dense.admitted_count,
                rerank_window_count=dense.rerank_window_count,
                rerank_tail_count=dense.rerank_tail_count,
                rerank_latency_ms=dense.rerank_latency_ms,
                rerank_cache_hits=dense.rerank_cache_hits,
                rerank_cache_misses=dense.rerank_cache_misses,
            )

        # --- 1. Dense candidates（§十一.1）---
        dense_candidates = backend.dense_search(
            request.query_vector, cfg.candidate_limit(cfg.top_k)
        )
        # --- 2. Identifier-filtered candidates（§十一.2；同一 query vector）---
        identifier_candidates = backend.identifier_search(
            request.query_vector, query_ids, cfg.identifier_candidate_cap
        )

        # --- 3. Union + dedup by identity（§十二）---
        merged, warnings = self._union(dense_candidates, identifier_candidates)

        # --- 4. Score components + admission（§十四 / §十五）---
        exact_count = 0
        confusable_count = 0
        rejected_threshold = 0
        rejected_floor = 0
        admitted: list[_Scored] = []
        for c in merged:
            cand_ids = list(c.identifiers)
            exact_matches = [i for i in query_ids if i in cand_ids]  # deterministic order
            conflicting = [i for i in cand_ids if i not in query_id_set]
            has_exact = bool(exact_matches)
            has_other = bool(conflicting)
            if has_exact:
                exact_count += 1

            boost = cfg.exact_match_boost if has_exact else 0.0
            # §十四：penalty 僅在 query 有 id、candidate 無任何 exact match、且含其他 id 時。
            penalty = cfg.confusable_penalty if (not has_exact and has_other) else 0.0
            if penalty > 0:
                confusable_count += 1
            ranking_score = c.vector_score + boost - penalty

            # --- Admission（§十五：以 raw vector_score 判定，不用 ranking_score）---
            if has_exact:
                if c.vector_score >= cfg.identifier_vector_floor:
                    reason = ADMISSION_EXACT_IDENTIFIER_FLOOR
                else:
                    rejected_floor += 1
                    continue
            else:
                if c.vector_score >= cfg.score_threshold:
                    reason = ADMISSION_DENSE_THRESHOLD
                else:
                    rejected_threshold += 1
                    continue

            admitted.append(
                _Scored(
                    candidate=c,
                    exact_matches=exact_matches,
                    conflicting=conflicting,
                    has_exact=has_exact,
                    boost=boost,
                    penalty=penalty,
                    ranking_score=ranking_score,
                    admission_reason=reason,
                )
            )

        # --- 5. Deterministic ranking（§十六）→ Top-K ---
        admitted.sort(key=_ranking_key)
        # v1.2.1：rerank 啟用時，A 完整保留交給 rerank stage 切 window/tail；
        # 未啟用時維持 v1.1 行為（直接截斷至 top_k），輸出 byte-identical。
        admitted_count = len(admitted)
        if not getattr(cfg, "is_rerank_enabled", False):
            admitted = admitted[: cfg.top_k]

        results = [
            RankedRetrievalResult(
                rank=rank,
                document_id=s.candidate.document_id,
                chunk_index=s.candidate.chunk_index,
                text=s.candidate.text,
                page=s.candidate.page,
                section=s.candidate.section,
                filename=s.candidate.filename,
                file_hash=s.candidate.file_hash,
                identifiers=list(s.candidate.identifiers),
                provenance_spans=list(s.candidate.provenance_spans),
                canonical_text_version=s.candidate.canonical_text_version,
                vector_score=s.candidate.vector_score,
                ranking_score=s.ranking_score,
                exact_identifier_matches=list(s.exact_matches),
                conflicting_identifiers=list(s.conflicting),
                exact_match_boost=s.boost,
                confusable_penalty=s.penalty,
                candidate_sources=list(s.candidate.sources),
                admission_reason=s.admission_reason,
            )
            for rank, s in enumerate(admitted, start=1)
        ]

        outcome = StrategyRetrievalOutcome(
            results=results,
            query_identifiers=list(query_ids),
            strategy_name=self.name,
            ranking_version=cfg.ranking_version,
            dense_candidate_count=len(dense_candidates),
            identifier_candidate_count=len(identifier_candidates),
            union_candidate_count=len(dense_candidates) + len(identifier_candidates),
            deduplicated_count=len(merged),
            exact_match_candidate_count=exact_count,
            confusable_candidate_count=confusable_count,
            threshold_rejected_count=rejected_threshold,
            identifier_floor_rejected_count=rejected_floor,
            warnings=warnings,
        )
        return maybe_rerank(outcome, request, admitted_count=admitted_count)

    # ------------------------------------------------------------------
    def _union(
        self, dense: list[Candidate], identifier: list[Candidate]
    ) -> tuple[list[Candidate], list[str]]:
        """Union + dedup by identity（§十二）。deterministic：dense 先、identifier 後。

        同一 chunk 兩來源：合併 candidate_sources、取較高 raw vector_score、保留較完整 metadata。
        """
        warnings: list[str] = []
        order: list[tuple] = []
        by_identity: dict[tuple, Candidate] = {}

        def _ingest(c: Candidate, source: str) -> None:
            key = c.identity()
            existing = by_identity.get(key)
            if existing is None:
                merged = Candidate(
                    document_id=c.document_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    vector_score=c.vector_score,
                    page=c.page,
                    section=c.section,
                    filename=c.filename,
                    file_hash=c.file_hash,
                    identifiers=list(c.identifiers),
                    provenance_spans=list(c.provenance_spans),
                    canonical_text_version=c.canonical_text_version,
                    sources=[source],
                )
                by_identity[key] = merged
                order.append(key)
                return
            # 合併：source、較高 score、較完整 metadata（§十二.2/3/4）。
            if source not in existing.sources:
                existing.sources.append(source)
            if abs(existing.vector_score - c.vector_score) > _SCORE_EPSILON:
                warnings.append(
                    f"score_divergence:{c.document_id}:{c.chunk_index}"
                )
            existing.vector_score = max(existing.vector_score, c.vector_score)
            _fill_missing(existing, c)

        for c in dense:
            _ingest(c, SOURCE_DENSE)
        for c in identifier:
            _ingest(c, SOURCE_IDENTIFIER)

        return [by_identity[k] for k in order], sorted(set(warnings))


def _fill_missing(dst: Candidate, src: Candidate) -> None:
    """保留較完整 metadata：dst 缺欄位時以 src 補（不覆蓋既有非空值）。"""
    if not dst.text and src.text:
        dst.text = src.text
    for attr in ("page", "section", "filename", "file_hash", "canonical_text_version"):
        if getattr(dst, attr) is None and getattr(src, attr) is not None:
            setattr(dst, attr, getattr(src, attr))
    if not dst.identifiers and src.identifiers:
        dst.identifiers = list(src.identifiers)
    if not dst.provenance_spans and src.provenance_spans:
        dst.provenance_spans = list(src.provenance_spans)


class _Scored:
    """admission 後暫存的評分結果（供排序）。"""

    __slots__ = (
        "candidate", "exact_matches", "conflicting", "has_exact",
        "boost", "penalty", "ranking_score", "admission_reason",
    )

    def __init__(self, candidate, exact_matches, conflicting, has_exact,
                 boost, penalty, ranking_score, admission_reason):
        self.candidate = candidate
        self.exact_matches = exact_matches
        self.conflicting = conflicting
        self.has_exact = has_exact
        self.boost = boost
        self.penalty = penalty
        self.ranking_score = ranking_score
        self.admission_reason = admission_reason


def _ranking_key(s: "_Scored") -> tuple:
    """Deterministic 排序鍵（§十六，全精度、不依賴 Qdrant / set 順序）：

    1. ranking_score desc
    2. exact match bool desc
    3. vector_score desc
    4. document_id asc（stable identity）
    5. chunk_index asc
    """
    return (
        -s.ranking_score,
        not s.has_exact,               # False(有 exact) 排在 True 前
        -s.candidate.vector_score,
        s.candidate.document_id or "",
        s.candidate.chunk_index if s.candidate.chunk_index is not None else -1,
    )
