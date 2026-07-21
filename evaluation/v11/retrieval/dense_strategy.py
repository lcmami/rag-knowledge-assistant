"""DenseRetrievalStrategy（v1.1 Slice 4 §十）——包裝既有 dense retrieval，parity。

複製正式 :class:`app.services.retrieval_service.RetrievalService` 的 dense 語意：

    candidate_limit = max(top_k, min(top_k * multiplier, cap))
    hits = dense_search(query_vector, candidate_limit)
    filtered = [h for h in hits if vector_score >= score_threshold]
    filtered.sort(vector_score desc, **stable**)   # 與 RetrievalService 相同：ties 保留後端順序
    top_k

``ranking_score == vector_score``、boost/penalty 皆為 0（§十.5/6）。Query 有無 identifier 都
不影響 dense（§十.7）。**不**套用 §十六 的 identifier tie-breaker，以維持與正式檢索 byte-parity。
"""
from __future__ import annotations

from evaluation.v11.retrieval.base import (
    ADMISSION_DENSE_THRESHOLD,
    SOURCE_DENSE,
    Candidate,
    RankedRetrievalResult,
    RetrievalRequest,
    RetrievalStrategy,
    SearchBackend,
    StrategyRetrievalOutcome,
)
from evaluation.v11.retrieval.config import DENSE_RANKING_VERSION


class DenseRetrievalStrategy(RetrievalStrategy):
    name = "dense"

    def retrieve(
        self, request: RetrievalRequest, backend: SearchBackend
    ) -> StrategyRetrievalOutcome:
        cfg = request.config
        candidate_limit = cfg.candidate_limit(cfg.top_k)
        candidates = backend.dense_search(request.query_vector, candidate_limit)

        admitted: list[Candidate] = []
        rejected = 0
        for c in candidates:
            if c.vector_score >= cfg.score_threshold:
                admitted.append(c)
            else:
                rejected += 1

        # 與 RetrievalService 相同：stable sort by vector_score desc（ties 保留後端順序）。
        admitted.sort(key=lambda c: c.vector_score, reverse=True)
        admitted = admitted[: cfg.top_k]

        results = [
            RankedRetrievalResult(
                rank=rank,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
                page=c.page,
                section=c.section,
                filename=c.filename,
                file_hash=c.file_hash,
                identifiers=list(c.identifiers),
                provenance_spans=list(c.provenance_spans),
                canonical_text_version=c.canonical_text_version,
                vector_score=c.vector_score,
                ranking_score=c.vector_score,  # §十.5：ranking_score == vector_score
                exact_identifier_matches=[],
                conflicting_identifiers=[],
                exact_match_boost=0.0,  # §十.6
                confusable_penalty=0.0,
                candidate_sources=[SOURCE_DENSE],
                admission_reason=ADMISSION_DENSE_THRESHOLD,
            )
            for rank, c in enumerate(admitted, start=1)
        ]

        return StrategyRetrievalOutcome(
            results=results,
            query_identifiers=[],  # dense 不使用 identifier
            strategy_name=self.name,
            ranking_version=DENSE_RANKING_VERSION,
            dense_candidate_count=len(candidates),
            identifier_candidate_count=0,
            union_candidate_count=len(candidates),
            deduplicated_count=len(candidates),
            exact_match_candidate_count=0,
            confusable_candidate_count=0,
            threshold_rejected_count=rejected,
            identifier_floor_rejected_count=0,
        )
