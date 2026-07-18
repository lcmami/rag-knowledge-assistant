"""Retrieval Service（規格 §16 / Phase 4 §5、§6）。

User Question → 加 Query Instruction → Query Embedding → Qdrant 候選檢索
→ Score Threshold 過濾 → 依 score 由高到低排序 → 取前 top_k → 分配 Source ID。
不呼叫 LLM。

候選數策略（§6.1）：先取 candidate_limit = min(top_k × multiplier, cap) 筆候選，
經 Threshold 過濾後再取前 top_k，避免先取 top_k 才過濾導致結果過少。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    source_id: str
    document_id: str
    filename: str
    page: int | None
    section: str | None
    chunk_index: int
    score: float
    text: str


@dataclass
class RetrievalResult:
    results: list[RetrievedChunk]
    retrieval_time_ms: int


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        score_threshold: float,
        candidate_multiplier: int = 3,
        candidate_cap: int = 50,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.score_threshold = score_threshold
        self.candidate_multiplier = candidate_multiplier
        self.candidate_cap = candidate_cap

    def _candidate_limit(self, top_k: int) -> int:
        return max(top_k, min(top_k * self.candidate_multiplier, self.candidate_cap))

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        start = time.perf_counter()

        query_vector = self.embedding_service.embed_query(question)
        # 先取較多候選再過濾（§6.1）
        candidate_limit = self._candidate_limit(top_k)
        hits = self.vector_store.search(query_vector, candidate_limit)

        # Threshold 過濾（§6）
        filtered = [h for h in hits if h["score"] >= self.score_threshold]
        # 依 score 由高到低排序（§5 規則：結果按 score 排序）
        filtered.sort(key=lambda h: h["score"], reverse=True)
        # 取前 top_k
        filtered = filtered[:top_k]

        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(filtered, start=1):
            payload = hit["payload"]
            results.append(
                RetrievedChunk(
                    source_id=f"S{rank}",  # Source ID 由後端依排序分配（§5 規則 1、2）
                    document_id=payload.get("document_id"),
                    filename=payload.get("filename"),
                    page=payload.get("page"),
                    section=payload.get("section"),
                    chunk_index=payload.get("chunk_index"),
                    score=hit["score"],
                    text=payload.get("text"),
                )
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return RetrievalResult(results=results, retrieval_time_ms=elapsed_ms)
