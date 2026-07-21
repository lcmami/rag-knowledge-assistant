"""Slice 4 retrieval strategy 單元測試共用 fake（不需 Qdrant / BGE）。"""
from __future__ import annotations

from evaluation.v11.retrieval import Candidate, RetrievalConfig, RetrievalRequest
from evaluation.v11.retrieval import build_retrieval_strategy


def cand(document_id, chunk_index, score, identifiers=None, **over):
    kw = dict(
        document_id=document_id,
        chunk_index=chunk_index,
        text=f"{document_id}-{chunk_index}",
        vector_score=score,
        identifiers=list(identifiers or []),
    )
    kw.update(over)
    return Candidate(**kw)


class FakeBackend:
    """記錄呼叫、依固定候選集合回應 dense / identifier search。"""

    def __init__(self, dense=None, identifier=None):
        self._dense = list(dense or [])
        self._identifier = list(identifier or [])
        self.dense_calls: list[int] = []
        self.identifier_calls: list[tuple[tuple, int]] = []

    def dense_search(self, query_vector, limit):
        self.dense_calls.append(limit)
        return [_clone(c) for c in self._dense[:limit]]

    def identifier_search(self, query_vector, identifiers, limit):
        self.identifier_calls.append((tuple(identifiers), limit))
        return [_clone(c) for c in self._identifier[:limit]]


def _clone(c: Candidate) -> Candidate:
    return Candidate(
        document_id=c.document_id, chunk_index=c.chunk_index, text=c.text,
        vector_score=c.vector_score, page=c.page, section=c.section,
        filename=c.filename, file_hash=c.file_hash, identifiers=list(c.identifiers),
        provenance_spans=list(c.provenance_spans),
        canonical_text_version=c.canonical_text_version, sources=list(c.sources),
    )


def run_strategy(strategy_name, config_raw, query, backend, query_vector=None):
    cfg = RetrievalConfig.from_raw(config_raw)
    strat = build_retrieval_strategy(strategy_name)
    req = RetrievalRequest(
        query=query, query_vector=query_vector or [0.1, 0.2],
        collection_name="c", config=cfg,
    )
    return strat.retrieve(req, backend)
