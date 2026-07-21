"""QdrantSearchBackend（v1.1 Slice 4 §十一）——真實 dense / identifier-filtered vector search。

實作 :class:`~evaluation.v11.retrieval.base.SearchBackend`，只讀取隔離 Trial collection：

- ``dense_search``：與正式 :class:`app.services.vector_store.VectorStore` 相同的 ``query_points``
  （query=vector, limit, with_payload），確保 dense parity。
- ``identifier_search``：同一 query vector + ``identifiers`` keyword payload filter（MatchAny），
  取回 raw vector_score（**不**使用 scroll 順序、**不**填假分數）。

**不修改**正式 app 服務；identifier keyword index 於 build_index 建立隔離 collection 時已存在。
"""
from __future__ import annotations

from evaluation.v11.retrieval.base import (
    SOURCE_DENSE,
    SOURCE_IDENTIFIER,
    Candidate,
)


def _payload_to_candidate(score: float, payload: dict, source: str) -> Candidate:
    payload = payload or {}
    return Candidate(
        document_id=payload.get("document_id"),
        chunk_index=payload.get("chunk_index"),
        text=payload.get("text") or "",
        vector_score=float(score),
        page=payload.get("page"),
        section=payload.get("section"),
        filename=payload.get("filename"),
        file_hash=payload.get("file_hash"),
        identifiers=list(payload.get("identifiers") or []),
        provenance_spans=list(payload.get("provenance_spans") or []),
        canonical_text_version=payload.get("canonical_text_version"),
        sources=[source],
    )


class QdrantSearchBackend:
    """對單一隔離 Trial collection 的 dense / identifier-filtered vector search。"""

    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection = collection_name

    def dense_search(self, query_vector: list[float], limit: int) -> list[Candidate]:
        points = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points
        return [_payload_to_candidate(p.score, p.payload, SOURCE_DENSE) for p in points]

    def identifier_search(
        self, query_vector: list[float], identifiers: list[str], limit: int
    ) -> list[Candidate]:
        if not identifiers:
            return []
        from qdrant_client import models

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="identifiers",
                    match=models.MatchAny(any=list(identifiers)),
                )
            ]
        )
        points = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        return [_payload_to_candidate(p.score, p.payload, SOURCE_IDENTIFIER) for p in points]
