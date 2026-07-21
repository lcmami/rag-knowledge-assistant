"""Qdrant Vector Store（規格 §13）。

- Collection 自動建立（Cosine + 動態維度）。
- 已存在則檢查距離函數與維度，不相容時不默默重建，拋 CollectionMismatchError。
- 為 document_id / file_hash 建 keyword payload index（§13.5）。
"""
from __future__ import annotations

from app.core.exceptions import CollectionMismatchError, VectorStoreError
from app.models.chunk import Chunk


class VectorStore:
    def __init__(self, client, collection: str):
        # client: qdrant_client.QdrantClient（server 或 :memory:）
        self.client = client
        self.collection = collection

    def ensure_collection(self, dimension: int) -> None:
        """啟動檢查（§13.4）：不存在則建立；存在則檢查相容性。"""
        from qdrant_client import models

        try:
            exists = self.client.collection_exists(self.collection)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"無法連線 Qdrant：{exc}") from exc

        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            self._create_payload_indexes()
            return

        info = self.client.get_collection(self.collection)
        existing_size = info.config.params.vectors.size
        existing_distance = info.config.params.vectors.distance
        if existing_size != dimension:
            raise CollectionMismatchError(
                "Qdrant Collection 向量維度與目前模型不符，需重新 indexing。",
                details={"expected": dimension, "actual": existing_size},
            )
        if existing_distance != models.Distance.COSINE:
            raise CollectionMismatchError(
                "Qdrant Collection 距離函數與預期不符。",
                details={"expected": "Cosine", "actual": str(existing_distance)},
            )

    # v1.1 Slice 1（§十一）：identifiers 以 keyword payload index 精確比對。
    # 僅於「新建 collection」時建立，既有正式 collection 不受影響（不會重建）。
    _KEYWORD_INDEX_FIELDS = ("document_id", "file_hash", "identifiers")

    def _create_payload_indexes(self) -> None:
        for field in self._KEYWORD_INDEX_FIELDS:
            self.create_keyword_index(field)

    def create_keyword_index(self, field_name: str) -> None:
        """在目前 collection 建立單一 keyword payload index（idempotent）。"""
        from qdrant_client import models

        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001 - index 可能已存在
            pass

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(id=chunk.id, vector=vector, payload=chunk.to_payload())
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            self.client.upsert(collection_name=self.collection, points=points)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"寫入 Qdrant 失敗：{exc}") from exc

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """回傳 [{score, payload}]。"""
        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant 檢索失敗：{exc}") from exc
        return [{"score": r.score, "payload": r.payload} for r in results]

    def delete_by_document_id(self, document_id: str) -> None:
        from qdrant_client import models

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"刪除 Qdrant Points 失敗：{exc}") from exc

    def count_by_document_id(self, document_id: str) -> int:
        from qdrant_client import models

        result = self.client.count(
            collection_name=self.collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
        )
        return result.count
