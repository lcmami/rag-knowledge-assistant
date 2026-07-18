"""共用測試 fixtures 與輕量假件（本階段：回歸測試固化）。

Unit / API 測試不得下載 Embedding 模型、依賴真實 Qdrant 或呼叫 LLM，
因此提供 Fake Embedding 與 Fake Vector Store，並以暫存 SQLite / 上傳目錄運行。
"""
from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.repositories.document_repository import DocumentRepository
from app.services.indexing_service import IndexingService


class FakeEmbeddingService:
    """以字元數近似 token 數；產生固定維度的零向量，不載入任何模型。"""

    dimension = 8

    def count_tokens(self, text: str) -> int:
        return len(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, question: str) -> list[float]:
        return [0.0] * self.dimension


class FakeVectorStore:
    """記錄 upsert / delete 呼叫的記憶體向量庫，可注入失敗以驗證回滾與刪除順序。"""

    def __init__(self) -> None:
        self.points: dict[str, list] = {}
        self.upsert_should_fail = False
        self.delete_should_fail = False
        # 若外部指派一個 list，delete 時會 append "qdrant"（供刪除順序測試）
        self.order: list[str] | None = None

    def upsert_chunks(self, chunks, vectors) -> None:
        if self.upsert_should_fail:
            raise VectorStoreError("注入的 upsert 失敗（測試）")
        for chunk in chunks:
            self.points.setdefault(chunk.document_id, []).append(chunk)

    def delete_by_document_id(self, document_id: str) -> None:
        if self.delete_should_fail:
            raise VectorStoreError("注入的 delete 失敗（測試）")
        if self.order is not None:
            self.order.append("qdrant")
        self.points.pop(document_id, None)

    def count_by_document_id(self, document_id: str) -> int:
        return len(self.points.get(document_id, []))


@pytest.fixture
def settings(tmp_path) -> Settings:
    """全部指向暫存路徑；qdrant_url 不會被 unit 測試使用。"""
    return Settings(
        app_env="test",
        upload_dir=str(tmp_path / "uploads"),
        sqlite_path=str(tmp_path / "metadata.db"),
        qdrant_url=":memory:",
        chunk_size_chars=600,
        chunk_overlap_chars=100,
        embedding_max_tokens=480,
    )


@pytest.fixture
def repository(settings) -> DocumentRepository:
    return DocumentRepository(settings.sqlite_path)


@pytest.fixture
def fake_embedding() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def indexing_service(settings, repository, fake_embedding, fake_vector_store) -> IndexingService:
    return IndexingService(
        settings=settings,
        repository=repository,
        embedding_service=fake_embedding,
        vector_store=fake_vector_store,
    )
