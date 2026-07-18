"""共用資源建構與 dependency injection（規格 §21）。

服務於 lifespan 初始化一次並掛在 app.state；透過 provider 取用，
讓測試可替換 Embedding / VectorStore / LLM / Repository。
"""
from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.services.llm_client import LLMClient
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore


def create_qdrant_client(url: str):
    """依 QDRANT_URL 建立 client；":memory:" 使用內嵌模式（供本機驗證 / 測試）。"""
    from qdrant_client import QdrantClient

    if url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url)


def build_services(settings: Settings) -> dict:
    """建立切片所需的核心服務。"""
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        max_tokens=settings.embedding_max_tokens,
    )
    client = create_qdrant_client(settings.qdrant_url)
    vector_store = VectorStore(client, settings.qdrant_collection)
    vector_store.ensure_collection(embedding_service.dimension)
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        score_threshold=settings.retrieval_score_threshold,
    )
    document_repository = DocumentRepository(settings.sqlite_path)
    indexing_service = IndexingService(
        settings=settings,
        repository=document_repository,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    # LLM Client 於 lifespan 初始化一次；未設定時 configured=False，/api/query 回明確 503。
    llm_client = LLMClient(settings)
    rag_service = RagService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
    )
    return {
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "retrieval_service": retrieval_service,
        "document_repository": document_repository,
        "indexing_service": indexing_service,
        "llm_client": llm_client,
        "rag_service": rag_service,
    }


# --- providers ---
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


def get_document_repository(request: Request) -> DocumentRepository:
    return request.app.state.document_repository


def get_indexing_service(request: Request) -> IndexingService:
    return request.app.state.indexing_service


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service
