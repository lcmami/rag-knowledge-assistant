"""RAG 問答 API（規格 Phase 5 §10）。

POST /api/query：Retrieval → 雙層拒答 → Citation 驗證 → 組裝 Sources。
驗證規則沿用 /api/retrieve；LLM 錯誤由應用層例外經 error handler 轉為明確狀態碼。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings
from app.dependencies import get_rag_service, get_settings
from app.schemas.query import QueryRequest, QueryResponse, QuerySource
from app.services.rag_service import RagService

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    rag_service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    # top_k：未提供沿用設定預設；上限沿用設定（與 /api/retrieve 一致）。
    top_k = payload.top_k if payload.top_k is not None else settings.retrieval_default_top_k
    if top_k > settings.retrieval_max_top_k:
        raise HTTPException(
            status_code=422,
            detail=f"top_k 不可超過 {settings.retrieval_max_top_k}。",
        )

    result = await rag_service.answer(payload.question, top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[QuerySource.from_chunk(c) for c in result.sources],
        retrieval_time_ms=result.retrieval_time_ms,
        generation_time_ms=result.generation_time_ms,
    )
