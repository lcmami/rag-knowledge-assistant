"""Retrieval-only API（規格 §19.6 / Phase 4 §5）。

POST /api/retrieve：直接執行檢索，不呼叫 LLM，供 evaluation 與獨立驗證使用。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings
from app.dependencies import get_retrieval_service, get_settings
from app.schemas.retrieve import RetrieveRequest, RetrieveResponse, RetrieveResultItem
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api", tags=["retrieve"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    payload: RetrieveRequest,
    settings: Settings = Depends(get_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrieveResponse:
    # top_k：未提供沿用設定預設；上限沿用設定（§5）。
    top_k = payload.top_k if payload.top_k is not None else settings.retrieval_default_top_k
    if top_k > settings.retrieval_max_top_k:
        raise HTTPException(
            status_code=422,
            detail=f"top_k 不可超過 {settings.retrieval_max_top_k}。",
        )

    result = retrieval_service.retrieve(payload.question, top_k)
    return RetrieveResponse(
        results=[RetrieveResultItem.from_chunk(c) for c in result.results],
        retrieval_time_ms=result.retrieval_time_ms,
    )
