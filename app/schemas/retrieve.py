"""Retrieval-only API schema（規格 §19.6 / Phase 4 §5）。"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.services.retrieval_service import RetrievedChunk

_MAX_QUESTION_CHARS = 2000


class RetrieveRequest(BaseModel):
    # question 必填；缺少時 FastAPI 直接回 422。
    question: str
    # top_k 選填；None 表示沿用設定預設值。最小 1；上限於 endpoint 依設定驗證。
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("question 不可為空白。")
        if len(trimmed) > _MAX_QUESTION_CHARS:
            raise ValueError(f"question 長度不可超過 {_MAX_QUESTION_CHARS} 字元。")
        return trimmed


class RetrieveResultItem(BaseModel):
    source_id: str
    document_id: str
    filename: str
    page: int | None
    section: str | None
    chunk_index: int
    score: float
    text: str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "RetrieveResultItem":
        return cls(
            source_id=chunk.source_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
            score=round(float(chunk.score), 4),
            text=chunk.text,
        )


class RetrieveResponse(BaseModel):
    results: list[RetrieveResultItem]
    retrieval_time_ms: int
