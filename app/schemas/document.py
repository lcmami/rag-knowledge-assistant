"""文件 API 回應 schema（規格 §19.3~§19.5，本切片 §二、§四）。"""
from __future__ import annotations

from pydantic import BaseModel

from app.repositories.document_repository import DocumentRecord


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str
    created_at: str

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "DocumentUploadResponse":
        return cls(
            document_id=record.id,
            filename=record.filename,
            chunk_count=record.chunk_count,
            status=record.status,
            created_at=record.created_at,
        )


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    error_message: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "DocumentListItem":
        return cls(
            document_id=record.id,
            filename=record.filename,
            file_type=record.file_type,
            file_size=record.file_size,
            status=record.status,
            chunk_count=record.chunk_count,
            error_message=record.error_message,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int
