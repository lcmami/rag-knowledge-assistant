"""Chunk（規格 §9.2）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    file_hash: str
    page: int | None
    section: str | None
    chunk_index: int
    text: str
    created_at: str

    def to_payload(self) -> dict:
        """轉為 Qdrant Point payload（規格 §9.3）。"""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "created_at": self.created_at,
        }
