"""Chunk（規格 §9.2 / v1.1 Slice 1 §六、§十一）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models.canonical import ProvenanceSpan


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
    # v1.1 Slice 1（§六）：一個 Chunk 可對應多個 provenance span（跨段 / 跨 section
    # 時保留各自來源，不偽裝成單一連續 offset）。以下欄位皆為新增，預設值確保 v1.0
    # 既有呼叫端與舊 Qdrant point 相容。
    provenance_spans: list[ProvenanceSpan] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    canonical_text_version: str | None = None
    loader_version: str | None = None
    identifier_normalization_version: str | None = None

    def to_payload(self) -> dict:
        """轉為 Qdrant Point payload（規格 §9.3 / v1.1 §十一）。

        v1.0 既有欄位保持不變；新增 v1.1 欄位為非破壞式擴充，舊 point 缺欄位時檢索端
        以 ``.get`` 讀取不會 crash。
        """
        return {
            # --- v1.0 既有欄位（不變）---
            "document_id": self.document_id,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "created_at": self.created_at,
            # --- v1.1 新增（非破壞）---
            "provenance_spans": [s.to_dict() for s in self.provenance_spans],
            "identifiers": list(self.identifiers),
            "canonical_text_version": self.canonical_text_version,
            "loader_version": self.loader_version,
            "identifier_normalization_version": self.identifier_normalization_version,
        }
