"""Canonical Document Representation（v1.1 Slice 1 §五）。

Gold Label 不得直接綁「原始 PDF 全文件 char offset」；改綁**版本化、可重現的
normalized canonical text** 座標。此模組定義三個核心資料結構：

- :class:`ProvenanceSpan`：一段可追蹤來源範圍（供 Chunk / Evidence 共用語意）。
- :class:`CanonicalSegment`：canonical 文件的一個段落單元（1:1 對應 Loader 的
  DocumentElement），帶版本化 normalized text 與 offset。
- :class:`CanonicalDocument`：整份版本化 canonical 表示。

offset scope 規則（§五）：
- PDF 使用 ``page`` scope：offset 於**每頁**各自從 0 起算（page-local）。
- Markdown / TXT 使用 ``document`` scope：offset 於**整份文件** canonical text 起算。

``char_start`` inclusive、``char_end`` exclusive；offset 指向 normalized canonical
text，不指向不可重現的原始 bytes。相同輸入 + 相同 loader/normalization 版本 → 輸出
deterministic。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# offset scope 兩種取值（§五）
OFFSET_SCOPE_PAGE = "page"
OFFSET_SCOPE_DOCUMENT = "document"


@dataclass
class ProvenanceSpan:
    """一段可追蹤的原文來源範圍。

    同時作為 Chunk 的多來源 provenance 與（概念上）Evidence 的定位單元；欄位對齊
    規格 §六，讓跨頁 / 跨段的 Chunk 不會被偽裝成單一連續 offset。
    """

    document_sha256: str
    canonical_text_version: str
    loader_version: str
    page: int | None
    section: str | None
    paragraph_index: int
    offset_scope: str
    char_start: int
    char_end: int
    # v1.1 Slice 3（§十二）：core = 核心 evidence 範圍；context = 相鄰段落 context。
    # 向後相容的選填欄位——舊資料缺此欄位時預設 ``core``；序列化時**僅在非預設值才輸出**，
    # 確保 v1.0 / Slice 1-2 既有 payload byte-identical（正式 Qdrant payload 不變）。
    role: str = "core"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "document_sha256": self.document_sha256,
            "canonical_text_version": self.canonical_text_version,
            "loader_version": self.loader_version,
            "page": self.page,
            "section": self.section,
            "paragraph_index": self.paragraph_index,
            "offset_scope": self.offset_scope,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }
        if self.role != "core":
            d["role"] = self.role
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProvenanceSpan":
        return cls(
            document_sha256=raw["document_sha256"],
            canonical_text_version=raw["canonical_text_version"],
            loader_version=raw["loader_version"],
            page=raw.get("page"),
            section=raw.get("section"),
            paragraph_index=raw["paragraph_index"],
            offset_scope=raw["offset_scope"],
            char_start=raw["char_start"],
            char_end=raw["char_end"],
            role=raw.get("role", "core"),
        )


@dataclass
class CanonicalSegment:
    """canonical 文件的一個段落單元（1:1 對應一個 DocumentElement）。"""

    segment_id: str
    page: int | None
    section: str | None
    paragraph_index: int
    normalized_text: str
    offset_scope: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "page": self.page,
            "section": self.section,
            "paragraph_index": self.paragraph_index,
            "normalized_text": self.normalized_text,
            "offset_scope": self.offset_scope,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class CanonicalDocument:
    """整份版本化、可重現的 canonical 表示。"""

    document_sha256: str
    filename: str
    document_type: str  # pdf | markdown | txt
    canonical_text_version: str
    loader_version: str
    segments: list[CanonicalSegment] = field(default_factory=list)

    def canonical_text(self, page: int | None = None) -> str:
        """重建某個 offset scope 的 canonical text（供除錯 / 對位驗證）。

        - ``document`` scope：``page`` 省略，回傳整份 doc-global canonical text。
        - ``page`` scope：傳入 ``page`` 回傳該頁 page-local canonical text。
        """
        segs = [
            s
            for s in self.segments
            if page is None or s.page == page
        ]
        segs.sort(key=lambda s: s.char_start)
        return "\n\n".join(s.normalized_text for s in segs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_sha256": self.document_sha256,
            "filename": self.filename,
            "document_type": self.document_type,
            "canonical_text_version": self.canonical_text_version,
            "loader_version": self.loader_version,
            "segments": [s.to_dict() for s in self.segments],
        }
