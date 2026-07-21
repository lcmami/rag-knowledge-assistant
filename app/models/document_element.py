"""DocumentElement（規格 §9.1）。

Loader 不得只回傳整份文件字串，須先解析為結構化 Element。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentElement:
    text: str
    page: int | None
    section: str | None
    element_type: str  # heading | paragraph | table | code | list
    # v1.1 Slice 1（§六）：可追蹤 provenance 的 canonical 座標（由
    # CanonicalDocumentBuilder 回填；Loader 產出時預設 None，保持 v1.0 相容）。
    paragraph_index: int | None = None
    offset_scope: str | None = None  # page | document
    char_start: int | None = None
    char_end: int | None = None
