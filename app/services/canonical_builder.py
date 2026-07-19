"""CanonicalDocumentBuilder（v1.1 Slice 1 §五 / architecture §7）。

輸入：Loader 產生的 ``list[DocumentElement]``（結構化，帶 page / section）。
輸出：版本化、可重現的 :class:`CanonicalDocument`。

- Segment 與 DocumentElement **1:1 對齊、同序**（不丟棄任何 element），供 TextSplitter
  以位置對映回 provenance。
- PDF → ``page`` scope：offset 於每頁各自從 0 起算；paragraph_index 於每頁各自從 0。
- Markdown / TXT → ``document`` scope：offset 於整份 doc-global canonical text；
  paragraph_index 於整份文件從 0。
- Segment 之間以 ``"\\n\\n"`` 連接構成 scope canonical text，故下一段起點為前一段
  ``char_end + 2``。
"""
from __future__ import annotations

from app.models.canonical import (
    OFFSET_SCOPE_DOCUMENT,
    OFFSET_SCOPE_PAGE,
    CanonicalDocument,
    CanonicalSegment,
)
from app.models.document_element import DocumentElement
from app.services.canonical_normalizer import (
    CANONICAL_TEXT_VERSION,
    LOADER_VERSION,
    normalize_text,
)

# scope canonical text 連接分隔字元長度（"\n\n"）。
_JOIN_LEN = 2

# document_type 對應 offset scope。
_PDF_TYPES = {"pdf"}


def _offset_scope(document_type: str) -> str:
    return OFFSET_SCOPE_PAGE if document_type in _PDF_TYPES else OFFSET_SCOPE_DOCUMENT


def build_canonical_document(
    *,
    filename: str,
    document_sha256: str,
    document_type: str,
    elements: list[DocumentElement],
) -> CanonicalDocument:
    """由 DocumentElement 清單建立版本化 CanonicalDocument。"""
    scope = _offset_scope(document_type)
    segments: list[CanonicalSegment] = []

    # page scope 需各頁獨立 cursor / paragraph_index；document scope 用單一鍵 None。
    cursor: dict[int | None, int] = {}
    para_index: dict[int | None, int] = {}

    for global_index, el in enumerate(elements):
        normalized = normalize_text(el.text)
        # scope key：page scope 依 page 分桶；document scope 全部同一桶。
        key = el.page if scope == OFFSET_SCOPE_PAGE else None
        start = cursor.get(key, 0)
        end = start + len(normalized)
        p_index = para_index.get(key, 0)

        segments.append(
            CanonicalSegment(
                segment_id=f"seg-{global_index}",
                page=el.page,
                section=el.section,
                paragraph_index=p_index,
                normalized_text=normalized,
                offset_scope=scope,
                char_start=start,
                char_end=end,
            )
        )

        # 下一段落於同一 scope 桶內，接續 "\n\n"。
        cursor[key] = end + _JOIN_LEN
        para_index[key] = p_index + 1

    return CanonicalDocument(
        document_sha256=document_sha256,
        filename=filename,
        document_type=document_type,
        canonical_text_version=CANONICAL_TEXT_VERSION,
        loader_version=LOADER_VERSION,
        segments=segments,
    )
