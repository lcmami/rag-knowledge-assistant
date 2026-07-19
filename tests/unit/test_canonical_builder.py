"""Canonical Document 單元測試（v1.1 Slice 1 §十三 Canonical Document）。"""
from __future__ import annotations

import unicodedata

from app.models.canonical import OFFSET_SCOPE_DOCUMENT, OFFSET_SCOPE_PAGE
from app.models.document_element import DocumentElement
from app.services.canonical_builder import build_canonical_document
from app.services.canonical_normalizer import (
    CANONICAL_TEXT_VERSION,
    LOADER_VERSION,
    normalize_text,
)


def _md_elements() -> list[DocumentElement]:
    return [
        DocumentElement("E102 感測器通訊異常", None, "E102 感測器通訊異常", "heading"),
        DocumentElement("E102 表示感測器通訊異常。", None, "E102 感測器通訊異常", "paragraph"),
        DocumentElement("E203 溫度過高警報", None, "E203 溫度過高警報", "heading"),
    ]


def _pdf_elements() -> list[DocumentElement]:
    return [
        DocumentElement("第一頁標題", 1, "第一頁標題", "heading"),
        DocumentElement("第一頁內文較長一些。", 1, "第一頁標題", "paragraph"),
        DocumentElement("第二頁標題", 2, "第二頁標題", "heading"),
        DocumentElement("第二頁內文。", 2, "第二頁標題", "paragraph"),
    ]


def test_deterministic_same_input_same_output():
    a = build_canonical_document(
        filename="a.md", document_sha256="h", document_type="markdown", elements=_md_elements()
    )
    b = build_canonical_document(
        filename="a.md", document_sha256="h", document_type="markdown", elements=_md_elements()
    )
    assert a.to_dict() == b.to_dict()


def test_line_ending_normalization():
    crlf = normalize_text("第一行\r\n第二行\r\n")
    lf = normalize_text("第一行\n第二行\n")
    assert crlf == lf
    assert "\r" not in crlf


def test_unicode_nfc_normalization():
    # 分解形（NFD）與組合形（NFC）正規化後一致
    nfc = "é"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc  # 前提：兩者位元不同
    assert normalize_text(nfd) == normalize_text(nfc)


def test_null_and_repeated_whitespace_removed():
    # null 直接移除（A 與 B 相鄰）；多個水平空白（空白 / Tab）壓成單一半形空白
    out = normalize_text("A\x00B   C\tD")
    assert "\x00" not in out
    assert "AB C D" == out


def test_paragraph_boundary_preserved():
    out = normalize_text("第一段\n第二段")
    assert "\n" in out  # 不得移除所有換行 / 段落結構


def test_markdown_uses_document_scope_offsets_accumulate():
    doc = build_canonical_document(
        filename="a.md", document_sha256="h", document_type="markdown", elements=_md_elements()
    )
    scopes = {s.offset_scope for s in doc.segments}
    assert scopes == {OFFSET_SCOPE_DOCUMENT}
    # doc-global：offset 隨段落累加，不重設
    assert doc.segments[0].char_start == 0
    assert doc.segments[1].char_start == doc.segments[0].char_end + 2
    assert doc.segments[2].char_start == doc.segments[1].char_end + 2
    # paragraph_index 於整份文件遞增
    assert [s.paragraph_index for s in doc.segments] == [0, 1, 2]


def test_pdf_uses_page_local_offsets_reset_per_page():
    doc = build_canonical_document(
        filename="a.pdf", document_sha256="h", document_type="pdf", elements=_pdf_elements()
    )
    assert {s.offset_scope for s in doc.segments} == {OFFSET_SCOPE_PAGE}
    p1 = [s for s in doc.segments if s.page == 1]
    p2 = [s for s in doc.segments if s.page == 2]
    # 每頁各自從 0 起算（page-local）
    assert p1[0].char_start == 0
    assert p2[0].char_start == 0
    # 每頁 paragraph_index 各自從 0
    assert [s.paragraph_index for s in p1] == [0, 1]
    assert [s.paragraph_index for s in p2] == [0, 1]


def test_versions_attached():
    doc = build_canonical_document(
        filename="a.md", document_sha256="h", document_type="markdown", elements=_md_elements()
    )
    assert doc.canonical_text_version == CANONICAL_TEXT_VERSION
    assert doc.loader_version == LOADER_VERSION


def test_paragraph_index_stable_across_builds():
    e = _md_elements()
    a = build_canonical_document(filename="a.md", document_sha256="h", document_type="markdown", elements=e)
    b = build_canonical_document(filename="a.md", document_sha256="h", document_type="markdown", elements=e)
    assert [s.paragraph_index for s in a.segments] == [s.paragraph_index for s in b.segments]


def test_canonical_text_reconstruction_matches_offsets():
    doc = build_canonical_document(
        filename="a.pdf", document_sha256="h", document_type="pdf", elements=_pdf_elements()
    )
    # page-local 重建：第 2 頁 canonical text 的 [char_start:char_end) 對應 segment normalized_text
    page2_text = doc.canonical_text(page=2)
    for s in [s for s in doc.segments if s.page == 2]:
        assert page2_text[s.char_start : s.char_end] == s.normalized_text
