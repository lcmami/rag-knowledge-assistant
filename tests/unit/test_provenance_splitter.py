"""Chunk Provenance 單元測試（v1.1 Slice 1 §六 / §十三 Provenance）。

驗證 TextSplitter 在提供 canonical 時附上多 span provenance，且**切分輸出（chunk
text / 數量）與未提供 canonical 時完全一致**（provenance 純為附加 metadata）。
"""
from __future__ import annotations

from app.models.document_element import DocumentElement
from app.services.canonical_builder import build_canonical_document
from app.services.text_splitter import TextSplitter


def _splitter() -> TextSplitter:
    # count_tokens=len、token 上限極大 → 只由 chunk_size_chars 決定視窗合併行為
    return TextSplitter(
        chunk_size_chars=600,
        chunk_overlap_chars=100,
        max_tokens=100000,
        count_tokens=len,
    )


def _split(elements, document_type="markdown", with_canonical=True):
    canonical = None
    if with_canonical:
        canonical = build_canonical_document(
            filename="a", document_sha256="hash-a", document_type=document_type, elements=elements
        )
    return _splitter().split(
        elements,
        document_id="doc1",
        filename="a",
        file_hash="hash-a",
        canonical=canonical,
    )


MULTI_SECTION = [
    DocumentElement("設備警報", None, "設備警報", "heading"),
    DocumentElement("第一段內容說明。", None, "設備警報", "paragraph"),
    DocumentElement("第二段補充內容。", None, "設備警報", "paragraph"),
    DocumentElement("獨立章節", None, "獨立章節", "heading"),
    DocumentElement("獨立章節的唯一段落。", None, "獨立章節", "paragraph"),
]


def test_single_element_group_has_one_span():
    els = [DocumentElement("只有一段內容。", None, "唯一章節", "paragraph")]
    chunks = _split(els)
    assert len(chunks) == 1
    assert len(chunks[0].provenance_spans) == 1
    assert chunks[0].provenance_spans[0].section == "唯一章節"


def test_multi_element_chunk_has_multiple_spans():
    chunks = _split(MULTI_SECTION)
    # 「設備警報」section 三個 element 合併為一個 chunk → 多個 span
    alarm_chunks = [c for c in chunks if c.section == "設備警報"]
    assert len(alarm_chunks) == 1
    spans = alarm_chunks[0].provenance_spans
    assert len(spans) >= 2
    # 多 span 的 paragraph_index 互異，且都屬同一 section
    assert len({s.paragraph_index for s in spans}) == len(spans)
    assert {s.section for s in spans} == {"設備警報"}


def test_chunk_never_crosses_section():
    chunks = _split(MULTI_SECTION)
    for c in chunks:
        sections = {s.section for s in c.provenance_spans}
        assert len(sections) == 1, f"chunk 跨了多個 section：{sections}"


def test_cross_page_does_not_fake_continuous_offset():
    pdf_els = [
        DocumentElement("第一頁標題", 1, "第一頁標題", "heading"),
        DocumentElement("第一頁的內文段落。", 1, "第一頁標題", "paragraph"),
        DocumentElement("第二頁標題", 2, "第二頁標題", "heading"),
        DocumentElement("第二頁的內文段落。", 2, "第二頁標題", "paragraph"),
    ]
    chunks = _split(pdf_els, document_type="pdf")
    # 每個 chunk 的所有 span 必須同一頁（chunking 不跨頁；證明而非假設）
    for c in chunks:
        pages = {s.page for s in c.provenance_spans}
        assert len(pages) == 1, f"chunk 的 provenance 跨頁：{pages}"
        # page-local scope
        assert {s.offset_scope for s in c.provenance_spans} == {"page"}
    # 存在分屬不同頁的 chunk，但沒有任何單一 chunk 混合兩頁
    all_pages = {c.page for c in chunks}
    assert all_pages == {1, 2}


def test_provenance_carries_versions_and_sha():
    chunks = _split(MULTI_SECTION)
    for c in chunks:
        assert c.canonical_text_version == "v1"
        assert c.loader_version == "v1"
        for s in c.provenance_spans:
            assert s.document_sha256 == "hash-a"
            assert s.canonical_text_version == "v1"


def test_legacy_without_canonical_is_compatible():
    # 未提供 canonical（v1.0 行為）：無 provenance、版本為 None
    chunks = _split(MULTI_SECTION, with_canonical=False)
    for c in chunks:
        assert c.provenance_spans == []
        assert c.canonical_text_version is None
        assert c.loader_version is None
    # 舊 payload 可安全序列化，且缺欄位以 .get 讀取不 crash
    payload = chunks[0].to_payload()
    assert payload["provenance_spans"] == []
    assert payload.get("canonical_text_version") is None


def test_chunk_text_and_count_identical_with_or_without_canonical():
    with_c = _split(MULTI_SECTION, with_canonical=True)
    without_c = _split(MULTI_SECTION, with_canonical=False)
    assert [c.text for c in with_c] == [c.text for c in without_c]
    assert len(with_c) == len(without_c)


def test_payload_includes_v11_fields():
    chunks = _split(MULTI_SECTION)
    payload = chunks[0].to_payload()
    for key in (
        "provenance_spans",
        "identifiers",
        "canonical_text_version",
        "loader_version",
        "identifier_normalization_version",
    ):
        assert key in payload
    # v1.0 既有欄位保持
    for key in ("document_id", "filename", "file_hash", "page", "section", "chunk_index", "text"):
        assert key in payload
