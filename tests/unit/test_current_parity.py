"""Current Strategy Golden Parity（v1.1 Slice 3 §八 / §二十二.2）。

``current`` strategy 委派正式 TextSplitter；此處以「已知穩定的 TextSplitter 輸出」為 golden
fixture，驗證 with/without Strategy Adapter 的 chunk 數量 / text / 順序 / page / section /
chunk_index 完全一致（**不**以固定向量分數作為 parity）。
"""
from __future__ import annotations

import pytest

from app.models.document_element import DocumentElement
from app.services.canonical_builder import build_canonical_document
from app.services.text_splitter import TextSplitter
from evaluation.v11.chunking import ChunkingConfig, ChunkingContext, build_strategy

# 涵蓋多段 / 多 section / 跨頁的語料（golden parity 用）。
CORPUS = [
    DocumentElement("設備警報總覽", None, "設備警報", "heading"),
    DocumentElement("第一段內容說明，警報碼 E102 需要注意處理流程與後續維護步驟安排。", None, "設備警報", "paragraph"),
    DocumentElement("第二段補充內容，關於 CX-500 泵浦的定期檢查與更換週期規範。", None, "設備警報", "paragraph"),
    DocumentElement("維護章節", None, "維護章節", "heading"),
    DocumentElement("維護章節的唯一段落敘述，涵蓋潤滑、緊固與清潔三項基本作業。", None, "維護章節", "paragraph"),
]

PDF_CORPUS = [
    DocumentElement("第一頁標題", 1, "第一頁標題", "heading"),
    DocumentElement("第一頁的內文段落，說明開機自檢流程。", 1, "第一頁標題", "paragraph"),
    DocumentElement("第二頁標題", 2, "第二頁標題", "heading"),
    DocumentElement("第二頁的內文段落，說明關機與保養。", 2, "第二頁標題", "paragraph"),
]


def _reference_chunks(elements, document_type):
    canonical = build_canonical_document(
        filename="doc", document_sha256="h", document_type=document_type, elements=elements
    )
    splitter = TextSplitter(chunk_size_chars=600, chunk_overlap_chars=100, max_tokens=480, count_tokens=len)
    return splitter.split(elements, document_id="d", filename="doc", file_hash="h", canonical=canonical)


def _adapter_chunks(elements, document_type, raw_config=None):
    canonical = build_canonical_document(
        filename="doc", document_sha256="h", document_type=document_type, elements=elements
    )
    cfg = ChunkingConfig.from_raw(raw_config or {"strategy": "current"})
    ctx = ChunkingContext(
        document_id="d", filename="doc", file_hash="h", canonical=canonical,
        elements=elements, config=cfg, count_tokens=len,
    )
    return build_strategy("current").chunk(ctx).chunks


def _fields(chunks):
    return [
        (c.text, c.page, c.section, c.chunk_index,
         tuple((s.page, s.section, s.paragraph_index, s.char_start, s.char_end) for s in c.provenance_spans))
        for c in chunks
    ]


@pytest.mark.parametrize("elements,dt", [(CORPUS, "markdown"), (PDF_CORPUS, "pdf")])
def test_current_adapter_parity_all_fields(elements, dt):
    ref = _reference_chunks(elements, dt)
    got = _adapter_chunks(elements, dt)
    assert len(got) == len(ref)
    assert _fields(got) == _fields(ref)


@pytest.mark.parametrize("elements,dt", [(CORPUS, "markdown"), (PDF_CORPUS, "pdf")])
def test_current_text_and_count_identical(elements, dt):
    ref = _reference_chunks(elements, dt)
    got = _adapter_chunks(elements, dt)
    assert [c.text for c in got] == [c.text for c in ref]
    assert [c.chunk_index for c in got] == [c.chunk_index for c in ref]
    assert [c.page for c in got] == [c.page for c in ref]
    assert [c.section for c in got] == [c.section for c in ref]


def test_current_provenance_roles_all_core():
    got = _adapter_chunks(CORPUS, "markdown")
    for c in got:
        assert all(s.role == "core" for s in c.provenance_spans)


def test_current_defaults_match_settings_defaults():
    # ChunkingConfig 預設 (600/100/480) == 正式 Settings 預設；确保 current baseline 不漂移。
    cfg = ChunkingConfig.from_raw({"strategy": "current"})
    assert (cfg.target_chars, cfg.overlap_chars, cfg.max_tokens) == (600, 100, 480)
