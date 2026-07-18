"""PDF Loader 單元測試（Phase 4 §10.1）。

在記憶體中以 PyMuPDF 建立測試用 PDF，不依賴 Docker / Qdrant / 模型 / LLM。
另針對 Repository 內的真實範例 PDF 做中文 roundtrip 與頁碼驗證（檔案不存在時 skip）。
"""
from __future__ import annotations

import os

import pytest

from app.core.exceptions import (
    CorruptPdfError,
    EncryptedPdfError,
    NoExtractableTextError,
)
from app.services.pdf_loader import load_pdf
from app.services.text_splitter import TextSplitter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_PDF = os.path.join(ROOT, "sample_docs", "equipment_maintenance_manual.pdf")


def _make_text_pdf(pages: list[str]) -> bytes:
    """以 Helvetica 建立多頁文字型 PDF（純結構測試，內容用 ASCII 即可）。"""
    import pymupdf

    doc = pymupdf.open()
    for content in pages:
        page = doc.new_page()
        if content:
            page.insert_text((72, 100), content, fontsize=12, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


def _make_empty_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()  # 一頁但完全沒有文字
    data = doc.tobytes()
    doc.close()
    return data


def _fake_count_tokens(text: str) -> int:
    return len(text)


# 1. 可解析文字型 PDF
def test_parses_text_pdf():
    raw = _make_text_pdf(["Hello alarm manual content", "Second page maintenance"])
    elements = load_pdf(raw)
    assert elements
    assert all(el.text.strip() for el in elements)


# 2. 頁碼從 1 開始
def test_page_numbers_are_one_based():
    raw = _make_text_pdf(["First page only text"])
    elements = load_pdf(raw)
    pages = {el.page for el in elements}
    assert pages == {1}
    assert min(pages) == 1  # 不得暴露 0-based index


# 3. 不同頁面產生不同 page metadata
def test_different_pages_have_distinct_page_metadata():
    raw = _make_text_pdf(["PAGEONEONLY body", "PAGETWOONLY body", "PAGETHREEONLY body"])
    elements = load_pdf(raw)
    pages = sorted({el.page for el in elements})
    assert pages == [1, 2, 3]
    p1_text = " ".join(el.text for el in elements if el.page == 1)
    p2_text = " ".join(el.text for el in elements if el.page == 2)
    assert "PAGEONEONLY" in p1_text and "PAGEONEONLY" not in p2_text


# 4. 空白 / 無可擷取文字 PDF 被拒絕
def test_blank_pdf_rejected():
    raw = _make_empty_pdf()
    with pytest.raises(NoExtractableTextError):
        load_pdf(raw)


# 5. 損壞 PDF 被拒絕
def test_corrupt_pdf_rejected():
    with pytest.raises(CorruptPdfError):
        load_pdf(b"this is definitely not a valid pdf file")


# 5b. 加密 PDF 被拒絕（§2.4 需可區分加密）
def test_encrypted_pdf_rejected():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "secret maintenance content", fontsize=12, fontname="helv")
    raw = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="pw-secret",
        owner_pw="pw-secret",
    )
    doc.close()
    with pytest.raises(EncryptedPdfError):
        load_pdf(raw)


# 6. PDF Chunk 不跨頁
def test_pdf_chunks_do_not_cross_pages():
    raw = _make_text_pdf(["ALPHA page one unique content here", "BRAVO page two unique content here"])
    elements = load_pdf(raw)
    splitter = TextSplitter(
        chunk_size_chars=600,
        chunk_overlap_chars=100,
        max_tokens=480,
        count_tokens=_fake_count_tokens,
    )
    chunks = splitter.split(elements, document_id="d1", filename="x.pdf", file_hash="h1")
    assert chunks
    for c in chunks:
        # 每個 chunk 只能來自單一頁面：不得同時包含兩頁的獨有標記
        assert not ("ALPHA" in c.text and "BRAVO" in c.text)
        assert c.page in (1, 2)


# 7. PDF Chunk 保留 filename、page、chunk_index
def test_pdf_chunks_preserve_metadata():
    raw = _make_text_pdf(["Metadata page one text", "Metadata page two text"])
    elements = load_pdf(raw)
    splitter = TextSplitter(
        chunk_size_chars=600,
        chunk_overlap_chars=100,
        max_tokens=480,
        count_tokens=_fake_count_tokens,
    )
    chunks = splitter.split(elements, document_id="doc-7", filename="manual.pdf", file_hash="hash7")
    assert chunks
    assert all(c.filename == "manual.pdf" for c in chunks)
    assert all(c.page is not None for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# 8. 真實範例 PDF：中文 roundtrip + 1-based 頁碼 + section 偵測
def test_sample_pdf_chinese_pages_and_sections():
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("尚未產生 sample PDF（執行 scripts/generate_sample_pdf.py）")
    raw = open(SAMPLE_PDF, "rb").read()
    elements = load_pdf(raw)
    pages = sorted({el.page for el in elements})
    assert pages[:3] == [1, 2, 3]  # 至少三頁且 1-based

    # 第 2 頁應包含「定期保養週期」section 與相關中文內容
    p2 = [el for el in elements if el.page == 2]
    assert any(el.section == "定期保養週期" for el in p2)
    assert any("500" in el.text for el in p2)  # 中文 roundtrip 後仍可讀到數字與內容

    # 第 3 頁應包含「通訊模組檢查」section
    p3 = [el for el in elements if el.page == 3]
    assert any(el.section == "通訊模組檢查" for el in p3)
