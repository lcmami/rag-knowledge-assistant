"""PDF Loader（規格 §11 / Phase 4 §2）。

使用 PyMuPDF 逐頁解析內含可擷取文字的 PDF，輸出結構化 DocumentElement 清單。

限制（§2.1）：
- 僅支援內含可擷取文字、非加密、非掃描圖片型 PDF。
- 不實作 OCR、表格結構辨識或複雜版面還原。

規則（§2.2 / §2.3）：
- 對外一律使用 1-based page number（PyMuPDF 內部 0-based index 不外洩）。
- 逐頁解析、忽略完全空白區塊、不同頁面不合併為同一 Element。
- Section 以頁內字級啟發式（明顯大於內文字級者視為標題）推得，
  無標題時 section=None。

錯誤（§2.4，皆為 AppError，API 統一轉譯且不外洩 traceback）：
- CorruptPdfError：無法開啟 / 損壞 / 非 PDF。
- EncryptedPdfError：加密或需要密碼。
- NoExtractableTextError：可解析但整份沒有可擷取文字。
"""
from __future__ import annotations

from collections import Counter

from app.core.constants import ELEMENT_HEADING, ELEMENT_PARAGRAPH
from app.core.exceptions import (
    CorruptPdfError,
    EncryptedPdfError,
    NoExtractableTextError,
)
from app.models.document_element import DocumentElement

# 標題判定：某行最大字級 >= 內文字級 × 此倍率即視為標題。
_HEADING_SIZE_RATIO = 1.15
# 過長的行不太可能是標題（避免把整段內文誤判為 section）。
_HEADING_MAX_CHARS = 40


def _page_lines(page) -> list[tuple[str, float]]:
    """取出單頁的 (行文字, 該行最大字級)，忽略空白行與非文字區塊。"""
    data = page.get_text("dict")
    lines: list[tuple[str, float]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 只處理文字區塊（type 0）
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            size = max((s.get("size", 0.0) for s in spans), default=0.0)
            lines.append((text, size))
    return lines


def _body_size(lines: list[tuple[str, float]]) -> float:
    """以出現最多的字級作為內文字級（standard body size）。"""
    counter = Counter(round(size, 1) for _, size in lines)
    if not counter:
        return 0.0
    return counter.most_common(1)[0][0]


def _parse_page(page, page_number: int) -> list[DocumentElement]:
    """將單一頁面解析為 DocumentElement；不跨頁合併（§2.3）。"""
    lines = _page_lines(page)
    if not lines:
        return []

    body = _body_size(lines)
    heading_threshold = body * _HEADING_SIZE_RATIO

    elements: list[DocumentElement] = []
    current_section: str | None = None
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = "".join(paragraph).strip()
        if text:
            elements.append(
                DocumentElement(
                    text=text,
                    page=page_number,
                    section=current_section,
                    element_type=ELEMENT_PARAGRAPH,
                )
            )
        paragraph = []

    for text, size in lines:
        is_heading = (
            size >= heading_threshold
            and size > body
            and len(text) <= _HEADING_MAX_CHARS
        )
        if is_heading:
            flush_paragraph()
            current_section = text
            elements.append(
                DocumentElement(
                    text=text,
                    page=page_number,
                    section=current_section,
                    element_type=ELEMENT_HEADING,
                )
            )
        else:
            paragraph.append(text)

    flush_paragraph()
    return elements


def load_pdf(raw_bytes: bytes) -> list[DocumentElement]:
    """解析 PDF bytes → DocumentElement 清單（1-based page）。"""
    import pymupdf

    try:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - 開檔失敗一律視為損壞 / 非 PDF
        raise CorruptPdfError(details={"reason": type(exc).__name__}) from exc

    try:
        # 加密 / 需要密碼：needs_pass 為 True 代表尚未解鎖
        if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
            raise EncryptedPdfError()

        elements: list[DocumentElement] = []
        try:
            page_count = doc.page_count
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                # 對外 1-based；不得暴露 PyMuPDF 的 0-based index（§2.2）
                elements.extend(_parse_page(page, page_number=page_index + 1))
        except EncryptedPdfError:
            raise
        except Exception as exc:  # noqa: BLE001 - 解析中途損壞
            raise CorruptPdfError(details={"reason": type(exc).__name__}) from exc

        # 整份 PDF 都沒有可用文字（含掃描圖片型）→ 明確錯誤（§2.3.6）
        if not any(el.text.strip() for el in elements):
            raise NoExtractableTextError()

        return elements
    finally:
        doc.close()
