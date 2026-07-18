"""Document Loader（規格 §11）。

將 Markdown / TXT（以及後續的 PDF）解析為結構化 DocumentElement 清單。
本切片先實作 Markdown 與 TXT；PDF 於 Phase 4 補上。
"""
from __future__ import annotations

import re

from app.core.constants import (
    ELEMENT_CODE,
    ELEMENT_HEADING,
    ELEMENT_LIST,
    ELEMENT_PARAGRAPH,
    ELEMENT_TABLE,
)
from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.models.document_element import DocumentElement

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def load_markdown(text: str) -> list[DocumentElement]:
    """依標題與空行分段，追蹤目前 section，盡量保持表格 / 程式碼區塊完整（§11.3）。"""
    elements: list[DocumentElement] = []
    current_section: str | None = None
    lines = text.splitlines()

    buffer: list[str] = []
    buffer_type = ELEMENT_PARAGRAPH
    in_code = False

    def flush() -> None:
        nonlocal buffer, buffer_type
        content = "\n".join(buffer).strip("\n")
        if content.strip():
            elements.append(
                DocumentElement(
                    text=content,
                    page=None,
                    section=current_section,
                    element_type=buffer_type,
                )
            )
        buffer = []
        buffer_type = ELEMENT_PARAGRAPH

    for line in lines:
        # 程式碼區塊圍欄
        if line.strip().startswith("```"):
            if not in_code:
                flush()
                in_code = True
                buffer_type = ELEMENT_CODE
                buffer.append(line)
            else:
                buffer.append(line)
                in_code = False
                flush()
            continue

        if in_code:
            buffer.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            title = heading.group(2).strip()
            current_section = title
            elements.append(
                DocumentElement(
                    text=title,
                    page=None,
                    section=current_section,
                    element_type=ELEMENT_HEADING,
                )
            )
            continue

        if _is_table_line(line):
            if buffer and buffer_type != ELEMENT_TABLE:
                flush()
            buffer_type = ELEMENT_TABLE
            buffer.append(line)
            continue

        if line.strip() == "":
            flush()
            continue

        # 清單
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            if buffer and buffer_type not in (ELEMENT_LIST, ELEMENT_PARAGRAPH):
                flush()
            if not buffer:
                buffer_type = ELEMENT_LIST
            buffer.append(line)
            continue

        if buffer_type == ELEMENT_TABLE:
            flush()
        buffer.append(line)

    flush()
    return elements


def load_txt(text: str) -> list[DocumentElement]:
    """依空行分段；無 section（None）；忽略完全空白段落（§11.4）。"""
    elements: list[DocumentElement] = []
    for block in re.split(r"\n\s*\n", text):
        content = block.strip()
        if not content:
            continue
        elements.append(
            DocumentElement(
                text=content,
                page=None,
                section=None,
                element_type=ELEMENT_PARAGRAPH,
            )
        )
    return elements


def load_document(filename: str, raw_bytes: bytes) -> list[DocumentElement]:
    """依副檔名分派 loader，並確保非空白文件（§11.5）。"""
    lower = filename.lower()
    if lower.endswith((".md", ".markdown")):
        elements = load_markdown(raw_bytes.decode("utf-8", errors="replace"))
    elif lower.endswith(".txt"):
        elements = load_txt(raw_bytes.decode("utf-8", errors="replace"))
    elif lower.endswith(".pdf"):
        from app.services.pdf_loader import load_pdf  # 延遲載入避免切片相依

        elements = load_pdf(raw_bytes)
    else:
        raise UnsupportedFileTypeError()

    if not any(el.text.strip() for el in elements):
        raise EmptyDocumentError()
    return elements
