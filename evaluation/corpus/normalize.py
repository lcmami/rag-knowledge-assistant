"""LawRecord -> Markdown 正規化（docs/external-corpus-api-plan.md §5.5）。

純函式，無 I/O。

**編章節（``ArticleType == "C"``）的處理是本模組的核心約束。**

``app/services/document_loader.py::load_markdown`` 的實測行為是：regex
``^(#{1,6})\\s+(.*)$`` 會讓**任何層級**的 heading 同時 (1) 設定
``current_section``、(2) 額外產生一個獨立的 ``ELEMENT_HEADING`` element。

因此只要把編章節寫成任何 Markdown heading，它就會成為獨立 section 與獨立
element，也就會成為可被視為**獨立法律條文 evidence** 的單位 — 這是不被允許的。

本模組的作法：編章節**永不輸出為 heading**，而是以深度堆疊維護章節路徑，
併入其後每一條 type-``A`` 條文自己的 ``##`` heading，形式為
``## {chapter_path}｜{ArticleNo}``。如此同時滿足：

1. 編章節從不獨立成 section／element；
2. 章節 context 完整保留於其後條文的 section 名稱中；
3. 零 loader 修改。
"""
from __future__ import annotations

from typing import Optional

from evaluation.corpus.models import LawRecord, NormalizedDocument

# 章節路徑與條號之間的分隔符。全形直線在中文排版中即為此用途，且不會與
# 來源文字衝突（實測 ArticleNo / 章節標題皆不含此字元）。
CHAPTER_SEPARATOR = "｜"


def _normalize_newlines(text: str) -> str:
    """來源含 CRLF（實測 ``LawForeword`` / ``LawHistories``）；統一為 LF。

    Snapshot 一律以 LF 寫出，避免 repo ``core.autocrlf=true`` 造成 hash 漂移。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _clean_block(text: str) -> str:
    """正規化換行、去除每行尾端空白與整體前後空白。"""
    normalized = _normalize_newlines(text)
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def build_chapter_path(stack: list[tuple[int, str]]) -> str:
    """把深度堆疊轉成單行章節路徑。"""
    return CHAPTER_SEPARATOR.join(title for _depth, title in stack)


def _push_chapter(stack: list[tuple[int, str]], depth: int, title: str) -> None:
    """依深度截斷堆疊後推入新章節。

    來源以行首空白（3 的倍數）編碼深度。深度 0 的編章節視為最外層，會清空堆疊。
    """
    while stack and stack[-1][0] >= depth:
        stack.pop()
    stack.append((depth, title))


def _header_lines(record: LawRecord) -> list[str]:
    """法規層級 metadata。以清單呈現，不使用 heading（避免多出 section）。"""
    lines = [
        f"- 法規名稱：{record.law_name}",
        f"- 法規位階：{record.law_level}",
        f"- 法規類別：{record.law_category}",
        f"- 最後修正日期：{record.law_modified_date}",
        f"- 廢止狀態：{'已廢止' if record.is_repealed else '現行'}",
        f"- 來源網址：{record.law_url}",
    ]
    if record.is_repealed:
        note = _clean_block(record.law_abandon_note).replace("\n", " ")
        if note:
            lines.append(f"- 廢止註記：{note}")
    return lines


def to_markdown(record: LawRecord) -> Optional[NormalizedDocument]:
    """把一部法規正規化為一份 Markdown 文件。

    回傳 ``None`` 表示該法規無任何可用條文（所有 ``ArticleContent`` 皆空），
    依計畫 §8 應整筆跳過。

    ``LawAttachements`` 於本階段明確忽略、不下載（PDF 屬 out of scope）；
    但其數量保留於 metadata，供 Slice D 的 corpus statistics 使用。
    """
    parts: list[str] = [f"# {record.law_name}", ""]
    parts.extend(_header_lines(record))
    parts.append("")

    foreword = _clean_block(record.law_foreword)
    if foreword:
        parts.append(foreword)
        parts.append("")

    chapter_stack: list[tuple[int, str]] = []
    emitted = 0
    skipped_articles = 0

    for article in record.articles:
        if article.is_chapter:
            title = article.chapter_title
            if title:
                _push_chapter(chapter_stack, article.chapter_depth, title)
            # 關鍵：編章節到此為止，不輸出任何 heading、不輸出任何內容。
            continue

        if not article.is_article:
            continue

        content = _clean_block(article.article_content)
        if not content:
            skipped_articles += 1
            continue

        article_no = article.article_no.strip()
        chapter_path = build_chapter_path(chapter_stack)
        if article_no and chapter_path:
            heading = f"{chapter_path}{CHAPTER_SEPARATOR}{article_no}"
        elif article_no:
            heading = article_no
        elif chapter_path:
            heading = chapter_path
        else:
            # ArticleNo 與章節皆空：無法構成可定位的 section，視同不可用。
            skipped_articles += 1
            continue

        parts.append(f"## {heading}")
        parts.append("")
        parts.append(content)
        parts.append("")
        emitted += 1

    if emitted == 0:
        return None

    markdown = "\n".join(parts).rstrip("\n") + "\n"

    return NormalizedDocument(
        doc_id=record.pcode,
        filename=f"{record.pcode}.md",
        markdown=markdown,
        metadata={
            "pcode": record.pcode,
            "law_name": record.law_name,
            "law_level": record.law_level,
            "law_category": record.law_category,
            "law_modified_date": record.law_modified_date,
            "law_url": record.law_url,
            "is_repealed": record.is_repealed,
            "article_count": emitted,
            "skipped_article_count": skipped_articles,
            "attachment_count": len(record.law_attachements),
        },
    )
