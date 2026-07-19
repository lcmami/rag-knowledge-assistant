"""Text Splitter（規格 §12）。

切分優先順序：頁面 → Markdown 標題 → 空行 → 段落 → 句尾符號 → 字元長度 → token 硬限制。

- CHUNK_SIZE_CHARS 為軟目標。
- EMBEDDING_MAX_TOKENS 為硬限制（emit 前用注入的 count_tokens 驗證）。
- overlap 不跨頁 / 不跨 section，且不得產生完全重複的 chunk。
- 單一超長句仍需依 token 安全長度硬切。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from app.models.canonical import CanonicalDocument, CanonicalSegment, ProvenanceSpan
from app.models.chunk import Chunk
from app.models.document_element import DocumentElement
from app.services.identifier_extractor import IdentifierExtractor

# 句尾符號（中英）
_SENTENCE_ENDINGS = "。！？；!?;"


def _split_sentences(text: str) -> list[str]:
    """依句尾符號與換行切句，保留分隔符。"""
    sentences: list[str] = []
    current = []
    for ch in text:
        current.append(ch)
        if ch in _SENTENCE_ENDINGS or ch == "\n":
            piece = "".join(current)
            if piece.strip():
                sentences.append(piece)
            current = []
    tail = "".join(current)
    if tail.strip():
        sentences.append(tail)
    return sentences


class TextSplitter:
    def __init__(
        self,
        chunk_size_chars: int,
        chunk_overlap_chars: int,
        max_tokens: int,
        count_tokens: Callable[[str], int],
        identifier_extractor: IdentifierExtractor | None = None,
    ):
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens
        # index-time 識別碼抽取（§十）；query-time 於未來 Slice 4 共用同一元件。
        self.identifier_extractor = identifier_extractor or IdentifierExtractor()

    # --- token 安全的硬切（處理單一超長句）---
    def _hard_split_by_tokens(self, text: str) -> list[str]:
        """對超過 token 硬限制的文字，用字元二分逼近安全長度切段。"""
        pieces: list[str] = []
        remaining = text
        while remaining:
            if self.count_tokens(remaining) <= self.max_tokens:
                pieces.append(remaining)
                break
            # 以字元為單位找出不超過 token 限制的最大前綴
            lo, hi = 1, len(remaining)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if self.count_tokens(remaining[:mid]) <= self.max_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            pieces.append(remaining[:best])
            remaining = remaining[best:]
        return [p for p in pieces if p.strip()]

    def _enforce_token_limit(self, text: str) -> list[str]:
        """emit 前確保每段不超 token 硬限制。"""
        if self.count_tokens(text) <= self.max_tokens:
            return [text]
        # 先嘗試依句子重組
        out: list[str] = []
        buf = ""
        for sentence in _split_sentences(text):
            if self.count_tokens(sentence) > self.max_tokens:
                if buf.strip():
                    out.append(buf)
                    buf = ""
                out.extend(self._hard_split_by_tokens(sentence))
                continue
            candidate = buf + sentence
            if buf and self.count_tokens(candidate) > self.max_tokens:
                out.append(buf)
                buf = sentence
            else:
                buf = candidate
        if buf.strip():
            out.append(buf)
        return out

    def _windows_from_group(self, group_text: str) -> list[str]:
        """對單一 (page, section) 群組文字做句子感知的滑動視窗切分。"""
        sentences = _split_sentences(group_text)
        if not sentences:
            return []

        chunks: list[str] = []
        i = 0
        n = len(sentences)
        while i < n:
            buf = ""
            j = i
            while j < n:
                candidate = buf + sentences[j]
                # 軟目標：字元數；硬限制：token 數
                if buf and (
                    len(candidate) > self.chunk_size_chars
                    or self.count_tokens(candidate) > self.max_tokens
                ):
                    break
                buf = candidate
                j += 1
            if not buf:  # 單句超過限制
                buf = sentences[i]
                j = i + 1

            for piece in self._enforce_token_limit(buf):
                if piece.strip():
                    chunks.append(piece.strip())

            if j >= n:
                break

            # overlap：從已消耗句子尾端回退，累積 <= overlap 字元，避免完全重複
            overlap_start = j
            acc = 0
            while overlap_start > i + 1 and acc < self.chunk_overlap_chars:
                acc += len(sentences[overlap_start - 1])
                overlap_start -= 1
            i = overlap_start if overlap_start > i else j
        return chunks

    def split(
        self,
        elements: list[DocumentElement],
        *,
        document_id: str,
        filename: str,
        file_hash: str,
        canonical: CanonicalDocument | None = None,
    ) -> list[Chunk]:
        """將 DocumentElement 清單切為保留 metadata 的 Chunk 清單。

        ``canonical`` 為 v1.1 Slice 1 新增（§六）：提供時，會為每個 Chunk 附上可追蹤的
        多個 :class:`ProvenanceSpan`（一個 Chunk 由多段落 / section 組成即多個 span），
        並回填 canonical / loader 版本。**切分邏輯與輸出 chunk text / 數量完全不變**，
        provenance 純為附加 metadata；``canonical`` 省略時退化為 v1.0 行為。
        """
        created_at = datetime.now(timezone.utc).isoformat()
        chunks: list[Chunk] = []
        chunk_index = 0

        # Segment 與 elements 1:1 同序對齊（CanonicalDocumentBuilder 不丟棄 element）。
        segments = canonical.segments if canonical is not None else None

        # 依 (page, section) 分群，群組邊界即為切分邊界（不跨頁 / 不跨 section）。
        # 保留 element 與其對應 segment，供 provenance 對映。
        groups: list[tuple[int | None, str | None, list[tuple[DocumentElement, CanonicalSegment | None]]]] = []
        for idx, el in enumerate(elements):
            seg = segments[idx] if segments is not None and idx < len(segments) else None
            if groups and groups[-1][0] == el.page and groups[-1][1] == el.section:
                groups[-1][2].append((el, seg))
            else:
                groups.append((el.page, el.section, [(el, seg)]))

        for page, section, items in groups:
            included = [(el, seg) for el, seg in items if el.text.strip()]
            group_text = "\n\n".join(el.text for el, _seg in included)

            # 每個 element 在 group_text 中的 [start, end)（原始文字座標），供 provenance 對映。
            elem_ranges = self._element_ranges(included)

            find_cursor = 0
            for piece in self._windows_from_group(group_text):
                g_start = group_text.find(piece, find_cursor)
                if g_start < 0:  # 理論上不會發生（piece 為 group_text 子字串）
                    g_start = group_text.find(piece)
                if g_start < 0:
                    g_start, g_end = 0, len(group_text)
                else:
                    g_end = g_start + len(piece)
                    find_cursor = g_start  # chunk 起點單調不減，允許 overlap

                provenance = self._provenance_for_chunk(
                    elem_ranges, g_start, g_end, canonical
                )
                identifiers = self.identifier_extractor.extract_normalized(piece)

                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        filename=filename,
                        file_hash=file_hash,
                        page=page,
                        section=section,
                        chunk_index=chunk_index,
                        text=piece,
                        created_at=created_at,
                        provenance_spans=provenance,
                        identifiers=identifiers,
                        canonical_text_version=(
                            canonical.canonical_text_version if canonical else None
                        ),
                        loader_version=(
                            canonical.loader_version if canonical else None
                        ),
                        identifier_normalization_version=self.identifier_extractor.version,
                    )
                )
                chunk_index += 1

        return chunks

    @staticmethod
    def _element_ranges(
        included: list[tuple[DocumentElement, CanonicalSegment | None]],
    ) -> list[tuple[int, int, CanonicalSegment | None]]:
        """計算每個 element 於 group_text（"\\n\\n" 連接）中的原始字元範圍。"""
        ranges: list[tuple[int, int, CanonicalSegment | None]] = []
        cursor = 0
        for el, seg in included:
            start = cursor
            end = start + len(el.text)
            ranges.append((start, end, seg))
            cursor = end + 2  # "\n\n"
        return ranges

    @staticmethod
    def _provenance_for_chunk(
        elem_ranges: list[tuple[int, int, CanonicalSegment | None]],
        g_start: int,
        g_end: int,
        canonical: CanonicalDocument | None,
    ) -> list[ProvenanceSpan]:
        """依 chunk 在 group_text 的範圍，回傳其覆蓋到的 segment 對應 provenance span。"""
        if canonical is None:
            return []
        spans: list[ProvenanceSpan] = []
        for e_start, e_end, seg in elem_ranges:
            if seg is None:
                continue
            # 半開區間重疊判定
            if e_start < g_end and g_start < e_end:
                spans.append(
                    ProvenanceSpan(
                        document_sha256=canonical.document_sha256,
                        canonical_text_version=canonical.canonical_text_version,
                        loader_version=canonical.loader_version,
                        page=seg.page,
                        section=seg.section,
                        paragraph_index=seg.paragraph_index,
                        offset_scope=seg.offset_scope,
                        char_start=seg.char_start,
                        char_end=seg.char_end,
                    )
                )
        return spans
