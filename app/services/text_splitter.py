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

from app.models.chunk import Chunk
from app.models.document_element import DocumentElement

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
    ):
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens

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
    ) -> list[Chunk]:
        """將 DocumentElement 清單切為保留 metadata 的 Chunk 清單。"""
        created_at = datetime.now(timezone.utc).isoformat()
        chunks: list[Chunk] = []
        chunk_index = 0

        # 依 (page, section) 分群，群組邊界即為切分邊界（不跨頁 / 不跨 section）
        groups: list[tuple[int | None, str | None, list[str]]] = []
        for el in elements:
            key = (el.page, el.section)
            if groups and groups[-1][0] == el.page and groups[-1][1] == el.section:
                groups[-1][2].append(el.text)
            else:
                groups.append((el.page, el.section, [el.text]))

        for page, section, texts in groups:
            group_text = "\n\n".join(t for t in texts if t.strip())
            for piece in self._windows_from_group(group_text):
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
                    )
                )
                chunk_index += 1

        return chunks
