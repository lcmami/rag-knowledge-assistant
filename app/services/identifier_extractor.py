"""Identifier Extractor / Normalizer（v1.1 Slice 1 §十）。

版本化的**共用**元件：index-time 與（未來 Slice 4）query-time 共用同一份規則，確保
「查詢代碼」與「索引代碼」以相同方式正規化。

v1.1 第一版明確規則（§十）：
- ``E102`` / ``e102`` / ``E-102`` → ``E102``
- ``CX-500`` / ``CX500`` → ``CX500``

正規化步驟：
1. Unicode NFC。
2. 英文字母轉大寫。
3. 移除識別碼內可選格式分隔符：hyphen ``-``、underscore ``_``、whitespace。

抽取（extract）保守規則，避免把一般中文句子或純年份大量誤判為識別碼：
- 需**至少一個英文字母開頭** + 至少兩位數字（純年份如 ``2024`` 不匹配）。
- 只接受 ``-`` / ``_`` 或無分隔符連接字母與數字（避免 ``表格 12`` / ``Table 12`` 這類
  以空白分隔的一般片語被誤判）。
- 邊界以「非 ASCII 英數」判定，故與中文相鄰（如 ``E102警報``）仍可抽出。

保留 ``raw_identifier`` 供顯示 / 除錯；``identifier_normalization_version`` 寫入
Chunk payload 與未來 Trial Config。規則變更 → bump 版本 → 需重建 identifiers payload
index，但**不需**重算相同 chunk text 的 embedding（identifier metadata hash 與 embedding
text hash 概念分離）。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# 規則變更必 bump（§十 要求 8/9）。
IDENTIFIER_NORMALIZATION_VERSION = "v1"

# 抽取樣式：字母(1-5) + 可選單一 -/_ + 數字(2-6)，前後不得緊鄰其他 ASCII 英數。
_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,5})[-_]?(\d{2,6})(?![A-Za-z0-9])")

# 正規化時移除的格式分隔符：hyphen / underscore / 各式空白。
_SEPARATORS = re.compile(r"[-_\s]+")


@dataclass(frozen=True)
class Identifier:
    """單一識別碼：保留原始值與正規化值，並記錄正規化版本。"""

    raw_identifier: str
    normalized_identifier: str
    identifier_normalization_version: str = IDENTIFIER_NORMALIZATION_VERSION


def normalize_identifier(raw: str) -> str:
    """將原始識別碼字串正規化（NFC → 大寫 → 去分隔符）。"""
    t = unicodedata.normalize("NFC", raw)
    t = _SEPARATORS.sub("", t)
    return t.upper()


class IdentifierExtractor:
    """版本化識別碼抽取 / 正規化器。"""

    version = IDENTIFIER_NORMALIZATION_VERSION

    def normalize(self, raw: str) -> str:
        return normalize_identifier(raw)

    def extract(self, text: str) -> list[Identifier]:
        """抽取文字中的識別碼，去重（依 normalized）但保持首次出現順序。"""
        if not text:
            return []
        seen: set[str] = set()
        out: list[Identifier] = []
        for match in _IDENTIFIER_RE.finditer(text):
            raw = match.group(0)
            normalized = normalize_identifier(raw)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(
                Identifier(
                    raw_identifier=raw,
                    normalized_identifier=normalized,
                    identifier_normalization_version=self.version,
                )
            )
        return out

    def extract_normalized(self, text: str) -> list[str]:
        """回傳去重、deterministic order 的 normalized 識別碼清單（供 payload）。"""
        return [i.normalized_identifier for i in self.extract(text)]
