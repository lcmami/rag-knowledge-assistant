"""Canonical Text Normalization（v1.1 Slice 1 §五）。

**單一集中模組**：三個 Loader（Markdown / TXT / PDF）都必須透過此模組正規化，
不得各自使用不同規則。正規化規則或版本一旦變更，必須 bump 下列版本常數，讓 Gold
Evidence 不會靜默沿用舊 offset（§五 / architecture §7）。

正規化規則（最小、可重現；deterministic）：
1. Unicode normalization（NFC）。
2. 移除 null characters（U+0000）。
3. Windows / 舊 Mac 換行 → Unix（``\\r\\n`` / ``\\r`` → ``\\n``）。
4. 重複水平空白（半形空白、Tab、全形空白 U+3000 等，但**不含換行**）壓成單一半形空白。
5. 去除每行行尾水平空白，以及每行行首水平空白（統一縮排噪音）。
6. 去除整段前後空白。
**保留換行與段落邊界**，不得為了方便 offset 而移除所有換行或段落結構。
"""
from __future__ import annotations

import re
import unicodedata

# --- 版本常數（規則變更必 bump）-------------------------------------------
# canonical normalized text 的版本；normalization 規則改變即 bump。
CANONICAL_TEXT_VERSION = "v1"
# Loader 行為版本；Loader 解析規則（分段 / page / section 推導）改變即 bump。
LOADER_VERSION = "v1"

# 水平空白（不含換行）：ASCII 空白 / Tab / 全形空白 / 其他 Unicode 空白。
_HORIZONTAL_WS = r"[^\S\n]+"

_horizontal_run = re.compile(_HORIZONTAL_WS)
_trailing_ws = re.compile(r"[^\S\n]+\n")
_leading_ws = re.compile(r"\n[^\S\n]+")


def normalize_text(text: str) -> str:
    """將原始 Loader 文字正規化為版本化 canonical text（deterministic）。"""
    if text is None:
        return ""
    # 1. Unicode NFC
    t = unicodedata.normalize("NFC", text)
    # 2. 移除 null characters
    t = t.replace("\x00", "")
    # 3. 換行統一為 \n
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # 4. 壓縮重複水平空白為單一半形空白（不動換行）
    t = _horizontal_run.sub(" ", t)
    # 5. 去除行尾 / 行首水平空白
    t = _trailing_ws.sub("\n", t)
    t = _leading_ws.sub("\n", t)
    # 6. 去除整段前後空白
    return t.strip()
