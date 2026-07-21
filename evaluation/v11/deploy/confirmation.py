"""Explicit Confirmation 字串格式與 hash（Slice 7 §十四 / §十五.3 / §十六.6）。

Confirmation 必須 deterministic、精確匹配、且**綁定關鍵語意**：

- initialize：``INITIALIZE <alias> -> <collection>``
- apply：``APPLY <alias> CONFIG <config_hash> FROM <expected_current>``（綁 config hash + alias + expected current target）
- rollback：``ROLLBACK <alias> FROM <current> TO <target>``

只保存 confirmation 的 **hash**（§十五.9 / §二十二）；原文不入 DB / report。
"""
from __future__ import annotations

import hashlib


def _norm(text: str) -> str:
    return (text or "").strip()


def initialize_confirmation(alias: str, collection: str) -> str:
    return f"INITIALIZE {alias} -> {collection}"


def apply_confirmation(alias: str, config_hash: str, expected_current: str) -> str:
    return f"APPLY {alias} CONFIG {config_hash} FROM {expected_current}"


def rollback_confirmation(alias: str, current: str, target: str) -> str:
    return f"ROLLBACK {alias} FROM {current} TO {target}"


def confirmation_matches(provided: str, expected: str) -> bool:
    """精確（去頭尾空白後）比對，deterministic（§十四 / §二十.7）。"""
    return _norm(provided) == _norm(expected)


def confirmation_hash(confirmation: str) -> str:
    """confirmation 原文的 SHA-256（只存 hash，不存原文）。"""
    return hashlib.sha256(_norm(confirmation).encode("utf-8")).hexdigest()
