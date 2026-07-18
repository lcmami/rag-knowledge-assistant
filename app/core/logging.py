"""Logging 設定（規格 §23）。

不得記錄 API Key 或未遮蔽的秘密資訊。
"""
from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def mask_hash(file_hash: str, keep: int = 8) -> str:
    """只保留雜湊前數碼（§23：記錄 file_hash 前數碼）。"""
    return file_hash[:keep] if file_hash else ""
