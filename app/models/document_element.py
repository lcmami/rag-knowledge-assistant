"""DocumentElement（規格 §9.1）。

Loader 不得只回傳整份文件字串，須先解析為結構化 Element。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentElement:
    text: str
    page: int | None
    section: str | None
    element_type: str  # heading | paragraph | table | code | list
