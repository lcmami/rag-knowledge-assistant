"""固定常數（規格 §16.3 拒答訊息、§17.3 Citation regex、§10.2 文件狀態）。"""
from __future__ import annotations

import re

# 後端固定拒答訊息（§16.3 / §18.5）
REFUSAL_MESSAGE = "目前提供的文件中沒有足夠資訊可以回答此問題。"

# Citation 解析（§17.3）
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

# 支援的副檔名（§11.1）
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}

# 文件狀態（§10.2）
STATUS_INDEXING = "indexing"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUS_DELETING = "deleting"

# DocumentElement.element_type（§9.1）
ELEMENT_HEADING = "heading"
ELEMENT_PARAGRAPH = "paragraph"
ELEMENT_TABLE = "table"
ELEMENT_CODE = "code"
ELEMENT_LIST = "list"
