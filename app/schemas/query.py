"""Query API schema（規格 Phase 5 §11）。

重用現有 Retrieval Source Schema，避免同欄位重複定義；驗證規則沿用 /api/retrieve。
"""
from __future__ import annotations

from pydantic import BaseModel

from app.schemas.retrieve import RetrieveRequest, RetrieveResultItem


class QueryRequest(RetrieveRequest):
    """沿用 /api/retrieve 的 question / top_k 驗證規則（必填、trim、上限 2000、top_k ≥ 1）。"""


class QuerySource(RetrieveResultItem):
    """回應來源；欄位與結構完全重用 Retrieval Source（source_id、document_id…text）。"""


class QueryResponse(BaseModel):
    answer: str
    sources: list[QuerySource]
    retrieval_time_ms: int
    generation_time_ms: int
