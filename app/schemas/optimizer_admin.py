"""Optimizer Admin API 的嚴格 Request Schema（Slice 7 §二十.5/.6 / §二十一.9）。

- ``extra="forbid"``：未知欄位一律拒絕（§二十.6）。
- **不接受** ``apply_eligible`` 之類的 eligibility boolean（§二十一.9）——eligibility 一律由
  DB locked outcome 重新讀取，request 只能提供 expected snapshot。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AliasInitializeRequest(_Strict):
    alias: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    confirmation: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class ApplyRequestModel(_Strict):
    search_id: str = Field(min_length=1)
    expected_config_hash: str = Field(min_length=1)
    expected_dataset_hash: str = Field(min_length=1)
    expected_corpus_hash: str = Field(min_length=1)
    expected_code_commit: str = Field(min_length=1)
    expected_current_collection: str = Field(min_length=1)
    confirmation: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    alias: Optional[str] = None
    expected_canonical_text_version: Optional[str] = None
    expected_identifier_normalization_version: Optional[str] = None
    expected_embedding_fingerprint_hash: Optional[str] = None
    expected_recommendation_locked_at: Optional[str] = None


class RollbackRequestModel(_Strict):
    expected_current_collection: str = Field(min_length=1)
    target_collection: str = Field(min_length=1)
    confirmation: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    alias: Optional[str] = None


class ReconcileRequestModel(_Strict):
    idempotency_key: str = Field(min_length=1)
    alias: Optional[str] = None
