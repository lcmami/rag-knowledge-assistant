"""ProvenanceSpan.role 向後相容（v1.1 Slice 3 §十二 / §二十一）。

role 為選填、預設 ``core``；序列化時**僅非預設值才輸出**，確保 v1.0 / Slice 1-2 既有
payload byte-identical（正式 Qdrant payload 不變）。
"""
from __future__ import annotations

from app.models.canonical import ProvenanceSpan


def _span(**over):
    base = dict(document_sha256="h", canonical_text_version="v1", loader_version="v1",
                page=None, section="S", paragraph_index=0, offset_scope="document",
                char_start=0, char_end=10)
    base.update(over)
    return ProvenanceSpan(**base)


def test_default_role_is_core():
    assert _span().role == "core"


def test_core_role_omitted_in_serialization():
    d = _span().to_dict()
    assert "role" not in d  # 預設 core → 不輸出，payload 與 Slice 1/2 一致


def test_context_role_serialized():
    d = _span(role="context").to_dict()
    assert d["role"] == "context"


def test_from_dict_defaults_core_when_missing():
    d = _span().to_dict()  # 無 role key
    assert ProvenanceSpan.from_dict(d).role == "core"


def test_from_dict_roundtrip_context():
    d = _span(role="context").to_dict()
    assert ProvenanceSpan.from_dict(d).role == "context"
