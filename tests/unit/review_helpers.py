"""Slice 6 Review 測試共用工具（不呼叫 LLM、不連 Qdrant、不碰正式資料）。

一律於 ``tmp_path`` 建立小型 corpus 與 base dataset，因此測試快速、可重現，
且完全不依賴 ``sample_docs`` 或正式 SQLite / Uploads。
"""
from __future__ import annotations

import json
import os

from evaluation.v11.review.corpus import load_canonical_corpus
from evaluation.v11.review.repository import ReviewRepository
from evaluation.v11.review.service import ReviewService

ALARM_DOC = """# 設備警報手冊

## E102 警報
E102 表示主軸溫度過高，請立即停機並通知維修人員。

## CX-500 保養
CX-500 每 500 小時需更換濾芯。
"""

SPEC_DOC = """# API 規格

## 認證與 Token
API Token 有效期限為 24 小時，逾期需重新申請。
"""


def make_corpus(tmp_path):
    """建立小型 corpus 目錄並回傳 :class:`CanonicalCorpus`。"""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    (corpus_dir / "equipment_alarm_manual.md").write_text(ALARM_DOC, encoding="utf-8")
    (corpus_dir / "api_integration_spec.md").write_text(SPEC_DOC, encoding="utf-8")
    return load_canonical_corpus([str(corpus_dir)], str(tmp_path))


def make_base_dataset(tmp_path, corpus, *, dataset_version: str = "v1.1-base-0001"):
    """建立一份合法的 base dataset（evidence 對齊 corpus 實際 offset）。"""
    doc = corpus.by_filename("api_integration_spec.md")
    seg = _segment_containing(doc, "API Token")
    start = seg.char_start + seg.normalized_text.index("API Token")
    payload = {
        "schema_version": "1.1",
        "dataset_version": dataset_version,
        "cases": [
            {
                "id": "base-token-001",
                "question": "API Token 的有效期限是多久？",
                "query_type": "semantic",
                "answerable": True,
                "group_id": "api-token",
                "split": "tune",
                "expected_evidence": [
                    {
                        "document_sha256": doc.document_sha256,
                        "filename": doc.filename,
                        "canonical_text_version": doc.canonical_text_version,
                        "loader_version": doc.loader_version,
                        "page": seg.page,
                        "section": seg.section,
                        "offset_scope": seg.offset_scope,
                        "char_start": start,
                        "char_end": start + len("API Token 有效期限為 24 小時"),
                        "anchor_text": ["API Token"],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "base-cases.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return str(path)


def _segment_containing(doc, needle: str):
    for seg in doc.segments:
        if needle in seg.normalized_text:
            return seg
    raise AssertionError(f"corpus 中找不到含 {needle!r} 的 segment")


def segment_containing(corpus, filename: str, needle: str):
    return _segment_containing(corpus.by_filename(filename), needle)


def evidence_for(corpus, filename: str, needle: str, *, anchor: str | None = None):
    """依實際 canonical offset 組出合法 evidence dict。"""
    doc = corpus.by_filename(filename)
    seg = _segment_containing(doc, needle)
    start = seg.char_start + seg.normalized_text.index(needle)
    return {
        "document_sha256": doc.document_sha256,
        "filename": doc.filename,
        "canonical_text_version": doc.canonical_text_version,
        "loader_version": doc.loader_version,
        "page": seg.page,
        "section": seg.section,
        "offset_scope": seg.offset_scope,
        "char_start": start,
        "char_end": start + len(needle),
        "anchor_text": [anchor or needle],
    }


def make_service(tmp_path, corpus) -> ReviewService:
    repo = ReviewRepository(str(tmp_path / "review.review.db"))
    return ReviewService(repo, corpus)


def make_session(service, base_dataset_path, **kwargs):
    return service.create_session(
        session_name=kwargs.pop("session_name", "test-session"),
        base_dataset_path=base_dataset_path,
        **kwargs,
    )
