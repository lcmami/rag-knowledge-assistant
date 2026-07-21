"""Slice 3 chunking 測試共用件（不載入模型；count_tokens=len）。"""
from __future__ import annotations

from app.models.document_element import DocumentElement
from app.services.canonical_builder import build_canonical_document
from evaluation.v11.chunking import ChunkingConfig, ChunkingContext, build_strategy


def make_elements(specs) -> list[DocumentElement]:
    """specs: [(text, page, section, element_type)]。"""
    return [DocumentElement(t, p, s, et) for (t, p, s, et) in specs]


def _coerce(elements) -> list[DocumentElement]:
    """接受 DocumentElement 清單，或 (text, page, section, element_type) tuple 清單。"""
    if elements and isinstance(elements[0], tuple):
        return make_elements(elements)
    return list(elements)


def build_ctx(elements, raw_config, *, document_type="markdown", count_tokens=len):
    elements = _coerce(elements)
    canonical = build_canonical_document(
        filename="doc.md", document_sha256="hash-doc", document_type=document_type, elements=elements
    )
    cfg = ChunkingConfig.from_raw(raw_config)
    return ChunkingContext(
        document_id="doc-1",
        filename="doc.md",
        file_hash="hash-doc",
        canonical=canonical,
        elements=elements,
        config=cfg,
        count_tokens=count_tokens,
    )


def run_strategy(elements, raw_config, *, document_type="markdown", count_tokens=len):
    ctx = build_ctx(elements, raw_config, document_type=document_type, count_tokens=count_tokens)
    return build_strategy(ctx.config.strategy).chunk(ctx)


def core_spans(chunk):
    return [s for s in chunk.provenance_spans if s.role == "core"]


def context_spans(chunk):
    return [s for s in chunk.provenance_spans if s.role == "context"]
