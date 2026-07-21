"""Review 用的 Canonical Corpus 載入（Slice 6 §四／§十四）。

只做「讀檔 → Loader → CanonicalDocument」，**不建索引、不呼叫 Embedding、
不連 Qdrant、不碰正式資料**。Evidence Offset 驗證與 Segment 定位都以此為唯一真相。

與 :mod:`evaluation.v11.real_services` 使用相同的 loader / builder，因此 Review 標註的
offset 與 Trial 索引時的 provenance 座標一致。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from app.models.canonical import CanonicalDocument
from evaluation.v11 import hashing

_DOCUMENT_TYPES = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
}


class CorpusError(Exception):
    """Corpus 載入或查詢失敗。"""


@dataclass
class CanonicalCorpus:
    """一份版本化 canonical corpus 快照。"""

    documents: dict[str, CanonicalDocument]  # sha256 -> CanonicalDocument
    corpus_hash: str

    def by_sha(self, document_sha256: str) -> CanonicalDocument:
        doc = self.documents.get(document_sha256)
        if doc is None:
            raise CorpusError(
                f"corpus 中找不到 document_sha256={document_sha256!r}；"
                f"已知文件數={len(self.documents)}。"
            )
        return doc

    def has_sha(self, document_sha256: str) -> bool:
        return document_sha256 in self.documents

    def by_filename(self, filename: str) -> CanonicalDocument:
        for doc in self.documents.values():
            if doc.filename == filename:
                return doc
        known = sorted(d.filename for d in self.documents.values())
        raise CorpusError(f"corpus 中找不到 filename={filename!r}；已知={known}")

    def segment(self, document_sha256: str, segment_id: str):
        doc = self.by_sha(document_sha256)
        for seg in doc.segments:
            if seg.segment_id == segment_id:
                return seg
        raise CorpusError(
            f"文件 {doc.filename!r} 沒有 segment_id={segment_id!r}。"
        )


def load_canonical_corpus(
    corpus_paths: list[str] | tuple[str, ...],
    base_dir: str | None = None,
) -> CanonicalCorpus:
    """載入 corpus 目錄並建立 canonical 表示（deterministic，不產生副作用）。"""
    from app.services.canonical_builder import build_canonical_document
    from app.services.document_loader import load_document

    files = hashing.collect_corpus_files(corpus_paths, base_dir)
    documents: dict[str, CanonicalDocument] = {}
    for _rel, abs_path in files:
        name = os.path.basename(abs_path)
        with open(abs_path, "rb") as f:
            raw = f.read()
        sha = hashlib.sha256(raw).hexdigest()
        ext = os.path.splitext(name.lower())[1]
        documents[sha] = build_canonical_document(
            filename=name,
            document_sha256=sha,
            document_type=_DOCUMENT_TYPES.get(ext, "txt"),
            elements=load_document(name, raw),
        )
    return CanonicalCorpus(
        documents=documents,
        corpus_hash=hashing.corpus_hash(corpus_paths, base_dir),
    )
