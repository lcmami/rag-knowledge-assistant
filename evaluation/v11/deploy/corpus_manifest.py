"""Target Active Corpus Manifest（Slice 7 §十一）。

Apply / Rollback / Readiness 前用來確認 candidate 的 corpus 與 target active corpus 為
**同一份、同一 hash 版本**。hash 只由「排序後相對／穩定名稱 + 檔案內容 SHA-256」組成，
**不含**絕對路徑或 mtime（§十一.2）。相同檔名但內容不同的 live corpus 不會被誤判為相同（§十一.4）。
"""
from __future__ import annotations

from typing import Optional

from app.services.canonical_normalizer import CANONICAL_TEXT_VERSION, LOADER_VERSION
from evaluation.v11 import hashing
from evaluation.v11.deploy.models import CorpusManifest

#: corpus hash 演算法版本（candidate 與 target 必須一致，§十一.1）。
CORPUS_HASH_VERSION = "v1"


def build_corpus_manifest(
    corpus_paths: list[str] | tuple[str, ...],
    *,
    base_dir: Optional[str] = None,
    chunk_count: Optional[int] = None,
) -> CorpusManifest:
    """由受追蹤 corpus 目錄建立 :class:`CorpusManifest`（穩定、可入 deployment record）。"""
    manifest = hashing.corpus_manifest(corpus_paths, base_dir)
    documents = [
        {"path": e["path"], "sha256": e["sha256"]} for e in manifest["documents"]
    ]
    return CorpusManifest(
        corpus_hash=manifest["corpus_hash"],
        corpus_hash_version=CORPUS_HASH_VERSION,
        document_count=manifest["document_count"],
        chunk_count=chunk_count,
        canonical_text_version=CANONICAL_TEXT_VERSION,
        loader_version=LOADER_VERSION,
        documents=documents,
    )
