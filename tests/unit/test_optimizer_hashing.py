"""Hashing 可重現性單元測試（Slice 2 §17.1）。"""
from __future__ import annotations

import os

from evaluation.v11 import hashing


def _write(path: str, content: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def test_corpus_hash_independent_of_absolute_path(tmp_path):
    # 相同相對結構 + 內容，位於不同絕對父目錄 → corpus hash 相同（不含絕對路徑）。
    for base in ("baseA", "baseB"):
        _write(str(tmp_path / base / "corpus" / "a.md"), b"alpha")
        _write(str(tmp_path / base / "corpus" / "b.txt"), b"bravo")
    h1 = hashing.corpus_hash(["corpus"], base_dir=str(tmp_path / "baseA"))
    h2 = hashing.corpus_hash(["corpus"], base_dir=str(tmp_path / "baseB"))
    assert h1 == h2


def test_corpus_hash_changes_with_content(tmp_path):
    _write(str(tmp_path / "c" / "a.md"), b"alpha")
    h1 = hashing.corpus_hash(["c"], base_dir=str(tmp_path))
    _write(str(tmp_path / "c" / "a.md"), b"ALPHA-changed")
    h2 = hashing.corpus_hash(["c"], base_dir=str(tmp_path))
    assert h1 != h2


def test_corpus_hash_independent_of_dir_argument_order(tmp_path):
    _write(str(tmp_path / "d1" / "a.md"), b"one")
    _write(str(tmp_path / "d2" / "b.md"), b"two")
    h1 = hashing.corpus_hash(["d1", "d2"], base_dir=str(tmp_path))
    h2 = hashing.corpus_hash(["d2", "d1"], base_dir=str(tmp_path))
    assert h1 == h2


def test_corpus_manifest_uses_posix_relative_paths(tmp_path):
    _write(str(tmp_path / "corpus" / "nested" / "x.md"), b"data")
    manifest = hashing.corpus_manifest(["corpus"], base_dir=str(tmp_path))
    paths = [d["path"] for d in manifest["documents"]]
    assert "corpus/nested/x.md" in paths
    assert all("\\" not in p for p in paths)


def test_dataset_hash_key_order_independent():
    a = {"schema_version": "1.1", "dataset_version": "v", "cases": [{"id": "1", "q": "x"}]}
    b = {"cases": [{"q": "x", "id": "1"}], "dataset_version": "v", "schema_version": "1.1"}
    assert hashing.dataset_hash(a) == hashing.dataset_hash(b)


def test_dataset_hash_changes_with_value():
    a = {"dataset_version": "v1", "cases": []}
    b = {"dataset_version": "v2", "cases": []}
    assert hashing.dataset_hash(a) != hashing.dataset_hash(b)


def test_config_hash_key_order_independent():
    a = {"top_k": 5, "score_threshold": 0.55, "chunk_config": {"strategy": "current"}}
    b = {"chunk_config": {"strategy": "current"}, "score_threshold": 0.55, "top_k": 5}
    assert hashing.config_hash(a) == hashing.config_hash(b)


def test_config_hash_changes_per_field():
    base = {"top_k": 5, "score_threshold": 0.55, "candidate_multiplier": 3,
            "candidate_cap": 50, "chunk_config": {"strategy": "current"},
            "canonical_text_version": "v1", "identifier_normalization_version": "v1"}
    h0 = hashing.config_hash(base)
    for field, new in [
        ("top_k", 3), ("score_threshold", 0.6), ("candidate_multiplier", 5),
        ("candidate_cap", 20), ("canonical_text_version", "v2"),
        ("identifier_normalization_version", "v2"),
    ]:
        variant = dict(base)
        variant[field] = new
        assert hashing.config_hash(variant) != h0, f"{field} 未改變 hash"
    # chunk_config 內容改變也要改變 hash
    variant = dict(base)
    variant["chunk_config"] = {"strategy": "current", "chunk_size_chars": 400}
    assert hashing.config_hash(variant) != h0


def test_real_corpus_hash_deterministic():
    h1 = hashing.corpus_hash(["sample_docs", "evaluation/fixtures"])
    h2 = hashing.corpus_hash(["evaluation/fixtures", "sample_docs"])
    assert h1 == h2 and len(h1) == 64
