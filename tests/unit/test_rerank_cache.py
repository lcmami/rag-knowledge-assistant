"""Rerank cache 與 fingerprint（v1.2.1 Slice R1）。

不載入任何模型 —— fingerprint 以 `make_fingerprint` 明確欄位建立。
"""
from __future__ import annotations

import os
import struct

import pytest

from evaluation.v11.rerank_cache import (
    FingerprintedRerankCache,
    RerankCacheRepository,
)
from evaluation.v11.rerank_fingerprint import (
    ACTIVATION_IDENTITY,
    MANIFEST_UNKNOWN,
    build_weights_manifest,
    make_fingerprint,
    weights_manifest_sha256,
)


def _fp(**over):
    kw = dict(
        provider="sentence_transformers",
        model_name="BAAI/bge-reranker-base",
        model_revision="2cfc18c9",
        num_labels=1,
        max_length=512,
        scored_activation=ACTIVATION_IDENTITY,
        weights_manifest_sha256_value="abc123",
        weights_shard_count=1,
    )
    kw.update(over)
    return make_fingerprint(**kw)


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------
def test_put_get_roundtrip(tmp_path):
    repo = RerankCacheRepository(str(tmp_path / "r.db"))
    repo.put("k1", -4.5)
    assert repo.get("k1") == pytest.approx(-4.5)
    assert repo.count() == 1
    repo.close()


def test_missing_key_is_a_miss(tmp_path):
    repo = RerankCacheRepository(str(tmp_path / "r.db"))
    assert repo.get("nope") is None
    repo.close()


def test_corrupt_blob_degrades_to_miss_and_is_deleted(tmp_path):
    repo = RerankCacheRepository(str(tmp_path / "r.db"))
    repo.put("k1", 1.0)
    repo._conn.execute(
        "UPDATE rerank_scores SET score = ? WHERE cache_key = ?", (b"\x00\x01", "k1")
    )
    repo._conn.commit()
    assert repo.get("k1") is None          # 不回傳可疑值
    assert repo.count() == 0               # 壞列已刪除，下次重算
    repo.close()


def test_storage_is_struct_not_pickle(tmp_path):
    repo = RerankCacheRepository(str(tmp_path / "r.db"))
    repo.put("k1", 2.5)
    blob = repo._conn.execute(
        "SELECT score FROM rerank_scores WHERE cache_key = 'k1'"
    ).fetchone()[0]
    assert bytes(blob) == struct.pack("<d", 2.5)
    repo.close()


def test_db_file_is_created_in_its_own_path(tmp_path):
    path = tmp_path / "nested" / "r.db"
    repo = RerankCacheRepository(str(path))
    assert os.path.isfile(path)
    repo.close()


# --------------------------------------------------------------------------
# Cache key 語義
# --------------------------------------------------------------------------
def test_cache_key_depends_only_on_fingerprint_and_the_two_texts():
    fp = _fp()
    k = fp.cache_key("query", "chunk")
    assert k == fp.cache_key("query", "chunk")           # deterministic
    assert k != fp.cache_key("query2", "chunk")
    assert k != fp.cache_key("query", "chunk2")


def test_cache_key_changes_when_the_model_changes():
    a, b = _fp(), _fp(weights_manifest_sha256_value="different")
    assert a.cache_key("q", "c") != b.cache_key("q", "c")


def test_cache_key_changes_when_activation_semantics_change():
    """若改為快取已激活值，舊 entry 必須自動失效，不得靜默混用兩種語意。"""
    a, b = _fp(), _fp(scored_activation="sigmoid")
    assert a.cache_key("q", "c") != b.cache_key("q", "c")


def test_batch_size_is_absent_from_the_fingerprint():
    """batch size 已由 R0 gate 驗證為純執行參數 —— 不得出現在 fingerprint。"""
    d = _fp().to_canonical_dict()
    assert not any("batch" in k for k in d)


def test_fingerprint_does_not_leak_absolute_paths():
    fp = _fp(model_name=r"C:\Users\someone\models\bge-reranker-base")
    assert fp.to_canonical_dict()["model_name"] == "bge-reranker-base"


# --------------------------------------------------------------------------
# Weights manifest（多 shard）
# --------------------------------------------------------------------------
def test_manifest_is_sorted_and_deterministic(tmp_path):
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"bbb")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"aaa")
    (tmp_path / "model.safetensors.index.json").write_bytes(b"{}")
    (tmp_path / "README.md").write_bytes(b"ignore me")

    manifest = build_weights_manifest(str(tmp_path))
    paths = [e["relative_posix_path"] for e in manifest]

    assert paths == sorted(paths)
    assert paths == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ]
    assert build_weights_manifest(str(tmp_path)) == manifest  # deterministic


def test_manifest_uses_relative_posix_paths_only(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "pytorch_model.bin").write_bytes(b"x")
    manifest = build_weights_manifest(str(tmp_path))
    assert manifest[0]["relative_posix_path"] == "sub/pytorch_model.bin"
    assert "\\" not in manifest[0]["relative_posix_path"]
    assert str(tmp_path) not in manifest[0]["relative_posix_path"]


def test_manifest_hash_changes_with_content(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"aaa")
    h1 = weights_manifest_sha256(build_weights_manifest(str(tmp_path)))
    f.write_bytes(b"bbb")
    h2 = weights_manifest_sha256(build_weights_manifest(str(tmp_path)))
    assert h1 != h2


def test_missing_dir_yields_unknown_not_a_fabricated_hash():
    assert build_weights_manifest(None) == []
    assert weights_manifest_sha256([]) == MANIFEST_UNKNOWN


# --------------------------------------------------------------------------
# FingerprintedRerankCache facade
# --------------------------------------------------------------------------
def test_facade_hit_miss_accounting(tmp_path):
    repo = RerankCacheRepository(str(tmp_path / "r.db"))
    cache = FingerprintedRerankCache(repo, _fp())

    assert cache.get_many("q", ["a", "b"]) == [None, None]
    assert (cache.hits, cache.misses) == (0, 2)

    cache.put_many("q", ["a"], [1.5])
    cache.reset_stats()
    assert cache.get_many("q", ["a", "b"]) == [1.5, None]
    assert (cache.hits, cache.misses) == (1, 1)
    repo.close()


def test_disabled_cache_always_misses_and_never_writes(tmp_path):
    repo = RerankCacheRepository(str(tmp_path / "r.db"))
    cache = FingerprintedRerankCache(repo, _fp(), enabled=False)
    cache.put_many("q", ["a"], [1.0])
    assert cache.get_many("q", ["a"]) == [None]
    assert repo.count() == 0  # uncached feasibility 量測不得被快取污染
    repo.close()


def test_put_many_length_mismatch_is_rejected(tmp_path):
    repo = RerankCacheRepository(str(tmp_path / "r.db"))
    cache = FingerprintedRerankCache(repo, _fp())
    with pytest.raises(ValueError, match="長度不符"):
        cache.put_many("q", ["a", "b"], [1.0])
    repo.close()
