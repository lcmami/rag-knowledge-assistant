"""Embedding Cache Repository 單元測試（v1.1 Slice 3 §十六 / §二十二.8）。"""
from __future__ import annotations

import sqlite3
import struct

import pytest

from evaluation.v11.embedding_cache import (
    CORRUPT,
    HIT,
    MISS,
    EmbeddingCacheRepository,
    decode_vector,
    encode_vector,
)


def _repo(tmp_path):
    return EmbeddingCacheRepository(str(tmp_path / "c.embedding-cache.db"))


def test_migration_idempotent(tmp_path):
    p = str(tmp_path / "c.embedding-cache.db")
    r1 = EmbeddingCacheRepository(p)
    r2 = EmbeddingCacheRepository(p)  # 再開一次 → 不重複套用
    assert r1.applied_migrations() == ["0001_embedding_cache"]
    assert r2.applied_migrations() == ["0001_embedding_cache"]


def test_migration_rollback_on_failure(tmp_path):
    from evaluation.v11 import embedding_cache as mod

    conn = sqlite3.connect(str(tmp_path / "x.db"), isolation_level=None)
    conn.execute(mod._SCHEMA_MIGRATIONS_DDL)
    orig = mod._MIGRATIONS
    try:
        mod._MIGRATIONS = [("bad", ["CREATE TABLE ok(a)", "THIS IS NOT SQL"])]
        with pytest.raises(mod.CacheMigrationError):
            mod._run_migrations(conn)
        # rollback：壞 migration 不得留下半套 table。
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "ok" not in tables
    finally:
        mod._MIGRATIONS = orig
        conn.close()


def test_put_get_float32_roundtrip(tmp_path):
    repo = _repo(tmp_path)
    vec = [0.1, -2.0, 3.5, 4.25]
    repo.put("k1", "fp", "tsha", 4, vec)
    got = repo.get("k1", 4)
    assert got.status == HIT
    assert got.vector == decode_vector(encode_vector(vec), 4)


def test_miss_returns_miss(tmp_path):
    assert _repo(tmp_path).get("absent", 4).status == MISS


def test_dimension_mismatch_is_corrupt(tmp_path):
    repo = _repo(tmp_path)
    repo.put("k", "fp", "t", 4, [1.0, 2.0, 3.0, 4.0])
    got = repo.get("k", 8)  # 要求 8 維 → 視為 corrupt（不重用）
    assert got.status == CORRUPT
    # corrupt 後壞 entry 已移除。
    assert repo.get("k", 8).status == MISS


def test_bad_blob_length_is_corrupt(tmp_path):
    repo = _repo(tmp_path)
    conn = sqlite3.connect(repo.db_path)
    conn.execute(
        "INSERT INTO embedding_cache (cache_key, cache_schema_version, embedding_fingerprint_hash,"
        " text_sha256, dimension, vector_blob, created_at, last_used_at, hit_count)"
        " VALUES (?,?,?,?,?,?,?,?,0)",
        ("bad", "v1", "fp", "t", 4, b"123", "now", "now"),
    )
    conn.commit()
    conn.close()
    assert repo.get("bad", 4).status == CORRUPT


def test_nan_inf_is_corrupt(tmp_path):
    repo = _repo(tmp_path)
    conn = sqlite3.connect(repo.db_path)
    blob = struct.pack("<4f", float("nan"), float("inf"), 0.0, 0.0)
    conn.execute(
        "INSERT INTO embedding_cache (cache_key, cache_schema_version, embedding_fingerprint_hash,"
        " text_sha256, dimension, vector_blob, created_at, last_used_at, hit_count)"
        " VALUES (?,?,?,?,?,?,?,?,0)",
        ("nan", "v1", "fp", "t", 4, blob, "now", "now"),
    )
    conn.commit()
    conn.close()
    assert repo.get("nan", 4).status == CORRUPT


def test_hit_count_and_last_used_update(tmp_path):
    repo = _repo(tmp_path)
    repo.put("k", "fp", "t", 4, [1.0, 2.0, 3.0, 4.0])
    repo.get("k", 4)
    repo.get("k", 4)
    conn = sqlite3.connect(repo.db_path)
    row = conn.execute("SELECT hit_count FROM embedding_cache WHERE cache_key='k'").fetchone()
    conn.close()
    assert row[0] == 2


def test_stats_and_clear(tmp_path):
    repo = _repo(tmp_path)
    repo.put("k1", "fp", "t", 4, [1.0, 2.0, 3.0, 4.0])
    repo.put("k2", "fp", "t", 4, [1.0, 2.0, 3.0, 4.0])
    assert repo.stats()["entry_count"] == 2
    assert repo.clear() == 2
    assert repo.stats()["entry_count"] == 0


def test_reject_non_finite_on_put(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ValueError):
        repo.put("k", "fp", "t", 4, [float("nan"), 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        repo.put("k", "fp", "t", 4, [1.0, 2.0])  # 維度不符


def test_db_path_injectable_and_no_documents_table(tmp_path):
    repo = _repo(tmp_path)
    assert repo.db_path.endswith("c.embedding-cache.db")
    assert repo.has_documents_table() is False
