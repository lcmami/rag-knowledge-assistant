"""Persistent content-hash Embedding Cache（v1.1 Slice 3 §十六 / §十七 / §十八）。

- :class:`EmbeddingCacheRepository`：獨立、可注入路徑的 SQLite Cache（**不**寫入正式
  ``metadata.db**，也不與 Optimizer run DB 混用；float32 binary、非 pickle；corrupt entry 視為
  miss 並安全重算）。
- :class:`CacheStats`：每個 Trial 的 cache 統計（§十八；batch dedup savings 與 persistent hit 分開）。
- :class:`CachingDocumentEmbeddingService`：以 decorator 包裝現有 Embedding Service，**只快取
  Document Chunk Embedding**（§十七.10）；Query Embedding 一律不快取，避免污染 latency 與語意。

Cache 只保存向量與 hash metadata，**不**保存 chunk 全文、API key（§十六.9 / §十四.1）。
"""
from __future__ import annotations

import math
import os
import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from evaluation.v11.embedding_fingerprint import (
    CACHE_SCHEMA_VERSION,
    EmbeddingFingerprint,
)

# get() 狀態
HIT = "hit"
MISS = "miss"
CORRUPT = "corrupt"


class CacheMigrationError(Exception):
    """Cache migration 失敗（已 rollback）。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# float32 binary 編解碼（明確 little-endian，非 pickle；§十六.3/4）
# --------------------------------------------------------------------------
def encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(blob: bytes, dimension: int) -> Optional[list[float]]:
    """解碼 float32 blob；長度不符 / 非有限值 → ``None``（視為 corrupt，§十六.5）。"""
    if len(blob) != dimension * 4:
        return None
    try:
        values = list(struct.unpack(f"<{dimension}f", blob))
    except struct.error:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _is_finite_vector(vector: list[float], dimension: int) -> bool:
    return len(vector) == dimension and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in vector
    )


# --------------------------------------------------------------------------
# get 結果
# --------------------------------------------------------------------------
@dataclass
class CacheGet:
    status: str  # HIT | MISS | CORRUPT
    vector: Optional[list[float]] = None


# --------------------------------------------------------------------------
# migration（獨立 schema_migrations；idempotent + rollback）
# --------------------------------------------------------------------------
_MIGRATION_0001 = [
    """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        cache_key                   TEXT PRIMARY KEY,
        cache_schema_version        TEXT NOT NULL,
        embedding_fingerprint_hash  TEXT NOT NULL,
        text_sha256                 TEXT NOT NULL,
        dimension                   INTEGER NOT NULL,
        vector_blob                 BLOB NOT NULL,
        created_at                  TEXT NOT NULL,
        last_used_at                TEXT NOT NULL,
        hit_count                   INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_embcache_fp ON embedding_cache(embedding_fingerprint_hash)",
]

_MIGRATIONS: list[tuple[str, list[str]]] = [("0001_embedding_cache", _MIGRATION_0001)]

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""


def _run_migrations(conn: sqlite3.Connection) -> list[str]:
    conn.execute(_SCHEMA_MIGRATIONS_DDL)
    conn.commit()
    applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    newly: list[str] = []
    for version, statements in _MIGRATIONS:
        if version in applied:
            continue
        try:
            conn.execute("BEGIN")
            for stmt in statements:
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, _now()),
            )
            conn.execute("COMMIT")
        except Exception as exc:  # noqa: BLE001
            conn.execute("ROLLBACK")
            raise CacheMigrationError(f"cache migration {version!r} 失敗，已 rollback：{exc}") from exc
        newly.append(version)
    return newly


class EmbeddingCacheRepository:
    """獨立、可注入路徑的 float32 Embedding Cache（預設僅供 Optimizer）。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        try:
            _run_migrations(conn)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def applied_migrations(self) -> list[str]:
        conn = self._connect()
        try:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
            ]
        finally:
            conn.close()

    # --- get（批次；corrupt 視為 miss 並移除壞資料）---
    def get_many(self, cache_keys: list[str], dimension: int) -> dict[str, CacheGet]:
        if not cache_keys:
            return {}
        out: dict[str, CacheGet] = {k: CacheGet(MISS) for k in cache_keys}
        conn = self._connect()
        try:
            corrupt_keys: list[str] = []
            hit_keys: list[str] = []
            # 分批 IN 查詢
            uniq = list(dict.fromkeys(cache_keys))
            for i in range(0, len(uniq), 400):
                batch = uniq[i : i + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT cache_key, dimension, vector_blob FROM embedding_cache "
                    f"WHERE cache_key IN ({placeholders})",
                    batch,
                ).fetchall()
                for r in rows:
                    key = r["cache_key"]
                    if r["dimension"] != dimension:
                        out[key] = CacheGet(CORRUPT)
                        corrupt_keys.append(key)
                        continue
                    vec = decode_vector(r["vector_blob"], dimension)
                    if vec is None:
                        out[key] = CacheGet(CORRUPT)
                        corrupt_keys.append(key)
                    else:
                        out[key] = CacheGet(HIT, vec)
                        hit_keys.append(key)
            # corrupt → 移除壞資料（安全重算，§十六.6）
            if corrupt_keys:
                conn.execute("BEGIN")
                conn.executemany(
                    "DELETE FROM embedding_cache WHERE cache_key=?",
                    [(k,) for k in corrupt_keys],
                )
                conn.execute("COMMIT")
            # hit → 更新 hit_count / last_used_at
            if hit_keys:
                ts = _now()
                conn.execute("BEGIN")
                conn.executemany(
                    "UPDATE embedding_cache SET hit_count=hit_count+1, last_used_at=? "
                    "WHERE cache_key=?",
                    [(ts, k) for k in hit_keys],
                )
                conn.execute("COMMIT")
        finally:
            conn.close()
        return out

    def get(self, cache_key: str, dimension: int) -> CacheGet:
        return self.get_many([cache_key], dimension)[cache_key]

    # --- put（float32；INSERT OR REPLACE 覆寫壞 / 舊資料）---
    def put_many(
        self,
        entries: list[tuple[str, str, str, int, list[float]]],
    ) -> None:
        """entries: ``[(cache_key, fingerprint_hash, text_sha256, dimension, vector)]``。"""
        if not entries:
            return
        ts = _now()
        rows = []
        for cache_key, fp_hash, tsha, dim, vector in entries:
            if not _is_finite_vector(vector, dim):
                raise ValueError(f"拒絕寫入非有限 / 維度不符向量 cache_key={cache_key[:12]}…")
            rows.append(
                (cache_key, CACHE_SCHEMA_VERSION, fp_hash, tsha, dim, encode_vector(vector), ts, ts)
            )
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.executemany(
                """
                INSERT INTO embedding_cache
                    (cache_key, cache_schema_version, embedding_fingerprint_hash,
                     text_sha256, dimension, vector_blob, created_at, last_used_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    cache_schema_version=excluded.cache_schema_version,
                    embedding_fingerprint_hash=excluded.embedding_fingerprint_hash,
                    text_sha256=excluded.text_sha256,
                    dimension=excluded.dimension,
                    vector_blob=excluded.vector_blob,
                    last_used_at=excluded.last_used_at
                """,
                rows,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def put(
        self, cache_key: str, fingerprint_hash: str, text_sha: str, dimension: int, vector: list[float]
    ) -> None:
        self.put_many([(cache_key, fingerprint_hash, text_sha, dimension, vector)])

    def delete(self, cache_key: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM embedding_cache WHERE cache_key=?", (cache_key,))
        finally:
            conn.close()

    def clear(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM embedding_cache")
            return cur.rowcount
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(hit_count),0) AS hits FROM embedding_cache"
            ).fetchone()
        finally:
            conn.close()
        return {"entry_count": row["n"], "total_hit_count": row["hits"]}

    def has_documents_table(self) -> bool:
        """驗證 Cache DB **不**含正式 documents 表（§十六：不混入 Document Registry）。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
            ).fetchone()
        finally:
            conn.close()
        return row is not None


# --------------------------------------------------------------------------
# Cache Stats（§十八）
# --------------------------------------------------------------------------
@dataclass
class CacheStats:
    cache_enabled: bool = True
    document_chunk_count: int = 0
    unique_embedding_text_count: int = 0
    embedding_cache_hit_count: int = 0
    embedding_cache_miss_count: int = 0
    embedding_model_call_text_count: int = 0
    embedding_cache_corrupt_count: int = 0
    embedding_cache_write_failure_count: int = 0

    @property
    def batch_dedup_savings(self) -> int:
        """同 Batch 重複文字去重節省（**非** persistent cache hit；§十八 note）。"""
        return max(0, self.document_chunk_count - self.unique_embedding_text_count)

    @property
    def hit_rate(self) -> Optional[float]:
        denom = self.embedding_cache_hit_count + self.embedding_cache_miss_count
        if denom == 0:
            return None  # 分母 0：不可計算（§十八：不得 crash / 不假裝）
        return self.embedding_cache_hit_count / denom

    def to_dict(self) -> dict[str, Any]:
        rate = self.hit_rate
        return {
            "cache_enabled": self.cache_enabled,
            "document_chunk_count": self.document_chunk_count,
            "unique_embedding_text_count": self.unique_embedding_text_count,
            "embedding_cache_hit_count": self.embedding_cache_hit_count,
            "embedding_cache_miss_count": self.embedding_cache_miss_count,
            "embedding_model_call_text_count": self.embedding_model_call_text_count,
            "embedding_cache_corrupt_count": self.embedding_cache_corrupt_count,
            "embedding_cache_write_failure_count": self.embedding_cache_write_failure_count,
            "batch_dedup_savings": self.batch_dedup_savings,
            "embedding_cache_hit_rate": rate,
            "embedding_cache_hit_rate_computable": rate is not None,
        }


# --------------------------------------------------------------------------
# Caching Document Embedding Service（decorator；只快取 document embedding）
# --------------------------------------------------------------------------
class CachingDocumentEmbeddingService:
    """包裝現有 Embedding Service，只對 **document chunk embedding** 加 persistent cache。

    Query Embedding 一律**不**快取（§十七.9/10）：``embed_query`` 直接委派底層服務。名稱明示
    「Document」以避免與 query 混淆（§十七.9）。
    """

    def __init__(
        self,
        underlying: Any,
        repository: Optional[EmbeddingCacheRepository],
        fingerprint: EmbeddingFingerprint,
        *,
        enabled: bool = True,
        warn: Optional[Callable[[str], None]] = None,
    ):
        self._underlying = underlying
        self._repo = repository
        self._fp = fingerprint
        self._enabled = bool(enabled and repository is not None)
        self._warn = warn or (lambda _m: None)
        self.stats = CacheStats(cache_enabled=self._enabled)

    # drop-in 相容 EmbeddingService 介面
    @property
    def dimension(self) -> int:
        return self._underlying.dimension

    @property
    def model_name(self) -> str:
        return getattr(self._underlying, "model_name", "")

    def count_tokens(self, text: str) -> int:
        return self._underlying.count_tokens(text)

    def embed_query(self, question: str) -> list[float]:
        # Query embedding 不快取（避免污染 latency / 語意；§十七.9/10）。
        return self._underlying.embed_query(question)

    def reset_stats(self) -> None:
        self.stats = CacheStats(cache_enabled=self._enabled)

    def snapshot_stats(self) -> CacheStats:
        return self.stats

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # cache 關閉 → 完全退回原服務（§十七.7）。
        if not self._enabled:
            self.stats.document_chunk_count += len(texts)
            self.stats.embedding_model_call_text_count += len(texts)
            return self._underlying.embed_documents(texts)

        dim = self._fp.dimension
        self.stats.document_chunk_count += len(texts)

        # 每筆算 cache key；同 batch 重複文字（相同 key）只 embed 一次（§十七.2）。
        keys = [self._fp.cache_key(t) for t in texts]
        # 依首次出現順序取唯一 key → text。
        unique_key_to_text: dict[str, str] = {}
        for key, text in zip(keys, texts):
            unique_key_to_text.setdefault(key, text)
        unique_keys = list(unique_key_to_text.keys())
        self.stats.unique_embedding_text_count += len(unique_keys)

        # 批次讀 cache。
        got = self._repo.get_many(unique_keys, dim)

        resolved: dict[str, list[float]] = {}
        miss_keys: list[str] = []
        for key in unique_keys:
            g = got.get(key, CacheGet(MISS))
            if g.status == HIT and g.vector is not None:
                resolved[key] = g.vector
                self.stats.embedding_cache_hit_count += 1
            else:
                if g.status == CORRUPT:
                    self.stats.embedding_cache_corrupt_count += 1
                miss_keys.append(key)
                self.stats.embedding_cache_miss_count += 1

        # 只把 miss 的**唯一**文字送底層（§十七.3 部分 hit/miss）。
        if miss_keys:
            miss_texts = [unique_key_to_text[k] for k in miss_keys]
            # 底層失敗 → 直接拋出，不寫入半成品（§十七.4）。
            vectors = self._underlying.embed_documents(miss_texts)
            if len(vectors) != len(miss_texts):
                raise ValueError("底層 embedding 回傳數量與輸入不符。")
            self.stats.embedding_model_call_text_count += len(miss_texts)

            put_entries = []
            for key, text, vec in zip(miss_keys, miss_texts, vectors):
                vec = list(vec)
                if not _is_finite_vector(vec, dim):
                    raise ValueError("底層 embedding 回傳非有限 / 維度不符向量。")
                resolved[key] = vec
                put_entries.append(
                    (key, self._fp.fingerprint_hash(), _text_sha(text), dim, vec)
                )
            # Cache 寫入失敗不得使 Trial 失敗（§十七.5）——記錄 warning + 計數後繼續。
            try:
                self._repo.put_many(put_entries)
            except Exception as exc:  # noqa: BLE001
                self.stats.embedding_cache_write_failure_count += len(put_entries)
                self._warn(f"embedding cache 寫入失敗（已略過，不影響 Trial）：{exc}")

        # 依原輸入順序組回（§十七.1）。
        return [resolved[key] for key in keys]


def _text_sha(text: str) -> str:
    from evaluation.v11.embedding_fingerprint import text_sha256

    return text_sha256(text)
