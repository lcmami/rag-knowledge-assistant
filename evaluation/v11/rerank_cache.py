"""Rerank Score Cache（v1.2.1 Slice R1）—— 比照 :mod:`evaluation.v11.embedding_cache`。

Optimizer 一輪會對同一組 (query, chunk) 重跑多個候選;R0 實測 cross-encoder 每 pair 約
200 ms,沒有快取的話 grid search 在 CPU 上跑不完。**快取是讓搜尋可行的工程手段。**

**硬性誠實規則**:快取**不得**用來壓低 feasibility latency。`p95_retrieval_latency_ms`
一律以 uncached 量測為準（見 R0/R0.5 baseline 產物）。

儲存格式沿用 embedding cache 的既定選擇:``float64`` struct little-endian,**非 pickle**;
資料損壞 → miss + 刪除該列 + 重算,絕不回傳可疑值。

Cache key = ``SHA256(key_algo_ver + cache_schema_ver + rerank_fingerprint_hash +
sha256(query) + sha256(chunk_text))`` —— 只依模型 fingerprint 與確切文字,
**不依** threshold / top_k / rerank_top_n / batch size / trial / collection
（§0(2)：batch size 已由 R0 gate 驗證為純執行參數）。
"""
from __future__ import annotations

import os
import sqlite3
import struct
from typing import Optional, Sequence

from evaluation.v11.rerank_fingerprint import RerankFingerprint

_SCORE_FORMAT = "<d"  # little-endian float64
_SCORE_SIZE = struct.calcsize(_SCORE_FORMAT)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rerank_scores (
    cache_key TEXT PRIMARY KEY,
    score BLOB NOT NULL
);
"""


class RerankCacheRepository:
    """單檔 SQLite 快取。與正式 metadata.db / optimizer run DB **完全分離**。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    def get(self, cache_key: str) -> Optional[float]:
        row = self._conn.execute(
            "SELECT score FROM rerank_scores WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        blob = row[0]
        if not isinstance(blob, (bytes, bytearray)) or len(blob) != _SCORE_SIZE:
            # 損壞 → 視為 miss 並刪除，讓呼叫端重算（絕不回傳可疑值）。
            self.delete(cache_key)
            return None
        try:
            return float(struct.unpack(_SCORE_FORMAT, bytes(blob))[0])
        except struct.error:
            self.delete(cache_key)
            return None

    def put(self, cache_key: str, score: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO rerank_scores (cache_key, score) VALUES (?, ?)",
            (cache_key, struct.pack(_SCORE_FORMAT, float(score))),
        )
        self._conn.commit()

    def delete(self, cache_key: str) -> None:
        self._conn.execute("DELETE FROM rerank_scores WHERE cache_key = ?", (cache_key,))
        self._conn.commit()

    def count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM rerank_scores").fetchone()[0]
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - 關閉失敗無副作用
            pass


class FingerprintedRerankCache:
    """把 fingerprint 綁進 key 的 cache facade（實作 rerank_stage 的 cache 介面）。

    ``enabled=False`` 時所有讀取皆 miss、寫入皆 no-op —— 用於 uncached feasibility 量測。
    """

    def __init__(
        self,
        repository: Optional[RerankCacheRepository],
        fingerprint: RerankFingerprint,
        *,
        enabled: bool = True,
    ):
        self._repo = repository
        self._fingerprint = fingerprint
        self.enabled = bool(enabled) and repository is not None
        self.hits = 0
        self.misses = 0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    def get_many(self, query: str, texts: Sequence[str]) -> list[Optional[float]]:
        if not self.enabled:
            self.misses += len(texts)
            return [None] * len(texts)
        out: list[Optional[float]] = []
        for text in texts:
            value = self._repo.get(self._fingerprint.cache_key(query, text))
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            out.append(value)
        return out

    def put_many(
        self, query: str, texts: Sequence[str], scores: Sequence[float]
    ) -> None:
        if not self.enabled:
            return
        if len(texts) != len(scores):
            raise ValueError(
                f"put_many: texts({len(texts)}) 與 scores({len(scores)}) 長度不符。"
            )
        for text, score in zip(texts, scores):
            self._repo.put(self._fingerprint.cache_key(query, text), float(score))
