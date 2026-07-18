"""SQLite 文件 Registry（規格 §10 / 本切片 §一）。

Qdrant 負責 Chunk 向量與 Payload；SQLite 負責文件層級資訊與生命週期。

- documents table：file_hash 具 UNIQUE constraint。
- status：indexing / indexed / failed。
- ping()：對 SQLite 做真實讀寫，供 /ready 的 metadata_db 檢查（本切片 §八）。

每個操作開啟短生命週期連線，避免跨執行緒共用問題
（FastAPI 同步端點於 threadpool 執行）。
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_hash     TEXT NOT NULL UNIQUE,
    file_type     TEXT NOT NULL,
    file_size     INTEGER NOT NULL,
    status        TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    storage_path  TEXT,
    error_message TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
)
"""

_CREATE_HEALTH = """
CREATE TABLE IF NOT EXISTS _health (
    id         INTEGER PRIMARY KEY,
    checked_at TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocumentRecord:
    id: str
    filename: str
    file_hash: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    storage_path: str | None
    error_message: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DocumentRecord":
        return cls(
            id=row["id"],
            filename=row["filename"],
            file_hash=row["file_hash"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            storage_path=row["storage_path"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class DocumentRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_DOCUMENTS)
            conn.execute(_CREATE_HEALTH)
            conn.commit()

    # --- readiness：真實讀寫檢查（本切片 §八）---
    def ping(self) -> None:
        """對 SQLite 執行一次真實寫入 + 讀回，任一步失敗則拋出例外。"""
        with self._connect() as conn:
            ts = _now()
            conn.execute(
                "INSERT INTO _health (id, checked_at) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET checked_at=excluded.checked_at",
                (ts,),
            )
            conn.commit()
            row = conn.execute("SELECT checked_at FROM _health WHERE id=1").fetchone()
            if row is None or row["checked_at"] != ts:
                raise RuntimeError("SQLite 讀寫檢查失敗。")

    # --- CRUD ---
    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash=?", (file_hash,)
            ).fetchone()
            return DocumentRecord.from_row(row) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            return DocumentRecord.from_row(row) if row else None

    def create(
        self,
        *,
        document_id: str,
        filename: str,
        file_hash: str,
        file_type: str,
        file_size: int,
        status: str,
        storage_path: str | None,
    ) -> DocumentRecord:
        ts = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents "
                "(id, filename, file_hash, file_type, file_size, status, "
                " chunk_count, storage_path, error_message, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, ?, ?)",
                (
                    document_id,
                    filename,
                    file_hash,
                    file_type,
                    file_size,
                    status,
                    storage_path,
                    ts,
                    ts,
                ),
            )
            conn.commit()
        return self.get(document_id)  # type: ignore[return-value]

    def mark_indexed(self, document_id: str, chunk_count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status=?, chunk_count=?, error_message=NULL, "
                "updated_at=? WHERE id=?",
                ("indexed", chunk_count, _now(), document_id),
            )
            conn.commit()

    def mark_failed(self, document_id: str, error_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status=?, error_message=?, updated_at=? WHERE id=?",
                ("failed", error_message, _now(), document_id),
            )
            conn.commit()

    def list_all(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
            return [DocumentRecord.from_row(r) for r in rows]

    def delete(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            conn.commit()
