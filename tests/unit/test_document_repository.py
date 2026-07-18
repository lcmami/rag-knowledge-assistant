"""Document Repository 單元測試（本階段 §二）。

不依賴 Qdrant / Embedding / LLM，只驗證 SQLite registry 的生命週期行為。
"""
from __future__ import annotations

import sqlite3

import pytest


def _connect(repository) -> sqlite3.Connection:
    conn = sqlite3.connect(repository.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _create(repository, *, document_id="doc-1", file_hash="hash-1", status="indexing"):
    return repository.create(
        document_id=document_id,
        filename="manual.md",
        file_hash=file_hash,
        file_type="text/markdown",
        file_size=123,
        status=status,
        storage_path=f"storage/uploads/{document_id}/manual.md",
    )


def test_documents_table_exists_after_init(repository):
    """1. SQLite 初始化後 documents table 存在。"""
    with _connect(repository) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
    assert row is not None and row["name"] == "documents"


def test_create_indexing_document(repository):
    """2. 可建立 status=indexing 的文件。"""
    record = _create(repository, status="indexing")
    assert record.status == "indexing"
    assert record.chunk_count == 0
    assert repository.get("doc-1").status == "indexing"


def test_mark_indexed_sets_chunk_count(repository):
    """3. 可更新為 indexed 並寫入 chunk_count。"""
    _create(repository)
    repository.mark_indexed("doc-1", chunk_count=7)
    record = repository.get("doc-1")
    assert record.status == "indexed"
    assert record.chunk_count == 7
    assert record.error_message is None


def test_mark_failed_preserves_error_message(repository):
    """4. 可更新為 failed 並保存 error_message。"""
    _create(repository)
    repository.mark_failed("doc-1", error_message="寫入 Qdrant 失敗")
    record = repository.get("doc-1")
    assert record.status == "failed"
    assert record.error_message == "寫入 Qdrant 失敗"


def test_file_hash_unique_constraint(repository):
    """5. file_hash UNIQUE constraint 能阻止重複記錄。"""
    _create(repository, document_id="doc-1", file_hash="dup")
    with pytest.raises(sqlite3.IntegrityError):
        _create(repository, document_id="doc-2", file_hash="dup")


def test_ping_performs_real_write_and_readback(repository):
    """6. ping() 確實執行 SQLite 寫入與讀回，而非固定回傳成功。"""
    # 正常情況：不應拋出例外，且 _health 有實際寫入的 row
    repository.ping()
    with _connect(repository) as conn:
        row = conn.execute("SELECT checked_at FROM _health WHERE id=1").fetchone()
    assert row is not None and row["checked_at"]

    # 破壞底層 table 後，ping() 必須失敗——證明它真的對 SQLite 讀寫，
    # 而不是無條件回傳成功。
    with _connect(repository) as conn:
        conn.execute("DROP TABLE _health")
        conn.commit()
    with pytest.raises(sqlite3.Error):
        repository.ping()
