"""Indexing Service 單元測試（本階段 §二）。

使用 Fake Embedding 與 Fake Vector Store，不依賴 Qdrant / 模型 / LLM。
"""
from __future__ import annotations

import os

import pytest

from app.core.exceptions import DuplicateDocumentError, VectorStoreError
import app.services.indexing_service as indexing_module

MD_BYTES = "# 設備警報\n\nE102 警報代表溫度過高，應先停機檢查冷卻系統。\n".encode("utf-8")


def test_duplicate_bytes_detected(indexing_service):
    """1. 相同 bytes 會辨識為重複文件。"""
    first = indexing_service.index_document("a.md", MD_BYTES)
    with pytest.raises(DuplicateDocumentError) as excinfo:
        indexing_service.index_document("a.md", MD_BYTES)
    # 409 應附既有 document_id
    assert excinfo.value.details.get("document_id") == first.id


def test_successful_index_marks_indexed(indexing_service, fake_vector_store):
    """2. 成功匯入後狀態為 indexed。"""
    record = indexing_service.index_document("a.md", MD_BYTES)
    assert record.status == "indexed"
    assert record.chunk_count > 0
    # Qdrant 內 point 數與 chunk_count 一致
    assert fake_vector_store.count_by_document_id(record.id) == record.chunk_count


def test_failure_marks_failed_and_does_not_return_indexed(indexing_service, repository, fake_vector_store):
    """3./5. 失敗時狀態為 failed，且不回傳成功或 indexed。"""
    fake_vector_store.upsert_should_fail = True
    with pytest.raises(VectorStoreError):
        indexing_service.index_document("a.md", MD_BYTES)

    records = repository.list_all()
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].status != "indexed"
    assert records[0].error_message


def test_failure_triggers_qdrant_rollback(indexing_service, repository, fake_vector_store):
    """4. 失敗時會呼叫 Qdrant rollback（刪除該 document_id 的 points）。"""
    fake_vector_store.upsert_should_fail = True
    order: list[str] = []
    fake_vector_store.order = order

    with pytest.raises(VectorStoreError):
        indexing_service.index_document("a.md", MD_BYTES)

    # rollback 應對 Qdrant 執行一次 delete
    assert order == ["qdrant"]
    doc_id = repository.list_all()[0].id
    assert fake_vector_store.count_by_document_id(doc_id) == 0


def test_delete_order_qdrant_then_file_then_sqlite(
    indexing_service, repository, fake_vector_store, monkeypatch
):
    """6. 刪除順序必須是：Qdrant → 本地檔案 → SQLite。"""
    record = indexing_service.index_document("a.md", MD_BYTES)
    doc_id = record.id
    doc_dir = os.path.dirname(record.storage_path)
    assert os.path.isdir(doc_dir)  # 匯入時已寫入本地檔案

    order: list[str] = []
    fake_vector_store.order = order  # delete 時 append "qdrant"

    orig_rmtree = indexing_module.shutil.rmtree

    def spy_rmtree(path, *args, **kwargs):
        order.append("file")
        return orig_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(indexing_module.shutil, "rmtree", spy_rmtree)

    orig_delete = repository.delete

    def spy_delete(document_id):
        order.append("sqlite")
        return orig_delete(document_id)

    monkeypatch.setattr(repository, "delete", spy_delete)

    indexing_service.delete_document(doc_id)

    assert order == ["qdrant", "file", "sqlite"]
    assert repository.get(doc_id) is None
    assert not os.path.isdir(doc_dir)


def test_qdrant_delete_failure_keeps_sqlite_record(
    indexing_service, repository, fake_vector_store
):
    """7. Qdrant 刪除失敗時，不得刪除 SQLite 記錄。"""
    record = indexing_service.index_document("a.md", MD_BYTES)
    doc_id = record.id
    doc_dir = os.path.dirname(record.storage_path)

    fake_vector_store.delete_should_fail = True
    with pytest.raises(VectorStoreError):
        indexing_service.delete_document(doc_id)

    # SQLite 記錄與本地檔案都必須保留
    assert repository.get(doc_id) is not None
    assert os.path.isdir(doc_dir)
