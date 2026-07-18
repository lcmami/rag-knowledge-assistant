"""文件生命週期整合測試（本階段 §四）。

將 scripts/verify_lifecycle.py 中最關鍵的行為固化為正式 integration test：
真實 Qdrant Server + 真實 SQLite 暫存 DB + 真實 storage 暫存目錄 + 真實 bge-small-zh 模型。
不呼叫 LLM。

執行：
    pytest -m integration
（需先啟動 localhost Qdrant；未就緒時整個模組會 skip 而非失敗。）
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_NAME = "equipment_alarm_manual.md"
MD_PATH = os.path.join(ROOT, "sample_docs", MD_NAME)


@pytest.fixture
def client(monkeypatch, tmp_path):
    # 真實 Qdrant；模型已由前階段快取，離線載入避免網路噪音
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    # 使用獨立 collection，避免污染正式資料
    monkeypatch.setenv("QDRANT_COLLECTION", "factory_test_lifecycle")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("SKIP_STARTUP_SERVICES", raising=False)

    # 若 Qdrant 未就緒，skip 而非 fail
    try:
        from qdrant_client import QdrantClient

        probe = QdrantClient(url=qdrant_url)
        probe.get_collections()
        probe.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"真實 Qdrant 未就緒（{qdrant_url}）：{exc}")

    from app.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        try:
            yield c
        finally:
            # 清理測試專用 collection
            try:
                app.state.vector_store.client.delete_collection("factory_test_lifecycle")
            except Exception:  # noqa: BLE001
                pass


def test_document_lifecycle_end_to_end(client, tmp_path):
    app = client.app
    store = app.state.vector_store
    repo = app.state.document_repository

    md_bytes = open(MD_PATH, "rb").read()

    # 1. 上傳 Markdown 成功
    r = client.post(
        "/api/documents",
        files={"file": (MD_NAME, md_bytes, "text/markdown")},
    )
    assert r.status_code == 201, r.text
    doc_id = r.json()["document_id"]
    record = repo.get(doc_id)
    assert record is not None and record.status == "indexed"

    # 2. SQLite chunk_count 等於 Qdrant point count
    point_count = store.count_by_document_id(doc_id)
    assert point_count > 0
    assert record.chunk_count == point_count == r.json()["chunk_count"]

    # 3. Top-K 能命中剛上傳的內容
    result = app.state.retrieval_service.retrieve(
        "設備出現 E102 警報時，應該先檢查什麼？", top_k=10
    )
    assert any(rc.document_id == doc_id for rc in result.results)

    # 4. 重複上傳回 409
    r_dup = client.post(
        "/api/documents",
        files={"file": (MD_NAME, md_bytes, "text/markdown")},
    )
    assert r_dup.status_code == 409
    assert r_dup.json()["error"]["details"]["document_id"] == doc_id

    # 5. 刪除後：Qdrant points 為 0、SQLite 記錄不存在、本地檔案不存在
    doc_dir = os.path.dirname(record.storage_path)
    r_del = client.delete(f"/api/documents/{doc_id}")
    assert r_del.status_code == 204
    assert store.count_by_document_id(doc_id) == 0
    assert repo.get(doc_id) is None
    assert not os.path.isdir(doc_dir)
