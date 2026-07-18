"""文件 API 測試（本階段 §三）。

使用 FastAPI TestClient 與 dependency override，套用真實 Router → Service → Repository
路徑，但以 Fake Embedding / Fake Vector Store 取代重量級依賴。
設定 SKIP_STARTUP_SERVICES=1，避免 lifespan 載入模型或連線 Qdrant。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_document_repository, get_indexing_service
from app.main import create_app

MD_BYTES = "# 品質須知\n\nAOI 檢測為 NG 時需記錄批號與缺陷座標。\n".encode("utf-8")


@pytest.fixture
def client(monkeypatch, indexing_service, repository):
    monkeypatch.setenv("SKIP_STARTUP_SERVICES", "1")
    app = create_app()
    app.dependency_overrides[get_indexing_service] = lambda: indexing_service
    app.dependency_overrides[get_document_repository] = lambda: repository
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_returns_200(client):
    """1. /health 回 200。"""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_valid_markdown_upload_returns_201(client):
    """2. 有效 Markdown 上傳回 201。"""
    r = client.post(
        "/api/documents",
        files={"file": ("note.md", MD_BYTES, "text/markdown")},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "indexed"
    assert body["chunk_count"] > 0


def test_duplicate_upload_returns_409_with_document_id(client):
    """3. 相同 bytes 第二次上傳回 409，並包含既有 document_id。"""
    r1 = client.post("/api/documents", files={"file": ("note.md", MD_BYTES, "text/markdown")})
    assert r1.status_code == 201
    existing_id = r1.json()["document_id"]

    r2 = client.post("/api/documents", files={"file": ("note.md", MD_BYTES, "text/markdown")})
    assert r2.status_code == 409
    assert r2.json()["error"]["details"]["document_id"] == existing_id


def test_blank_document_returns_400(client):
    """4. 空白文件回 400。"""
    r = client.post(
        "/api/documents",
        files={"file": ("blank.txt", b"   \n  \t\n", "text/plain")},
    )
    assert r.status_code == 400


def test_unsupported_extension_returns_415(client):
    """5. 不支援副檔名回 415（.pdf 自 Phase 4 起已支援，改用 .docx 驗證）。"""
    r = client.post(
        "/api/documents",
        files={
            "file": (
                "report.docx",
                b"PK\x03\x04 dummy docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert r.status_code == 415


def test_list_documents_returns_documents_and_total(client):
    """6. /api/documents 回傳 documents 與 total。"""
    client.post("/api/documents", files={"file": ("note.md", MD_BYTES, "text/markdown")})
    r = client.get("/api/documents")
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body and "total" in body
    assert body["total"] == len(body["documents"]) == 1


def test_delete_missing_document_returns_404(client):
    """7. 刪除不存在文件回 404。"""
    r = client.delete("/api/documents/does-not-exist")
    assert r.status_code == 404


def test_ready_reports_not_initialized_metadata_db(monkeypatch):
    """8. 未初始化 metadata repository 時 metadata_db=not_initialized 且 /ready 回 503。"""
    monkeypatch.setenv("SKIP_STARTUP_SERVICES", "1")
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["metadata_db"] == "not_initialized"


def test_unhandled_error_does_not_leak_traceback(monkeypatch):
    """9. 不得在 API response 中外洩 traceback。"""
    monkeypatch.setenv("SKIP_STARTUP_SERVICES", "1")
    app = create_app()

    class Boom:
        def index_document(self, *args, **kwargs):
            raise RuntimeError("SENSITIVE_INTERNAL_/secret/path 詳細堆疊")

    app.dependency_overrides[get_indexing_service] = lambda: Boom()
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/api/documents", files={"file": ("a.md", MD_BYTES, "text/markdown")})

    assert r.status_code == 500
    text = r.text
    assert "Traceback" not in text
    assert "SENSITIVE_INTERNAL" not in text
    assert r.json()["error"]["code"] == "INTERNAL_ERROR"
