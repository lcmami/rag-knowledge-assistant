"""Retrieval 整合測試（Phase 4 §10.3）。

真實 Qdrant Server + 真實 BGE Embedding + 真實 SQLite + 真實 storage 暫存目錄，
匯入 Markdown / TXT / PDF 各一份，透過 /api/retrieve 驗證跨格式檢索與 PDF 頁碼。
不呼叫 LLM，使用獨立 collection，teardown 清除，不污染 factory_knowledge。

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
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
TEST_COLLECTION = "factory_test_retrieval"

MD_NAME = "equipment_alarm_manual.md"
TXT_NAME = "manufacturing_data_dictionary.txt"
PDF_NAME = "equipment_maintenance_manual.pdf"


@pytest.fixture
def client(monkeypatch, tmp_path):
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("QDRANT_COLLECTION", TEST_COLLECTION)  # 獨立 collection
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("SKIP_STARTUP_SERVICES", raising=False)

    assert TEST_COLLECTION != "factory_knowledge"  # 不得污染正式 collection

    try:
        from qdrant_client import QdrantClient

        probe = QdrantClient(url=qdrant_url)
        probe.get_collections()
        probe.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"真實 Qdrant 未就緒（{qdrant_url}）：{exc}")

    if not os.path.exists(os.path.join(SAMPLE_DIR, PDF_NAME)):
        pytest.skip("尚未產生 sample PDF（執行 scripts/generate_sample_pdf.py）")

    from app.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        try:
            yield c
        finally:
            try:
                app.state.vector_store.client.delete_collection(TEST_COLLECTION)
            except Exception:  # noqa: BLE001
                pass


def _upload(client, name: str, content_type: str) -> str:
    with open(os.path.join(SAMPLE_DIR, name), "rb") as f:
        raw = f.read()
    r = client.post("/api/documents", files={"file": (name, raw, content_type)})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "indexed"
    assert body["chunk_count"] > 0
    return body["document_id"]


def _retrieve(client, question: str, top_k: int = 5) -> dict:
    r = client.post("/api/retrieve", json={"question": question, "top_k": top_k})
    assert r.status_code == 200, r.text
    return r.json()


def test_retrieval_across_md_txt_pdf(client):
    # 1. 三種文件皆可成功 Index
    _upload(client, MD_NAME, "text/markdown")
    _upload(client, TXT_NAME, "text/plain")
    _upload(client, PDF_NAME, "application/pdf")

    # 2. 每種文件至少一題命中正確文件（出現在 top-3）
    md_res = _retrieve(client, "設備出現 E102 警報代表什麼？")
    assert MD_NAME in [x["filename"] for x in md_res["results"][:3]]

    txt_res = _retrieve(client, "PM-100 與 PM-1000 分別代表什麼模組？")
    assert TXT_NAME in [x["filename"] for x in txt_res["results"][:3]]

    pdf_res = _retrieve(client, "空氣過濾網每幾小時需要清潔一次？")
    pdf_hits = [x for x in pdf_res["results"] if x["filename"] == PDF_NAME]
    assert pdf_hits, f"PDF 未命中：{[x['filename'] for x in pdf_res['results']]}"

    # 3. PDF 問題命中正確頁碼（空氣過濾網位於第 2 頁）
    assert pdf_hits[0]["page"] == 2

    # 4. /api/retrieve 回傳實際結果（score 為真實相似度、含 retrieval_time_ms）
    assert isinstance(md_res["results"][0]["score"], float)
    assert md_res["results"][0]["score"] > 0
    assert "retrieval_time_ms" in md_res

    # 8. 不呼叫 LLM：回應僅含檢索欄位
    assert set(md_res.keys()) == {"results", "retrieval_time_ms"}


def test_retrieval_uses_independent_collection(client):
    # 5./6./7. 使用獨立 collection，且不建立 / 污染 factory_knowledge
    app = client.app
    assert app.state.vector_store.collection == TEST_COLLECTION
    collections = [c.name for c in app.state.vector_store.client.get_collections().collections]
    # 本測試只操作獨立 collection；不主動建立正式 collection
    assert TEST_COLLECTION in collections
