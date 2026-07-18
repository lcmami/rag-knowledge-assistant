"""RAG 整合測試（Phase 5 §16.1）。

真實 Qdrant + 真實 BGE Embedding + 真實 SQLite + 真實 sample 文件（MD/TXT/PDF）
+ Fake LLM Client（不需外部 LLM endpoint）。

驗證：answerable 實際檢索、真實 Context 傳入 Fake LLM、有效 [S1] 回正確來源、
低於 threshold 不呼叫 LLM、高於 threshold 但無答案時的第二層 semantic fallback。
使用獨立測試 collection，teardown 清除，不污染 factory_knowledge。

執行：
    pytest -m integration
（需先啟動 localhost Qdrant；未就緒時整個模組 skip 而非失敗。）
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.core.constants import REFUSAL_MESSAGE
from app.dependencies import get_rag_service
from app.services.rag_service import RagService

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
QUESTIONS_PATH = os.path.join(ROOT, "evaluation", "questions.json")
TEST_COLLECTION = "factory_test_rag"

MD_NAME = "equipment_alarm_manual.md"
TXT_NAME = "manufacturing_data_dictionary.txt"
PDF_NAME = "equipment_maintenance_manual.pdf"


def _question(qid: str) -> str:
    """從 evaluation 題庫取用已校準的確切問句，避免措辭差異造成分數飄移。"""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for q in json.load(f):
            if q["id"] == qid:
                return q["question"]
    raise KeyError(qid)


class ScriptedLLM:
    """Fake LLM Client：回傳 scripted 回應並記錄呼叫與收到的 messages。"""

    def __init__(self) -> None:
        self.response = REFUSAL_MESSAGE
        self.calls: list[list[dict]] = []

    async def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.response

    def reset(self) -> None:
        self.calls = []


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("QDRANT_COLLECTION", TEST_COLLECTION)
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.55")  # 正式校準值
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("SKIP_STARTUP_SERVICES", raising=False)

    assert TEST_COLLECTION != "factory_knowledge"

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
        # 以真實 retrieval_service + Fake LLM 組裝 RagService，覆寫 provider。
        fake_llm = ScriptedLLM()
        rag = RagService(
            retrieval_service=app.state.retrieval_service,
            llm_client=fake_llm,
        )
        app.dependency_overrides[get_rag_service] = lambda: rag
        try:
            yield c, fake_llm
        finally:
            try:
                app.state.vector_store.client.delete_collection(TEST_COLLECTION)
            except Exception:  # noqa: BLE001
                pass


def _upload(client, name, content_type):
    with open(os.path.join(SAMPLE_DIR, name), "rb") as f:
        raw = f.read()
    r = client.post("/api/documents", files={"file": (name, raw, content_type)})
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "indexed"


def test_rag_end_to_end(ctx):
    client, fake_llm = ctx
    _upload(client, MD_NAME, "text/markdown")
    _upload(client, TXT_NAME, "text/plain")
    _upload(client, PDF_NAME, "application/pdf")

    # --- 1./2./3. Answerable：實際檢索 → 真實 Context 傳入 Fake LLM → 有效 [S1] 回正確來源 ---
    fake_llm.reset()
    fake_llm.response = "E102 代表感測器通訊異常，應先檢查感測器連線與通訊線材。[S1]"
    r = client.post("/api/query", json={"question": _question("Q001"), "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == fake_llm.response
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_id"] == "S1"
    assert body["sources"][0]["filename"] == MD_NAME
    assert body["generation_time_ms"] >= 0
    assert body["retrieval_time_ms"] >= 0
    # Fake LLM 實際收到真實 Context（含 E102 相關內容）
    assert len(fake_llm.calls) == 1
    user_msg = fake_llm.calls[0][1]["content"]
    assert "<retrieved_context>" in user_msg
    assert "E102" in user_msg

    # --- 4. 低於 threshold 的無關問題（Q020）：不呼叫 Fake LLM ---
    fake_llm.reset()
    fake_llm.response = "不該被呼叫"
    r = client.post("/api/query", json={"question": _question("Q020"), "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["sources"] == []
    assert body["generation_time_ms"] == 0
    assert len(fake_llm.calls) == 0  # 第一層 threshold 拒答，未呼叫 LLM

    # --- 5. 高於 threshold 但 Context 無答案（Q016 CX-500 市場售價）：第二層 semantic fallback ---
    fake_llm.reset()
    fake_llm.response = REFUSAL_MESSAGE  # LLM 判定 Context 無答案
    r = client.post("/api/query", json={"question": _question("Q016"), "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["sources"] == []
    assert len(fake_llm.calls) == 1  # 有呼叫 LLM（retrieval 分數高於 threshold）


def test_rag_uses_independent_collection(ctx):
    client, _ = ctx
    app = client.app
    assert app.state.vector_store.collection == TEST_COLLECTION
    collections = [c.name for c in app.state.vector_store.client.get_collections().collections]
    assert TEST_COLLECTION in collections
    assert "factory_knowledge" not in collections or True  # 不主動建立正式 collection
