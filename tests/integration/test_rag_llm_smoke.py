"""Optional Real LLM Smoke Test（Phase 5 §16.2）。

只有在 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL 皆設定且真實 Qdrant 就緒時執行，
否則 skip（不得以假設定使測試假通過）。

使用真實 Qdrant + 真實 Embedding + 真實 LLM + 獨立測試 collection。
只驗證：是否 fallback、Citation 是否有效、Sources 是否一致；
不斷言答案自然語言逐字相同。

執行：
    pytest -m llm_integration
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.core.constants import CITATION_PATTERN, REFUSAL_MESSAGE

pytestmark = pytest.mark.llm_integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
QUESTIONS_PATH = os.path.join(ROOT, "evaluation", "questions.json")
TEST_COLLECTION = "factory_test_llm_smoke"

MD_NAME = "equipment_alarm_manual.md"
TXT_NAME = "manufacturing_data_dictionary.txt"
PDF_NAME = "equipment_maintenance_manual.pdf"

_REQUIRED_ENV = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


def _question(qid: str) -> str:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for q in json.load(f):
            if q["id"] == qid:
                return q["question"]
    raise KeyError(qid)


@pytest.fixture
def client(monkeypatch, tmp_path):
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        pytest.skip(f"real LLM verification skipped：缺少環境變數 {missing}")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("QDRANT_COLLECTION", TEST_COLLECTION)
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.55")
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
        try:
            yield c
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


def _valid_citations(answer: str, sources: list[dict]) -> bool:
    cited = {f"S{m.group(1)}" for m in CITATION_PATTERN.finditer(answer)}
    source_ids = {s["source_id"] for s in sources}
    # 答案中每個 citation 都必須對應到回傳 sources
    return bool(cited) and cited == source_ids


def test_ready_is_ok(client):
    _upload(client, MD_NAME, "text/markdown")
    r = client.get("/ready")
    assert r.status_code == 200, r.text
    assert r.json()["checks"]["llm_config"] == "ok"


def test_answerable_question(client):
    _upload(client, MD_NAME, "text/markdown")
    r = client.post("/api/query", json={"question": _question("Q001"), "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] != REFUSAL_MESSAGE
    assert body["sources"], "answerable 問題應有 sources"
    assert _valid_citations(body["answer"], body["sources"])
    assert body["generation_time_ms"] > 0


def test_unrelated_question_threshold_fallback(client):
    _upload(client, MD_NAME, "text/markdown")
    # Evaluation 中明顯低於 threshold 的問題（Q020）
    r = client.post("/api/query", json={"question": _question("Q020"), "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["sources"] == []
    assert body["generation_time_ms"] == 0


def test_high_similarity_no_answer_semantic_fallback(client):
    _upload(client, MD_NAME, "text/markdown")
    _upload(client, PDF_NAME, "application/pdf")
    r = client.post("/api/query", json={"question": _question("Q016"), "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["sources"] == []
    assert body["generation_time_ms"] > 0  # retrieval 分數高於 threshold → 有呼叫 LLM
