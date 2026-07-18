"""/api/query API 測試（Phase 5 §15.3）。

套用真實 Router → RagService 路徑，但以 Fake Retrieval + Fake LLM 注入，
不載入模型、不連 Qdrant、不呼叫真實 LLM。驗證輸入驗證、雙層拒答、
LLM 錯誤狀態碼與不外洩敏感資訊。
"""
from __future__ import annotations

import os

from fastapi.testclient import TestClient

from app.core.constants import REFUSAL_MESSAGE
from app.core.exceptions import (
    LLMNotConfiguredError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.dependencies import get_rag_service
from app.main import create_app
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalResult, RetrievedChunk


class FakeRetrievalService:
    def __init__(self, chunks, retrieval_time_ms=22):
        self._chunks = chunks
        self._ms = retrieval_time_ms

    def retrieve(self, question, top_k):
        return RetrievalResult(results=list(self._chunks), retrieval_time_ms=self._ms)


class FakeLLMClient:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc

    async def complete(self, messages):
        if self.exc is not None:
            raise self.exc
        return self.response


def _chunk(rank, text="E102 表示感測器通訊異常。"):
    return RetrievedChunk(
        source_id=f"S{rank}",
        document_id=f"doc-{rank}",
        filename="equipment_alarm_manual.md",
        page=None,
        section="E102 感測器通訊異常",
        chunk_index=rank,
        score=0.6997,
        text=text,
    )


def make_client(chunks, llm_response=None, llm_exc=None) -> TestClient:
    os.environ["SKIP_STARTUP_SERVICES"] = "1"
    app = create_app()
    service = RagService(
        retrieval_service=FakeRetrievalService(chunks),
        llm_client=FakeLLMClient(response=llm_response, exc=llm_exc),
    )
    app.dependency_overrides[get_rag_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


# 1. 缺少 question → 422
def test_missing_question_422():
    with make_client([_chunk(1)], llm_response="a [S1]") as c:
        r = c.post("/api/query", json={"top_k": 5})
    assert r.status_code == 422


# 2. question 空白 → 422
def test_blank_question_422():
    with make_client([_chunk(1)], llm_response="a [S1]") as c:
        r = c.post("/api/query", json={"question": "   "})
    assert r.status_code == 422


# 3. question 超過 2000 字元 → 422
def test_question_too_long_422():
    with make_client([_chunk(1)], llm_response="a [S1]") as c:
        r = c.post("/api/query", json={"question": "字" * 2001})
    assert r.status_code == 422


# 4. top_k < 1 → 422
def test_top_k_too_small_422():
    with make_client([_chunk(1)], llm_response="a [S1]") as c:
        r = c.post("/api/query", json={"question": "E102?", "top_k": 0})
    assert r.status_code == 422


# 5. top_k 超過最大值 → 422
def test_top_k_too_large_422():
    with make_client([_chunk(1)], llm_response="a [S1]") as c:
        r = c.post("/api/query", json={"question": "E102?", "top_k": 9999})
    assert r.status_code == 422


# 6. 有效回答包含 answer、sources、兩種 latency
def test_valid_answer_shape():
    with make_client([_chunk(1), _chunk(2)], llm_response="E102 是通訊異常。[S1]") as c:
        r = c.post("/api/query", json={"question": "E102?", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "E102 是通訊異常。[S1]"
    assert [s["source_id"] for s in body["sources"]] == ["S1"]
    assert body["sources"][0]["filename"] == "equipment_alarm_manual.md"
    assert isinstance(body["retrieval_time_ms"], int)
    assert isinstance(body["generation_time_ms"], int)
    assert body["generation_time_ms"] >= 0


# 7. Threshold 前置拒答 → 200、generation_time_ms=0、sources=[]
def test_threshold_fallback_200():
    with make_client([], llm_response="不該被呼叫") as c:
        r = c.post("/api/query", json={"question": "台灣的首都?", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["sources"] == []
    assert body["generation_time_ms"] == 0


# 8. Semantic 拒答 → 200、generation_time_ms > 0（有呼叫 LLM）、sources=[]
def test_semantic_fallback_200():
    with make_client([_chunk(1)], llm_response=REFUSAL_MESSAGE) as c:
        r = c.post("/api/query", json={"question": "CX-500 市場售價是多少？", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["sources"] == []


# 9. LLM 未設定 → 503（明確 LLM_NOT_CONFIGURED，非一般 500）
def test_llm_not_configured_503():
    with make_client([_chunk(1)], llm_exc=LLMNotConfiguredError()) as c:
        r = c.post("/api/query", json={"question": "E102?", "top_k": 5})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "LLM_NOT_CONFIGURED"


# 10. LLM Timeout → 504
def test_llm_timeout_504():
    with make_client([_chunk(1)], llm_exc=LLMTimeoutError()) as c:
        r = c.post("/api/query", json={"question": "E102?", "top_k": 5})
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "LLM_TIMEOUT"


# 11. Provider 失敗 → 502
def test_llm_provider_error_502():
    with make_client([_chunk(1)], llm_exc=LLMServiceError()) as c:
        r = c.post("/api/query", json={"question": "E102?", "top_k": 5})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_SERVICE_ERROR"


# 12. 錯誤回應不外洩 traceback 或 API Key
def test_error_does_not_leak_sensitive():
    secret = "sk-leak-me-not-42"
    with make_client([_chunk(1)], llm_exc=LLMServiceError(details={"note": "internal"})) as c:
        r = c.post("/api/query", json={"question": "E102?", "top_k": 5})
    text = r.text
    assert secret not in text
    assert "Traceback" not in text
    assert "chat.completions" not in text
