"""/api/retrieve 單元 / API 測試（Phase 4 §10.2）。

使用 Fake Embedding 與可預設命中結果的 Fake Vector Store，套用真實
Router → RetrievalService 路徑，但不載入模型、不連 Qdrant、不呼叫 LLM。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_retrieval_service
from app.main import create_app
from app.services.retrieval_service import RetrievalService


class FakeEmbedding:
    def embed_query(self, question: str) -> list[float]:
        return [0.0] * 8


class PresetVectorStore:
    """search() 一律回傳預設 hits，讓 threshold / 排序 / source id 可被確定性驗證。"""

    def __init__(self, hits: list[dict]):
        self.hits = hits

    def search(self, query_vector, top_k: int) -> list[dict]:
        return [dict(h) for h in self.hits]


def _payload(**kw) -> dict:
    base = {
        "document_id": "doc-1",
        "filename": "equipment_alarm_manual.md",
        "page": None,
        "section": "E102 感測器通訊異常",
        "chunk_index": 0,
        "text": "E102 表示感測器通訊異常。",
    }
    base.update(kw)
    return base


def make_client(hits: list[dict], threshold: float = 0.0) -> TestClient:
    import os

    os.environ["SKIP_STARTUP_SERVICES"] = "1"
    app = create_app()
    service = RetrievalService(
        embedding_service=FakeEmbedding(),
        vector_store=PresetVectorStore(hits),
        score_threshold=threshold,
    )
    app.dependency_overrides[get_retrieval_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


# 1. question 缺少 → 422
def test_missing_question_returns_422():
    with make_client([]) as c:
        r = c.post("/api/retrieve", json={"top_k": 5})
    assert r.status_code == 422


# 2. question 空白 → 422
def test_blank_question_returns_422():
    with make_client([]) as c:
        r = c.post("/api/retrieve", json={"question": "   ", "top_k": 5})
    assert r.status_code == 422


# 3. top_k 超出範圍 → 422（下限與上限）
def test_top_k_out_of_range_returns_422():
    with make_client([]) as c:
        r_low = c.post("/api/retrieve", json={"question": "E102?", "top_k": 0})
        r_high = c.post("/api/retrieve", json={"question": "E102?", "top_k": 9999})
    assert r_low.status_code == 422
    assert r_high.status_code == 422


# 4. 結果按 score 由高到低排序
def test_results_sorted_by_score_desc():
    hits = [
        {"score": 0.42, "payload": _payload(chunk_index=1, text="low")},
        {"score": 0.88, "payload": _payload(chunk_index=2, text="high")},
        {"score": 0.61, "payload": _payload(chunk_index=3, text="mid")},
    ]
    with make_client(hits, threshold=0.0) as c:
        r = c.post("/api/retrieve", json={"question": "E102?", "top_k": 5})
    assert r.status_code == 200
    scores = [item["score"] for item in r.json()["results"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 0.88


# 5. Source ID 為 S1、S2…
def test_source_ids_assigned_s1_s2():
    hits = [
        {"score": 0.9, "payload": _payload(chunk_index=1)},
        {"score": 0.8, "payload": _payload(chunk_index=2)},
    ]
    with make_client(hits, threshold=0.0) as c:
        r = c.post("/api/retrieve", json={"question": "E102?", "top_k": 5})
    ids = [item["source_id"] for item in r.json()["results"]]
    assert ids == ["S1", "S2"]


# 6. Threshold 能過濾低分結果
def test_threshold_filters_low_scores():
    hits = [
        {"score": 0.80, "payload": _payload(chunk_index=1, text="keep")},
        {"score": 0.20, "payload": _payload(chunk_index=2, text="drop")},
    ]
    with make_client(hits, threshold=0.5) as c:
        r = c.post("/api/retrieve", json={"question": "E102?", "top_k": 5})
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["score"] == 0.80


# 7. 無合格結果 → results 為空、仍回 200
def test_no_qualified_results_returns_empty():
    hits = [{"score": 0.10, "payload": _payload()}]
    with make_client(hits, threshold=0.5) as c:
        r = c.post("/api/retrieve", json={"question": "無關問題", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    assert "retrieval_time_ms" in body


# 8. /api/retrieve 不呼叫 LLM（回應僅含檢索欄位，無 answer / sources 生成內容）
def test_retrieve_does_not_call_llm():
    hits = [{"score": 0.9, "payload": _payload()}]
    with make_client(hits, threshold=0.0) as c:
        r = c.post("/api/retrieve", json={"question": "E102?", "top_k": 5})
    body = r.json()
    assert set(body.keys()) == {"results", "retrieval_time_ms"}
    assert "answer" not in body and "sources" not in body


# 9. Response 包含 retrieval_time_ms
def test_response_contains_retrieval_time_ms():
    hits = [{"score": 0.9, "payload": _payload()}]
    with make_client(hits, threshold=0.0) as c:
        r = c.post("/api/retrieve", json={"question": "E102?", "top_k": 5})
    body = r.json()
    assert isinstance(body["retrieval_time_ms"], int)
    assert body["retrieval_time_ms"] >= 0


# 10. PDF 結果包含 1-based page
def test_pdf_result_has_one_based_page():
    hits = [
        {
            "score": 0.75,
            "payload": _payload(
                document_id="doc-pdf",
                filename="equipment_maintenance_manual.pdf",
                page=2,
                section="定期保養週期",
                chunk_index=4,
                text="CX-500 的基本保養每運轉 500 小時執行一次。",
            ),
        }
    ]
    with make_client(hits, threshold=0.0) as c:
        r = c.post("/api/retrieve", json={"question": "保養週期?", "top_k": 5})
    item = r.json()["results"][0]
    assert item["page"] == 2
    assert item["filename"].endswith(".pdf")
