"""RAG Service 單元測試（Phase 5 §15.2）。

以 Fake Retrieval Service 提供確定性 Chunk、Fake LLM Client 提供確定性回應，
驗證雙層拒答、Citation 驗證與 Sources 組裝。以 asyncio.run 驅動 async 方法。
"""
from __future__ import annotations

import asyncio

import pytest

from app.core.constants import REFUSAL_MESSAGE
from app.core.exceptions import LLMServiceError
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalResult, RetrievedChunk


class FakeRetrievalService:
    """回傳預設（已 threshold 過濾）結果，並記錄 retrieval 是否被呼叫。"""

    def __init__(self, chunks: list[RetrievedChunk], retrieval_time_ms: int = 20):
        self._chunks = chunks
        self._retrieval_time_ms = retrieval_time_ms
        self.calls = 0

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            results=list(self._chunks), retrieval_time_ms=self._retrieval_time_ms
        )


class FakeLLMClient:
    """回傳預設文字或拋出預設例外，並記錄呼叫次數。"""

    def __init__(self, response: str | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.call_count = 0
        self.last_messages = None

    async def complete(self, messages: list[dict]) -> str:
        self.call_count += 1
        self.last_messages = messages
        if self.exc is not None:
            raise self.exc
        return self.response


def _chunk(rank: int, text: str = "内容") -> RetrievedChunk:
    return RetrievedChunk(
        source_id=f"S{rank}",
        document_id=f"doc-{rank}",
        filename=f"file_{rank}.md",
        page=None,
        section=f"section-{rank}",
        chunk_index=rank,
        score=0.7 - rank * 0.01,
        text=text,
    )


def _service(chunks, llm) -> tuple[RagService, FakeRetrievalService, FakeLLMClient]:
    retrieval = FakeRetrievalService(chunks)
    service = RagService(retrieval_service=retrieval, llm_client=llm)
    return service, retrieval, llm


def _run(service, question="問題", top_k=5):
    return asyncio.run(service.answer(question, top_k))


# 1. Retrieval 無結果 → 固定 fallback、sources=[]、generation_time_ms=0、LLM 未被呼叫
def test_first_layer_refusal_no_retrieval():
    llm = FakeLLMClient(response="不該被呼叫")
    service, _, llm = _service([], llm)
    result = _run(service)
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []
    assert result.generation_time_ms == 0
    assert llm.call_count == 0


# 2. Retrieval 有結果 + LLM 回有效 [S1] → 回答 + sources 只含 S1
def test_valid_answer_single_citation():
    llm = FakeLLMClient(response="E102 代表感測器通訊異常。[S1]")
    service, _, _ = _service([_chunk(1), _chunk(2)], llm)
    result = _run(service)
    assert result.answer == "E102 代表感測器通訊異常。[S1]"
    assert [s.source_id for s in result.sources] == ["S1"]
    assert result.generation_time_ms >= 0


# 3. LLM 只引用 [S2] → sources 不含未使用的 S1
def test_only_cited_sources_returned():
    llm = FakeLLMClient(response="檢查通訊線材。[S2]")
    service, _, _ = _service([_chunk(1), _chunk(2)], llm)
    result = _run(service)
    assert [s.source_id for s in result.sources] == ["S2"]


# 4. 重複引用 [S1][S1] → sources 只含一份 S1
def test_duplicate_citation_deduped():
    llm = FakeLLMClient(response="重點一 [S1] 重點二 [S1]")
    service, _, _ = _service([_chunk(1), _chunk(2)], llm)
    result = _run(service)
    assert [s.source_id for s in result.sources] == ["S1"]


# 5. LLM 回固定 fallback → sources=[]、generation_time_ms > 0
def test_llm_exact_fallback():
    llm = FakeLLMClient(response=REFUSAL_MESSAGE)
    service, _, _ = _service([_chunk(1)], llm)
    result = _run(service)
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []
    assert llm.call_count == 1


# 6. 回答含 fallback 句子與多餘文字 → 正規化為精確 fallback、sources=[]
def test_llm_fallback_with_extra_text_normalized():
    llm = FakeLLMClient(response=f"很抱歉，{REFUSAL_MESSAGE} 建議聯繫工程師。[S1]")
    service, _, _ = _service([_chunk(1)], llm)
    result = _run(service)
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []


# 7. 回答沒有 Citation → fail-closed
def test_no_citation_fail_closed():
    llm = FakeLLMClient(response="E102 代表感測器通訊異常。")
    service, _, _ = _service([_chunk(1)], llm)
    result = _run(service)
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []


# 8. 引用不存在的 [S99] → fail-closed，不建立 S99 source
def test_invalid_citation_fail_closed():
    llm = FakeLLMClient(response="答案 [S99]")
    service, _, _ = _service([_chunk(1), _chunk(2)], llm)
    result = _run(service)
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []


# 9. 混合 [S1][S99] → fail-closed（不部分保留 S1）
def test_mixed_valid_invalid_citation_fail_closed():
    llm = FakeLLMClient(response="答案內容。[S1][S99]")
    service, _, _ = _service([_chunk(1), _chunk(2)], llm)
    result = _run(service)
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []


# 10. 引用 [S2][S1] → sources 順序依首次引用順序
def test_sources_order_follows_first_citation():
    llm = FakeLLMClient(response="先講 [S2] 再講 [S1]")
    service, _, _ = _service([_chunk(1), _chunk(2)], llm)
    result = _run(service)
    assert [s.source_id for s in result.sources] == ["S2", "S1"]


# 11. Context 有高分但無答案（第二層拒答） → Fake LLM 回 fallback
def test_second_layer_refusal_high_score_no_answer():
    # 模擬 CX-500 市場售價：retrieval 有高分結果，但 LLM 判定 Context 無答案
    high = _chunk(1, text="CX-500 潤滑油規格...")
    high.score = 0.6319
    llm = FakeLLMClient(response=REFUSAL_MESSAGE)
    service, retrieval, llm = _service([high], llm)
    result = _run(service, question="CX-500 市場售價是多少？")
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []
    assert llm.call_count == 1  # 第二層確實呼叫過 LLM
    assert result.generation_time_ms >= 0


# 12. LLM 發生錯誤 → 例外向上拋，不偽造 fallback 成功結果
def test_llm_error_propagates():
    llm = FakeLLMClient(exc=LLMServiceError())
    service, _, _ = _service([_chunk(1)], llm)
    with pytest.raises(LLMServiceError):
        _run(service)


# 補充：有效答案時實際傳給 LLM 的 messages 含 context 與問題
def test_llm_receives_context_and_question():
    llm = FakeLLMClient(response="答案 [S1]")
    service, _, llm = _service([_chunk(1, text="E102 表示感測器通訊異常")], llm)
    _run(service, question="E102 是什麼？")
    system, user = llm.last_messages
    assert system["role"] == "system"
    assert "<retrieved_context>" in user["content"]
    assert "E102 表示感測器通訊異常" in user["content"]
    assert "E102 是什麼？" in user["content"]
