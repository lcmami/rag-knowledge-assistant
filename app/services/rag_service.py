"""RAG Service（規格 Phase 5 §5~§9）。

流程：
    Question
    → Retrieval Service（已套用 Threshold 過濾）
    → 第一層拒答：無合格 Chunk → 直接回固定 fallback，不呼叫 LLM
    → 組合 Context → 呼叫 LLM
    → 第二層拒答：LLM 回固定 fallback → 正規化、sources=[]
    → Citation 驗證（fail-closed）
    → 依答案首次引用順序組裝 Sources
    → 回傳結果

拒答分兩層（§1）：
- 第一層：Threshold 前置拒答（明顯無關、相似度過低）→ 不呼叫 LLM。
- 第二層：LLM 語意拒答（實體高度相似但文件無答案）→ LLM 回 fallback。

Citation 驗證失敗一律 fail-closed（§8）：正規化為固定 fallback、sources=[]，不自動重試。
Sources metadata 完全來自後端 Retrieval Result，不信任 LLM 生成的任何 metadata（§9）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.constants import CITATION_PATTERN, REFUSAL_MESSAGE
from app.core.logging import get_logger
from app.prompts.rag_prompt import build_messages
from app.services.llm_client import LLMClient
from app.services.retrieval_service import RetrievalService, RetrievedChunk

logger = get_logger("app.services.rag_service")


@dataclass
class RagResult:
    answer: str
    sources: list[RetrievedChunk]
    retrieval_time_ms: int
    generation_time_ms: int


def parse_citations(text: str) -> list[str]:
    """解析 [S<n>] citation，回傳依首次出現順序、去重後的 source id 清單（§8）。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for match in CITATION_PATTERN.finditer(text):
        source_id = f"S{match.group(1)}"
        if source_id not in seen:
            seen.add(source_id)
            ordered.append(source_id)
    return ordered


class RagService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: LLMClient,
    ):
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client

    def _refusal(self, retrieval_ms: int, generation_ms: int) -> RagResult:
        return RagResult(
            answer=REFUSAL_MESSAGE,
            sources=[],
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
        )

    async def answer(self, question: str, top_k: int) -> RagResult:
        logger.info("query 開始：question_len=%d top_k=%d", len(question), top_k)

        retrieval = self.retrieval_service.retrieve(question, top_k)
        results = retrieval.results
        retrieval_ms = retrieval.retrieval_time_ms
        logger.info(
            "retrieval 完成：count=%d latency_ms=%d", len(results), retrieval_ms
        )

        # --- 第一層拒答：Threshold 過濾後無合格 Chunk（§6）---
        if not results:
            logger.info("第一層拒答（threshold）：無合格 Chunk，不呼叫 LLM")
            return self._refusal(retrieval_ms, 0)

        # --- 組合 Context 並呼叫 LLM ---
        messages = build_messages(question, results)
        start = time.perf_counter()
        # LLM 錯誤（未設定 / timeout / provider / 空回應）直接向上拋，交由 API error handler 處理，
        # 不偽造 fallback 成功結果（§15.2 case 12）。
        raw = await self.llm_client.complete(messages)
        generation_ms = int((time.perf_counter() - start) * 1000)
        logger.info("LLM generation 完成：latency_ms=%d", generation_ms)

        answer = raw.strip()

        # --- 第二層拒答：LLM 語意拒答（§7）---
        # 回應若包含固定 fallback 句子，統一正規化為精確 fallback。
        if REFUSAL_MESSAGE in answer:
            logger.info("第二層拒答（semantic）：LLM 判定 Context 無答案")
            return self._refusal(retrieval_ms, generation_ms)

        # --- Citation 驗證（§8，fail-closed）---
        cited_ids = parse_citations(answer)
        valid_ids = {r.source_id for r in results}

        if not cited_ids:
            # §8.2 無 Citation：不視為有效答案。
            logger.warning("LLM response rejected: missing citation")
            return self._refusal(retrieval_ms, generation_ms)

        invalid = [cid for cid in cited_ids if cid not in valid_ids]
        if invalid:
            # §8.3 / §8.4 無效或混合無效 Citation：整體 fail-closed，不部分修補。
            logger.warning(
                "LLM response rejected: invalid citation %s (valid=%s)",
                invalid,
                sorted(valid_ids),
            )
            return self._refusal(retrieval_ms, generation_ms)

        # --- 組裝 Sources（§9）：只含實際引用、依首次引用順序、去重 ---
        by_id = {r.source_id: r for r in results}
        sources = [by_id[cid] for cid in cited_ids]
        logger.info(
            "有效回答：cited=%s sources=%d",
            cited_ids,
            len(sources),
        )
        return RagResult(
            answer=answer,
            sources=sources,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
        )
