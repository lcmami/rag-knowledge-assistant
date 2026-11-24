"""Embedding Service。

改為呼叫外部 OpenAI-compatible Embedding API，不再於本地載入
sentence-transformers 模型。對外介面（embed_documents / embed_query /
dimension / count_tokens）與原本完全相同，呼叫端無需修改。

設計原則：
- 於 lifespan 初始化一次（build_services），不每次 Request 重建 client。
- Query 加上 query_instruction 前綴；Document Chunk 不加。
- 對回傳向量做 L2 normalize，保留既有檢索語意。
- 第一次取得 embedding 後驗證維度與設定一致，不一致拋 EmbeddingError。
- API 呼叫失敗包裝為既有 EmbeddingError，不外洩 provider 內部資訊。
"""
from __future__ import annotations

import math

from openai import OpenAI, OpenAIError

from app.core.exceptions import EmbeddingError


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        query_instruction: str,
        max_tokens: int,
        base_url: str,
        api_key: str,
        dimension: int,
    ):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.max_tokens = max_tokens
        self._dimension = dimension
        # 延遲建立 client：未設定 base_url / api_key 時允許建構（測試 / 設定未就緒），
        # 實際呼叫時再拋 EmbeddingError。
        self._client: OpenAI | None = None
        if base_url and api_key:
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._dimension_verified = False

    @property
    def dimension(self) -> int:
        return self._dimension

    def count_tokens(self, text: str) -> int:
        """保守估算 token 數。

        注意：此實作為字元數近似，**不是**模型 tokenizer 的精準 token count。
        沒有 embedding model 對應的本地 tokenizer 前，先以 len(text) 作為
        上限估算，避免低估中文等非 ASCII 內容的 token 數。
        """
        return len(text)

    def _check_tokens(self, texts: list[str]) -> None:
        for t in texts:
            if self.count_tokens(t) > self.max_tokens:
                raise EmbeddingError(
                    "Chunk 超過 embedding token 硬限制。",
                    details={"max_tokens": self.max_tokens},
                )

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            raise EmbeddingError(
                "Embedding API 尚未設定（EMBEDDING_BASE_URL / EMBEDDING_API_KEY）。",
            )
        return self._client

    def _normalize(self, vector: list[float]) -> list[float]:
        """L2 normalize 向量，保留既有檢索語意（與原 sentence-transformers
        normalize_embeddings=True 行為一致）。"""
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]

    def _verify_dimension(self, vector: list[float]) -> None:
        """第一次取得 embedding 後驗證維度與設定一致。"""
        if self._dimension_verified:
            return
        if self._dimension and len(vector) != self._dimension:
            raise EmbeddingError(
                "Embedding API 回傳維度與 EMBEDDING_DIMENSION 設定不一致。",
                details={
                    "expected": self._dimension,
                    "actual": len(vector),
                },
            )
        self._dimension_verified = True

    def _call_embeddings(self, texts: list[str]) -> list[list[float]]:
        """呼叫 OpenAI-compatible Embeddings API，回傳 L2-normalized 向量清單。"""
        client = self._ensure_client()
        try:
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
        except OpenAIError as exc:
            # 不外洩 provider 內部訊息；只記錄型別。
            raise EmbeddingError(
                "Embedding API 呼叫失敗。",
                details={"error_type": type(exc).__name__},
            ) from exc

        vectors: list[list[float]] = []
        for item in response.data:
            vector = list(item.embedding)
            self._verify_dimension(vector)
            vectors.append(self._normalize(vector))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """對文件 Chunk 產生 normalized embedding（不加 query instruction）。"""
        if not texts:
            return []
        self._check_tokens(texts)
        return self._call_embeddings(texts)

    def embed_query(self, question: str) -> list[float]:
        """對 Query 加上 query instruction 後產生 normalized embedding。"""
        text = f"{self.query_instruction}{question}"
        self._check_tokens([text])
        vectors = self._call_embeddings([text])
        return vectors[0]

