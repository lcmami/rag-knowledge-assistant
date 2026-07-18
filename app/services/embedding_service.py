"""Embedding Service（規格 §6.2）。

- 於應用程式啟動時載入一次，不得每次 request 重載。
- 使用 normalized embedding。
- 從模型動態取得向量維度。
- Query 加入可設定的 query instruction；文件 Chunk 不加。
- Embedding 前用 tokenizer 檢查 token 數，不依賴靜默 truncation。
"""
from __future__ import annotations

from app.core.exceptions import EmbeddingError


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        query_instruction: str,
        max_tokens: int,
    ):
        # 延遲 import，避免測試 / 純邏輯路徑載入重量級套件
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_instruction = query_instruction
        self.max_tokens = max_tokens
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def count_tokens(self, text: str) -> int:
        """用模型 tokenizer 計算 token 數（不含特殊 token 的內容長度）。"""
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def _check_tokens(self, texts: list[str]) -> None:
        for t in texts:
            if self.count_tokens(t) > self.max_tokens:
                raise EmbeddingError(
                    "Chunk 超過 embedding token 硬限制。",
                    details={"max_tokens": self.max_tokens},
                )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """對文件 Chunk 產生 normalized embedding（不加 query instruction）。"""
        if not texts:
            return []
        self._check_tokens(texts)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, question: str) -> list[float]:
        """對 Query 加上 query instruction 後產生 normalized embedding。"""
        text = f"{self.query_instruction}{question}"
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()
