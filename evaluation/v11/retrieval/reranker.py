"""Cross-Encoder Reranker（v1.2.1 Slice R1）。

介面刻意極小 —— :class:`Reranker` 只需要 ``score(query, texts) -> list[float]``,
因此單元測試可用 fake 完全取代,**不需要載入真實模型**（沿用 `tests/conftest.py` 的既定規則）。

**兩個由 R0/R0.5 實測決定、不可隨意更動的實作常數:**

1. :data:`RERANK_BATCH_SIZE` = 1 —— Repository Owner 依 R0 實測裁示。
   跨 batch 的分數偏差達 4.292e-06（約 3.5× float32 ULP),成因是 **padding 改變
   reduction order**;「完全相同輸入」與「同 batch 重跑」兩個對照組的 delta 皆為 0。
   batch=1 定義上無 padding,故偏差恆為 0,且 CPU 上比 batch=16 更快。
   這是 **runtime implementation constant**:不進 RetrievalConfig、不進 config hash、
   不進 cache key、不進 optimizer search space。

2. :data:`SCORED_ACTIVATION` = identity —— ``CrossEncoder.predict()`` 在 ``num_labels==1``
   時**預設套用 Sigmoid**。本專案以 **raw logit** 作排序主鍵,若沿用預設會**靜默失效**
   （分數被壓進 [0,1],飽和端在報表捨入後塌成假平手）。因此一律顯式傳入 Identity。
"""
from __future__ import annotations

import math
from typing import Any, Optional, Protocol, Sequence

#: 見模組 docstring 第 1 點。**不是**可調參數。
RERANK_BATCH_SIZE = 1

#: 見模組 docstring 第 2 點。被快取與排序使用的分數一律是 raw logit。
SCORED_ACTIVATION = "identity"


class Reranker(Protocol):
    """Rerank stage 所需的最小介面（真實 = CrossEncoder;測試 = fake）。"""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """回傳與 ``texts`` 等長、同順序的 **raw** 分數（越大越相關）。"""
        ...


class NoOpReranker:
    """``rerank_strategy=none`` 時使用;不應被呼叫,呼叫即為程式錯誤。"""

    name = "none"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        raise RuntimeError(
            "NoOpReranker.score 不應被呼叫 —— rerank_strategy=none 時不得進入 rerank stage。"
        )


class CrossEncoderReranker:
    """以本地 sentence-transformers ``CrossEncoder`` 計分,回傳 raw logit。

    Deterministic 保證（R0 已實測驗證）:
    - ``torch.no_grad()`` + ``model.eval()``
    - 固定 ``batch_size=RERANK_BATCH_SIZE``（=1,無 padding）
    - 固定送入順序（呼叫端給什麼順序就用什麼順序）

    模型為惰性載入,故建構此物件本身不會觸發下載或載入。
    """

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str,
        *,
        model: Any = None,
        batch_size: int = RERANK_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self._model = model
        # 允許注入以便測試,但預設一律 RERANK_BATCH_SIZE。
        self._batch_size = int(batch_size)

    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name)
            try:
                model.model.eval()
            except AttributeError:  # pragma: no cover - 介面差異時不致命
                pass
            self._model = model
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        items = list(texts)
        if not items:
            return []

        import torch

        model = self._ensure_model()
        pairs = [(query, t) for t in items]
        with torch.no_grad():
            raw = model.predict(
                pairs,
                # §0.9(1)：必須顯式 Identity,否則 predict() 會套 Sigmoid。
                activation_fn=torch.nn.Identity(),
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        scores = [float(x) for x in raw]
        if len(scores) != len(items):  # pragma: no cover - 防禦性
            raise RuntimeError(
                f"reranker 回傳 {len(scores)} 筆,與輸入 {len(items)} 筆不符。"
            )
        return scores


def sigmoid(value: float) -> float:
    """報表 / audit 用的正規化分數（**不參與任何排序或 tie-break**）。

    以 ``math.exp`` 的分支寫法避免大 |x| 溢位（logit 實測可達 ±10 以上）。
    """
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def build_reranker(
    strategy: str, *, model_name: Optional[str] = None, model: Any = None
) -> Reranker:
    """由 ``rerank_strategy`` 名稱建立 reranker。"""
    from evaluation.v11.retrieval.config import (
        DEFAULT_RERANK_MODEL,
        RERANK_CROSS_ENCODER,
        RERANK_NONE,
    )

    if strategy == RERANK_NONE:
        return NoOpReranker()
    if strategy == RERANK_CROSS_ENCODER:
        return CrossEncoderReranker(model_name or DEFAULT_RERANK_MODEL, model=model)
    raise KeyError(f"未知的 rerank_strategy={strategy!r}。")
