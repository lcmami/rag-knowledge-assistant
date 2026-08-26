"""Rerank stage（v1.2.1 Slice R1）—— dense 與 identifier_aware **共用同一份實作**。

位置（locked design）::

    candidates -> boost/penalty -> admission（以 RAW vector_score，完全不動）
               -> W = A[:rerank_top_n]  送進 cross-encoder，依 raw logit 重排
               -> T = A[rerank_top_n:]  不丟棄，維持原相對順序，接在 W 之後
               -> (rerank(W) ++ T)[:top_k]

**放在 admission 之後**的理由:admission 是 v1.1 花 Slice 4/5 調出的安全閘門。
rerank 若在其之前就能把低於 threshold 的 chunk 救回來，直接摧毀 unanswerable FP 的控制。

**H3 安全不變量（結構性保證，不是經驗觀察）**::

    rerank stage 不具備 admission 或 filtering 權限。
    對任一 case：results 為空  <=>  admitted set A 為空。

依據 :mod:`evaluation.v11.retrieval_metrics`（`:205`、`:583-586`）——
``unanswerable_false_positive`` 只判定「回傳清單是否為空」，與 relevance / 排序 / 分數無關。
A 非空 => W 非空 => 輸出非空，故 FP rate 恆等。
**一旦在此加入任何 rerank 分數門檻或過濾，H3 立刻失效** —— 此禁令不可移除。

**排序主鍵是 raw logit**（§0.9(1)）:``ranking_score = rerank_score``。
``rerank_score_normalized = sigmoid(...)`` **僅供 report / audit**，不作排序鍵也不作
tie-break 鍵;raw score 完全相等時直接落入既有 5 鍵 deterministic chain。
"""
from __future__ import annotations

import time
from typing import Any, Optional, Protocol, Sequence

from evaluation.v11.retrieval.base import RankedRetrievalResult
from evaluation.v11.retrieval.reranker import Reranker, sigmoid


class RerankScoreCache(Protocol):
    """(query, chunk_text) -> raw logit 的快取介面（R1 以 fake 測試，實體實作見 rerank_cache）。"""

    def get_many(
        self, query: str, texts: Sequence[str]
    ) -> list[Optional[float]]:  # pragma: no cover - Protocol
        ...

    def put_many(
        self, query: str, texts: Sequence[str], scores: Sequence[float]
    ) -> None:  # pragma: no cover - Protocol
        ...


class RerankStageError(RuntimeError):
    """rerank stage 的內部不變量被破壞（屬程式錯誤，不是使用者輸入問題）。"""


def _tie_break_key(item: tuple[int, RankedRetrievalResult]) -> tuple:
    """window 內排序鍵。

    主鍵 = **raw** ``rerank_score``。其後**完全沿用既有 5 鍵 chain 的語義**:
    exact match、vector_score、document_id、chunk_index。
    ``rerank_score_normalized`` **不出現在此**。
    最後再加原始序位，確保完全相等時仍為穩定排序。
    """
    idx, r = item
    return (
        -(r.rerank_score if r.rerank_score is not None else 0.0),
        not bool(r.exact_identifier_matches),
        -r.vector_score,
        r.document_id or "",
        r.chunk_index if r.chunk_index is not None else -1,
        idx,
    )


def apply_rerank(
    admitted: list[RankedRetrievalResult],
    *,
    query: str,
    reranker: Reranker,
    rerank_top_n: int,
    top_k: int,
    cache: Optional[RerankScoreCache] = None,
) -> dict[str, Any]:
    """對已通過 admission 的候選序列套用 rerank，回傳結果與統計。

    :param admitted: admission 後、依既有 deterministic ranking key 排序的候選序列 ``A``。
                     **本函式不做任何過濾**，只重排與截斷至 ``top_k``。
    :returns: dict，含 ``results`` 與 window/tail/latency/cache 統計。
    """
    if rerank_top_n < top_k:
        raise RerankStageError(
            f"rerank_top_n ({rerank_top_n}) 必須 >= top_k ({top_k})；"
            "否則 tail 會擠掉已被 rerank 的結果。"
        )

    admitted_count = len(admitted)
    if admitted_count == 0:
        # H3：A 為空 -> 輸出為空。rerank 不會、也不得改變這件事。
        return {
            "results": [],
            "admitted_count": 0,
            "window_count": 0,
            "tail_count": 0,
            "latency_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    window = admitted[:rerank_top_n]
    tail = admitted[rerank_top_n:]

    texts = [r.text for r in window]
    cache_hits = 0
    cache_misses = 0

    start = time.perf_counter()
    cached: list[Optional[float]] = (
        list(cache.get_many(query, texts)) if cache is not None else [None] * len(texts)
    )
    missing_idx = [i for i, v in enumerate(cached) if v is None]
    cache_hits = len(texts) - len(missing_idx)
    cache_misses = len(missing_idx)

    if missing_idx:
        computed = reranker.score(query, [texts[i] for i in missing_idx])
        if len(computed) != len(missing_idx):  # pragma: no cover - 防禦性
            raise RerankStageError(
                f"reranker 回傳 {len(computed)} 筆，與待算 {len(missing_idx)} 筆不符。"
            )
        for i, score in zip(missing_idx, computed):
            cached[i] = float(score)
        if cache is not None:
            cache.put_many(
                query,
                [texts[i] for i in missing_idx],
                [float(s) for s in computed],
            )
    latency_ms = (time.perf_counter() - start) * 1000.0

    for r, raw in zip(window, cached):
        value = float(raw)  # 至此不可能為 None
        r.rerank_score = value
        # 僅供 report / audit —— 不參與排序。
        r.rerank_score_normalized = sigmoid(value)
        r.rerank_applied = True
        # 排序主鍵改為 raw logit（§0.9(1)）。vector_score 維持 RAW cosine，不被覆寫。
        r.ranking_score = value

    reordered = [r for _, r in sorted(enumerate(window), key=_tie_break_key)]

    # tail 不丟棄、維持原相對順序，接在 window 之後（W ++ T）。
    for r in tail:
        r.rerank_applied = False

    merged = reordered + tail
    if len(merged) != admitted_count:  # pragma: no cover - 不變量防禦
        raise RerankStageError(
            f"rerank 後候選數由 {admitted_count} 變為 {len(merged)}；"
            "rerank stage 不得新增或移除任何候選。"
        )

    final = merged[:top_k]
    for rank, r in enumerate(final, start=1):
        r.rank = rank

    return {
        "results": final,
        "admitted_count": admitted_count,
        "window_count": len(window),
        "tail_count": len(tail),
        "latency_ms": latency_ms,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }


def maybe_rerank(
    outcome: Any, request: Any, *, admitted_count: Optional[int] = None
) -> Any:
    """Strategy 共用入口:未啟用 rerank 時**原樣返回**，啟用時套用 rerank stage。

    呼叫端的約定:啟用 rerank 時，``outcome.results`` 必須是**完整的 admitted 序列 A**
    （尚未截斷至 ``top_k``）;未啟用時則已是 v1.1 的 top_k 結果。
    """
    cfg = request.config
    if not getattr(cfg, "is_rerank_enabled", False):
        return outcome

    reranker = getattr(request, "reranker", None)
    if reranker is None:
        raise RerankStageError(
            f"rerank_strategy={cfg.rerank_strategy!r} 但 RetrievalRequest.reranker 為 None。"
        )

    stage = apply_rerank(
        outcome.results,
        query=request.query,
        reranker=reranker,
        rerank_top_n=cfg.rerank_top_n,
        top_k=cfg.top_k,
        cache=getattr(request, "rerank_cache", None),
    )

    outcome.results = stage["results"]
    outcome.rerank_strategy = cfg.rerank_strategy
    outcome.admitted_count = (
        admitted_count if admitted_count is not None else stage["admitted_count"]
    )
    outcome.rerank_window_count = stage["window_count"]
    outcome.rerank_tail_count = stage["tail_count"]
    outcome.rerank_latency_ms = stage["latency_ms"]
    outcome.rerank_cache_hits = stage["cache_hits"]
    outcome.rerank_cache_misses = stage["cache_misses"]
    return outcome
