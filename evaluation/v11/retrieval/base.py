"""RetrievalStrategy 介面、候選 / 結果資料結構、SearchBackend、Registry（v1.1 Slice 4 §九）。

可插拔的版本化 Retrieval Strategy：

- :class:`DenseRetrievalStrategy` 包裝既有 dense retrieval（parity）。
- :class:`IdentifierAwareRetrievalStrategy` 加入 identifier candidate union + boost/penalty。

策略只依賴抽象 :class:`SearchBackend`（dense / identifier-filtered vector search），故可用
fake backend 單元測試，不需真實 Qdrant / BGE。

**不覆蓋原始 Vector Score**（§九）：每個結果同時保留 ``vector_score``（raw cosine）與
``ranking_score``（boost/penalty 後）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

# candidate 來源標籤（§十二 candidate_sources）。
SOURCE_DENSE = "dense"
SOURCE_IDENTIFIER = "identifier"

# admission_reason 常數（§十五）。
ADMISSION_DENSE_THRESHOLD = "dense_threshold"
ADMISSION_EXACT_IDENTIFIER_FLOOR = "exact_identifier_floor"
ADMISSION_REJECTED_BELOW_THRESHOLD = "rejected_below_threshold"
ADMISSION_REJECTED_BELOW_IDENTIFIER_FLOOR = "rejected_below_identifier_floor"


# --------------------------------------------------------------------------
# 候選 / 結果
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    """單一 retrieval 候選（dense 或 identifier search 取得，含 raw vector score）。"""

    document_id: Optional[str]
    chunk_index: Optional[int]
    text: str
    vector_score: float
    page: Optional[int] = None
    section: Optional[str] = None
    filename: Optional[str] = None
    file_hash: Optional[str] = None
    identifiers: list[str] = field(default_factory=list)
    provenance_spans: list[dict] = field(default_factory=list)
    canonical_text_version: Optional[str] = None
    # 此候選由哪些來源產生（dense / identifier）；union 去重時合併。
    sources: list[str] = field(default_factory=list)

    def identity(self) -> tuple:
        """穩定 chunk identity（§十二）：document_id + chunk_index。"""
        return (self.document_id, self.chunk_index)


@dataclass
class RankedRetrievalResult:
    """最終排序後的結果，保留完整、可解釋的 ranking 明細（§九 per-result）。"""

    rank: int
    document_id: Optional[str]
    chunk_index: Optional[int]
    text: str
    page: Optional[int]
    section: Optional[str]
    filename: Optional[str]
    file_hash: Optional[str]
    identifiers: list[str]
    provenance_spans: list[dict]
    canonical_text_version: Optional[str]
    vector_score: float
    ranking_score: float
    exact_identifier_matches: list[str]
    conflicting_identifiers: list[str]
    exact_match_boost: float
    confusable_penalty: float
    candidate_sources: list[str]
    admission_reason: str
    # --- v1.2.1 rerank（附加欄位；rerank 未啟用時維持預設，輸出不變）---
    #: raw cross-encoder logit。**啟用 rerank 時這就是 ranking_score（排序主鍵）**。
    rerank_score: Optional[float] = None
    #: sigmoid(rerank_score)。**僅供 report / audit，不參與排序或 tie-break**。
    rerank_score_normalized: Optional[float] = None
    #: 此結果是否落在 rerank window W 內（tail T 為 False）。
    rerank_applied: bool = False

    def ranking_details(self) -> dict[str, Any]:
        """供 Optimizer Trial Result 持久化的 ranking 明細（§十七）。

        rerank 未啟用時**完全不輸出** rerank 欄位，故既有持久化內容不變。
        """
        d = {
            "vector_score": self.vector_score,
            "ranking_score": self.ranking_score,
            "exact_identifier_matches": list(self.exact_identifier_matches),
            "conflicting_identifiers": list(self.conflicting_identifiers),
            "exact_match_boost": self.exact_match_boost,
            "confusable_penalty": self.confusable_penalty,
            "candidate_sources": list(self.candidate_sources),
            "admission_reason": self.admission_reason,
        }
        if self.rerank_score is not None:
            d["rerank_score"] = self.rerank_score
            d["rerank_score_normalized"] = self.rerank_score_normalized
            d["rerank_applied"] = self.rerank_applied
        return d


@dataclass
class StrategyRetrievalOutcome:
    """Strategy 執行結果（§九 概念輸出 RetrievalOutcome）。"""

    results: list[RankedRetrievalResult]
    query_identifiers: list[str]
    strategy_name: str
    ranking_version: str
    dense_candidate_count: int = 0
    identifier_candidate_count: int = 0
    union_candidate_count: int = 0
    deduplicated_count: int = 0
    exact_match_candidate_count: int = 0
    confusable_candidate_count: int = 0
    threshold_rejected_count: int = 0
    identifier_floor_rejected_count: int = 0
    warnings: list[str] = field(default_factory=list)
    # --- v1.2.1 rerank（附加；未啟用時 rerank_strategy 為 None → meta 不輸出）---
    rerank_strategy: Optional[str] = None
    #: admission 後的候選數 |A|（rerank window 由此切出）。
    admitted_count: int = 0
    #: 實際送進 cross-encoder 的數量 |W|。
    rerank_window_count: int = 0
    #: 未進入 window 的 tail |T|（保留，不被移除）。
    rerank_tail_count: int = 0
    rerank_latency_ms: float = 0.0
    rerank_cache_hits: int = 0
    rerank_cache_misses: int = 0

    def meta(self) -> dict[str, Any]:
        """per-case retrieval metadata（持久化 / Report §十九）。

        rerank 未啟用時**完全不輸出** rerank 欄位，故既有 report / 持久化內容不變。
        """
        d = {
            "strategy_name": self.strategy_name,
            "ranking_version": self.ranking_version,
            "query_identifiers": list(self.query_identifiers),
            "dense_candidate_count": self.dense_candidate_count,
            "identifier_candidate_count": self.identifier_candidate_count,
            "union_candidate_count": self.union_candidate_count,
            "deduplicated_count": self.deduplicated_count,
            "exact_match_candidate_count": self.exact_match_candidate_count,
            "confusable_candidate_count": self.confusable_candidate_count,
            "threshold_rejected_count": self.threshold_rejected_count,
            "identifier_floor_rejected_count": self.identifier_floor_rejected_count,
            "warnings": list(self.warnings),
        }
        if self.rerank_strategy is not None:
            d["rerank_strategy"] = self.rerank_strategy
            d["admitted_count"] = self.admitted_count
            d["rerank_window_count"] = self.rerank_window_count
            d["rerank_tail_count"] = self.rerank_tail_count
            d["rerank_latency_ms"] = self.rerank_latency_ms
            d["rerank_cache_hits"] = self.rerank_cache_hits
            d["rerank_cache_misses"] = self.rerank_cache_misses
        return d


# --------------------------------------------------------------------------
# 輸入 + 後端抽象
# --------------------------------------------------------------------------
@dataclass
class RetrievalRequest:
    """Strategy 輸入（§九 概念輸入）。query_vector 由呼叫端一次算好（§十一.5）。"""

    query: str
    query_vector: list[float]
    collection_name: str
    config: Any  # RetrievalConfig（避免循環 import）
    # --- v1.2.1 rerank（附加、選填）---
    #: :class:`evaluation.v11.retrieval.reranker.Reranker`。
    #: ``rerank_strategy != none`` 時必填；未啟用 rerank 時忽略。
    reranker: Any = None
    #: 選填的 (query, chunk_text) -> raw logit 快取。
    rerank_cache: Any = None


class SearchBackend(Protocol):
    """Strategy 所需的最小向量檢索介面（真實 = Qdrant；測試 = fake）。"""

    def dense_search(self, query_vector: list[float], limit: int) -> list[Candidate]:
        """一般 dense vector search（無 filter），回傳含 raw vector_score 的候選。"""
        ...

    def identifier_search(
        self, query_vector: list[float], identifiers: list[str], limit: int
    ) -> list[Candidate]:
        """identifier keyword filtered vector search（MatchAny），回傳含 raw vector_score 的候選。"""
        ...


# --------------------------------------------------------------------------
# Strategy 介面 + Registry
# --------------------------------------------------------------------------
class RetrievalStrategy(ABC):
    """可插拔 Retrieval Strategy 介面（§九）。所有實作必須 deterministic。"""

    name: str

    @abstractmethod
    def retrieve(
        self, request: RetrievalRequest, backend: SearchBackend
    ) -> StrategyRetrievalOutcome:  # pragma: no cover - 介面
        ...


class RetrievalStrategyRegistry:
    """Strategy 名稱 → 實作的註冊表。"""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], RetrievalStrategy]] = {}

    def register(self, name: str, factory: Callable[[], RetrievalStrategy]) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> RetrievalStrategy:
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(
                f"未註冊的 retrieval strategy={name!r}；已註冊：{sorted(self._factories)}。"
            )
        return factory()

    def names(self) -> list[str]:
        return sorted(self._factories)


def default_registry() -> RetrievalStrategyRegistry:
    from evaluation.v11.retrieval.dense_strategy import DenseRetrievalStrategy
    from evaluation.v11.retrieval.identifier_aware_strategy import (
        IdentifierAwareRetrievalStrategy,
    )

    reg = RetrievalStrategyRegistry()
    reg.register("dense", DenseRetrievalStrategy)
    reg.register("identifier_aware", IdentifierAwareRetrievalStrategy)
    return reg


def build_retrieval_strategy(name: str) -> RetrievalStrategy:
    return default_registry().create(name)
