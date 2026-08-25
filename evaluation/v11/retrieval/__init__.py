"""版本化 RetrievalStrategy（dense / identifier_aware）+ v1.2.1 正交 rerank stage。"""
from __future__ import annotations

from evaluation.v11.retrieval.base import (
    Candidate,
    RankedRetrievalResult,
    RetrievalRequest,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
    SearchBackend,
    StrategyRetrievalOutcome,
    build_retrieval_strategy,
    default_registry,
)
from evaluation.v11.retrieval.config import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_N,
    RERANK_CROSS_ENCODER,
    RERANK_NONE,
    RetrievalConfig,
    RetrievalConfigError,
    SUPPORTED_RERANK_STRATEGIES,
    SUPPORTED_STRATEGIES,
)
from evaluation.v11.retrieval.dense_strategy import DenseRetrievalStrategy
from evaluation.v11.retrieval.identifier_aware_strategy import (
    IdentifierAwareRetrievalStrategy,
)
from evaluation.v11.retrieval.rerank_stage import RerankStageError, apply_rerank
from evaluation.v11.retrieval.reranker import (
    RERANK_BATCH_SIZE,
    CrossEncoderReranker,
    NoOpReranker,
    Reranker,
    build_reranker,
)

__all__ = [
    "Candidate",
    "RankedRetrievalResult",
    "RetrievalRequest",
    "RetrievalStrategy",
    "RetrievalStrategyRegistry",
    "SearchBackend",
    "StrategyRetrievalOutcome",
    "build_retrieval_strategy",
    "default_registry",
    "RetrievalConfig",
    "RetrievalConfigError",
    "SUPPORTED_STRATEGIES",
    "DenseRetrievalStrategy",
    "IdentifierAwareRetrievalStrategy",
    # v1.2.1 rerank
    "RERANK_NONE",
    "RERANK_CROSS_ENCODER",
    "SUPPORTED_RERANK_STRATEGIES",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RERANK_TOP_N",
    "RERANK_BATCH_SIZE",
    "Reranker",
    "NoOpReranker",
    "CrossEncoderReranker",
    "build_reranker",
    "apply_rerank",
    "RerankStageError",
]
