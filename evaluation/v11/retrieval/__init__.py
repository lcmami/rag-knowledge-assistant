"""v1.1 Slice 4：版本化 RetrievalStrategy（dense / identifier_aware）。"""
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
    RetrievalConfig,
    RetrievalConfigError,
    SUPPORTED_STRATEGIES,
)
from evaluation.v11.retrieval.dense_strategy import DenseRetrievalStrategy
from evaluation.v11.retrieval.identifier_aware_strategy import (
    IdentifierAwareRetrievalStrategy,
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
]
