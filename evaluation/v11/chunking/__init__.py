"""版本化 Chunking Strategy 搜尋空間（v1.1 Slice 3 §六–§十三）。

**Optimizer 專用**、與正式索引路徑完全隔離：正式 :class:`app.services.text_splitter.TextSplitter`
與 :class:`app.services.indexing_service.IndexingService` 於本 Slice **完全不變**；本套件僅供
離線 Trial Runner 探索不同 Chunking 設定。

- :mod:`.config`：版本化、嚴格驗證的 :class:`ChunkingConfig` + deterministic canonical 序列化。
- :mod:`.base`：:class:`ChunkingStrategy` 介面、:class:`ChunkingContext` 輸入、共用 provenance /
  section-title prefix / 相鄰段落 context 工具，以及 :class:`ChunkingStrategyRegistry`。
- :mod:`.current_strategy`：包裝正式 TextSplitter 的 ``current`` strategy（byte-identical parity）。
- :mod:`.fixed_strategy` / :mod:`.paragraph_strategy` / :mod:`.heading_strategy`：新增策略。

本 Slice **不**實作自動 grid / random / bayesian search、Pareto、best-candidate 自動選擇。
Trial Config 只能**明確列出**要嘗試的 Chunking 設定。
"""
from __future__ import annotations

from evaluation.v11.chunking.base import (
    ChunkingContext,
    ChunkingStrategy,
    ChunkingStrategyRegistry,
    StrategyOutcome,
    build_strategy,
    default_registry,
)
from evaluation.v11.chunking.config import (
    PREFIX_FORMAT_VERSION,
    SUPPORTED_STRATEGIES,
    ChunkConfigError,
    ChunkingConfig,
)

__all__ = [
    "ChunkConfigError",
    "ChunkingConfig",
    "ChunkingContext",
    "ChunkingStrategy",
    "ChunkingStrategyRegistry",
    "StrategyOutcome",
    "SUPPORTED_STRATEGIES",
    "PREFIX_FORMAT_VERSION",
    "build_strategy",
    "default_registry",
]
