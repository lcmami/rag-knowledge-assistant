"""``paragraph`` strategy（v1.1 Slice 3 §十）——完整段落優先，超限才 fixed fallback。

預設不跨 page / 不跨 section（由 (page, section) 分群保證）；本 Slice 不提供跨 section 選項。
paragraph 不支援 section-title prefix（由 ChunkingConfig 驗證拒絕），故無 prefix token 預算問題。
"""
from __future__ import annotations

from evaluation.v11.chunking.base import (
    ChunkingContext,
    ChunkingStrategy,
    ChunkPlan,
    StrategyOutcome,
    build_chunks,
    group_segments,
)
from evaluation.v11.chunking.strategies_common import paragraph_plans


class ParagraphAwareStrategy(ChunkingStrategy):
    name = "paragraph"

    def chunk(self, context: ChunkingContext) -> StrategyOutcome:
        cfg = context.config
        count = context.count_tokens
        plans: list[ChunkPlan] = []
        reasons: list[str] = []

        for group in group_segments(context.canonical):
            # paragraph 無 prefix → effective_max = max_tokens。
            group_plans, group_reasons = paragraph_plans(group, cfg, count, cfg.max_tokens)
            plans.extend(group_plans)
            reasons.extend(group_reasons)

        outcome = build_chunks(plans, context)
        outcome.fallback_reasons = reasons + outcome.fallback_reasons
        outcome.fallback_count = len(outcome.fallback_reasons)
        return outcome
