"""``heading`` strategy（v1.1 Slice 3 §十一）——依既有 section metadata 分組。

使用 Loader / Canonical Document **已存在**的 section metadata，**不**用 LLM / ML 推測標題。
先依明確 section 分組（由 (page, section) 分群保證），section 內套用 paragraph-aware 行為；
預設不跨 section。``prepend_section_title=true`` 時於 chunk text 前加一致格式的 title prefix；
section metadata 缺失時明確 fallback 到 paragraph 行為（不加 "None"/空白/虛構標題），且 fallback
於 Trial Report 可見（§十一.6/8/9）。
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
from evaluation.v11.chunking.config import section_title_prefix
from evaluation.v11.chunking.strategies_common import paragraph_plans


class HeadingAwareStrategy(ChunkingStrategy):
    name = "heading"

    def chunk(self, context: ChunkingContext) -> StrategyOutcome:
        cfg = context.config
        count = context.count_tokens
        plans: list[ChunkPlan] = []
        reasons: list[str] = []

        for group in group_segments(context.canonical):
            section = group[0].section
            has_section = bool(section and section.strip())

            # section 存在且啟用 prefix → 為 prefix 預留 token 預算。
            effective_max = cfg.max_tokens
            if cfg.prepend_section_title and has_section:
                effective_max = max(1, cfg.max_tokens - count(section_title_prefix(section)))

            # 缺 section metadata → 明確 fallback paragraph 行為（§十一.8/9，Report 可見）。
            if not has_section:
                reasons.append("heading:no_section_metadata->paragraph")

            group_plans, group_reasons = paragraph_plans(group, cfg, count, effective_max)
            plans.extend(group_plans)
            reasons.extend(group_reasons)

        outcome = build_chunks(plans, context)
        outcome.fallback_reasons = reasons + outcome.fallback_reasons
        outcome.fallback_count = len(outcome.fallback_reasons)
        return outcome
