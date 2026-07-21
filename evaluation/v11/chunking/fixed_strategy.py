"""``fixed`` strategy（v1.1 Slice 3 §九）——固定視窗 + configurable overlap + token 硬限制。

在每個 (page, section) 群組的 canonical group_text 上滑動視窗；優先於合理文字邊界切分，
找不到才硬切；overlap 不產生空 chunk / 不無限迴圈。Provenance 只涵蓋實際進入 chunk 的 canonical
子範圍（§九.8）。section-title prefix 只影響 embedding/retrieval text，不偽造 canonical offset
（§九.9/10：prefix 不作為 gold evidence span——由 build_chunks 以「無 span 的純文字 prefix」實現）。
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
from evaluation.v11.chunking.strategies_common import (
    context_spans,
    covered_indices,
    fixed_windows,
    group_text_and_ranges,
    spans_for_window,
)


class FixedWindowStrategy(ChunkingStrategy):
    name = "fixed"

    def chunk(self, context: ChunkingContext) -> StrategyOutcome:
        cfg = context.config
        count = context.count_tokens
        plans: list[ChunkPlan] = []

        for group in group_segments(context.canonical):
            section = group[0].section
            group_text, ranges, base = group_text_and_ranges(group)

            # core 需為 prefix 預留 token 預算，確保 prefix+core <= max_tokens（context 另由
            # build_chunks 依 max_tokens 縮減）。
            effective_max = cfg.max_tokens
            if cfg.prepend_section_title and section and section.strip():
                effective_max = max(1, cfg.max_tokens - count(section_title_prefix(section)))

            for ws, we in fixed_windows(
                group_text, cfg.target_chars, cfg.overlap_chars, effective_max, count
            ):
                core_text = group_text[ws:we]
                if not core_text.strip():
                    continue
                core_spans = spans_for_window(group_text, ranges, ws, we, base)
                first_idx, last_idx = covered_indices(ranges, ws, we)
                before, after = context_spans(
                    group, first_idx, last_idx,
                    cfg.context_before_paragraphs, cfg.context_after_paragraphs,
                )
                plans.append(
                    ChunkPlan(
                        page=group[0].page,
                        section=section,
                        core_text=core_text,
                        core_spans=core_spans,
                        before_context=before,
                        after_context=after,
                    )
                )

        return build_chunks(plans, context)
