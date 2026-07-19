"""``current`` strategy（v1.1 Slice 3 §八）——包裝正式 TextSplitter，byte-identical parity。

**不複製**切分核心邏輯：直接委派 :class:`app.services.text_splitter.TextSplitter`（§八：若需
重構須保留相容 Adapter，不得複製兩套易漂移邏輯）。以現有真實預設 config（target/overlap/max_tokens）
執行時，Sample Corpus 的 chunk 數量 / 排序 / text / page / section / chunk_index 與改造前一致。
"""
from __future__ import annotations

from app.services.text_splitter import TextSplitter
from evaluation.v11.chunking.base import ChunkingContext, ChunkingStrategy, StrategyOutcome


class CurrentStrategy(ChunkingStrategy):
    name = "current"

    def chunk(self, context: ChunkingContext) -> StrategyOutcome:
        cfg = context.config
        splitter = TextSplitter(
            chunk_size_chars=cfg.target_chars,
            chunk_overlap_chars=cfg.overlap_chars,
            max_tokens=cfg.max_tokens,
            count_tokens=context.count_tokens,
            identifier_extractor=context.identifier_extractor,
        )
        chunks = splitter.split(
            context.elements,
            document_id=context.document_id,
            filename=context.filename,
            file_hash=context.file_hash,
            canonical=context.canonical,
        )
        return StrategyOutcome(chunks=chunks, strategy=self.name)
