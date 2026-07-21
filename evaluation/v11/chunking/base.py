"""ChunkingStrategy 介面、輸入 context、共用 assembly / registry（v1.1 Slice 3 §七/§十二/§十三）。

策略只需產生 :class:`ChunkPlan`（core 文字 + core canonical spans + 相鄰段落 context）；
:func:`build_chunks` 統一處理：

- section-title prefix（§十三：影響 embedding text、**不**偽造 canonical offset）。
- 相鄰段落 context 的 token-limit 縮減（§十二.7：先減 after、再減 before、不擠掉 core）。
- Provenance span 的 ``role`` 標記與**與文字一致的順序**（context → core → context）。
- deterministic chunk_index、identifier 抽取、空 chunk 過濾、max_tokens 硬限制。

所有策略共用同一 canonical 座標語意：``ChunkPlan`` 內的 char_start/char_end 皆為
**normalized canonical text** 座標（與 Slice 1 ProvenanceSpan 一致）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from app.models.canonical import CanonicalDocument, CanonicalSegment, ProvenanceSpan
from app.models.chunk import Chunk
from app.models.document_element import DocumentElement
from app.services.identifier_extractor import IdentifierExtractor
from evaluation.v11.chunking.config import ChunkingConfig, section_title_prefix

# core / context 之間、以及與 prefix 之後的文字連接分隔（與 canonical builder 一致）。
_JOIN = "\n\n"

ROLE_CORE = "core"
ROLE_CONTEXT = "context"


# --------------------------------------------------------------------------
# 輸入 / 輸出
# --------------------------------------------------------------------------
@dataclass
class ChunkingContext:
    """Chunking Strategy 的輸入（§七）。"""

    document_id: str
    filename: str
    file_hash: str
    canonical: CanonicalDocument
    # 原始 Loader elements：``current`` strategy 委派正式 TextSplitter 需要（parity）。
    elements: list[DocumentElement]
    config: ChunkingConfig
    count_tokens: Callable[[str], int]
    identifier_extractor: IdentifierExtractor = field(default_factory=IdentifierExtractor)


@dataclass
class _SpanContrib:
    """一段進入 chunk 的 canonical 貢獻（core 或 context）。

    ``text`` 為實際併入 chunk 的文字；char_start/char_end 為其 normalized canonical 範圍
    （§九.8 / §十二.4：只涵蓋實際進入 chunk 的範圍，不偽造）。
    """

    char_start: int
    char_end: int
    page: Optional[int]
    section: Optional[str]
    paragraph_index: int
    offset_scope: str
    text: str


@dataclass
class ChunkPlan:
    """策略產出的單一 chunk 規劃（尚未套 prefix / token 縮減）。"""

    page: Optional[int]
    section: Optional[str]
    core_text: str
    core_spans: list[_SpanContrib]
    # before：由**遠到近**排序（縮減時從最遠端移除）。after：由**近到遠**排序。
    before_context: list[_SpanContrib] = field(default_factory=list)
    after_context: list[_SpanContrib] = field(default_factory=list)


@dataclass
class StrategyOutcome:
    """策略執行結果（chunks + fallback / context 統計，供 Report §二十三）。"""

    chunks: list[Chunk]
    strategy: str
    fallback_count: int = 0
    fallback_reasons: list[str] = field(default_factory=list)
    context_added_count: int = 0
    context_reduced_count: int = 0

    def stats(self) -> dict:
        # deterministic reasons（去重 + 排序）供 Report。
        return {
            "strategy": self.strategy,
            "chunk_count": len(self.chunks),
            "fallback_count": self.fallback_count,
            "fallback_reasons": sorted(set(self.fallback_reasons)),
            "context_added_count": self.context_added_count,
            "context_reduced_count": self.context_reduced_count,
        }


# --------------------------------------------------------------------------
# 共用 canonical 工具
# --------------------------------------------------------------------------
def group_segments(canonical: CanonicalDocument) -> list[list[CanonicalSegment]]:
    """依連續 (page, section) 將 canonical segments 分群（不跨 page / 不跨 section）。"""
    groups: list[list[CanonicalSegment]] = []
    for seg in canonical.segments:
        if groups and groups[-1][0].page == seg.page and groups[-1][0].section == seg.section:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return groups


def segment_span(seg: CanonicalSegment) -> _SpanContrib:
    """整段 segment 作為一個貢獻（char 範圍即 segment 全域範圍）。"""
    return _SpanContrib(
        char_start=seg.char_start,
        char_end=seg.char_end,
        page=seg.page,
        section=seg.section,
        paragraph_index=seg.paragraph_index,
        offset_scope=seg.offset_scope,
        text=seg.normalized_text,
    )


def _to_provenance(canonical: CanonicalDocument, sc: _SpanContrib, role: str) -> ProvenanceSpan:
    return ProvenanceSpan(
        document_sha256=canonical.document_sha256,
        canonical_text_version=canonical.canonical_text_version,
        loader_version=canonical.loader_version,
        page=sc.page,
        section=sc.section,
        paragraph_index=sc.paragraph_index,
        offset_scope=sc.offset_scope,
        char_start=sc.char_start,
        char_end=sc.char_end,
        role=role,
    )


# --------------------------------------------------------------------------
# 統一 chunk assembly：prefix + context 縮減 + provenance + chunk_index
# --------------------------------------------------------------------------
def build_chunks(plans: list[ChunkPlan], context: ChunkingContext) -> StrategyOutcome:
    """把策略產出的 ChunkPlan 清單組成最終 Chunk 清單（§十二/§十三統一處理）。"""
    cfg = context.config
    canonical = context.canonical
    count = context.count_tokens
    created_at = datetime.now(timezone.utc).isoformat()

    chunks: list[Chunk] = []
    fallback_reasons: list[str] = []
    context_added = 0
    context_reduced = 0
    chunk_index = 0

    for plan in plans:
        core_text = plan.core_text
        if not core_text.strip():
            continue  # §七.4：不產生空 chunk

        # --- section-title prefix（§十三）---
        prefix = ""
        if cfg.prepend_section_title:
            prefix = section_title_prefix(plan.section)
            if prefix and count(prefix) >= cfg.max_tokens:
                # 極端：prefix 本身即超限 → 停用 prefix 並記錄 fallback（不擠掉 core）。
                prefix = ""
                fallback_reasons.append("prefix_dropped_exceeds_max_tokens")

        # --- 相鄰段落 context（可縮減）---
        before = list(plan.before_context)
        after = list(plan.after_context)
        requested_ctx = len(before) + len(after)

        def _assemble(pre: str, befs: list[_SpanContrib], afts: list[_SpanContrib]) -> str:
            parts = [c.text for c in befs] + [core_text] + [c.text for c in afts]
            return pre + _JOIN.join(parts)

        text = _assemble(prefix, before, after)
        # §十二.6/7：超 token 時先減 after（最遠端）、再減 before（最遠端），永不動 core。
        while count(text) > cfg.max_tokens and (after or before):
            if after:
                after.pop()          # after 由近到遠 → tail 為最遠
            else:
                before.pop(0)        # before 由遠到近 → head 為最遠
            text = _assemble(prefix, before, after)

        added_ctx = len(before) + len(after)
        context_added += added_ctx
        context_reduced += requested_ctx - added_ctx

        # --- provenance（順序與文字一致：before(context) → core → after(context)）---
        spans: list[ProvenanceSpan] = []
        for sc in before:
            spans.append(_to_provenance(canonical, sc, ROLE_CONTEXT))
        for sc in plan.core_spans:
            spans.append(_to_provenance(canonical, sc, ROLE_CORE))
        for sc in after:
            spans.append(_to_provenance(canonical, sc, ROLE_CONTEXT))

        identifiers = context.identifier_extractor.extract_normalized(text)

        chunks.append(
            Chunk(
                id=_new_id(),
                document_id=context.document_id,
                filename=context.filename,
                file_hash=context.file_hash,
                page=plan.page,
                section=plan.section,
                chunk_index=chunk_index,
                text=text,
                created_at=created_at,
                provenance_spans=spans,
                identifiers=identifiers,
                canonical_text_version=canonical.canonical_text_version,
                loader_version=canonical.loader_version,
                identifier_normalization_version=context.identifier_extractor.version,
            )
        )
        chunk_index += 1

    return StrategyOutcome(
        chunks=chunks,
        strategy=cfg.strategy,
        fallback_reasons=fallback_reasons,
        context_added_count=context_added,
        context_reduced_count=context_reduced,
    )


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())


# --------------------------------------------------------------------------
# Strategy 介面 + Registry
# --------------------------------------------------------------------------
class ChunkingStrategy(ABC):
    """可插拔 Chunking Strategy 介面（§七）。所有實作必須 deterministic。"""

    name: str

    @abstractmethod
    def chunk(self, context: ChunkingContext) -> StrategyOutcome:  # pragma: no cover - 介面
        ...


class ChunkingStrategyRegistry:
    """Strategy 名稱 → 實作的註冊表。"""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ChunkingStrategy]] = {}

    def register(self, name: str, factory: Callable[[], ChunkingStrategy]) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> ChunkingStrategy:
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(
                f"未註冊的 chunk strategy={name!r}；已註冊：{sorted(self._factories)}。"
            )
        return factory()

    def names(self) -> list[str]:
        return sorted(self._factories)


def default_registry() -> ChunkingStrategyRegistry:
    """建立含全部內建策略的 registry（延遲 import 避免循環相依）。"""
    from evaluation.v11.chunking.current_strategy import CurrentStrategy
    from evaluation.v11.chunking.fixed_strategy import FixedWindowStrategy
    from evaluation.v11.chunking.heading_strategy import HeadingAwareStrategy
    from evaluation.v11.chunking.paragraph_strategy import ParagraphAwareStrategy

    reg = ChunkingStrategyRegistry()
    reg.register("current", CurrentStrategy)
    reg.register("fixed", FixedWindowStrategy)
    reg.register("paragraph", ParagraphAwareStrategy)
    reg.register("heading", HeadingAwareStrategy)
    return reg


def build_strategy(name: str) -> ChunkingStrategy:
    return default_registry().create(name)
