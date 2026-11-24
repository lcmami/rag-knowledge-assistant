"""Corpus statistics（docs/external-corpus-api-plan.md §10 Slice D）。

由一份**既有 snapshot** 產生可重現的 corpus 統計報告。輸入只有 snapshot 目錄
（``manifest.json`` + ``normalized/*.md``）與一個 ``count_tokens`` callable；
不碰網路、不碰 Qdrant、不做 retrieval。

三個設計約束：

1. **決定性** —— 同一份 snapshot 必得同一份統計。百分位採 nearest-rank，不做插值；
   所有集合輸出皆排序。
2. **沿用既有管線** —— chunk 統計走 ``app.services.document_loader`` /
   ``canonical_builder`` 與 ``evaluation.v11.chunking``，**不**另寫一套切分邏輯。
   ``count_tokens`` 由呼叫端注入（正式執行注入 ``EmbeddingService.count_tokens``；
   unit test 注入決定性 fake，故測試完全不載入模型）。
3. **不引入新方法論** —— 混淆性指標只用確定性的字串規則（完全同名、詞幹碰撞、
   共同前綴）。**不**加入 ML clustering，**不**做新的 embedding 分析。

「hard token split」在本模組的定義是**明確的**：一個條文 section 的完整文字若本身
就超過 ``max_tokens``，它無論如何都會被切在 token 邊界而非語意邊界。這是可由語料
單獨決定的性質，故可重現；它與「chunk 因 target_chars 而分段」是兩件事，分開統計。
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from evaluation.corpus.snapshot import MANIFEST_FILENAME, NORMALIZED_DIRNAME

STATS_SCHEMA_VERSION = 1

#: 計畫 §7.4 / L2 的 pilot chunk 值。與 Slice C integration test 使用同一組數字。
PILOT_CHUNK_CONFIG = {
    "strategy": "current",
    "target_chars": 450,
    "overlap_chars": 80,
    "max_tokens": 480,
}

#: ``normalize.py`` 寫出的 metadata 欄位（順序即 Markdown 中的順序）。
METADATA_KEYS = (
    "法規名稱",
    "法規位階",
    "法規類別",
    "最後修正日期",
    "廢止狀態",
    "來源網址",
)
REPEALED_VALUE = "已廢止"

#: 混淆性：標題詞幹碰撞用的尾綴。剝除後相同即視為「同一主題的不同法制形式」。
TITLE_SUFFIXES = (
    "施行細則", "組織條例", "組織規程", "組織法", "條例", "通則", "準則",
    "細則", "辦法", "規程", "規則", "標準", "綱要", "法",
)
#: 計畫 §5.3 用 6 字前綴衡量共用前綴的法規數，此處沿用同一個定義。
TITLE_PREFIX_LEN = 6
#: 詞幹至少要有這麼長才算數。否則 ``憲法`` 會被剝成 ``憲``，與所有「憲*法」
#: 假性碰撞 —— 單字詞幹沒有辨識力，只會製造雜訊。
MIN_TITLE_STEM_LEN = 2

_METADATA_RE = re.compile(r"^-\s*([^：]+)：(.*)$")
_ARTICLE_HEADING_RE = re.compile(r"^##\s+(.*)$")


# ---------------------------------------------------------------------------
# 1. 由 normalized Markdown 還原每份文件的事實
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentFacts:
    """一份 normalized 文件的可稽核事實（全部由 Markdown 本身還原）。"""

    filename: str
    pcode: str
    law_name: str
    metadata: dict[str, str]
    has_foreword: bool
    sections: tuple[str, ...]  # type-A 條文的 section 名稱，依文件順序
    char_count: int

    @property
    def is_repealed(self) -> bool:
        return self.metadata.get("廢止狀態", "") == REPEALED_VALUE

    @property
    def article_count(self) -> int:
        return len(self.sections)


def parse_normalized_document(filename: str, markdown: str) -> DocumentFacts:
    """由 normalized Markdown 還原文件事實。

    只依賴 ``normalize.py`` 實際寫出的結構：``# 法規名稱`` → metadata 清單 →
    （選用）前言 → 一連串 ``## {章節路徑｜條號}``。前言的判定是「preamble 中既非
    標題、也非 metadata 清單的非空行」，故不需要回頭讀原始來源。
    """
    lines = markdown.split("\n")
    law_name = ""
    metadata: dict[str, str] = {}
    sections: list[str] = []
    foreword_lines: list[str] = []
    in_preamble = True

    for index, line in enumerate(lines):
        heading = _ARTICLE_HEADING_RE.match(line)
        if heading:
            in_preamble = False
            sections.append(heading.group(1).strip())
            continue
        if not in_preamble:
            continue
        if index == 0 and line.startswith("# "):
            law_name = line[2:].strip()
            continue
        stripped = line.strip()
        if not stripped:
            continue
        match = _METADATA_RE.match(stripped)
        if match:
            metadata[match.group(1).strip()] = match.group(2).strip()
            continue
        foreword_lines.append(stripped)

    return DocumentFacts(
        filename=filename,
        pcode=filename[:-3] if filename.endswith(".md") else filename,
        law_name=law_name,
        metadata=metadata,
        has_foreword=bool(foreword_lines),
        sections=tuple(sections),
        char_count=len(markdown),
    )


def collect_documents(snapshot_dir: str) -> list[DocumentFacts]:
    """讀取 snapshot 的 ``normalized/``，依檔名排序回傳文件事實。"""
    normalized = os.path.join(snapshot_dir, NORMALIZED_DIRNAME)
    facts: list[DocumentFacts] = []
    for name in sorted(os.listdir(normalized)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(normalized, name), "r", encoding="utf-8") as handle:
            facts.append(parse_normalized_document(name, handle.read()))
    return facts


# ---------------------------------------------------------------------------
# 2. 決定性百分位
# ---------------------------------------------------------------------------
def percentile(values: list[int], pct: float) -> Optional[int]:
    """Nearest-rank 百分位（不插值），故結果恆為語料中真實存在的值。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def distribution(values: Iterable[int]) -> dict[str, Any]:
    """min / p50 / p90 / p95 / p99 / max / mean / count 的決定性摘要。"""
    data = list(values)
    if not data:
        return {"count": 0, "min": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(data),
        "min": min(data),
        "p50": percentile(data, 50),
        "p90": percentile(data, 90),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": max(data),
        # 四捨五入到小數兩位，避免浮點尾數影響輸出的可重現性
        "mean": round(sum(data) / len(data), 2),
    }


# ---------------------------------------------------------------------------
# 3. metadata 完整度
# ---------------------------------------------------------------------------
def metadata_stats(documents: list[DocumentFacts]) -> dict[str, Any]:
    """每個 metadata 欄位的非空比例，以及前言 / 廢止標記的完整度。"""
    total = len(documents)
    present = {
        key: sum(1 for d in documents if d.metadata.get(key, "").strip())
        for key in METADATA_KEYS
    }
    repealed = [d for d in documents if d.is_repealed]
    return {
        "document_count": total,
        "field_present_count": dict(sorted(present.items())),
        "field_complete": sorted(k for k, v in present.items() if v == total),
        "field_incomplete": sorted(k for k, v in present.items() if v != total),
        "all_fields_complete_count": sum(
            1 for d in documents if all(d.metadata.get(k, "").strip() for k in METADATA_KEYS)
        ),
        "laws_with_foreword": sum(1 for d in documents if d.has_foreword),
        "laws_without_foreword": sum(1 for d in documents if not d.has_foreword),
        # 已廢止法規的廢止註記完整度（normalize.py 僅在有註記時才寫出該行）
        "repealed_law_count": len(repealed),
        "repealed_with_abandon_note": sum(
            1 for d in repealed if d.metadata.get("廢止註記", "").strip()
        ),
    }


# ---------------------------------------------------------------------------
# 4. 混淆性（確定性字串規則，無 ML）
# ---------------------------------------------------------------------------
def title_stem(law_name: str) -> str:
    """剝除單一法制形式尾綴後的標題詞幹（如 ``組織法`` / ``組織條例`` → 同幹）。

    剝除後不足 :data:`MIN_TITLE_STEM_LEN` 字者保留原標題 —— 單字詞幹沒有辨識力。
    """
    for suffix in TITLE_SUFFIXES:
        if law_name.endswith(suffix) and len(law_name) - len(suffix) >= MIN_TITLE_STEM_LEN:
            return law_name[: -len(suffix)]
    return law_name


def _groups(pairs: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, value in pairs:
        out.setdefault(key, []).append(value)
    return {k: sorted(v) for k, v in sorted(out.items()) if len(v) > 1}


def confusability_stats(documents: list[DocumentFacts]) -> dict[str, Any]:
    """標題混淆性指標：完全同名、詞幹碰撞、共用前綴、現行／已廢止交錯。"""
    identical = _groups((d.law_name, d.pcode) for d in documents)
    stems = _groups((title_stem(d.law_name), d.pcode) for d in documents)
    prefixes = _groups(
        (d.law_name[:TITLE_PREFIX_LEN], d.pcode)
        for d in documents
        if len(d.law_name) >= TITLE_PREFIX_LEN
    )

    by_pcode = {d.pcode: d for d in documents}
    # 詞幹相同但一現行一已廢止 —— 最容易讓 retrieval 取錯版本的那一類
    mixed = {
        stem: members
        for stem, members in stems.items()
        if len({by_pcode[p].is_repealed for p in members}) > 1
    }
    return {
        "identical_title_count": len(identical),
        "identical_title_groups": identical,
        "title_stem_collision_group_count": len(stems),
        "title_stem_collision_law_count": sum(len(v) for v in stems.values()),
        "title_stem_collision_groups": stems,
        "shared_title_prefix_len": TITLE_PREFIX_LEN,
        "shared_title_prefix_group_count": len(prefixes),
        "shared_title_prefix_law_count": sum(len(v) for v in prefixes.values()),
        "active_repealed_confuser_group_count": len(mixed),
        "active_repealed_confuser_groups": mixed,
    }


def identity_stats(documents: list[DocumentFacts]) -> dict[str, Any]:
    """pcode / 標題的唯一性（重複即代表來源契約假設被打破）。"""
    names = [d.law_name for d in documents]
    pcodes = [d.pcode for d in documents]
    return {
        "unique_law_names": len(set(names)),
        "unique_pcodes": len(set(pcodes)),
        "duplicate_law_name_count": len(names) - len(set(names)),
        "duplicate_pcode_count": len(pcodes) - len(set(pcodes)),
    }


# ---------------------------------------------------------------------------
# 5. Chunk 統計（走既有管線）
# ---------------------------------------------------------------------------
def _chunk_document(
    filename: str,
    abs_path: str,
    chunk_config: dict[str, Any],
    count_tokens: Callable[[str], int],
):
    """走**既有** loader / canonical builder / chunking strategy，不另寫切分邏輯。"""
    import hashlib

    from app.services.canonical_builder import build_canonical_document
    from app.services.document_loader import load_document
    from evaluation.v11.chunking import ChunkingConfig, ChunkingContext, build_strategy

    with open(abs_path, "rb") as handle:
        raw = handle.read()
    file_hash = hashlib.sha256(raw).hexdigest()
    elements = load_document(filename, raw)
    canonical = build_canonical_document(
        filename=filename,
        document_sha256=file_hash,
        document_type="markdown",
        elements=elements,
    )
    cfg = ChunkingConfig.from_raw(chunk_config)
    ctx = ChunkingContext(
        document_id="corpus-stats",
        filename=filename,
        file_hash=file_hash,
        canonical=canonical,
        elements=elements,
        config=cfg,
        count_tokens=count_tokens,
    )
    return canonical, build_strategy(cfg.strategy).chunk(ctx)


def chunk_stats(
    snapshot_dir: str,
    documents: list[DocumentFacts],
    *,
    count_tokens: Callable[[str], int],
    chunk_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """chunk 規模、大小分布、header/article 組成與 hard token split。

    **header chunk 的判定**：``section == 法規名稱``。不用 ``chunk_index == 0``——
    前言很長時法規標頭可能不只一個 chunk，用 index 會低估。
    """
    cfg = dict(chunk_config or PILOT_CHUNK_CONFIG)
    max_tokens = int(cfg["max_tokens"])
    normalized = os.path.join(snapshot_dir, NORMALIZED_DIRNAME)

    segment_total = 0
    header_chunks = 0
    article_chunks = 0
    fallback_total = 0
    chunk_chars: list[int] = []
    chunk_tokens: list[int] = []
    chunks_per_law: list[int] = []
    chunks_per_article: list[int] = []
    header_with_foreword = 0
    header_metadata_only = 0
    oversize_sections = 0
    total_sections = 0
    chunks_over_limit = 0

    for doc in documents:
        canonical, outcome = _chunk_document(
            doc.filename, os.path.join(normalized, doc.filename), cfg, count_tokens
        )
        segment_total += len(canonical.segments)
        chunks_per_law.append(len(outcome.chunks))
        fallback_total += outcome.fallback_count

        per_section: dict[str, int] = {}
        for chunk in outcome.chunks:
            tokens = count_tokens(chunk.text)
            chunk_chars.append(len(chunk.text))
            chunk_tokens.append(tokens)
            if tokens > max_tokens:
                chunks_over_limit += 1
            if chunk.section == doc.law_name:
                header_chunks += 1
            else:
                article_chunks += 1
                per_section[chunk.section] = per_section.get(chunk.section, 0) + 1

        chunks_per_article.extend(per_section[s] for s in sorted(per_section))
        if any(c.section == doc.law_name for c in outcome.chunks):
            if doc.has_foreword:
                header_with_foreword += 1
            else:
                header_metadata_only += 1

        # 條文本身即超過 token 上限者，必然被切在 token 邊界（hard token split）
        for section in doc.sections:
            total_sections += 1
            text = "\n".join(
                s.normalized_text for s in canonical.segments if s.section == section
            )
            if count_tokens(text) > max_tokens:
                oversize_sections += 1

    total_chunks = header_chunks + article_chunks
    return {
        "chunk_config": {k: cfg[k] for k in sorted(cfg)},
        "canonical_segment_count": segment_total,
        "total_chunk_count": total_chunks,
        "article_chunk_count": article_chunks,
        "law_header_chunk_count": header_chunks,
        "law_header_chunk_share_pct": (
            round(100.0 * header_chunks / total_chunks, 2) if total_chunks else 0.0
        ),
        "law_header_with_foreword": header_with_foreword,
        "law_header_metadata_only": header_metadata_only,
        "chunks_per_law": distribution(chunks_per_law),
        "chunks_per_article": distribution(chunks_per_article),
        "chunk_char_length": distribution(chunk_chars),
        "chunk_token_length": distribution(chunk_tokens),
        # hard token split：條文全文本身就超過 max_tokens，無論如何都會被硬切
        "hard_token_split_count": oversize_sections,
        "hard_token_split_rate_pct": (
            round(100.0 * oversize_sections / total_sections, 2) if total_sections else 0.0
        ),
        "article_section_count": total_sections,
        # 切分後仍超過上限者必須為 0（代表 token 上限強制有生效）
        "chunks_over_token_limit": chunks_over_limit,
        "strategy_fallback_count": fallback_total,
    }


# ---------------------------------------------------------------------------
# 6. 組合
# ---------------------------------------------------------------------------
def build_stats(
    snapshot_dir: str,
    *,
    count_tokens: Callable[[str], int],
    chunk_config: Optional[dict[str, Any]] = None,
    runtime: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """由 snapshot 產生完整統計 dict（可直接 canonical_json 序列化）。"""
    import json

    with open(os.path.join(snapshot_dir, MANIFEST_FILENAME), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    documents = collect_documents(snapshot_dir)
    repealed = sum(1 for d in documents if d.is_repealed)

    return {
        "stats_schema_version": STATS_SCHEMA_VERSION,
        "snapshot_id": manifest["snapshot_id"],
        "corpus_hash": manifest["corpus_hash"],
        "source": {
            "provider": manifest["provider"],
            "source_endpoint": manifest["source_endpoint"],
            "source_update_date": manifest["source_update_date"],
            "raw_sha256": manifest["raw_sha256"],
            "raw_bytes": manifest["raw_bytes"],
            "selection": manifest["selection"],
            "selection_hash": manifest["selection_hash"],
        },
        "scale": {
            "source_law_count": manifest["source_record_count"],
            "selected_law_count": manifest["selected_record_count"],
            "normalized_document_count": manifest["normalized_document_count"],
            "skipped_record_count": manifest["skipped_record_count"],
            "skipped_article_count": manifest["skipped_article_count"],
            "article_count": manifest["article_count"],
            "repealed_law_count": repealed,
            "active_law_count": len(documents) - repealed,
            "normalized_bytes": sum(d.char_count for d in documents),
        },
        "identity": identity_stats(documents),
        "metadata": metadata_stats(documents),
        "confusability": confusability_stats(documents),
        "chunks": chunk_stats(
            snapshot_dir, documents, count_tokens=count_tokens, chunk_config=chunk_config
        ),
        # runtime 只是觀察值，不建立任何 SLO
        "runtime": dict(sorted((runtime or {}).items())),
    }


def render_markdown(stats: dict[str, Any]) -> str:
    """把統計 dict 呈現為簡短 Markdown（不含任何本機絕對路徑）。"""
    scale, chunks = stats["scale"], stats["chunks"]
    conf, meta = stats["confusability"], stats["metadata"]

    def row(label: str, value: Any) -> str:
        return f"| {label} | {value} |"

    def dist_row(label: str, d: dict[str, Any]) -> str:
        return (
            f"| {label} | {d['min']} | {d['p50']} | {d['p90']} | {d['p95']} "
            f"| {d['p99']} | {d['max']} | {d['mean']} |"
        )

    lines = [
        f"### Snapshot `{stats['snapshot_id']}`",
        "",
        "| 項目 | 值 |",
        "| --- | --- |",
        row("source_update_date", stats["source"]["source_update_date"]),
        row("corpus_hash", f"`{stats['corpus_hash']}`"),
        row("來源法規數", scale["source_law_count"]),
        row("挑選法規數", scale["selected_law_count"]),
        row("normalized 文件數", scale["normalized_document_count"]),
        row("type-A 條文數", scale["article_count"]),
        row("已廢止 / 現行", f"{scale['repealed_law_count']} / {scale['active_law_count']}"),
        row("skipped record / article", f"{scale['skipped_record_count']} / {scale['skipped_article_count']}"),
        "",
        "### Chunk",
        "",
        "| 項目 | 值 |",
        "| --- | --- |",
        row("canonical segment", chunks["canonical_segment_count"]),
        row("chunk 總數", chunks["total_chunk_count"]),
        row("條文 chunk", chunks["article_chunk_count"]),
        row("法規標頭 chunk", chunks["law_header_chunk_count"]),
        row("標頭占比", f"{chunks['law_header_chunk_share_pct']}%"),
        row("標頭含前言 / 僅 metadata",
            f"{chunks['law_header_with_foreword']} / {chunks['law_header_metadata_only']}"),
        row("hard token split",
            f"{chunks['hard_token_split_count']} / {chunks['article_section_count']}"
            f"（{chunks['hard_token_split_rate_pct']}%）"),
        row("切分後超過 token 上限", chunks["chunks_over_token_limit"]),
        row("strategy fallback", chunks["strategy_fallback_count"]),
        "",
        "| 分布 | min | p50 | p90 | p95 | p99 | max | mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        dist_row("chunk 字元數", chunks["chunk_char_length"]),
        dist_row("chunk token 數", chunks["chunk_token_length"]),
        dist_row("每部法規 chunk 數", chunks["chunks_per_law"]),
        dist_row("每條條文 chunk 數", chunks["chunks_per_article"]),
        "",
        "### Identity / metadata / 混淆性",
        "",
        "| 項目 | 值 |",
        "| --- | --- |",
        row("unique pcode / 標題", f"{stats['identity']['unique_pcodes']} / {stats['identity']['unique_law_names']}"),
        row("重複 pcode / 標題",
            f"{stats['identity']['duplicate_pcode_count']} / {stats['identity']['duplicate_law_name_count']}"),
        row("metadata 六欄全備", f"{meta['all_fields_complete_count']} / {meta['document_count']}"),
        row("有前言 / 無前言", f"{meta['laws_with_foreword']} / {meta['laws_without_foreword']}"),
        row("已廢止且有廢止註記",
            f"{meta['repealed_with_abandon_note']} / {meta['repealed_law_count']}"),
        row("完全同名群組", conf["identical_title_count"]),
        row("標題詞幹碰撞",
            f"{conf['title_stem_collision_group_count']} 群 / "
            f"{conf['title_stem_collision_law_count']} 部"),
        row(f"共用 {conf['shared_title_prefix_len']} 字前綴",
            f"{conf['shared_title_prefix_group_count']} 群 / "
            f"{conf['shared_title_prefix_law_count']} 部"),
        row("現行／已廢止交錯詞幹", conf["active_repealed_confuser_group_count"]),
    ]
    if stats["runtime"]:
        lines += ["", "### Runtime（觀察值，非 SLO）", "", "| 項目 | 值 |", "| --- | --- |"]
        lines += [row(k, v) for k, v in stats["runtime"].items()]
    return "\n".join(lines) + "\n"
