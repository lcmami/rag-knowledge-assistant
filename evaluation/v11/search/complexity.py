"""版本化 Config Complexity（Slice 5 §十八「Config Complexity」）。

用途**僅限最後的 tie-break**：當所有 retrieval 品質指標完全相同時，偏好較簡單、
較容易解釋與維護的設定。它**不得蓋過 retrieval quality** —— 這由
:mod:`recommendation` 的 lexicographic 順序保證（complexity 必須排在所有 metric 之後）。

複雜度以「可解釋的分量 tuple」表示（依序比較，數字越小越簡單）：

1. retrieval strategy：``dense`` (0) < ``identifier_aware`` (1)
2. chunking strategy：``current`` (0) < ``paragraph`` (1) < ``heading`` / ``fixed`` (2)
3. 是否使用 section prefix / context：無 (0) < 有 (1)
4. 非預設參數個數（越少越簡單）

規則變更 → bump :data:`COMPLEXITY_VERSION`，報告據此標示可比較性。
"""
from __future__ import annotations

from typing import Any

from evaluation.v11.chunking.config import ChunkingConfig
from evaluation.v11.retrieval.config import RetrievalConfig

COMPLEXITY_VERSION = "v1"

_RETRIEVAL_STRATEGY_RANK = {"dense": 0, "identifier_aware": 1}
_CHUNK_STRATEGY_RANK = {"current": 0, "paragraph": 1, "heading": 2, "fixed": 2}


def _chunk_defaults() -> dict[str, Any]:
    return ChunkingConfig.from_raw({"strategy": "current"}).to_canonical_dict()


def _retrieval_defaults(strategy: str) -> dict[str, Any]:
    return RetrievalConfig.from_raw({"strategy": strategy}).to_canonical_dict()


def _non_default_count(
    chunk_config: dict[str, Any], retrieval_config: dict[str, Any]
) -> int:
    """相對於「同 strategy 的預設值」有多少參數被改動。

    strategy / schema_version / ranking_version / prefix_format_version 屬版本或身分欄位，
    不計入「調了幾個旋鈕」。
    """
    skip = {"strategy", "schema_version", "ranking_version", "prefix_format_version"}
    count = 0

    chunk_ref = _chunk_defaults()
    for key, value in chunk_config.items():
        if key in skip:
            continue
        # current 以外的 strategy 沿用同一組數值預設（600/100/480 等）作為比較基準。
        if chunk_ref.get(key) != value:
            count += 1

    retr_ref = _retrieval_defaults(retrieval_config.get("strategy", "dense"))
    for key, value in retrieval_config.items():
        if key in skip:
            continue
        if retr_ref.get(key) != value:
            count += 1
    return count


def complexity_vector(
    chunk_config: dict[str, Any], retrieval_config: dict[str, Any]
) -> tuple[int, int, int, int]:
    """回傳可依序比較的複雜度分量（越小越簡單）。"""
    retr_rank = _RETRIEVAL_STRATEGY_RANK.get(retrieval_config.get("strategy", "dense"), 99)
    chunk_rank = _CHUNK_STRATEGY_RANK.get(chunk_config.get("strategy", "current"), 99)

    uses_prefix = bool(chunk_config.get("prepend_section_title"))
    uses_context = bool(
        chunk_config.get("context_before_paragraphs")
        or chunk_config.get("context_after_paragraphs")
    )
    prefix_context_rank = 1 if (uses_prefix or uses_context) else 0

    return (
        retr_rank,
        chunk_rank,
        prefix_context_rank,
        _non_default_count(chunk_config, retrieval_config),
    )


def describe(
    chunk_config: dict[str, Any], retrieval_config: dict[str, Any]
) -> dict[str, Any]:
    """報告用的可解釋複雜度說明。"""
    vec = complexity_vector(chunk_config, retrieval_config)
    return {
        "complexity_version": COMPLEXITY_VERSION,
        "vector": list(vec),
        "components": {
            "retrieval_strategy_rank": vec[0],
            "chunk_strategy_rank": vec[1],
            "prefix_or_context_rank": vec[2],
            "non_default_param_count": vec[3],
        },
        "note": (
            "complexity 僅在所有 retrieval 品質指標完全相同時作為 tie-break，"
            "不得蓋過品質指標（§十八）。"
        ),
    }
