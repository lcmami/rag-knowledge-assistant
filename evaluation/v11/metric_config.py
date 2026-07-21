"""版本化 Metric Config（Slice 2 §十二 / §十三）。

所有 metric 的可調參數集中於此、且**版本化**，不得散落硬編碼於計算邏輯：

- graded NDCG gain 對映（Level 2 / 1 / 0）。
- offset coverage 門檻（EvidenceMatcher 判 Level 2）。
- near-duplicate 判定（character n-gram Jaccard 門檻 + n）。
- 主要 K 值。

規則變更 → bump :data:`METRIC_CONFIG_VERSION`，報告據此標示可比較性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

METRIC_CONFIG_VERSION = "v1"


@dataclass(frozen=True)
class MetricConfig:
    version: str = METRIC_CONFIG_VERSION

    # 主要 K 值（Recall@K / Precision@K / Coverage@K / duplicate 皆用 top_k）。
    top_k: int = 5

    # graded NDCG gain：Level 2 → 3、Level 1 → 1、Level 0 → 0（§12.3）。
    ndcg_gains: dict[int, int] = field(
        default_factory=lambda: {2: 3, 1: 1, 0: 0}
    )
    # NDCG@K 的 K。
    ndcg_k: int = 5
    # NDCG 變體：trial_index_gold —— IDCG 由「該 Trial Index 全部 Chunk 的 gain 降序」
    # 取前 K 計算，衡量相對於 Trial Index 全部可用相關結果的 graded ranking（含漏檢懲罰）。
    # 舊的 retrieved-only 定義（僅衡量取回集合內部排序）已移除，不得作為 NDCG。
    ndcg_variant: str = "trial_index_gold_v1"
    # ideal universe 來源：trial_index（該 Trial 實際建立的 Index，不含正式 collection）。
    ndcg_ideal_universe: str = "trial_index"

    # EvidenceMatcher 的 offset 覆蓋門檻（判定 Level 2）。
    offset_coverage_threshold: float = 0.5

    # near-duplicate：normalized chunk text 的 character n-gram Jaccard >= 門檻（§12.9）。
    near_duplicate_ngram: int = 3
    near_duplicate_jaccard_threshold: float = 0.9

    def gain(self, relevance_level: int) -> int:
        return self.ndcg_gains.get(relevance_level, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "top_k": self.top_k,
            "ndcg_gains": {str(k): v for k, v in self.ndcg_gains.items()},
            "gain_mapping": {str(k): v for k, v in self.ndcg_gains.items()},
            "ndcg_k": self.ndcg_k,
            "ndcg_variant": self.ndcg_variant,
            "ndcg_ideal_universe": self.ndcg_ideal_universe,
            "offset_coverage_threshold": self.offset_coverage_threshold,
            "near_duplicate_ngram": self.near_duplicate_ngram,
            "near_duplicate_jaccard_threshold": self.near_duplicate_jaccard_threshold,
        }


DEFAULT_METRIC_CONFIG = MetricConfig()
