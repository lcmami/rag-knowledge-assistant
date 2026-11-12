"""Grid search candidate generator（v1.2 Optimizer 第一版）。

從 baseline :class:`RetrievalConfig` 與 search space 產生所有
``top_k`` × ``score_threshold`` 組合的候選。

- 不修改 baseline（``RetrievalConfig`` 為 frozen dataclass,自動保證）。
- 自動過濾與 baseline 完全相同的 candidate。
- 保留 baseline 的其他欄位（strategy / candidate_multiplier / candidate_cap /
  identifier_* / rerank_* / schema_version）。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from evaluation.v11.retrieval.config import RetrievalConfig


class GridCandidateGenerator:
    """Grid search candidate generator。

    ``search_space`` 格式::

        {
            "top_k": [3, 5, 7, 10],
            "score_threshold": [0.3, 0.5, 0.7],
        }

    對每個 ``(top_k, score_threshold)`` 組合產生一個 ``RetrievalConfig``。
    若 search space 缺少某維度,則該維度只取 baseline 值（不會產生額外 candidate）。
    """

    def generate(
        self,
        baseline: RetrievalConfig,
        search_space: dict,
    ) -> Iterable[RetrievalConfig]:
        top_k_values = search_space.get("top_k", [baseline.top_k])
        threshold_values = search_space.get("score_threshold", [baseline.score_threshold])

        for top_k in top_k_values:
            for score_threshold in threshold_values:
                candidate = replace(
                    baseline,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )

                # 過濾與 baseline 完全相同的 candidate。
                if candidate == baseline:
                    continue

                yield candidate
