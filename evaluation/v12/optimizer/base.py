"""v1.2 Optimizer 模組（最小版本）。

第一版只做 candidate generation,不執行 evaluation。
重用既有 :class:`evaluation.v11.retrieval.config.RetrievalConfig` 作為候選型別,
不另外發明 schema。
"""
from __future__ import annotations

from typing import Iterable, Protocol

from evaluation.v11.retrieval.config import RetrievalConfig


class CandidateGenerator(Protocol):
    """最小 candidate generator 介面。

    接收 baseline config 與 search space,產生候選 ``RetrievalConfig`` 集合。
    實作不得修改傳入的 baseline object。
    """

    def generate(
        self,
        baseline: RetrievalConfig,
        search_space: dict,
    ) -> Iterable[RetrievalConfig]:
        ...
