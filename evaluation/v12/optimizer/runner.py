"""Grid candidate runner（v1.2 Optimizer 第一版）。

給定一個**已建立完成**的 baseline index（同一個 Qdrant collection）,
依序執行 :class:`GridCandidateGenerator` 產生的 ``RetrievalConfig``,
對每個 candidate 在既有 index 上跑相同 tune evaluation,
記錄 MRR 等既有 metric（直接重用 :func:`evaluation.v11.retrieval_metrics.evaluate_trial`）。

設計約束：
- 不重新 chunk / embedding / 建 collection（重用既有 index 與 index_universe）。
- 不修改 Agent / evaluator / trial runner。
- 不新增 metric（沿用 ``evaluate_trial`` 既有輸出）。
- 不執行完整 SciFact benchmark（呼叫端決定 dataset 範圍）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from evaluation.v11.case_schema import EvaluationCaseV11
from evaluation.v11.retrieval_metrics import (
    CaseRetrieval,
    IndexChunk,
    evaluate_trial,
)
from evaluation.v11.run_models import TrialConfig, TrialMetric
from evaluation.v11.trial_runner import (
    CANONICAL_TEXT_VERSION,
    IDENTIFIER_NORMALIZATION_VERSION,
    TrialServiceProvider,
)
from evaluation.v12.optimizer.grid import GridCandidateGenerator


@dataclass
class CandidateOutcome:
    """單一 candidate 的 evaluation 結果。"""

    candidate_hash: str
    retrieval_config: dict
    metrics: tuple[TrialMetric, ...] = field(default_factory=tuple)


def run_grid_candidates(
    *,
    baseline_trial: TrialConfig,
    collection_name: str,
    provider: TrialServiceProvider,
    cases: list[EvaluationCaseV11],
    index_universe: list[IndexChunk],
    search_space: dict,
    candidate_generator: Optional[GridCandidateGenerator] = None,
) -> list[CandidateOutcome]:
    """對 GridCandidateGenerator 產生的每個 candidate,在既有 index 上跑 evaluation。

    Parameters
    ----------
    baseline_trial:
        Baseline ``TrialConfig``。其 ``chunk_config`` / ``collection_options`` /
        ``schema_version`` 會被所有 candidate 重用;只有 ``retrieval`` 會被替換。
    collection_name:
        已建立完成的 Qdrant collection 名稱（**不重建**）。
    provider:
        :class:`TrialServiceProvider`（真實或 fake）。
    cases:
        Tune dataset cases。
    index_universe:
        該 collection 的全部 chunk（``evaluate_trial`` 算 NDCG IDCG 用;
        所有 candidate 共用同一份）。
    search_space:
        傳給 ``candidate_generator.generate`` 的搜尋空間。
    candidate_generator:
        可注入的 candidate generator;預設 :class:`GridCandidateGenerator`。
    """
    generator = candidate_generator or GridCandidateGenerator()
    outcomes: list[CandidateOutcome] = []

    for candidate in generator.generate(
        baseline_trial.retrieval_config_obj(), search_space
    ):
        # 構造 candidate TrialConfig：重用 baseline 的 chunk_config / collection_options,
        # 只換 retrieval（query-time config）。chunk / embedding / collection 全部沿用。
        candidate_trial = replace(
            baseline_trial,
            trial_name=f"grid-{candidate.top_k}-{candidate.score_threshold}",
            retrieval=candidate.to_canonical_dict(),
        )

        # 對每個 case 在既有 collection 上做 retrieval（不重建 index、不重新 embedding）。
        case_retrievals: list[CaseRetrieval] = []
        for case in cases:
            ro = provider.retrieve(
                collection_name,
                candidate_trial,
                case.question,
                candidate_trial.top_k,
            )
            case_retrievals.append(
                CaseRetrieval(
                    case=case,
                    results=ro.results,
                    latency_ms=ro.retrieval_time_ms,
                    rerank_strategy=ro.meta.get("rerank_strategy"),
                    rerank_latency_ms=ro.meta.get("rerank_latency_ms"),
                    rerank_cache_hits=ro.meta.get("rerank_cache_hits"),
                    rerank_cache_misses=ro.meta.get("rerank_cache_misses"),
                    rerank_window_count=ro.meta.get("rerank_window_count"),
                    rerank_tail_count=ro.meta.get("rerank_tail_count"),
                    admitted_count=ro.meta.get("admitted_count"),
                )
            )

        # 重用既有 evaluate_trial（MRR / NDCG / hit@k 等 metric 全部沿用,不新增 metric）。
        evaluation = evaluate_trial(
            case_retrievals,
            index_universe=index_universe,
        )

        outcomes.append(
            CandidateOutcome(
                candidate_hash=candidate_trial.compute_hash(
                    canonical_text_version=CANONICAL_TEXT_VERSION,
                    identifier_normalization_version=IDENTIFIER_NORMALIZATION_VERSION,
                ),
                retrieval_config=candidate.to_canonical_dict(),
                metrics=tuple(evaluation.metrics),
            )
        )

    return outcomes
