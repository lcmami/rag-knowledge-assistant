"""v1.2 Agent 測試用 fakes（Slice 3/4 共用）。

- :class:`FakeTrialExecutor`：以可注入的 ``metric_fn`` / ``fail_predicate`` 產生 scripted
  per-split metrics 與 trial 失敗，並記錄 cleanup 呼叫（不碰 Qdrant）。
- ``make_proposal`` / helpers：組出可被 parser/validator 接受的 proposal payload。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from evaluation.v11.run_models import TrialConfig, TrialMetric
from evaluation.v12.agent.context import SegmentExcerpt, TuneFailureView
from evaluation.v12.agent.orchestrator import TrialOutcome

MetricFn = Callable[[TrialConfig, str], dict[str, float]]
FailPredicate = Callable[[TrialConfig, str], bool]


def default_metric_fn(trial: TrialConfig, split: str) -> dict[str, float]:
    """單調於 top_k 的示範 metric：top_k 越大，exact/mrr 越高、unanswerable FP 越低。"""
    tk = trial.top_k
    return {
        "exact_code_top1_accuracy": min(0.5 + 0.05 * (tk - 5), 1.0),
        "unanswerable_false_positive_rate": max(0.4 - 0.02 * (tk - 5), 0.0),
        "mrr": min(0.6 + 0.03 * (tk - 5), 1.0),
    }


class FakeTrialExecutor:
    def __init__(
        self,
        metric_fn: MetricFn = default_metric_fn,
        *,
        fail_predicate: Optional[FailPredicate] = None,
        segments: tuple[SegmentExcerpt, ...] = (),
        tune_failures: tuple[TuneFailureView, ...] = (),
    ):
        self._metric_fn = metric_fn
        self._fail = fail_predicate or (lambda t, s: False)
        self._segments = segments
        self._tune_failures = tune_failures
        self.cleaned: list[str] = []
        self.retrieved: list[tuple[str, str]] = []
        self._counter = 0

    def run(
        self, *, trial: TrialConfig, split: str, dataset_path: str, run_label: str, config_hash: str
    ) -> TrialOutcome:
        self._counter += 1
        collection = f"optimizer_fake_{split}_{self._counter}"
        if self._fail(trial, split):
            return TrialOutcome(
                split=split,
                config_hash=config_hash,
                status="failed",
                collection_name=collection,
                error="fake_trial_failure",
            )
        metrics = tuple(
            TrialMetric(
                trial_id="",
                metric_name=name,
                scope="aggregate",
                query_type=None,
                value=value,
                computable=True,
            )
            for name, value in self._metric_fn(trial, split).items()
        )
        outcome = TrialOutcome(
            split=split,
            config_hash=config_hash,
            status="completed",
            trial_id=f"trial-{self._counter}",
            collection_name=collection,
            metrics=metrics,
        )
        if split == "tune":
            outcome.tune_failures = self._tune_failures
            outcome.segments = self._segments
        return outcome

    def retrieve(
        self, *, collection_name: str, trial: TrialConfig, question: str, top_k: int
    ) -> list[dict[str, Any]]:
        self.retrieved.append((collection_name, question))
        return []

    def cleanup(self, collection_name: Optional[str]) -> None:
        if collection_name:
            self.cleaned.append(collection_name)


# --------------------------------------------------------------------------
def make_proposal(
    *,
    top_k: int,
    chunk_strategy: str = "current",
    stop: bool = False,
    with_probe: bool = False,
    segment_id: str = "seg-1",
    anchor: str = "D900",
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    if with_probe:
        probes.append(
            {
                "question": "probe: retrieved?",
                "query_type": "exact_code",
                "source_segment_ids": [segment_id],
                "anchor_text_suggestions": [anchor],
                "rationale": "probe",
            }
        )
    return {
        "diagnoses": [],
        "proposed_config": {
            "chunk_config": {"strategy": chunk_strategy},
            "retrieval_config": {
                "strategy": "dense",
                "top_k": top_k,
                "score_threshold": 0.55,
                "candidate_multiplier": 3,
                "candidate_cap": 50,
            },
        },
        "probe_questions": probes,
        "expected_effects": [],
        "confidence": 0.5,
        "stop_recommendation": stop,
    }


def sample_segment() -> SegmentExcerpt:
    return SegmentExcerpt(
        segment_id="seg-1",
        section="Alarms",
        text="The alarm code D900 indicates an overheat fault in the pump.",
    )


def sample_tune_failure() -> TuneFailureView:
    return TuneFailureView(
        case_id="c1",
        question="What is D900?",
        query_type="exact_code",
        case_status="miss",
        first_relevant_rank=None,
        expected_evidence_excerpt="alarm code D900",
    )
