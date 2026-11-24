"""RealTrialExecutor（Slice 6）：以 v1.1 ``execute_run`` 在 split dataset 上跑單一 config。

包住既有 :func:`evaluation.v11.trial_runner.execute_run` + :class:`RealTrialServiceProvider`
+ :class:`OptimizerRepository`，實作 Orchestrator 的 :class:`TrialExecutor` 介面：

- ``run``：在指定 split dataset 建一個單 Trial Run 並執行；讀回 trial + metrics。tune split
  另外萃取受控 context inputs（segments / per-case 檢索實況）供 LLM。
- ``retrieve``：在既有 collection 上做 probe 檢索（Slice 4）。
- ``cleanup``：只刪 ``optimizer_`` 前綴 collection；``factory_knowledge`` 永不受影響。

集合生命週期由 Orchestrator 掌控：``build_run_config`` 設 ``keep_trial_collections=True``，
故 execute_run 不會在回傳前刪 collection；probe / 下一輪比較後由 Orchestrator 呼叫 ``cleanup``。
"""
from __future__ import annotations

import os
from typing import Any, Optional

from evaluation.v11 import hashing
from evaluation.v11.case_schema import load_dataset
from evaluation.v11.run_models import TrialConfig, TrialStatus
from evaluation.v11.trial_collections import is_optimizer_collection
from evaluation.v11.trial_runner import execute_run
from evaluation.v12.agent.context import SegmentExcerpt, TuneFailureView
from evaluation.v12.agent.orchestrator import TrialOutcome
from evaluation.v12.agent.run_config import AgentRunConfig
from evaluation.v12.agent.diagnosis import classify_failure

ROOT = hashing.repo_root()

_MAX_SEGMENTS = 4
_MAX_TUNE_CASES = 4
#_PROBE_TOP_K = 3
_EXCERPT_CAP = 800


class RealTrialExecutor:
    def __init__(
        self,
        run_config: AgentRunConfig,
        provider,
        repo,
        *,
        output_dir: str,
        base_dir: str = ROOT,
    ):
        self._cfg = run_config
        self._provider = provider
        self._repo = repo
        self._output_dir = output_dir
        self._base_dir = base_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def run(
        self, *, trial: TrialConfig, split: str, dataset_path: str, run_label: str, config_hash: str
    ) -> TrialOutcome:
        rc = self._cfg.build_run_config(dataset_path, trial, run_label)
        result = execute_run(
            rc,
            self._repo,
            self._provider,
            output_dir=self._output_dir,
            base_dir=self._base_dir,
            write_report=False,
        )
        trials = self._repo.list_trials(result.run.run_id)
        if not trials:
            return TrialOutcome(split=split, config_hash=config_hash, status="failed", error="no_trial")
        t = trials[0]
        metrics = tuple(self._repo.get_trial_metrics(t.trial_id))
        status = "completed" if t.status == TrialStatus.COMPLETED else "failed"
        outcome = TrialOutcome(
            split=split,
            config_hash=config_hash,
            status=status,
            trial_id=t.trial_id,
            collection_name=t.collection_name,
            metrics=metrics,
            error=t.error,
        )
        if split == "tune" and outcome.ok and t.collection_name:
            outcome.segments, outcome.tune_failures = self._context_inputs(
                t.collection_name,
                trial,
                dataset_path,
                t.trial_id,
            )
        return outcome

    def retrieve(
        self, *, collection_name: str, trial: TrialConfig, question: str, top_k: int
    ) -> list[dict[str, Any]]:
        ro = self._provider.retrieve(collection_name, trial, question, top_k)
        return [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_index,
                "section": r.section,
                "chunk": (r.text or "")[:_EXCERPT_CAP],
                "identifiers": list(r.identifiers or []),
                "conflicting_identifiers": list(getattr(r, "conflicting_identifiers", []) or []),
            }
            for r in ro.results
        ]

    def cleanup(self, collection_name: Optional[str]) -> None:
        if not collection_name or not is_optimizer_collection(collection_name):
            return
        try:
            if self._provider.collection_exists(collection_name):
                self._provider.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 - 清理失敗不得中斷流程
            pass

    # ------------------------------------------------------------------
    def _context_inputs(
        self,
        collection_name: str,
        trial: TrialConfig,
        dataset_path: str,
        trial_id: str,
    ) -> tuple[tuple[SegmentExcerpt, ...], tuple[TuneFailureView, ...]]:
        # segments：由該 Trial index 取前 K 個 chunk 作為受控原文片段。
        segments: list[SegmentExcerpt] = []
        try:
            manifest = self._provider.index_manifest(collection_name)
        except Exception:  # noqa: BLE001
            manifest = []
        for i, chunk in enumerate(manifest[:_MAX_SEGMENTS]):
            segments.append(
                SegmentExcerpt(
                    segment_id=f"chunk-{chunk.chunk_index if chunk.chunk_index is not None else i}",
                    section=chunk.section,
                    text=chunk.text or "",
                )
            )

        # tune failure context：
        # 正式 failure/rank 直接使用 execute_run 已持久化的 per-case result；
        # 只對少量 failure 重新 retrieve，以取得給 LLM 看的原文 excerpt。
        failures: list[TuneFailureView] = []

        abs_path = (
            dataset_path
            if os.path.isabs(dataset_path)
            else os.path.join(self._base_dir, dataset_path)
        )

        try:
            dataset = load_dataset(abs_path)
        except Exception:  # noqa: BLE001
            dataset = None

        if dataset is not None:
            cases_by_id = {case.id: case for case in dataset.cases}

            try:
                case_results = self._repo.get_trial_case_results(trial_id)
            except Exception:  # noqa: BLE001
                case_results = []

            for row in case_results:
                case = cases_by_id.get(row.case_id)
                if case is None:
                    continue

                # 明確成功的 case 不需要浪費一次 retrieval。
                if (
                    row.case_status == "unanswerable_ok"
                    or (
                        row.answerable
                        and row.first_relevant_rank == 1
                    )
                ):
                    continue

                # 只為真正可能進 context 的 failure 重取 excerpt。
                # 必須使用正式 trial 的 top_k，不能再硬寫 _PROBE_TOP_K=3。
                try:
                    results = self.retrieve(
                        collection_name=collection_name,
                        trial=trial,
                        question=case.question,
                        top_k=trial.top_k,
                    )
                except Exception:  # noqa: BLE001
                    results = []

                conflicting = (
                    results[0]["conflicting_identifiers"]
                    if results
                    else []
                )

                failure_type = classify_failure(
                    case_status=row.case_status,
                    answerable=row.answerable,
                    first_relevant_rank=row.first_relevant_rank,
                    conflicting_identifiers=conflicting,
                    retrieved_any=bool(results),
                )

                if failure_type is None:
                    continue

                anchor = ""
                if (
                    case.expected_evidence
                    and case.expected_evidence[0].anchor_text
                ):
                    anchor = case.expected_evidence[0].anchor_text[0]

                failures.append(
                    TuneFailureView(
                        case_id=case.id,
                        question=case.question,
                        query_type=row.query_type or case.query_type,
                        case_status=row.case_status,
                        first_relevant_rank=row.first_relevant_rank,
                        failure_type=failure_type,
                        expected_evidence_excerpt=anchor,
                        actual_top_excerpts=tuple(
                            r["chunk"] for r in results
                        ),
                        identifier_detail={
                            "conflicting_identifiers": conflicting
                        },
                    )
                )

                if len(failures) >= _MAX_TUNE_CASES:
                    break

        return tuple(segments), tuple(failures)
