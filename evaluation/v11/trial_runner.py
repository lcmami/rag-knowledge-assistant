"""Trial Runner（Slice 2 §九、§十、§十一）——CLI 與 import 共用的單一核心。

流程：Validate Run Config → 建立/讀取 Run → 計算 Hash 與環境資訊 → 逐一執行明確列出的
Trial（每個 Trial 獨立隔離 collection、固定 Corpus 建索引、跑 Dataset queries、
EvidenceMatcher 判 relevance、算 Metrics、持久化、清理 collection）→ 產生 Run Report。

隔離與安全：
- 每 Trial 專屬 collection ``optimizer_<run>_<trial>``，永不觸碰 ``factory_knowledge``。
- 一個 Trial 失敗不阻止其他 Trial（除非 ``fail_fast``）。
- 支援中斷後恢復；Completed 不重跑；Failed/Interrupted 需明確 retry 才重跑。
- 狀態轉換一律先持久化；Cleanup 失敗不得讓 Metrics 消失。
- 本 Slice 不自動選 Best Trial、不改任何正式設定、不呼叫 LLM。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from evaluation.v11 import hashing, report as report_mod
from evaluation.v11.case_schema import parse_dataset
from evaluation.v11.metric_config import DEFAULT_METRIC_CONFIG, MetricConfig
from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.retrieval_metrics import (
    CaseEvaluation,
    CaseRetrieval,
    IndexChunk,
    RankedResult,
    evaluate_trial,
)
from evaluation.v11.run_models import (
    CleanupStatus,
    EvaluationRun,
    RetrievalTrial,
    RunConfig,
    RunStatus,
    TrialCaseResult,
    TrialConfig,
    TrialMetric,
    TrialStatus,
)
from evaluation.v11.trial_collections import (
    is_optimizer_collection,
    trial_collection_name,
)

ROOT = hashing.repo_root()

# 版本常數（與正式索引服務一致；hash / 報告誠實記錄）。
CANONICAL_TEXT_VERSION = "v1"
LOADER_VERSION = "v1"
IDENTIFIER_NORMALIZATION_VERSION = "v1"


# --------------------------------------------------------------------------
# Service Provider 抽象（真實 = app 服務；測試 = fake）
# --------------------------------------------------------------------------
@dataclass
class IndexOutcome:
    point_count: int
    build_time_ms: int
    # Slice 3（§十八 / §二十三）：Chunking + Embedding Cache 統計（fake / 舊 provider 用預設值）。
    chunk_count: Optional[int] = None
    strategy: Optional[str] = None
    chunk_config_hash: Optional[str] = None
    # 版本化 stats blob：{strategy, chunk_config, fallback, section_prefix, context,
    # cache_stats, embedding_fingerprint, ...}（持久化於 retrieval_trials.chunk_cache_stats_json）。
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalOutcome:
    results: list[RankedResult]
    retrieval_time_ms: int
    # Slice 4（§十八.4）：identifier-aware 每題 strategy 明細（dense / fake 為空 → 不持久化）。
    meta: dict[str, Any] = field(default_factory=dict)


class TrialServiceProvider(Protocol):
    """Trial 執行所需的最小服務介面（可用 fake 替換以免載入 BGE / 真實 Qdrant）。"""

    def build_index(
        self, collection_name: str, trial_config: TrialConfig, files: list[tuple[str, str]]
    ) -> IndexOutcome: ...

    def index_manifest(self, collection_name: str) -> list[IndexChunk]:
        """回傳該 Trial Index 建立的全部 Chunk（Index Universe，供 trial_index_gold NDCG）。"""
        ...

    def retrieve(
        self, collection_name: str, trial_config: TrialConfig, question: str, top_k: int
    ) -> RetrievalOutcome: ...

    def collection_exists(self, collection_name: str) -> bool: ...

    def delete_collection(self, collection_name: str) -> None: ...


# --------------------------------------------------------------------------
# 預設時鐘 / id（可注入以利 deterministic 測試）
# --------------------------------------------------------------------------
def _default_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_id() -> str:
    return uuid.uuid4().hex


@dataclass
class RunResult:
    run: EvaluationRun
    report_json_path: Optional[str]
    report_markdown_path: Optional[str]
    remaining_collections: list[str] = field(default_factory=list)
    executed_trial_ids: list[str] = field(default_factory=list)
    skipped_trial_ids: list[str] = field(default_factory=list)


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass


def _sanitize_error(exc: Exception) -> str:
    """保存可理解錯誤，不含敏感環境資訊；截斷長度。"""
    return f"{type(exc).__name__}: {exc}"[:500]


# --------------------------------------------------------------------------
# 核心：執行一個 Run（新建或 resume 皆走此函式）
# --------------------------------------------------------------------------
def execute_run(
    run_config: RunConfig,
    repo: OptimizerRepository,
    service_provider: TrialServiceProvider,
    *,
    output_dir: str,
    base_dir: str = ROOT,
    run_id: Optional[str] = None,
    metric_config: MetricConfig = DEFAULT_METRIC_CONFIG,
    now_fn: Callable[[], str] = _default_now,
    id_factory: Callable[[], str] = _default_id,
    write_report: bool = True,
) -> RunResult:
    """執行（或 resume）一個 Run 的所有明確 Trial，回傳 :class:`RunResult`。"""
    _seed_everything(run_config.random_seed)

    # --- 計算 hash / 環境資訊 ---
    corpus_manifest = hashing.corpus_manifest(run_config.corpus_paths, base_dir)
    files = [
        (os.path.basename(rel), abs_path)
        for rel, abs_path in hashing.collect_corpus_files(run_config.corpus_paths, base_dir)
    ]
    dataset_abs = (
        run_config.dataset_path
        if os.path.isabs(run_config.dataset_path)
        else os.path.join(base_dir, run_config.dataset_path)
    )
    with open(dataset_abs, encoding="utf-8") as f:
        raw_dataset = json.load(f)
    dataset = parse_dataset(raw_dataset)
    git = hashing.git_state(base_dir)

    remaining_collections: list[str] = []
    executed: list[str] = []
    skipped: list[str] = []

    # --- 建立或讀取 Run ---
    run = repo.get_run(run_id) if run_id else None
    if run is None:
        run = EvaluationRun(
            run_id=run_id or id_factory(),
            status=RunStatus.RUNNING,
            run_name=run_config.run_name,
            dataset_version=dataset.dataset_version,
            dataset_hash=hashing.dataset_hash(raw_dataset),
            corpus_hash=corpus_manifest["corpus_hash"],
            code_commit=git["commit"],
            working_tree_dirty=git["working_tree_dirty"],
            canonical_text_version=CANONICAL_TEXT_VERSION,
            loader_version=LOADER_VERSION,
            identifier_normalization_version=IDENTIFIER_NORMALIZATION_VERSION,
            random_seed=run_config.random_seed,
            config_json=hashing.canonical_json(run_config.sanitized()),
            created_at=now_fn(),
            started_at=now_fn(),
        )
        repo.upsert_run(run)
        _create_trial_rows(run, run_config, repo, id_factory)
    else:
        run.status = RunStatus.RUNNING
        run.started_at = run.started_at or now_fn()
        repo.upsert_run(run)

    # --- resume：遺留 running trial 標記 interrupted + 嘗試清理其 collection ---
    _recover_interrupted_trials(run, repo, service_provider, remaining_collections, now_fn)

    # --- 逐一執行 Trial（依 ordinal）---
    tc_by_hash = {
        t.compute_hash(
            canonical_text_version=CANONICAL_TEXT_VERSION,
            identifier_normalization_version=IDENTIFIER_NORMALIZATION_VERSION,
        ): t
        for t in run_config.trials
    }
    for trial in repo.list_trials(run.run_id):
        decision = _should_run(trial.status, run_config)
        if not decision:
            skipped.append(trial.trial_id)
            continue
        tc = tc_by_hash.get(trial.config_hash)
        if tc is None:
            # config 與 DB trial 不符（不應發生）；跳過並記錄
            skipped.append(trial.trial_id)
            continue
        ok = _execute_trial(
            trial, tc, dataset, files, repo, service_provider,
            run_config, metric_config, remaining_collections, now_fn,
        )
        executed.append(trial.trial_id)
        if not ok and run_config.fail_fast:
            break

    # --- 收斂 Run 狀態 ---
    all_trials = repo.list_trials(run.run_id)
    failed = [t for t in all_trials if t.status == TrialStatus.FAILED]
    incomplete = [t for t in all_trials if t.status not in (TrialStatus.COMPLETED,)]
    if not incomplete:
        run.status = RunStatus.COMPLETED
        run.error = None
    else:
        run.status = RunStatus.FAILED
        parts = [f"{t.trial_name}:{t.status}" for t in incomplete]
        run.error = "未全部完成：" + ", ".join(parts)
    run.completed_at = now_fn()
    repo.upsert_run(run)

    # --- 報告 ---
    json_path = md_path = None
    if write_report:
        report = report_mod.build_report(
            repo, run, run_config, corpus_manifest,
            remaining_collections=remaining_collections,
            metric_config=metric_config,
        )
        json_path, md_path = report_mod.write_report(report, output_dir)
        run.report_json_path = json_path
        run.report_markdown_path = md_path
        repo.upsert_run(run)

    return RunResult(
        run=run,
        report_json_path=json_path,
        report_markdown_path=md_path,
        remaining_collections=remaining_collections,
        executed_trial_ids=executed,
        skipped_trial_ids=skipped,
    )


def _create_trial_rows(
    run: EvaluationRun,
    run_config: RunConfig,
    repo: OptimizerRepository,
    id_factory: Callable[[], str],
) -> None:
    """為新 Run 建立 pending Trial 記錄；重複 config_hash 不建立重複 Trial。"""
    seen_hashes: set[str] = set()
    for ordinal, tc in enumerate(run_config.trials):
        config_hash = tc.compute_hash(
            canonical_text_version=CANONICAL_TEXT_VERSION,
            identifier_normalization_version=IDENTIFIER_NORMALIZATION_VERSION,
        )
        if config_hash in seen_hashes:
            continue  # §九：Config Hash 重複不產生重複 Trial
        if repo.get_trial_by_config_hash(run.run_id, config_hash) is not None:
            continue
        seen_hashes.add(config_hash)
        trial = RetrievalTrial(
            trial_id=id_factory(),
            run_id=run.run_id,
            ordinal=ordinal,
            trial_name=tc.trial_name,
            status=TrialStatus.PENDING,
            config_json=tc.config_json(),
            config_hash=config_hash,
            cleanup_status=CleanupStatus.NOT_CREATED,
            attempt_count=0,
        )
        repo.upsert_trial(trial)


def _should_run(status: str, run_config: RunConfig) -> bool:
    if status == TrialStatus.COMPLETED:
        return False
    if status == TrialStatus.PENDING:
        return True
    if status == TrialStatus.FAILED:
        return run_config.retry_failed
    if status == TrialStatus.INTERRUPTED:
        return run_config.retry_interrupted
    if status == TrialStatus.RUNNING:
        # 已由 recover 轉為 interrupted；保守處理
        return run_config.retry_interrupted
    return False


def _recover_interrupted_trials(
    run: EvaluationRun,
    repo: OptimizerRepository,
    provider: TrialServiceProvider,
    remaining_collections: list[str],
    now_fn: Callable[[], str],
) -> None:
    for trial in repo.list_trials(run.run_id):
        if trial.status != TrialStatus.RUNNING:
            continue
        trial.status = TrialStatus.INTERRUPTED
        # 嘗試清理其暫時 collection（記錄結果）
        if trial.collection_name and is_optimizer_collection(trial.collection_name):
            try:
                if provider.collection_exists(trial.collection_name):
                    provider.delete_collection(trial.collection_name)
                trial.cleanup_status = CleanupStatus.CLEANED
            except Exception:  # noqa: BLE001
                trial.cleanup_status = CleanupStatus.CLEANUP_FAILED
                remaining_collections.append(trial.collection_name)
        repo.upsert_trial(trial)


def _execute_trial(
    trial: RetrievalTrial,
    tc: TrialConfig,
    dataset,
    files: list[tuple[str, str]],
    repo: OptimizerRepository,
    provider: TrialServiceProvider,
    run_config: RunConfig,
    metric_config: MetricConfig,
    remaining_collections: list[str],
    now_fn: Callable[[], str],
) -> bool:
    """執行單一 Trial；回傳是否成功。狀態轉換先持久化，finally 一律嘗試清理。"""
    trial.attempt_count += 1
    trial.status = TrialStatus.RUNNING
    trial.started_at = now_fn()
    trial.error = None
    trial.collection_name = trial.collection_name or trial_collection_name(
        trial.run_id, trial.trial_id
    )
    repo.upsert_trial(trial)  # §九：狀態轉換先持久化

    collection = trial.collection_name
    try:
        idx = provider.build_index(collection, tc, files)
        trial.cleanup_status = CleanupStatus.ACTIVE
        trial.point_count = idx.point_count
        trial.index_build_time_ms = idx.build_time_ms
        # Slice 3：持久化 Chunking + Cache 統計（在 metrics / cleanup 之前）。
        trial.chunk_count = idx.chunk_count if idx.chunk_count is not None else idx.point_count
        trial.strategy = idx.strategy
        trial.chunk_config_hash = idx.chunk_config_hash
        trial.chunk_cache_stats_json = (
            hashing.canonical_json(idx.stats) if idx.stats else None
        )
        repo.upsert_trial(trial)

        # Index Universe（該 Trial 全部 Chunk）供 trial_index_gold NDCG 的 IDCG；
        # 重用已建索引，不為每題重新檢索或重新 scroll 整個 collection。
        index_universe = provider.index_manifest(collection)

        case_retrievals: list[CaseRetrieval] = []
        retrieval_meta_by_case: dict[str, dict] = {}
        total_retrieval = 0
        for case in dataset.cases:
            ro = provider.retrieve(collection, tc, case.question, tc.top_k)
            total_retrieval += ro.retrieval_time_ms
            case_retrievals.append(
                CaseRetrieval(case=case, results=ro.results, latency_ms=ro.retrieval_time_ms)
            )
            if ro.meta:
                retrieval_meta_by_case[case.id] = ro.meta

        evaluation = evaluate_trial(
            case_retrievals,
            index_universe=index_universe,
            metric_config=metric_config,
            point_count=idx.point_count,
            index_build_time_ms=idx.build_time_ms,
        )

        # 持久化 case results + metrics（在 cleanup 之前，確保清理失敗仍保有 metrics）
        repo.save_trial_case_results(
            trial.trial_id,
            [
                _case_result_row(trial.trial_id, ce, retrieval_meta_by_case.get(ce.case_id))
                for ce in evaluation.case_evaluations
            ],
        )
        for m in evaluation.metrics:
            m.trial_id = trial.trial_id
        repo.save_trial_metrics(trial.trial_id, evaluation.metrics)

        trial.retrieval_time_ms = total_retrieval
        trial.status = TrialStatus.COMPLETED
        trial.completed_at = now_fn()
        repo.upsert_trial(trial)
        success = True
    except Exception as exc:  # noqa: BLE001 - 單一 Trial 失敗須隔離
        trial.status = TrialStatus.FAILED
        trial.error = _sanitize_error(exc)
        trial.completed_at = now_fn()
        repo.upsert_trial(trial)
        success = False
    finally:
        _cleanup_trial_collection(
            trial, provider, keep=run_config.keep_trial_collections,
            remaining=remaining_collections,
        )
        repo.upsert_trial(trial)

    return success


def _cleanup_trial_collection(
    trial: RetrievalTrial,
    provider: TrialServiceProvider,
    *,
    keep: bool,
    remaining: list[str],
) -> None:
    collection = trial.collection_name
    if not collection or not is_optimizer_collection(collection):
        return
    try:
        exists = provider.collection_exists(collection)
    except Exception:  # noqa: BLE001
        exists = False

    if keep:
        if exists:
            remaining.append(collection)
            trial.cleanup_status = CleanupStatus.ACTIVE
        elif trial.cleanup_status == CleanupStatus.ACTIVE:
            trial.cleanup_status = CleanupStatus.CLEANED
        return

    if not exists:
        if trial.cleanup_status == CleanupStatus.ACTIVE:
            trial.cleanup_status = CleanupStatus.CLEANED
        return
    try:
        provider.delete_collection(collection)
        trial.cleanup_status = CleanupStatus.CLEANED
    except Exception as exc:  # noqa: BLE001 - cleanup 失敗不得讓 metrics 消失
        trial.cleanup_status = CleanupStatus.CLEANUP_FAILED
        remaining.append(collection)
        note = f"cleanup_failed: {exc}"[:200]
        trial.error = (trial.error + " | " + note) if trial.error else note


def _case_result_row(
    trial_id: str, ce: CaseEvaluation, retrieval_meta: Optional[dict] = None
) -> TrialCaseResult:
    return TrialCaseResult(
        trial_id=trial_id,
        case_id=ce.case_id,
        case_status=ce.case_status,
        query_type=ce.query_type,
        answerable=ce.answerable,
        latency_ms=ce.latency_ms,
        first_relevant_rank=ce.first_relevant_rank,
        top_results_json=hashing.canonical_json(ce.top_results_payload()),
        evidence_match_json=hashing.canonical_json(ce.evidence_match_payload()),
        retrieval_meta_json=(
            hashing.canonical_json(retrieval_meta) if retrieval_meta else None
        ),
    )


# --------------------------------------------------------------------------
# Cleanup subcommand 核心：刪除一個 Run 的所有 Trial collection
# --------------------------------------------------------------------------
def cleanup_run(
    run_id: str,
    repo: OptimizerRepository,
    provider: TrialServiceProvider,
) -> dict[str, str]:
    """刪除指定 Run 的所有 Trial collection（防誤刪：只刪 optimizer_ 前綴）。"""
    results: dict[str, str] = {}
    for trial in repo.list_trials(run_id):
        name = trial.collection_name
        if not name:
            continue
        if not is_optimizer_collection(name):
            results[name] = "skipped_not_optimizer"
            continue
        try:
            if provider.collection_exists(name):
                provider.delete_collection(name)
                results[name] = "cleaned"
            else:
                results[name] = "absent"
            trial.cleanup_status = CleanupStatus.CLEANED
        except Exception as exc:  # noqa: BLE001
            results[name] = f"cleanup_failed: {exc}"
            trial.cleanup_status = CleanupStatus.CLEANUP_FAILED
        repo.upsert_trial(trial)
    return results
