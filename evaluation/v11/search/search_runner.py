"""Optimizer Search 流程編排（Slice 5 §十二、§十六、§十七、§十八、§二十一）。

    baseline → tune → validation → hard constraints → pareto → recommendation
    → **LOCK** → test（僅 baseline + recommended 兩個 config）

設計要點：**完全重用 Slice 2 的 :func:`trial_runner.execute_run`**，不修改 Slice 2–4 任何程式。
每個 split 各開一個 ``evaluation_run``，並餵給它一份「只含該 split case」的 dataset 檔 ——
隔離因此是物理性的：runner 根本拿不到其他 split 的題目，而不是靠呼叫端自律。
同時免費繼承既有的 resume / retry / fail-isolation / metrics / cleanup。

不變式：

- Test 階段在 ``recommendation_locked`` 之前**絕不**被建立或執行（§12.4 / §13.3），
  並由 :class:`SearchRepository` 在 persistence 層再守一次。
- Lock 之後 recommendation **不因 test 結果改變**（§12.5）；test 只能改 final status。
- 沒有 feasible candidate → 明確失敗，不硬選（§18.5 / §18.6）。
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from evaluation.v11 import hashing, trial_runner
from evaluation.v11.metric_config import DEFAULT_METRIC_CONFIG, MetricConfig
from evaluation.v11.run_models import RunConfig, TrialConfig, TrialStatus
from evaluation.v11.search import pareto as pareto_mod
from evaluation.v11.search import recommendation as rec_mod
from evaluation.v11.search import split_dataset as split_mod
from evaluation.v11.search.constraints import (
    ConstraintEvaluation,
    MetricLookup,
    evaluate_constraints,
)
from evaluation.v11.search.search_config import SearchConfig
from evaluation.v11.search.search_models import (
    CandidateStageStatus,
    OptimizerSearch,
    RecommendationLockError,
    SearchCandidate,
    SearchResumeError,
    SearchStatus,
)
from evaluation.v11.search.search_repository import SearchRepository
from evaluation.v11.search.search_space import (
    GenerationResult,
    candidate_manifest_signature,
    generate_candidates,
)

BASELINE_TRIAL_NAME = "baseline"


def _default_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_id() -> str:
    return uuid.uuid4().hex


def _dataset_ref(dataset_path: str, base_dir: str) -> str:
    """split dataset 檔相對於 repo 的參照。

    ``output_dir`` 可能在 repo 之外（例如測試用的暫存目錄，甚至不同磁碟機），
    此時 :func:`os.path.relpath` 會拋 ``ValueError`` —— 退回絕對路徑。
    ``os.path.join(base_dir, <abs>)`` 仍會得到正確結果，故 runner 兩種形式都能解析。
    """
    try:
        return os.path.relpath(dataset_path, base_dir).replace("\\", "/")
    except ValueError:
        return dataset_path


@dataclass
class StageOutcome:
    """單一 split 的執行結果。"""

    split: str
    run_id: Optional[str] = None
    executed: bool = False
    trial_by_hash: dict[str, Any] = field(default_factory=dict)
    metrics_by_hash: dict[str, MetricLookup] = field(default_factory=dict)
    remaining_collections: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    search: OptimizerSearch
    candidates: list[SearchCandidate]
    generation: GenerationResult
    split_manifest: split_mod.SplitManifest
    coverage: Optional[split_mod.ValidationCoverageVerdict] = None
    tune: Optional[StageOutcome] = None
    validation: Optional[StageOutcome] = None
    test: Optional[StageOutcome] = None
    constraint_evals: dict[str, ConstraintEvaluation] = field(default_factory=dict)
    pareto: Optional[pareto_mod.ParetoResult] = None
    recommendation: Optional[rec_mod.RecommendationResult] = None
    tune_ranking: list[rec_mod.RankedCandidate] = field(default_factory=list)
    report_json_path: Optional[str] = None
    report_markdown_path: Optional[str] = None
    remaining_collections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
def _run_split(
    *,
    search: OptimizerSearch,
    split: str,
    dataset_path: str,
    trials: Sequence[TrialConfig],
    search_config: SearchConfig,
    repo: SearchRepository,
    provider,
    output_dir: str,
    base_dir: str,
    run_id: Optional[str],
    metric_config: MetricConfig,
    now_fn: Callable[[], str],
    id_factory: Callable[[], str],
) -> StageOutcome:
    """在單一 split 上跑一個 evaluation_run（完全委派 Slice 2 runner）。"""
    outcome = StageOutcome(split=split)
    if not trials:
        outcome.errors.append(f"split={split} 沒有任何 trial 可執行。")
        return outcome

    run_config = RunConfig(
        run_name=f"{search.search_name}-{split}",
        dataset_path=_dataset_ref(dataset_path, base_dir),
        corpus_paths=search_config.corpus_paths,
        trials=tuple(trials),
        random_seed=search_config.random_seed,
        fail_fast=False,
        retry_failed=False,
        retry_interrupted=True,
        keep_trial_collections=search_config.keep_trial_collections,
    )

    result = trial_runner.execute_run(
        run_config,
        repo,
        provider,
        output_dir=os.path.join(output_dir, "runs", split),
        base_dir=base_dir,
        run_id=run_id,
        metric_config=metric_config,
        now_fn=now_fn,
        id_factory=id_factory,
        write_report=True,
    )
    outcome.run_id = result.run.run_id
    outcome.executed = True
    outcome.remaining_collections = list(result.remaining_collections)

    for trial in repo.list_trials(result.run.run_id):
        outcome.trial_by_hash[trial.config_hash] = trial
        if trial.status == TrialStatus.COMPLETED:
            outcome.metrics_by_hash[trial.config_hash] = MetricLookup(
                repo.get_trial_metrics(trial.trial_id)
            )
        elif trial.error:
            outcome.errors.append(f"[{split}] trial {trial.trial_name}: {trial.error}")
    return outcome


def _trial_configs(
    baseline: TrialConfig,
    baseline_hash: str,
    candidates: Sequence[SearchCandidate],
    hash_to_config: dict[str, TrialConfig],
) -> list[TrialConfig]:
    """baseline 永遠第一個；與 baseline 同 hash 的 candidate 不重複加入。"""
    trials = [baseline]
    seen = {baseline_hash}
    for c in candidates:
        if not c.config_hash or c.config_hash in seen:
            continue
        trials.append(hash_to_config[c.config_hash])
        seen.add(c.config_hash)
    return trials


def _rankable(
    c: SearchCandidate,
    metrics: MetricLookup,
    constraint_eval: Optional[ConstraintEvaluation],
) -> rec_mod.RankableCandidate:
    cfg = json.loads(c.config_json)
    return rec_mod.RankableCandidate(
        config_hash=c.config_hash or "",
        metrics=metrics,
        constraint_eval=constraint_eval,
        chunk_config=cfg.get("chunk_config", {}),
        retrieval_config=cfg.get("retrieval_config", {}),
    )


# --------------------------------------------------------------------------
def execute_search(
    search_config: SearchConfig,
    repo: SearchRepository,
    provider,
    *,
    output_dir: str,
    base_dir: str = trial_runner.ROOT,
    search_id: Optional[str] = None,
    metric_config: MetricConfig = DEFAULT_METRIC_CONFIG,
    now_fn: Callable[[], str] = _default_now,
    id_factory: Callable[[], str] = _default_id,
    write_report: bool = True,
) -> SearchResult:
    """執行（或 resume）一次完整的 optimizer search。"""
    from evaluation.v11.search import search_report

    created = now_fn()
    search_hash = search_config.compute_hash()
    git = hashing.git_state(base_dir)
    corpus_hash = hashing.corpus_hash(search_config.corpus_paths, base_dir)

    search_output = os.path.join(output_dir, search_id or "")
    dataset_abs = os.path.join(base_dir, search_config.dataset_path)

    # -- 1. Split 隔離（§十二 / §十三）--
    existing = repo.get_search(search_id) if search_id else None
    sid = search_id or id_factory()
    search_output = os.path.join(output_dir, sid)
    splits_dir = os.path.join(search_output, "splits")
    split_manifest = split_mod.prepare_split_datasets(dataset_abs, splits_dir)

    # -- 2. Candidate 生成（§九 / §十）--
    generation = generate_candidates(
        search_config,
        canonical_text_version=trial_runner.CANONICAL_TEXT_VERSION,
        identifier_normalization_version=trial_runner.IDENTIFIER_NORMALIZATION_VERSION,
    )
    manifest_sig = candidate_manifest_signature(generation)

    # -- 3. Resume 前置驗證（§21.6 / §21.7）--
    if existing is not None:
        if existing.search_config_hash != search_hash:
            raise SearchResumeError(
                f"search_config_hash 不符（DB={existing.search_config_hash!r} / "
                f"目前={search_hash!r}），拒絕 resume。",
                code="search_config_hash_mismatch",
            )
        if existing.dataset_hash != split_manifest.full_dataset_hash:
            raise SearchResumeError(
                f"dataset_hash 已改變（DB={existing.dataset_hash!r} / "
                f"目前={split_manifest.full_dataset_hash!r}），拒絕 resume（§21.7 / §13.6）。",
                code="dataset_hash_mismatch",
            )
        if existing.corpus_hash != corpus_hash:
            raise SearchResumeError(
                f"corpus_hash 已改變（DB={existing.corpus_hash!r} / 目前={corpus_hash!r}），"
                "拒絕 resume（§21.7）。",
                code="corpus_hash_mismatch",
            )
        prior_sig = existing.candidate_manifest_json
        if prior_sig is not None and prior_sig != manifest_sig:
            raise SearchResumeError(
                "candidate manifest 與原 search 不符，拒絕 resume（§21.6）。",
                code="candidate_manifest_mismatch",
            )

    baseline_tc = TrialConfig(
        trial_name=BASELINE_TRIAL_NAME,
        chunk_config=dict(search_config.baseline_chunk_config),
        retrieval=dict(search_config.baseline_retrieval_config),
    )
    baseline_hash = baseline_tc.compute_hash(
        canonical_text_version=trial_runner.CANONICAL_TEXT_VERSION,
        identifier_normalization_version=trial_runner.IDENTIFIER_NORMALIZATION_VERSION,
    )

    search = existing or OptimizerSearch(
        search_id=sid,
        search_name=search_config.search_name,
        status=SearchStatus.RUNNING,
        search_mode=search_config.search_mode,
        search_config_json=hashing.canonical_json(search_config.sanitized()),
        search_config_hash=search_hash,
        dataset_version=split_manifest.dataset_version,
        dataset_hash=split_manifest.full_dataset_hash,
        corpus_hash=corpus_hash,
        code_commit=git.get("commit"),
        working_tree_dirty=git.get("working_tree_dirty"),
        random_seed=search_config.random_seed,
        max_candidates=search_config.max_candidates,
        validation_top_n=search_config.validation_top_n,
        baseline_config_hash=baseline_hash,
        created_at=created,
    )
    search.status = SearchStatus.RUNNING
    search.started_at = search.started_at or created
    search.generation_stats_json = hashing.canonical_json(generation.stats())
    search.candidate_manifest_json = manifest_sig
    search.split_manifest_json = hashing.canonical_json(split_manifest.to_dict())
    repo.upsert_search(search)

    # -- 4. 持久化 candidate（resume 時沿用既有 candidate_id，不產生重複）--
    prior = {c.ordinal: c for c in repo.list_candidates(sid)}
    hash_to_config: dict[str, TrialConfig] = {}
    candidates: list[SearchCandidate] = []

    for gc in sorted(generation.executable + generation.rejected, key=lambda c: c.ordinal):
        cfg_json = hashing.canonical_json(
            {
                "chunk_config": gc.chunk_config,
                "retrieval_config": gc.retrieval_config,
                "assignment": gc.assignment,
                "projected_fields": list(gc.projected_fields),
            }
        )
        old = prior.get(gc.ordinal)
        cand = SearchCandidate(
            search_id=sid,
            candidate_id=old.candidate_id if old else id_factory(),
            ordinal=gc.ordinal,
            config_json=cfg_json,
            config_hash=gc.config_hash if gc.is_executable else None,
            generation_status=gc.generation_status,
            generation_rejection_reason=gc.generation_rejection_reason,
            duplicate_of_hash=gc.duplicate_of_hash,
            is_baseline=bool(gc.is_executable and gc.config_hash == baseline_hash),
            tune_status=(
                CandidateStageStatus.PENDING
                if gc.is_executable
                else CandidateStageStatus.NOT_SELECTED
            ),
            validation_status=CandidateStageStatus.NOT_SELECTED,
            test_status=CandidateStageStatus.LOCKED_OUT,
        )
        if old is not None:
            # 保留既有階段狀態（resume 不重跑已完成階段）。
            cand.tune_trial_id = old.tune_trial_id
            cand.validation_trial_id = old.validation_trial_id
            cand.test_trial_id = old.test_trial_id
        candidates.append(cand)
        if gc.is_executable and gc.config_hash:
            hash_to_config[gc.config_hash] = gc.to_trial_config()

    repo.save_candidates(candidates)
    executable = [c for c in candidates if c.config_hash]

    result = SearchResult(
        search=search,
        candidates=candidates,
        generation=generation,
        split_manifest=split_manifest,
    )

    # -- 5. Coverage 事前評估（§五）--
    coverage = split_mod.assess_validation_coverage(
        search_config.hard_constraints, split_manifest.splits[split_mod.SPLIT_VALIDATION]
    )
    result.coverage = coverage
    if not coverage.sufficient:
        result.notes.append(coverage.message)

    # ------------------------------------------------------------------
    # 6. Tune stage（§12.2）—— 所有合法 candidate，只在 tune split
    # ------------------------------------------------------------------
    tune_info = split_manifest.splits[split_mod.SPLIT_TUNE]
    if not tune_info.dataset_path:
        raise SearchResumeError("tune split 沒有任何 case，無法執行搜尋。", code="empty_tune_split")

    tune = _run_split(
        search=search,
        split=split_mod.SPLIT_TUNE,
        dataset_path=tune_info.dataset_path,
        trials=_trial_configs(baseline_tc, baseline_hash, executable, hash_to_config),
        search_config=search_config,
        repo=repo,
        provider=provider,
        output_dir=search_output,
        base_dir=base_dir,
        run_id=search.tune_run_id,
        metric_config=metric_config,
        now_fn=now_fn,
        id_factory=id_factory,
    )
    result.tune = tune
    search.tune_run_id = tune.run_id
    repo.upsert_search(search)

    baseline_tune_metrics = tune.metrics_by_hash.get(baseline_hash)
    for c in executable:
        trial = tune.trial_by_hash.get(c.config_hash or "")
        c.tune_trial_id = getattr(trial, "trial_id", None)
        c.tune_status = (
            CandidateStageStatus.COMPLETED
            if c.config_hash in tune.metrics_by_hash
            else CandidateStageStatus.FAILED
        )

    # -- Tune 預篩 + ranking（§十六）--
    prefilter = search_config.recommendation_policy.tune_prefilter
    survivors: list[SearchCandidate] = []
    tune_rejections: dict[str, list[str]] = {}
    for c in executable:
        reasons: list[str] = []
        metrics = tune.metrics_by_hash.get(c.config_hash or "")
        if metrics is None:
            if prefilter.eliminate_on_failed_trial:
                reasons.append("tune trial 未成功完成")
        else:
            ce = evaluate_constraints(
                search_config.hard_constraints, metrics, baseline_tune_metrics
            )
            for r in ce.results:
                if r.status != "fail":
                    continue
                if r.computable:
                    if r.safety_critical and prefilter.eliminate_on_computable_safety_violation:
                        reasons.append(
                            f"tune 明確違反 safety constraint {r.name}："
                            f"observed={r.observed_value!r} 未滿足 {r.operator} {r.required_value!r}"
                        )
                elif prefilter.eliminate_on_not_computable:
                    reasons.append(f"tune 上 {r.name} 不可計算（{r.reason}）")
        if reasons:
            tune_rejections[c.config_hash or ""] = reasons
        else:
            survivors.append(c)

    rankables = [
        _rankable(c, tune.metrics_by_hash[c.config_hash or ""], None)
        for c in survivors
        if c.config_hash in tune.metrics_by_hash
    ]
    tune_ranking = rec_mod.rank_candidates(
        rankables,
        search_config.recommendation_policy.tune_ranking,
        tolerance=search_config.recommendation_policy.float_tolerance,
    )
    result.tune_ranking = tune_ranking
    rank_by_hash = {r.config_hash: r.rank for r in tune_ranking}
    for c in executable:
        c.tune_rank = rank_by_hash.get(c.config_hash or "")

    selected_hashes = [
        r.config_hash for r in tune_ranking[: search_config.validation_top_n]
    ]

    # ------------------------------------------------------------------
    # 7. Validation stage（§12.3）—— 只跑 tune 選出的 top-N
    # ------------------------------------------------------------------
    val_info = split_manifest.splits[split_mod.SPLIT_VALIDATION]
    if not val_info.dataset_path:
        raise SearchResumeError(
            "validation split 沒有任何 case，無法進行 candidate selection。",
            code="empty_validation_split",
        )
    selected = [c for c in executable if c.config_hash in set(selected_hashes)]
    validation = _run_split(
        search=search,
        split=split_mod.SPLIT_VALIDATION,
        dataset_path=val_info.dataset_path,
        trials=_trial_configs(baseline_tc, baseline_hash, selected, hash_to_config),
        search_config=search_config,
        repo=repo,
        provider=provider,
        output_dir=search_output,
        base_dir=base_dir,
        run_id=search.validation_run_id,
        metric_config=metric_config,
        now_fn=now_fn,
        id_factory=id_factory,
    )
    result.validation = validation
    search.validation_run_id = validation.run_id
    repo.upsert_search(search)

    baseline_val_metrics = validation.metrics_by_hash.get(baseline_hash)
    for c in selected:
        trial = validation.trial_by_hash.get(c.config_hash or "")
        c.validation_trial_id = getattr(trial, "trial_id", None)
        c.validation_status = (
            CandidateStageStatus.COMPLETED
            if c.config_hash in validation.metrics_by_hash
            else CandidateStageStatus.FAILED
        )

    # ------------------------------------------------------------------
    # 8. Hard constraints（§十四）—— 正式判定用 validation
    # ------------------------------------------------------------------
    feasible: list[SearchCandidate] = []
    infeasible_rankables: list[rec_mod.RankableCandidate] = []
    for c in selected:
        metrics = validation.metrics_by_hash.get(c.config_hash or "")
        if metrics is None:
            c.feasible = False
            c.rejection_reasons_json = hashing.canonical_json(
                ["validation trial 未成功完成"]
            )
            continue
        ce = evaluate_constraints(
            search_config.hard_constraints, metrics, baseline_val_metrics
        )
        result.constraint_evals[c.config_hash or ""] = ce
        c.constraint_results_json = hashing.canonical_json(ce.to_dict())
        c.feasible = ce.feasible
        if ce.feasible:
            feasible.append(c)
        else:
            c.rejection_reasons_json = hashing.canonical_json(ce.rejection_reasons())
            infeasible_rankables.append(_rankable(c, metrics, ce))

    # ------------------------------------------------------------------
    # 9. Pareto（§十七）—— 只在 feasible candidate 上建立
    # ------------------------------------------------------------------
    pareto_result = pareto_mod.compute_pareto(
        {
            c.config_hash: validation.metrics_by_hash[c.config_hash]  # type: ignore[index]
            for c in feasible
            if c.config_hash
        },
        search_config.pareto_objectives,
        tolerance=search_config.recommendation_policy.float_tolerance,
    )
    result.pareto = pareto_result
    point_by_hash = {p.key: p for p in pareto_result.points}
    for c in feasible:
        p = point_by_hash.get(c.config_hash or "")
        if p is not None:
            c.pareto_member = p.pareto_member
            c.pareto_rank = p.pareto_rank

    # ------------------------------------------------------------------
    # 10. Recommendation（§十八）
    # ------------------------------------------------------------------
    eligible = [
        _rankable(
            c,
            validation.metrics_by_hash[c.config_hash],  # type: ignore[index]
            result.constraint_evals.get(c.config_hash or ""),
        )
        for c in feasible
        if c.pareto_member and c.config_hash in validation.metrics_by_hash
    ]
    recommendation = rec_mod.select_recommendation(
        eligible,
        search_config.recommendation_policy.recommendation_ranking,
        tolerance=search_config.recommendation_policy.float_tolerance,
        infeasible=infeasible_rankables,
    )
    result.recommendation = recommendation
    rec_rank = {r.config_hash: r.rank for r in recommendation.ranking}
    for c in executable:
        c.recommendation_rank = rec_rank.get(c.config_hash or "")

    repo.save_candidates(candidates)

    # ------------------------------------------------------------------
    # 11. Lock + Test（§12.4 / §12.5）
    # ------------------------------------------------------------------
    if recommendation.status != rec_mod.STATUS_OK or not recommendation.recommended_config_hash:
        # §18.5 / §五：明確失敗，不 lock、不跑 test、不硬選。
        search.status = (
            SearchStatus.INSUFFICIENT_VALIDATION_COVERAGE
            if not coverage.sufficient
            else SearchStatus.NO_FEASIBLE_CANDIDATE
        )
        search.recommended_config_hash = None
        search.completed_at = now_fn()
        repo.upsert_search(search)
        result.notes.append(
            recommendation.reason
            or "沒有可行 candidate；未進入 test 階段（test metric 因此不存在）。"
        )
    else:
        search = repo.lock_recommendation(
            sid, recommendation.recommended_config_hash, now_fn()
        )
        result.search = search

        test_info = split_manifest.splits[split_mod.SPLIT_TEST]
        rec_hash = recommendation.recommended_config_hash
        test_trials = [baseline_tc]
        if rec_hash != baseline_hash:
            test_trials.append(hash_to_config[rec_hash])

        if not test_info.dataset_path:
            search.status = SearchStatus.TEST_FAILED
            search.error = "test split 沒有任何 case。"
        else:
            test = _run_split(
                search=search,
                split=split_mod.SPLIT_TEST,
                dataset_path=test_info.dataset_path,
                trials=test_trials,
                search_config=search_config,
                repo=repo,
                provider=provider,
                output_dir=search_output,
                base_dir=base_dir,
                run_id=search.test_run_id,
                metric_config=metric_config,
                now_fn=now_fn,
                id_factory=id_factory,
            )
            result.test = test
            search.test_run_id = test.run_id
            search.test_evaluated = True

            for c in candidates:
                if c.config_hash in (baseline_hash, rec_hash):
                    trial = test.trial_by_hash.get(c.config_hash or "")
                    c.test_trial_id = getattr(trial, "trial_id", None)
                    c.test_status = (
                        CandidateStageStatus.COMPLETED
                        if c.config_hash in test.metrics_by_hash
                        else CandidateStageStatus.FAILED
                    )

            # §12.5：test 只影響 final status，**不改變** recommendation。
            if rec_hash not in test.metrics_by_hash or baseline_hash not in test.metrics_by_hash:
                search.status = SearchStatus.TEST_FAILED
            else:
                regressed = _test_regressions(
                    search_config,
                    test.metrics_by_hash[baseline_hash],
                    test.metrics_by_hash[rec_hash],
                )
                search.status = (
                    SearchStatus.TEST_REGRESSION if regressed else SearchStatus.COMPLETED
                )
                if regressed:
                    result.notes.append(
                        "test split 上偵測到 regression："
                        + "；".join(regressed)
                        + "。依 §12.5，recommendation 維持為原鎖定結果，不改選第二名。"
                    )
        search.completed_at = now_fn()
        repo.upsert_search(search)
        repo.save_candidates(candidates)

    result.candidates = repo.list_candidates(sid)
    result.search = repo.get_search(sid) or search
    result.remaining_collections = sorted(
        set(
            (result.tune.remaining_collections if result.tune else [])
            + (result.validation.remaining_collections if result.validation else [])
            + (result.test.remaining_collections if result.test else [])
        )
    )

    if write_report:
        report = search_report.build_search_report(result, search_config, repo)
        json_path, md_path = search_report.write_search_report(report, search_output)
        result.report_json_path = json_path
        result.report_markdown_path = md_path
        search.report_json_path = json_path
        search.report_markdown_path = md_path
        repo.upsert_search(search)
        result.search = repo.get_search(sid) or search

    return result


def _test_regressions(
    search_config: SearchConfig,
    baseline: MetricLookup,
    recommended: MetricLookup,
) -> list[str]:
    """§12.5：以同一組 hard constraint 檢查 test split 上是否退化。

    僅供標示 final status；**不得**用來重新選擇 candidate。
    """
    ce = evaluate_constraints(search_config.hard_constraints, recommended, baseline)
    return [
        f"{r.name}（observed={r.observed_value!r} / required={r.required_value!r}）"
        for r in ce.results
        if r.status == "fail"
    ]


def cleanup_search(search_id: str, repo: SearchRepository, provider) -> dict[str, str]:
    """刪除該 search 全部 split run 的 trial collection（§20.7）。"""
    search = repo.get_search(search_id)
    if search is None:
        raise SearchResumeError(f"找不到 search_id={search_id!r}", code="search_not_found")
    out: dict[str, str] = {}
    for run_id in (search.tune_run_id, search.validation_run_id, search.test_run_id):
        if run_id:
            out.update(trial_runner.cleanup_run(run_id, repo, provider))
    return out
