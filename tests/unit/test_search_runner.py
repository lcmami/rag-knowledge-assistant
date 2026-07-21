"""Slice 5 §25.2 / §25.7 —— Split isolation、Runner 流程、Lock、Resume、Report。

全部使用 fake provider：不載入 BGE、不連 Qdrant、不呼叫任何 LLM。
"""
from __future__ import annotations

import json
import os
import shutil
import uuid

import pytest

from evaluation.v11.search import search_runner
from evaluation.v11.search.search_config import SearchConfig
from evaluation.v11.search.search_models import (
    RecommendationLockError,
    SearchResumeError,
    SearchStatus,
)
from evaluation.v11.search.search_repository import SearchRepository
from tests.unit.search_helpers import case, raw_search_config, write_dataset
from tests.unit.v11_fakes import FakeTrialServiceProvider, ranked

# 與 search_helpers.case() 的 expected_evidence 對齊（doc-a / sha-a / section "S"）
GOLD_DOC = "doc-a"
GOLD_SECTION = "S"

TUNE_Q = "tune 問題一"
TUNE_Q2 = "tune 問題二"
VAL_Q = "validation 問題"
TEST_Q = "test 問題"


def dataset_cases():
    return [
        dict(case("t1", split="tune", query_type="exact_code", question=TUNE_Q)),
        dict(case("t2", split="tune", query_type="semantic", question=TUNE_Q2)),
        dict(case("v1", split="validation", query_type="semantic", question=VAL_Q)),
        dict(case("x1", split="test", query_type="exact_code", question=TEST_Q)),
    ]


class RecordingProvider(FakeTrialServiceProvider):
    """記錄每次 retrieve 的 (collection, question)，用於證明 split 隔離。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: list[tuple[str, str]] = []

    def retrieve(self, collection_name, trial_config, question, top_k):
        self.seen.append((collection_name, question))
        return super().retrieve(collection_name, trial_config, question, top_k)


def results_for(trial_name, collection, question):
    """讓不同 candidate 有不同品質：ordinal 越小，命中排名越前。"""
    hit = ranked(1, doc=GOLD_DOC, sha="sha-a", section=GOLD_SECTION, score=0.9)
    filler = ranked(1, doc="other", sha="sha-other", section="無關", score=0.8,
                    chunk_index=9)
    if trial_name == "baseline" or trial_name.endswith("000"):
        return [hit, filler]
    return [
        ranked(1, doc="other", sha="sha-other", section="無關", score=0.8, chunk_index=9),
        ranked(2, doc=GOLD_DOC, sha="sha-a", section=GOLD_SECTION, score=0.7),
    ]


@pytest.fixture()
def repo_tmp():
    """repo 內的暫存目錄。

    ``SearchConfig.dataset_path`` 依 §6.9 必須是 repo 相對路徑，而 pytest 的 ``tmp_path``
    可能位於不同磁碟機，無法表示成相對路徑 —— 因此 dataset fixture 必須放在 repo 樹內。
    測試結束即刪除；目錄名已納入 .gitignore。
    """
    d = os.path.join(
        search_runner.trial_runner.ROOT, "evaluation", ".pytest-tmp", uuid.uuid4().hex
    )
    os.makedirs(d, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_config(repo_tmp, **overrides):
    ds = write_dataset(os.path.join(repo_tmp, "data", "cases.json"), dataset_cases())
    raw = raw_search_config(
        search_name="runner-test",
        dataset_path=os.path.relpath(ds, search_runner.trial_runner.ROOT).replace("\\", "/"),
        # 只用一定可計算、且寬鬆到必然通過的限制 → 流程可推進到 lock + test
        hard_constraints=[
            {
                "name": "latency_ceiling",
                "metric": "p95_retrieval_latency_ms",
                "operator": "<=",
                "value": 100000,
                "safety_critical": False,
                "on_not_computable": "fail",
            }
        ],
        search_space={"retrieval": {"top_k": [3, 5]}},
        validation_top_n=4,
    )
    raw.update(overrides)
    return SearchConfig.from_raw(raw)


def run_search(repo_tmp, *, provider=None, config=None, search_id=None, **overrides):
    cfg = config or make_config(repo_tmp, **overrides)
    repo = SearchRepository(os.path.join(repo_tmp, "s.optimizer.db"))
    prov = provider or RecordingProvider(results_for=results_for)
    result = search_runner.execute_search(
        cfg, repo, prov, output_dir=os.path.join(repo_tmp, "out"), search_id=search_id
    )
    return result, repo, prov, cfg


# --------------------------------------------------------------------------
# Split isolation（§25.2）
# --------------------------------------------------------------------------
def test_each_split_run_only_sees_its_own_cases(repo_tmp):
    result, repo, prov, _ = run_search(repo_tmp)
    run_of_collection = {}
    for run_id in (result.search.tune_run_id, result.search.validation_run_id,
                   result.search.test_run_id):
        if not run_id:
            continue
        for t in repo.list_trials(run_id):
            run_of_collection[t.collection_name] = run_id

    expected = {
        result.search.tune_run_id: {TUNE_Q, TUNE_Q2},
        result.search.validation_run_id: {VAL_Q},
        result.search.test_run_id: {TEST_Q},
    }
    seen: dict[str, set[str]] = {}
    for collection, question in prov.seen:
        seen.setdefault(run_of_collection[collection], set()).add(question)

    for run_id, questions in seen.items():
        assert questions == expected[run_id], f"run {run_id} 看到了不屬於它的 split 題目"


def test_split_dataset_files_are_written_and_disjoint(repo_tmp):
    result, _, _, _ = run_search(repo_tmp)
    ids = {}
    for name, info in result.split_manifest.splits.items():
        with open(info.dataset_path, encoding="utf-8") as f:
            ids[name] = {c["id"] for c in json.load(f)["cases"]}
    assert ids["tune"] == {"t1", "t2"}
    assert ids["validation"] == {"v1"}
    assert ids["test"] == {"x1"}
    assert not (ids["tune"] & ids["validation"] & ids["test"])


def test_group_leakage_is_rejected(repo_tmp):
    from evaluation.v11.case_schema import CaseSchemaError

    bad = dataset_cases()
    bad[2]["group_id"] = bad[0]["group_id"]  # validation 與 tune 共用 group
    ds = write_dataset(os.path.join(repo_tmp, "bad", "cases.json"), bad)
    raw = raw_search_config(
        dataset_path=os.path.relpath(ds, search_runner.trial_runner.ROOT).replace("\\", "/")
    )
    repo = SearchRepository(os.path.join(repo_tmp, "s.db"))
    with pytest.raises(CaseSchemaError, match="leakage"):
        search_runner.execute_search(
            SearchConfig.from_raw(raw), repo,
            RecordingProvider(results_for=results_for),
            output_dir=os.path.join(repo_tmp, "out"),
        )


# --------------------------------------------------------------------------
# Lock 與 test 階段（§12.4 / §12.5 / §25.2）
# --------------------------------------------------------------------------
def test_happy_path_locks_and_runs_test(repo_tmp):
    result, repo, _, _ = run_search(repo_tmp)
    s = result.search
    assert s.recommendation_locked is True
    assert s.recommended_config_hash
    assert s.test_evaluated is True
    assert s.status in (SearchStatus.COMPLETED, SearchStatus.TEST_REGRESSION)
    assert s.recommendation_locked_at


def test_test_stage_runs_only_baseline_and_recommendation(repo_tmp):
    result, repo, _, _ = run_search(repo_tmp)
    trials = repo.list_trials(result.search.test_run_id)
    hashes = {t.config_hash for t in trials}
    assert len(trials) <= 2, "test 只能跑 baseline + recommendation"
    assert result.search.baseline_config_hash in hashes
    assert result.search.recommended_config_hash in hashes


# --------------------------------------------------------------------------
# Tune 預篩 vs Validation Safety Gate 的不對稱性（已核准的 §十六 決策）
# --------------------------------------------------------------------------
SAFETY_FP_CONSTRAINT = {
    "name": "unanswerable_false_positive_ceiling",
    "metric": "unanswerable_false_positive_rate",
    "operator": "<=",
    "value": 0.25,
    "safety_critical": True,
    "on_not_computable": "fail",
}


def test_tune_keeps_candidate_but_validation_rejects_when_not_computable(repo_tmp):
    """核准的 tune_prefilter 決策：不可計算在兩階段的效果必須不對稱。

    - Tune（``eliminate_on_not_computable=false``）：候選**保留**，仍進入 validation。
    - Validation（safety-critical + ``on_not_computable=fail``）：候選 **infeasible**。

    本 fixture 的 tune 與 validation 都沒有 unanswerable case，
    因此 ``unanswerable_false_positive_rate`` 在兩階段都不可計算。
    """
    result, repo, _, _ = run_search(repo_tmp, hard_constraints=[SAFETY_FP_CONSTRAINT])

    # 1) Tune 沒有把候選刷掉 —— 它們有進入 validation 並實際跑完。
    reached_validation = [
        c for c in result.candidates
        if c.config_hash and c.validation_status == "completed"
    ]
    assert reached_validation, "tune 不得因不可計算而淘汰全部候選"

    # 2) Validation 階段全部 infeasible，且原因就是「不可計算 + 政策 fail」。
    assert result.constraint_evals, "應有候選進入 validation constraint 判定"
    for chash, ce in result.constraint_evals.items():
        assert ce.feasible is False, f"{chash} 不得在 safety metric 不可計算時被判為可行"
        fp = [r for r in ce.results if r.name == SAFETY_FP_CONSTRAINT["name"]]
        assert fp and fp[0].status == "fail"
        assert fp[0].computable is False
        assert fp[0].observed_value is None, "不可計算不得被代換成 0"

    # 3) 不可計算不得灌水 constraint pass count（§十六 限制）。
    for ce in result.constraint_evals.values():
        assert ce.pass_count == 0

    # 4) 未通過 Safety Gate → 不 lock、不跑 test。
    s = result.search
    assert s.recommendation_locked is False
    assert s.recommended_config_hash is None
    assert repo.count_test_trials(s.search_id) == 0


def test_tune_prefilter_policy_is_disclosed_in_report(repo_tmp):
    """§十六：實際使用的預篩欄位必須寫進 Report 的 recommendation policy。"""
    result, _, _, _ = run_search(repo_tmp, hard_constraints=[SAFETY_FP_CONSTRAINT])
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    pf = report["recommendation_policy"]["tune_prefilter"]
    assert pf["eliminate_on_not_computable"] is False
    assert pf["eliminate_on_failed_trial"] is True
    assert pf["eliminate_on_computable_safety_violation"] is True


def test_tune_still_eliminates_computable_safety_violation(repo_tmp):
    """預篩放寬只限『不可計算』；明確違反的 safety constraint 仍必須在 tune 淘汰。"""
    result, _, _, _ = run_search(
        repo_tmp,
        hard_constraints=[
            {
                "name": "impossible_but_computable",
                "metric": "p95_retrieval_latency_ms",
                "operator": "<=",
                "value": -1,
                "safety_critical": True,
                "on_not_computable": "fail",
            }
        ],
    )
    # 全部候選在 tune 就被刷掉 → 沒有任何候選進入 validation constraint 判定
    assert result.tune_ranking == []
    assert result.search.recommended_config_hash is None
    assert result.search.recommendation_locked is False


def test_no_test_trials_exist_when_no_feasible_candidate(repo_tmp):
    """限制不可能達成 → 不 lock、不跑 test、test metric 不存在。"""
    result, repo, _, _ = run_search(
        repo_tmp,
        hard_constraints=[
            {
                "name": "impossible",
                "metric": "p95_retrieval_latency_ms",
                "operator": "<=",
                "value": -1,
                "safety_critical": False,
                "on_not_computable": "fail",
            }
        ],
    )
    s = result.search
    assert s.status == SearchStatus.NO_FEASIBLE_CANDIDATE
    assert s.recommended_config_hash is None
    assert s.recommendation_locked is False
    assert s.test_evaluated is False
    assert s.test_run_id is None
    assert repo.count_test_trials(s.search_id) == 0
    assert all(c.test_trial_id is None for c in result.candidates)


def test_report_hides_test_section_before_lock(repo_tmp):
    result, _, _, _ = run_search(
        repo_tmp,
        hard_constraints=[
            {"name": "impossible", "metric": "p95_retrieval_latency_ms",
             "operator": "<=", "value": -1, "safety_critical": False,
             "on_not_computable": "fail"}
        ],
    )
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["test"]["available"] is False
    assert report["test"]["test_trial_count"] == 0
    assert "baseline_metrics" not in report["test"]
    assert report["c01_same_chunk_multi_identifier"]["observable"] is False
    assert report["no_feasible_candidate_state"]["is_no_feasible_candidate"] is True
    assert report["recommendation"]["recommended_config_hash"] is None


def test_report_contains_required_sections(repo_tmp):
    result, _, _, _ = run_search(repo_tmp)
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    for key in (
        "search_id", "search_mode", "search_config_hash", "random_seed",
        "candidate_budget", "generation_stats", "candidate_manifest",
        "split_manifest", "group_leakage_check", "corpus_hash", "dataset_hash",
        "git_commit", "working_tree_dirty", "baseline_config",
        "hard_constraint_policy", "constraint_matrix", "pareto", "recommendation",
        "recommendation_policy", "per_case_diffs", "cache_and_latency",
        "known_limitations", "errors", "no_feasible_candidate_state",
    ):
        assert key in report, f"報告缺少 {key}"
    assert os.path.exists(result.report_markdown_path)


def test_report_contains_outcome_semantics(repo_tmp):
    """Slice 5A §二～§六：兩種狀態與證據強度必須都進 Report。"""
    result, _, _, _ = run_search(repo_tmp)
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    oc = report["outcome"]
    for key in (
        "test_constraint_status", "failed_test_constraints",
        "baseline_comparison_status", "metrics",
        "recommendation_apply_eligible", "recommendation_confidence",
        "recommendation_selection_basis", "apply_ineligible_reasons",
        "quality_tie_candidate_count", "latency_measurement_repetitions",
        "latency_selection_stable", "metric_sample_granularity",
        "identifier_unanswerable_coverage",
        # 四段說明各自獨立保存，不得互相覆蓋。
        "test_constraint_note", "baseline_comparison_note",
        "selection_note", "apply_eligibility_note",
    ):
        assert key in oc, f"outcome 缺少 {key}"
    # recommendation 區塊也要能直接看到 apply 資格
    assert report["recommendation"]["recommendation_apply_eligible"] is False

    # §六 的 tie-break 揭露與 §三 的 apply 說明必須**都**出現在 Markdown 報告裡。
    with open(result.report_markdown_path, encoding="utf-8") as f:
        md = f.read()
    assert oc["apply_eligibility_note"].split("\n")[0] in md
    assert oc["selection_note"].split("\n")[0] in md


def test_outcome_is_persisted_on_the_search_record(repo_tmp):
    """§三：Slice 7 需能從 search record（而非只有報告）讀到 apply 資格。"""
    result, repo, _, _ = run_search(repo_tmp)
    stored = repo.get_search(result.search.search_id)
    assert stored.outcome_json
    oc = json.loads(stored.outcome_json)
    assert oc["recommendation_apply_eligible"] is False
    assert oc["recommendation_confidence"] == "low"


def test_locked_recommendation_exists_but_is_not_apply_eligible(repo_tmp):
    """§三：lock 與 apply 資格是兩件事。"""
    result, repo, _, _ = run_search(repo_tmp)
    stored = repo.get_search(result.search.search_id)
    assert stored.recommendation_locked is True
    assert stored.recommended_config_hash, "lock 仍應保有固定的 candidate"
    assert json.loads(stored.outcome_json)["recommendation_apply_eligible"] is False


def test_constraint_failed_and_regression_are_distinct_statuses():
    """§2.3：不得只輸出 test_regression。"""
    assert SearchStatus.TEST_CONSTRAINT_FAILED == "test_constraint_failed"
    assert SearchStatus.TEST_REGRESSION == "test_regression"
    assert SearchStatus.TEST_CONSTRAINT_FAILED != SearchStatus.TEST_REGRESSION


def test_report_never_labels_recommendation_as_deployed(repo_tmp):
    result, _, _, _ = run_search(repo_tmp)
    blob = open(result.report_json_path, encoding="utf-8").read()
    for banned in ("deployed_candidate", "active_candidate", "production_candidate"):
        assert banned not in blob


def test_test_result_does_not_change_recommendation(repo_tmp):
    """即使 test 退化，recommended_config_hash 仍維持鎖定值。"""
    result, repo, _, _ = run_search(repo_tmp)
    locked = result.search.recommended_config_hash
    # 直接嘗試改選第二名 → 必須被拒絕
    with pytest.raises(RecommendationLockError):
        repo.lock_recommendation(result.search.search_id, "some-other-hash", "now")
    assert repo.get_search(result.search.search_id).recommended_config_hash == locked


# --------------------------------------------------------------------------
# 生成 / 排序 / 可行性
# --------------------------------------------------------------------------
def test_candidates_and_generation_stats_persisted(repo_tmp):
    result, repo, _, _ = run_search(repo_tmp)
    stored = repo.list_candidates(result.search.search_id)
    assert len(stored) == result.generation.raw_combination_count
    assert json.loads(result.search.generation_stats_json)["executable_candidate_count"] == 2
    assert result.search.candidate_manifest_json


def test_validation_top_n_limits_validation_trials(repo_tmp):
    result, repo, _, _ = run_search(repo_tmp, validation_top_n=1)
    val_trials = repo.list_trials(result.search.validation_run_id)
    # baseline + 至多 1 個 candidate
    assert len(val_trials) <= 2


def test_failed_trial_is_isolated_and_recorded(repo_tmp):
    prov = RecordingProvider(results_for=results_for, fail_build_trials={"candidate-001"})
    result, repo, _, _ = run_search(repo_tmp, provider=prov)
    # 其他 candidate 仍完成，search 不因單一 trial 失敗而中止
    assert result.search.tune_run_id is not None
    statuses = {c.tune_status for c in result.candidates if c.config_hash}
    assert "failed" in statuses or "completed" in statuses


def test_all_trial_collections_cleaned_up(repo_tmp):
    result, _, prov, _ = run_search(repo_tmp)
    assert prov.collections == set(), f"殘留 collection：{prov.collections}"
    assert result.remaining_collections == []


def test_runner_never_applies_to_production(repo_tmp):
    """Slice 5 不得寫入任何正式設定：provider 只被要求建立 optimizer_ collection。"""
    result, _, prov, _ = run_search(repo_tmp)
    for name in prov.deleted:
        assert name.startswith("optimizer_"), f"動到非 optimizer collection：{name}"


# --------------------------------------------------------------------------
# Resume（§21）
# --------------------------------------------------------------------------
def test_resume_reuses_candidates_and_does_not_duplicate(repo_tmp):
    result, repo, _, cfg = run_search(repo_tmp)
    sid = result.search.search_id
    before = [(c.candidate_id, c.ordinal) for c in repo.list_candidates(sid)]

    again = search_runner.execute_search(
        cfg, repo, RecordingProvider(results_for=results_for),
        output_dir=os.path.join(repo_tmp, "out"), search_id=sid,
    )
    after = [(c.candidate_id, c.ordinal) for c in repo.list_candidates(sid)]
    assert before == after, "resume 不得產生重複 candidate 或改變 candidate_id"
    assert again.search.search_id == sid


def test_resume_keeps_existing_lock(repo_tmp):
    result, repo, _, cfg = run_search(repo_tmp)
    sid = result.search.search_id
    locked = result.search.recommended_config_hash
    again = search_runner.execute_search(
        cfg, repo, RecordingProvider(results_for=results_for),
        output_dir=os.path.join(repo_tmp, "out"), search_id=sid,
    )
    assert again.search.recommended_config_hash == locked
    assert again.search.recommendation_locked is True


def test_resume_rejected_when_search_config_hash_changes(repo_tmp):
    result, repo, _, _ = run_search(repo_tmp)
    other = make_config(repo_tmp, random_seed=999)
    with pytest.raises(SearchResumeError) as exc:
        search_runner.execute_search(
            other, repo, RecordingProvider(results_for=results_for),
            output_dir=os.path.join(repo_tmp, "out"), search_id=result.search.search_id,
        )
    assert exc.value.code == "search_config_hash_mismatch"


def test_resume_rejected_when_dataset_hash_changes(repo_tmp):
    result, repo, _, cfg = run_search(repo_tmp)
    # 改動 dataset 內容（不改路徑）→ dataset_hash 變 → 拒絕 resume
    ds_path = os.path.join(search_runner.trial_runner.ROOT, cfg.dataset_path)
    with open(ds_path, encoding="utf-8") as f:
        raw = json.load(f)
    raw["cases"][0]["question"] = "被竄改的問題"
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)

    with pytest.raises(SearchResumeError) as exc:
        search_runner.execute_search(
            cfg, repo, RecordingProvider(results_for=results_for),
            output_dir=os.path.join(repo_tmp, "out"), search_id=result.search.search_id,
        )
    assert exc.value.code == "dataset_hash_mismatch"


def test_cleanup_search_removes_all_split_collections(repo_tmp):
    result, repo, prov, _ = run_search(repo_tmp)
    out = search_runner.cleanup_search(result.search.search_id, repo, prov)
    assert isinstance(out, dict)
    assert prov.collections == set()
