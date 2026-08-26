"""Slice 5 §25.6 —— Repository / migration / lock guard / resume 持久化。"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from evaluation.v11.optimizer_repository import MIGRATIONS, MigrationError, run_migrations
from evaluation.v11.search.search_models import (
    CandidateStageStatus,
    OptimizerSearch,
    RecommendationLockError,
    SearchCandidate,
    SearchStatus,
)
from evaluation.v11.search.search_repository import SearchRepository


@pytest.fixture()
def repo(tmp_path):
    return SearchRepository(str(tmp_path / "s.optimizer.db"))


def make_search(**kw) -> OptimizerSearch:
    base = dict(
        search_id="s1", search_name="unit", status=SearchStatus.RUNNING,
        search_mode="grid", search_config_json="{}", search_config_hash="h-cfg",
        dataset_version="1.1", dataset_hash="h-ds", corpus_hash="h-corpus",
        code_commit="abc123", working_tree_dirty=False, random_seed=42,
        max_candidates=8, validation_top_n=4, baseline_config_hash="h-base",
        created_at="2026-07-20T00:00:00Z",
    )
    base.update(kw)
    return OptimizerSearch(**base)


def make_candidate(**kw) -> SearchCandidate:
    base = dict(
        search_id="s1", candidate_id="c1", ordinal=0, config_json="{}",
        config_hash="h1", generation_status="executable",
        tune_status=CandidateStageStatus.PENDING,
        validation_status=CandidateStageStatus.NOT_SELECTED,
        test_status=CandidateStageStatus.LOCKED_OUT,
    )
    base.update(kw)
    return SearchCandidate(**base)


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------
def test_migrations_registered_in_order():
    # 0007（review）/ 0008（deployment）屬各自獨立的 DB namespace，刻意不在 base list。
    assert [v for v, _ in MIGRATIONS] == [
        "0001_initial", "0002_runs_trials", "0003_chunk_cache_stats",
        "0004_retrieval_meta", "0005_optimizer_search", "0006_search_outcome",
        "0009_rerank",
    ]


def test_migrations_are_idempotent(tmp_path):
    path = str(tmp_path / "x.db")
    first = SearchRepository(path).applied_migrations()
    second = SearchRepository(path).applied_migrations()
    assert first == second == [v for v, _ in MIGRATIONS]


def test_tables_exist_and_no_documents_table(repo):
    conn = sqlite3.connect(repo.db_path)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {"optimizer_searches", "optimizer_candidates"} <= names
    assert "documents" not in names, "Optimizer DB 不得建立正式 documents 表"


def test_failed_migration_rolls_back(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "bad.db"), isolation_level=None)
    try:
        with pytest.raises(MigrationError):
            run_migrations(conn, [("0001_bad", [
                "CREATE TABLE ok_table (id TEXT)",
                "THIS IS NOT SQL",
            ])])
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "ok_table" not in names, "失敗的 migration 必須整批 rollback"
    finally:
        conn.close()


def test_foreign_keys_enforced(repo):
    repo.upsert_search(make_search())
    repo.upsert_candidate(make_candidate())
    repo.delete_search("s1")
    assert repo.list_candidates("s1") == []


# --------------------------------------------------------------------------
# CRUD + round trip
# --------------------------------------------------------------------------
def test_search_round_trip(repo):
    s = make_search(
        generation_stats_json=json.dumps({"raw_combination_count": 4}),
        split_manifest_json=json.dumps({"a": 1}),
        tune_run_id="run-tune",
    )
    repo.upsert_search(s)
    got = repo.get_search("s1")
    assert got.search_config_hash == "h-cfg"
    assert got.working_tree_dirty is False
    assert got.recommendation_locked is False
    assert got.test_evaluated is False
    assert json.loads(got.generation_stats_json)["raw_combination_count"] == 4
    assert got.tune_run_id == "run-tune"


def test_upsert_search_is_idempotent(repo):
    repo.upsert_search(make_search())
    repo.upsert_search(make_search(status=SearchStatus.COMPLETED))
    assert len(repo.list_searches()) == 1
    assert repo.get_search("s1").status == SearchStatus.COMPLETED


def test_candidate_round_trip(repo):
    repo.upsert_search(make_search())
    repo.upsert_candidate(make_candidate(
        feasible=True, pareto_member=True, pareto_rank=1, recommendation_rank=2,
        constraint_results_json=json.dumps({"feasible": True}),
        rejection_reasons_json=json.dumps(["nope"]),
    ))
    got = repo.list_candidates("s1")[0]
    assert got.feasible is True and got.pareto_member is True
    assert got.pareto_rank == 1 and got.recommendation_rank == 2
    assert json.loads(got.rejection_reasons_json) == ["nope"]


def test_unique_search_id_ordinal(repo):
    repo.upsert_search(make_search())
    repo.upsert_candidate(make_candidate(candidate_id="c1", ordinal=0, config_hash="h1"))
    with pytest.raises(sqlite3.IntegrityError):
        repo.upsert_candidate(make_candidate(candidate_id="c2", ordinal=0, config_hash="h2"))


def test_unique_search_id_config_hash(repo):
    repo.upsert_search(make_search())
    repo.upsert_candidate(make_candidate(candidate_id="c1", ordinal=0, config_hash="h1"))
    with pytest.raises(sqlite3.IntegrityError):
        repo.upsert_candidate(make_candidate(candidate_id="c2", ordinal=1, config_hash="h1"))


def test_null_config_hash_allows_multiple_non_executable_candidates(repo):
    """invalid / duplicate / not_sampled 都存 NULL hash，必須能並存。"""
    repo.upsert_search(make_search())
    for i in range(3):
        repo.upsert_candidate(make_candidate(
            candidate_id=f"c{i}", ordinal=i, config_hash=None,
            generation_status="invalid", generation_rejection_reason="bad",
        ))
    assert len(repo.list_candidates("s1")) == 3


def test_get_candidate_by_config_hash(repo):
    repo.upsert_search(make_search())
    repo.upsert_candidate(make_candidate(config_hash="target"))
    assert repo.get_candidate_by_config_hash("s1", "target").candidate_id == "c1"
    assert repo.get_candidate_by_config_hash("s1", "missing") is None


def test_save_candidates_rolls_back_on_failure(repo):
    repo.upsert_search(make_search())
    good = make_candidate(candidate_id="c1", ordinal=0, config_hash="h1")
    clash = make_candidate(candidate_id="c2", ordinal=0, config_hash="h2")  # ordinal 衝突
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_candidates([good, clash])
    assert repo.list_candidates("s1") == []


# --------------------------------------------------------------------------
# Recommendation Lock（§12.4 / §25.6）
# --------------------------------------------------------------------------
def test_test_trial_id_rejected_before_lock(repo):
    repo.upsert_search(make_search())
    with pytest.raises(RecommendationLockError, match="尚未 lock"):
        repo.upsert_candidate(make_candidate(test_trial_id="t-test"))
    assert repo.count_test_trials("s1") == 0


def test_save_candidates_also_guards_test_trial(repo):
    repo.upsert_search(make_search())
    with pytest.raises(RecommendationLockError):
        repo.save_candidates([make_candidate(test_trial_id="t-test")])


def test_test_trial_id_allowed_after_lock(repo):
    repo.upsert_search(make_search())
    repo.lock_recommendation("s1", "h-winner", "2026-07-20T01:00:00Z")
    repo.upsert_candidate(make_candidate(
        config_hash="h-winner", test_trial_id="t-test",
        test_status=CandidateStageStatus.COMPLETED))
    assert repo.count_test_trials("s1") == 1


def test_lock_persists_and_is_idempotent(repo):
    repo.upsert_search(make_search())
    a = repo.lock_recommendation("s1", "h-winner", "2026-07-20T01:00:00Z")
    b = repo.lock_recommendation("s1", "h-winner", "2026-07-20T02:00:00Z")
    assert a.recommendation_locked is True
    assert b.recommendation_locked_at == "2026-07-20T01:00:00Z", "重複 lock 不得改時間"
    assert repo.is_recommendation_locked("s1") is True


def test_locked_recommendation_cannot_be_changed(repo):
    """§12.5：test 結果不好也不得改選第二名。"""
    repo.upsert_search(make_search())
    repo.lock_recommendation("s1", "h-winner", "2026-07-20T01:00:00Z")
    with pytest.raises(RecommendationLockError, match="不得改為"):
        repo.lock_recommendation("s1", "h-runner-up", "2026-07-20T03:00:00Z")
    assert repo.get_search("s1").recommended_config_hash == "h-winner"


def test_cannot_lock_null_recommendation(repo):
    repo.upsert_search(make_search())
    with pytest.raises(RecommendationLockError, match="不得 lock"):
        repo.lock_recommendation("s1", "", "2026-07-20T01:00:00Z")


def test_lock_survives_reopen(repo, tmp_path):
    repo.upsert_search(make_search())
    repo.lock_recommendation("s1", "h-winner", "2026-07-20T01:00:00Z")
    reopened = SearchRepository(repo.db_path)
    got = reopened.get_search("s1")
    assert got.recommendation_locked is True
    assert got.recommended_config_hash == "h-winner"


def test_assert_test_allowed_rejects_locked_without_hash():
    s = make_search(recommendation_locked=True, recommended_config_hash=None)
    with pytest.raises(RecommendationLockError, match="狀態不一致"):
        s.assert_test_allowed()


def test_slice2_repository_still_works(repo):
    """Slice 5 的 migration 不得破壞 Slice 2 的 run/trial 儲存。"""
    from evaluation.v11.run_models import EvaluationRun, RetrievalTrial, RunStatus, TrialStatus

    repo.upsert_run(EvaluationRun(run_id="r1", status=RunStatus.PENDING, run_name="n"))
    repo.upsert_trial(RetrievalTrial(
        trial_id="t1", run_id="r1", ordinal=0, trial_name="a",
        status=TrialStatus.PENDING, config_json="{}", config_hash="h"))
    assert repo.get_run("r1") is not None
    assert len(repo.list_trials("r1")) == 1
