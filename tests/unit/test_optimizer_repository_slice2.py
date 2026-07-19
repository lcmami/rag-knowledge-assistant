"""Optimizer Repository Slice 2 表 / migration / CRUD 單元測試（§17.2）。"""
from __future__ import annotations

import sqlite3

import pytest

from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.run_models import (
    CleanupStatus,
    EvaluationRun,
    RetrievalTrial,
    RunStatus,
    TrialCaseResult,
    TrialMetric,
    TrialStatus,
)


def _repo(tmp_path) -> OptimizerRepository:
    return OptimizerRepository(str(tmp_path / "optimizer.db"))


def _run(run_id="r1", **over) -> EvaluationRun:
    base = dict(run_id=run_id, status=RunStatus.RUNNING, run_name="n", random_seed=42,
                working_tree_dirty=True, corpus_hash="ch", dataset_hash="dh")
    base.update(over)
    return EvaluationRun(**base)


def _trial(trial_id="t1", run_id="r1", ordinal=0, config_hash="h1", **over) -> RetrievalTrial:
    base = dict(trial_id=trial_id, run_id=run_id, ordinal=ordinal, trial_name="a",
                status=TrialStatus.PENDING, config_json="{}", config_hash=config_hash)
    base.update(over)
    return RetrievalTrial(**base)


def test_migration_0002_applied(tmp_path):
    repo = _repo(tmp_path)
    # Slice 3 追加 additive migration 0003_chunk_cache_stats（0001/0002 不變）。
    assert repo.applied_migrations() == [
        "0001_initial",
        "0002_runs_trials",
        "0003_chunk_cache_stats",
    ]


def test_slice2_tables_exist_no_documents_table(tmp_path):
    repo = _repo(tmp_path)
    conn = sqlite3.connect(repo.db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"evaluation_runs", "retrieval_trials", "trial_metrics",
            "trial_case_results", "evaluation_cases", "schema_migrations"} <= tables
    assert "documents" not in tables


def test_run_crud_roundtrip_and_bool(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_run(_run(working_tree_dirty=False))
    got = repo.get_run("r1")
    assert got is not None and got.working_tree_dirty is False and got.random_seed == 42
    # idempotent upsert（相同 run_id 不產生重複）
    repo.upsert_run(_run(status=RunStatus.COMPLETED))
    conn = sqlite3.connect(repo.db_path)
    n = conn.execute("SELECT COUNT(*) FROM evaluation_runs WHERE run_id='r1'").fetchone()[0]
    conn.close()
    assert n == 1 and repo.get_run("r1").status == RunStatus.COMPLETED


def test_trial_crud_and_lookup(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_run(_run())
    repo.upsert_trial(_trial())
    assert repo.get_trial("t1").trial_name == "a"
    assert repo.get_trial_by_ordinal("r1", 0).trial_id == "t1"
    assert repo.get_trial_by_config_hash("r1", "h1").trial_id == "t1"
    assert [t.trial_id for t in repo.list_trials("r1")] == ["t1"]


def test_duplicate_ordinal_constraint(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_run(_run())
    repo.upsert_trial(_trial(trial_id="t1", ordinal=0, config_hash="h1"))
    with pytest.raises(sqlite3.IntegrityError):
        repo.upsert_trial(_trial(trial_id="t2", ordinal=0, config_hash="h2"))


def test_duplicate_config_hash_constraint(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_run(_run())
    repo.upsert_trial(_trial(trial_id="t1", ordinal=0, config_hash="dup"))
    with pytest.raises(sqlite3.IntegrityError):
        repo.upsert_trial(_trial(trial_id="t2", ordinal=1, config_hash="dup"))


def test_foreign_key_enforced(tmp_path):
    repo = _repo(tmp_path)
    # 無對應 run → 插入 trial 應被 FK 阻擋
    with pytest.raises(sqlite3.IntegrityError):
        repo.upsert_trial(_trial(run_id="missing"))


def test_metrics_and_case_results_crud_and_json_roundtrip(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_run(_run())
    repo.upsert_trial(_trial())
    metrics = [
        TrialMetric(trial_id="t1", metric_name="mrr", scope="aggregate", query_type=None,
                    value=0.5, numerator=1.5, denominator=3, sample_count=3),
        TrialMetric(trial_id="t1", metric_name="graded_ndcg_at_5", scope="aggregate",
                    query_type=None, value=None, computable=False,
                    not_computable_reason="no_offset_grounded_gold_evidence"),
        TrialMetric(trial_id="t1", metric_name="recall_at_1", scope="query_type",
                    query_type="exact_code", value=1.0, sample_count=2),
    ]
    repo.save_trial_metrics("t1", metrics)
    got = {(m.metric_name, m.scope, m.query_type): m for m in repo.get_trial_metrics("t1")}
    assert got[("mrr", "aggregate", None)].value == 0.5
    assert got[("graded_ndcg_at_5", "aggregate", None)].computable is False
    assert got[("recall_at_1", "query_type", "exact_code")].value == 1.0

    cases = [TrialCaseResult(trial_id="t1", case_id="c1", case_status="relevant_hit",
                             query_type="exact_code", answerable=True, latency_ms=12,
                             first_relevant_rank=1, top_results_json='[{"rank":1}]',
                             evidence_match_json='[{"rank":1,"relevance_level":2}]')]
    repo.save_trial_case_results("t1", cases)
    cr = repo.get_trial_case_results("t1")[0]
    assert cr.answerable is True and cr.first_relevant_rank == 1
    import json
    assert json.loads(cr.evidence_match_json)[0]["relevance_level"] == 2

    # delete-then-insert idempotent（重存不重複）
    repo.save_trial_metrics("t1", metrics)
    assert len(repo.get_trial_metrics("t1")) == 3


def test_run_delete_cascades(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_run(_run())
    repo.upsert_trial(_trial())
    repo.save_trial_metrics("t1", [TrialMetric(trial_id="t1", metric_name="mrr",
                                               scope="aggregate", query_type=None, value=0.5)])
    repo.save_trial_case_results("t1", [TrialCaseResult(trial_id="t1", case_id="c1",
                                        case_status="x", query_type=None, answerable=False,
                                        latency_ms=1, first_relevant_rank=None,
                                        top_results_json="[]", evidence_match_json="[]")])
    repo.delete_run("r1")
    assert repo.list_trials("r1") == []
    assert repo.get_trial_metrics("t1") == []
    assert repo.get_trial_case_results("t1") == []


def test_db_path_injection_separate_files(tmp_path):
    r1 = OptimizerRepository(str(tmp_path / "a.optimizer.db"))
    r2 = OptimizerRepository(str(tmp_path / "b.optimizer.db"))
    r1.upsert_run(_run(run_id="only-in-a"))
    assert r1.get_run("only-in-a") is not None
    assert r2.get_run("only-in-a") is None


def test_list_running_trials(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_run(_run())
    repo.upsert_trial(_trial(trial_id="t1", ordinal=0, config_hash="h1",
                             status=TrialStatus.RUNNING))
    repo.upsert_trial(_trial(trial_id="t2", ordinal=1, config_hash="h2",
                             status=TrialStatus.COMPLETED))
    assert [t.trial_id for t in repo.list_running_trials()] == ["t1"]
