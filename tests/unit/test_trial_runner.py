"""Trial Runner 單元測試（Slice 2 §17.4）——全部使用 Fake Services。"""
from __future__ import annotations

import itertools
import json
import os

import pytest

import json as _json

from evaluation.v11 import cli, trial_runner
from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.retrieval_metrics import IndexChunk, RankedResult
from evaluation.v11.run_models import RunConfig, TrialStatus, RunStatus, CleanupStatus
from evaluation.v11.trial_collections import (
    UnsafeCollectionError,
    assert_deletable,
    trial_collection_name,
)
from tests.unit.v11_fakes import FakeTrialServiceProvider


def _ids():
    c = itertools.count()
    return lambda: f"t{next(c)}"


def _clock():
    c = itertools.count()
    return lambda: f"2026-01-01T00:00:{next(c) % 60:02d}+00:00"


def _write_dataset(tmp_path) -> str:
    ds = {
        "schema_version": "1.1",
        "dataset_version": "v-test",
        "cases": [
            {"id": "ans", "question": "E102 是什麼", "query_type": "exact_code",
             "answerable": True, "group_id": "g1", "split": "test",
             "expected_evidence": [{"document_sha256": "sha-doc", "canonical_text_version": "v1",
                                    "section": "S1", "offset_scope": "document",
                                    "char_start": 0, "char_end": 100, "anchor_text": ["E102"]}]},
            {"id": "un", "question": "售價多少", "query_type": "unanswerable",
             "answerable": False, "group_id": "g2", "split": "test", "expected_evidence": []},
        ],
    }
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(ds), encoding="utf-8")
    return str(p)


def _results(trial_name, collection, question):
    if "E102" in question:
        prov = [{"document_sha256": "sha-doc", "canonical_text_version": "v1",
                 "loader_version": "v1", "page": None, "section": "S1", "paragraph_index": 0,
                 "offset_scope": "document", "char_start": 0, "char_end": 100}]
        return [RankedResult(rank=1, document_id="doc", document_sha256="sha-doc",
                             filename="f.md", section="S1", page=None, chunk_index=0, score=0.9,
                             text="E102", provenance_spans=prov, identifiers=["E102"],
                             canonical_text_version="v1")]
    return []


def _config(tmp_path, trials, **flags) -> RunConfig:
    raw = {"schema_version": "1.0", "run_name": "t", "random_seed": 42,
           "dataset_path": _write_dataset(tmp_path),
           "corpus_paths": ["sample_docs", "evaluation/fixtures"], "trials": trials}
    raw.update(flags)
    return RunConfig.from_raw(raw)


def _repo(tmp_path) -> OptimizerRepository:
    return OptimizerRepository(str(tmp_path / "o.optimizer.db"))


def _out(tmp_path) -> str:
    return str(tmp_path / "runs")


def _run(repo, cfg, provider, tmp_path, run_id="run-1"):
    return trial_runner.execute_run(
        cfg, repo, provider, output_dir=_out(tmp_path), run_id=run_id,
        now_fn=_clock(), id_factory=_ids(),
    )


# --------------------------------------------------------------------------
def test_single_trial_success(tmp_path):
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}])
    repo = _repo(tmp_path)
    provider = FakeTrialServiceProvider(results_for=_results, point_count=10)
    result = _run(repo, cfg, provider, tmp_path)

    assert result.run.status == RunStatus.COMPLETED
    trials = repo.list_trials("run-1")
    assert len(trials) == 1
    t = trials[0]
    assert t.status == TrialStatus.COMPLETED
    assert t.attempt_count == 1
    assert t.point_count == 10
    assert t.started_at and t.completed_at
    assert t.cleanup_status == CleanupStatus.CLEANED
    # metrics + case results 持久化
    assert repo.get_trial_metrics(t.trial_id)
    assert {c.case_id for c in repo.get_trial_case_results(t.trial_id)} == {"ans", "un"}
    # collection 已清理，不殘留
    assert provider.collections == set()
    assert result.remaining_collections == []
    # report 產生
    assert os.path.exists(result.report_json_path)
    assert os.path.exists(result.report_markdown_path)


def test_multiple_trials_sequential(tmp_path):
    cfg = _config(tmp_path, [
        {"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
        {"trial_name": "b", "retrieval_config": {"top_k": 5, "score_threshold": 0.60}},
    ])
    repo = _repo(tmp_path)
    provider = FakeTrialServiceProvider(results_for=_results)
    _run(repo, cfg, provider, tmp_path)
    assert provider.build_calls == ["a", "b"]
    assert all(t.status == TrialStatus.COMPLETED for t in repo.list_trials("run-1"))


def test_trial_failure_isolation(tmp_path):
    cfg = _config(tmp_path, [
        {"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
        {"trial_name": "b", "retrieval_config": {"top_k": 5, "score_threshold": 0.60}},
    ])
    repo = _repo(tmp_path)
    provider = FakeTrialServiceProvider(results_for=_results, fail_build_trials={"a"})
    result = _run(repo, cfg, provider, tmp_path)
    by_name = {t.trial_name: t for t in repo.list_trials("run-1")}
    assert by_name["a"].status == TrialStatus.FAILED and by_name["a"].error
    assert by_name["b"].status == TrialStatus.COMPLETED
    assert result.run.status == RunStatus.FAILED  # 有 trial 失敗 → run 誠實標記 failed


def test_fail_fast_stops(tmp_path):
    cfg = _config(tmp_path, [
        {"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
        {"trial_name": "b", "retrieval_config": {"top_k": 5, "score_threshold": 0.60}},
    ], fail_fast=True)
    repo = _repo(tmp_path)
    provider = FakeTrialServiceProvider(results_for=_results, fail_build_trials={"a"})
    _run(repo, cfg, provider, tmp_path)
    by_name = {t.trial_name: t for t in repo.list_trials("run-1")}
    assert by_name["a"].status == TrialStatus.FAILED
    assert by_name["b"].status == TrialStatus.PENDING  # fail_fast → 未執行
    assert provider.build_calls == ["a"]


def test_completed_trial_skipped_on_resume(tmp_path):
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}])
    repo = _repo(tmp_path)
    p1 = FakeTrialServiceProvider(results_for=_results)
    _run(repo, cfg, p1, tmp_path)
    # resume：completed 應跳過（不重跑）
    p2 = FakeTrialServiceProvider(results_for=_results)
    _run(repo, cfg, p2, tmp_path)
    assert p2.build_calls == []
    assert repo.list_trials("run-1")[0].attempt_count == 1


def test_failed_trial_not_retried_without_flag(tmp_path):
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}])
    repo = _repo(tmp_path)
    _run(repo, cfg, FakeTrialServiceProvider(results_for=_results, fail_build_trials={"a"}), tmp_path)
    # 未允許 retry → 跳過
    p2 = FakeTrialServiceProvider(results_for=_results)
    _run(repo, cfg, p2, tmp_path)
    assert p2.build_calls == []
    assert repo.list_trials("run-1")[0].attempt_count == 1


def test_failed_trial_retried_with_flag(tmp_path):
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}],
                  retry_failed=True)
    repo = _repo(tmp_path)
    _run(repo, cfg, FakeTrialServiceProvider(results_for=_results, fail_build_trials={"a"}), tmp_path)
    p2 = FakeTrialServiceProvider(results_for=_results)
    _run(repo, cfg, p2, tmp_path)
    t = repo.list_trials("run-1")[0]
    assert p2.build_calls == ["a"]
    assert t.status == TrialStatus.COMPLETED and t.attempt_count == 2


def test_interrupted_recovery_and_cleanup(tmp_path):
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}],
                  retry_interrupted=True)
    repo = _repo(tmp_path)
    _run(repo, cfg, FakeTrialServiceProvider(results_for=_results), tmp_path)
    t = repo.list_trials("run-1")[0]
    leftover = t.collection_name
    # 模擬中斷：trial 卡在 running、collection 殘留
    t.status = TrialStatus.RUNNING
    repo.upsert_trial(t)
    p2 = FakeTrialServiceProvider(results_for=_results)
    p2.collections.add(leftover)
    _run(repo, cfg, p2, tmp_path)
    t2 = repo.list_trials("run-1")[0]
    assert leftover in p2.deleted  # recovery 清理了殘留 collection
    assert t2.status == TrialStatus.COMPLETED
    assert t2.attempt_count == 2


def test_duplicate_config_hash_no_duplicate_trial(tmp_path):
    # 兩個 trial 名稱不同但語意 config 相同 → 只建立一個 Trial
    cfg = _config(tmp_path, [
        {"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
        {"trial_name": "b-same-config", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
    ])
    repo = _repo(tmp_path)
    _run(repo, cfg, FakeTrialServiceProvider(results_for=_results), tmp_path)
    assert len(repo.list_trials("run-1")) == 1


def test_cleanup_failure_keeps_metrics(tmp_path):
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}])
    repo = _repo(tmp_path)
    ids = _ids()
    # 預先算出 collection 名稱（deterministic id）以注入 cleanup 失敗
    trial_id = "t0"  # 第一個 id_factory() 用於 trial row
    coll = trial_collection_name("run-1", trial_id)
    provider = FakeTrialServiceProvider(results_for=_results, fail_cleanup_collections={coll})
    result = trial_runner.execute_run(cfg, repo, provider, output_dir=_out(tmp_path),
                                      run_id="run-1", now_fn=_clock(), id_factory=ids)
    t = repo.list_trials("run-1")[0]
    assert t.cleanup_status == CleanupStatus.CLEANUP_FAILED
    assert coll in result.remaining_collections
    # Metrics 未消失
    assert repo.get_trial_metrics(t.trial_id)
    assert repo.get_trial_case_results(t.trial_id)


def test_keep_trial_collections_reported(tmp_path):
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}],
                  keep_trial_collections=True)
    repo = _repo(tmp_path)
    provider = FakeTrialServiceProvider(results_for=_results)
    result = _run(repo, cfg, provider, tmp_path)
    t = repo.list_trials("run-1")[0]
    assert t.cleanup_status == CleanupStatus.ACTIVE
    assert t.collection_name in result.remaining_collections
    assert t.collection_name in provider.collections  # 保留未刪


def test_cleanup_guard_rejects_non_optimizer():
    assert_deletable("optimizer_run_trial")  # 不拋
    with pytest.raises(UnsafeCollectionError):
        assert_deletable("factory_knowledge")
    with pytest.raises(UnsafeCollectionError):
        assert_deletable("some_other_collection")


def test_cleanup_run_skips_non_optimizer(tmp_path):
    repo = _repo(tmp_path)
    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}],
                  keep_trial_collections=True)
    provider = FakeTrialServiceProvider(results_for=_results)
    _run(repo, cfg, provider, tmp_path)
    # 竄改 collection_name 為正式名稱 → cleanup 必須跳過、不刪
    t = repo.list_trials("run-1")[0]
    t.collection_name = "factory_knowledge"
    repo.upsert_trial(t)
    results = trial_runner.cleanup_run("run-1", repo, provider)
    assert results["factory_knowledge"] == "skipped_not_optimizer"


def test_runner_passes_full_index_universe_to_ndcg(tmp_path):
    # 取回只給 L1（section 命中），但 Index Universe 另含未被取回的 L2 →
    # 若 runner 只傳 retrieved top-K，NDCG 會=1；傳完整 universe 才會 <1（漏檢受罰）。
    def prov(cs, ce, section="S1"):
        return {"document_sha256": "sha-doc", "canonical_text_version": "v1",
                "loader_version": "v1", "page": None, "section": section, "paragraph_index": 0,
                "offset_scope": "document", "char_start": cs, "char_end": ce}

    def results_l1(trial_name, collection, question):
        if "E102" in question:
            return [RankedResult(rank=1, document_id="doc", document_sha256="sha-doc",
                                 filename="f.md", section="S1", page=None, chunk_index=5,
                                 score=0.7, text="ctx", provenance_spans=[prov(500, 600)],
                                 identifiers=["E102"], canonical_text_version="v1")]
        return []

    def universe(coll_a, coll_b):
        return [
            IndexChunk("doc", "sha-doc", 0, "cover", None, "S1", [prov(0, 100)], ["E102"], "v1"),
            IndexChunk("doc", "sha-doc", 5, "ctx", None, "S1", [prov(500, 600)], ["E102"], "v1"),
        ]

    cfg = _config(tmp_path, [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}])
    repo = _repo(tmp_path)
    provider = FakeTrialServiceProvider(results_for=results_l1, universe_for=universe, point_count=2)
    _run(repo, cfg, provider, tmp_path)
    assert provider.manifest_calls  # runner 有取 index manifest
    t = repo.list_trials("run-1")[0]
    ans = next(c for c in repo.get_trial_case_results(t.trial_id) if c.case_id == "ans")
    ndcg = _json.loads(ans.evidence_match_json)["ndcg"]
    assert ndcg["computable"] is True
    assert 0.0 < ndcg["value"] < 1.0  # 漏掉 L2 → NDCG < 1
    assert ndcg["relevant_chunk_count"] == 2
    # index_universe_size 應等於 point_count
    metrics = {mm.metric_name: mm for mm in repo.get_trial_metrics(t.trial_id)}
    assert metrics["index_universe_size"].value == metrics["index_point_count"].value == 2.0


def test_cli_shares_core_runner(tmp_path, monkeypatch):
    # CLI run 與 import 共用 execute_run；以 fake provider 驗證整條路徑
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "schema_version": "1.0", "run_name": "cli-run", "random_seed": 42,
        "dataset_path": _write_dataset(tmp_path),
        "corpus_paths": ["sample_docs", "evaluation/fixtures"],
        "trials": [{"trial_name": "a", "retrieval_config": {"top_k": 5, "score_threshold": 0.55}}],
    }), encoding="utf-8")
    db = str(tmp_path / "cli.optimizer.db")
    out = _out(tmp_path)
    monkeypatch.setattr(cli, "_make_provider",
                        lambda url, wd: FakeTrialServiceProvider(results_for=_results))
    rc = cli.main(["run", "--config", str(cfg_path), "--db", db, "--output-dir", out,
                   "--run-id", "cli-1"])
    assert rc == 0
    repo = OptimizerRepository(db)
    assert repo.get_run("cli-1").status == RunStatus.COMPLETED
