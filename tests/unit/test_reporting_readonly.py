"""Read-only Optimizer Reporting store 測試（Slice 8 §六 / §二十）。

重點：read-only 保證（不建立 DB、不改 mtime / 內容 / 不新增 table）、unavailable / schema
incompatible 語意、去識別化、以及 demo fixture 的資料正確性。
"""
from __future__ import annotations

import os
import sqlite3
import time

import pytest

from evaluation.v11.reporting import (
    REASON_NOT_AVAILABLE,
    REASON_SCHEMA_INCOMPATIBLE,
    OptimizerDataUnavailable,
    OptimizerReportingStore,
    ReadOnlySearchRepository,
)
from evaluation.v11.reporting.demo_fixture import build_demo_fixtures


@pytest.fixture()
def fixtures(tmp_path):
    paths = build_demo_fixtures(str(tmp_path / "demo"))
    return paths


@pytest.fixture()
def store(fixtures):
    return OptimizerReportingStore(fixtures["search"], fixtures["review"], fixtures["deployment"])


# --------------------------------------------------------------------------
# Read-only 保證（§六.4 / §六.7）
# --------------------------------------------------------------------------
def test_missing_db_not_created(tmp_path):
    missing = str(tmp_path / "nope.search.db")
    with pytest.raises(OptimizerDataUnavailable) as ei:
        ReadOnlySearchRepository(missing)
    assert ei.value.reason == REASON_NOT_AVAILABLE
    assert not os.path.exists(missing)


def test_store_missing_paths_report_unavailable_without_creating(tmp_path):
    s = str(tmp_path / "a.search.db")
    store = OptimizerReportingStore(s, None, None)
    ov = store.overview(active_alias="factory_knowledge_active", admin_write_enabled=False)
    assert ov["search"]["configured"] is True
    assert ov["search"]["available"] is False
    assert ov["search"]["reason"] == REASON_NOT_AVAILABLE
    assert ov["review"]["configured"] is False
    assert not os.path.exists(s)


def test_schema_incompatible(tmp_path):
    bad = str(tmp_path / "bad.search.db")
    conn = sqlite3.connect(bad)
    conn.execute("CREATE TABLE something_else (x TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(OptimizerDataUnavailable) as ei:
        ReadOnlySearchRepository(bad)
    assert ei.value.reason == REASON_SCHEMA_INCOMPATIBLE


def test_reads_do_not_modify_db(store, fixtures):
    path = fixtures["search"]
    before_mtime = os.path.getmtime(path)
    before_tables = _tables(path)
    time.sleep(0.01)
    # 一輪讀取
    store.list_searches()
    store.get_search("search-demo")
    store.list_candidates("search-demo")
    store.comparison("search-demo")
    store.coverage()
    assert os.path.getmtime(path) == before_mtime
    assert _tables(path) == before_tables


def _tables(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 資料正確性
# --------------------------------------------------------------------------
def test_overview_demo_fixture(store):
    ov = store.overview(active_alias="factory_knowledge_active", admin_write_enabled=False)
    assert ov["is_demo_fixture"] is True
    assert ov["search"]["available"] is True
    ls = ov["search"]["latest_search"]
    assert ls["status"] == "test_constraint_failed"
    assert ls["apply_eligible"] is False
    assert ls["test_constraint_status"] == "failed"
    assert "small_validation_sample" in ls["apply_ineligible_reasons"]
    assert ov["deployment"]["admin_write_enabled"] is False
    assert ov["deployment"]["alias_initialized"] is False


def test_candidate_config_fields(store):
    cands = store.list_candidates("search-demo")["candidates"]
    rec = next(c for c in cands if c["candidate_id"] == "cand-recommended")
    assert rec["chunk_strategy"] == "current"
    assert rec["retrieval_strategy"] == "identifier_aware"
    assert rec["identifier_vector_floor"] == 0.5
    base = next(c for c in cands if c["is_baseline"])
    assert base["retrieval_strategy"] == "dense"


def test_comparison_metrics_and_per_case(store):
    cmp = store.comparison("search-demo")["comparison"]
    names = {m["metric_name"] for m in cmp["metrics"]}
    assert "ndcg_at_5" in names and "latency_p95_ms" in names
    ndcg = next(m for m in cmp["metrics"] if m["metric_name"] == "ndcg_at_5")
    assert ndcg["baseline_value"] == 0.8367 and ndcg["candidate_value"] == 0.9193
    assert ndcg["direction"] == "higher_is_better"
    p95 = next(m for m in cmp["metrics"] if m["metric_name"] == "latency_p95_ms")
    assert p95["direction"] == "lower_is_better"
    # C01 誠實：baseline/candidate first relevant rank 皆為 2 → verdict same（§九）
    c01 = next(c for c in cmp["per_case"]["cases"] if c["case_id"] == "eval-e102-001")
    assert c01["baseline_first_relevant_rank"] == 2
    assert c01["candidate_first_relevant_rank"] == 2
    assert c01["verdict"] == "same"
    assert c01["question"]  # 題目文字有補上（來自 evaluation_cases）
    assert len(c01["candidate_top5"]) >= 1


def test_review_detail(store):
    rv = store.get_review("review-demo")["session"]
    assert rv["counts"]["accepted"] == 1
    assert rv["counts"]["rejected"] == 1
    assert len(rv["candidates"]) == 3
    assert rv["audit_events"]


def test_coverage_gap(store):
    cov = store.coverage()["coverage"]
    assert any("identifier_bearing_unanswerable" in w for w in cov["coverage_warnings"])


# --------------------------------------------------------------------------
# 去識別化（§五.4/.5）
# --------------------------------------------------------------------------
def test_no_private_paths_in_outputs(store, fixtures):
    import json

    payloads = [
        store.overview(active_alias="a", admin_write_enabled=False),
        store.get_search("search-demo"),
        store.list_candidates("search-demo"),
        store.comparison("search-demo"),
        store.get_review("review-demo"),
        store.coverage(),
        store.deployment_summary("factory_knowledge_active"),
    ]
    for p in payloads:
        blob = json.dumps(p, ensure_ascii=False)
        assert ".db" not in blob
        assert "cases-v2.json" not in blob
        assert "dataset_path" not in blob
        # tmp_path 的絕對路徑片段不得外洩
        assert str(fixtures["search"]).replace("\\", "/") not in blob.replace("\\", "/")


def test_unknown_ids(store):
    assert store.get_search("nope")["found"] is False
    assert store.list_candidates("nope")["found"] is False
    assert store.get_review("nope")["found"] is False
