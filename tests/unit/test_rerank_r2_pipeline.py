"""Slice R2（v1.2.1）—— rerank 接進 pipeline 後的 metrics / 持久化 / 報表。

不載入任何真實模型、不連 Qdrant（沿用 `tests/conftest.py` 的既定規則）：
rerank telemetry 一律以既有 meta 通道模擬，驗證的是**接線與呈現**，不是模型行為。

核心回歸不變量（每一節都覆蓋）：
**rerank 未啟用時，所有輸出與 v1.1 byte-identical。**
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from evaluation.v11.case_schema import EvaluationCaseV11
from evaluation.v11.optimizer_repository import MIGRATIONS, OptimizerRepository
from evaluation.v11.retrieval_metrics import (
    CaseRetrieval,
    RankedResult,
    evaluate_trial,
    score_case,
)
from evaluation.v11.evidence_matcher import EvidenceMatcher
from evaluation.v11.metric_config import DEFAULT_METRIC_CONFIG


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _case(cid: str = "c1", *, answerable: bool = True) -> EvaluationCaseV11:
    return EvaluationCaseV11(
        id=cid,
        question=f"question for {cid}",
        query_type="semantic",
        answerable=answerable,
        group_id=cid,
        split="tune",
        expected_evidence=[],
    )


def _result(rank: int, *, rerank_score=None, normalized=None, applied=None) -> RankedResult:
    return RankedResult(
        rank=rank,
        document_id="doc",
        document_sha256="sha",
        filename="doc.md",
        section="S",
        page=None,
        chunk_index=rank,
        score=0.9 - rank * 0.01,
        text=f"chunk {rank}",
        ranking_score=0.9 - rank * 0.01,
        rerank_score=rerank_score,
        rerank_score_normalized=normalized,
        rerank_applied=applied,
    )


def _rerank_case(cid: str, *, latency: float, hits: int, misses: int) -> CaseRetrieval:
    return CaseRetrieval(
        case=_case(cid),
        results=[
            _result(1, rerank_score=3.5, normalized=0.970688, applied=True),
            _result(2, rerank_score=-10.19, normalized=0.0000373, applied=True),
            _result(3, applied=False),  # tail：未計分，保留原相對順序
        ],
        latency_ms=1200,
        rerank_strategy="cross_encoder",
        rerank_latency_ms=latency,
        rerank_cache_hits=hits,
        rerank_cache_misses=misses,
        rerank_window_count=2,
        rerank_tail_count=1,
        admitted_count=3,
    )


def _plain_case(cid: str = "c1") -> CaseRetrieval:
    return CaseRetrieval(case=_case(cid), results=[_result(1), _result(2)], latency_ms=40)


def _evaluate(cases):
    return evaluate_trial(cases, metric_config=DEFAULT_METRIC_CONFIG)


def _by_name(metrics):
    return {m.metric_name: m for m in metrics if m.scope == "aggregate"}


# --------------------------------------------------------------------------
# 1. 未啟用 rerank → 完全不變
# --------------------------------------------------------------------------
def test_rerank_disabled_emits_no_rerank_metrics_at_all():
    """未啟用時**一列都不產生** —— 不是產生 n/a 列，否則既有 trial_metrics 內容會改變。"""
    metrics = _evaluate([_plain_case()]).metrics
    assert not [m for m in metrics if "rerank" in m.metric_name]


def test_rerank_disabled_leaves_top_results_payload_unchanged():
    ce = score_case(_plain_case(), EvidenceMatcher(), DEFAULT_METRIC_CONFIG)
    payload = ce.top_results_payload()
    for item in payload:
        assert not [k for k in item if "rerank" in k]
        assert "ranking_score_source" not in item


def test_rerank_disabled_yields_no_per_case_rerank_meta():
    ce = score_case(_plain_case(), EvidenceMatcher(), DEFAULT_METRIC_CONFIG)
    assert ce.rerank_enabled is False
    assert ce.rerank_meta_payload() is None


# --------------------------------------------------------------------------
# 2. Metrics：mean / p50 / p95 + cache telemetry
# --------------------------------------------------------------------------
def test_rerank_latency_metrics_are_emitted_when_enabled():
    cases = [
        _rerank_case("c1", latency=100.0, hits=0, misses=2),
        _rerank_case("c2", latency=300.0, hits=1, misses=1),
        _rerank_case("c3", latency=200.0, hits=2, misses=0),
    ]
    m = _by_name(_evaluate(cases).metrics)

    assert m["mean_rerank_latency_ms"].value == pytest.approx(200.0)
    assert m["p50_rerank_latency_ms"].value == pytest.approx(200.0)
    # p95 沿用既有 _percentile（與 p95_retrieval_latency_ms 同一實作，故可比較）。
    assert m["p95_rerank_latency_ms"].value == pytest.approx(290.0)
    assert all(m[k].computable for k in
               ("mean_rerank_latency_ms", "p50_rerank_latency_ms", "p95_rerank_latency_ms"))


def test_rerank_cache_hit_miss_telemetry():
    cases = [
        _rerank_case("c1", latency=100.0, hits=0, misses=2),
        _rerank_case("c2", latency=100.0, hits=1, misses=1),
        _rerank_case("c3", latency=100.0, hits=2, misses=0),
    ]
    m = _by_name(_evaluate(cases).metrics)

    assert m["rerank_cache_hit_count"].value == 3.0
    assert m["rerank_cache_miss_count"].value == 3.0
    assert m["rerank_cache_hit_rate"].value == pytest.approx(0.5)
    assert m["rerank_case_count"].value == 3.0
    assert m["rerank_scored_pair_count"].value == 6.0  # 每 case window=2


def test_rerank_cache_hit_rate_not_computable_without_lookups():
    case = _rerank_case("c1", latency=0.0, hits=0, misses=0)
    m = _by_name(_evaluate([case]).metrics)
    assert m["rerank_cache_hit_rate"].computable is False
    assert m["rerank_cache_hit_rate"].not_computable_reason == "no_rerank_cache_lookups"


def test_retrieval_latency_metrics_are_end_to_end_not_rerank_only():
    """``*_retrieval_latency_ms`` 必須是整條 pipeline 的時間，不是 rerank 分項。

    這是 R2 最重要的量測不變量：feasibility 只能引用 end-to-end 值。
    """
    case = _rerank_case("c1", latency=900.0, hits=0, misses=2)  # 其中 900ms 花在 rerank
    m = _by_name(_evaluate([case]).metrics)
    assert m["p95_retrieval_latency_ms"].value == pytest.approx(1200.0)
    assert m["p95_rerank_latency_ms"].value == pytest.approx(900.0)
    assert m["p95_retrieval_latency_ms"].value > m["p95_rerank_latency_ms"].value


def test_mixed_cases_only_count_the_rerank_enabled_ones():
    cases = [_rerank_case("c1", latency=100.0, hits=1, misses=1), _plain_case("c2")]
    m = _by_name(_evaluate(cases).metrics)
    assert m["rerank_case_count"].value == 1.0
    assert m["rerank_case_count"].sample_count == 2  # 分母仍為全部 case


# --------------------------------------------------------------------------
# 3. per-case rerank metadata + raw/normalized provenance
# --------------------------------------------------------------------------
def test_per_case_rerank_meta_payload():
    ce = score_case(
        _rerank_case("c1", latency=123.456789, hits=1, misses=3),
        EvidenceMatcher(),
        DEFAULT_METRIC_CONFIG,
    )
    meta = ce.rerank_meta_payload()
    assert meta == {
        "rerank_strategy": "cross_encoder",
        "admitted_count": 3,
        "rerank_window_count": 2,
        "rerank_tail_count": 1,
        "rerank_latency_ms": 123.4568,
        "rerank_cache_hits": 1,
        "rerank_cache_misses": 3,
        "rerank_cache_hit_rate": 0.25,
        "ranking_score_source": "rerank_raw_logit",
    }


def test_top_results_payload_carries_raw_and_normalized_provenance():
    ce = score_case(
        _rerank_case("c1", latency=10.0, hits=0, misses=2),
        EvidenceMatcher(),
        DEFAULT_METRIC_CONFIG,
    )
    items = ce.top_results_payload()

    window = items[0]
    assert window["rerank_applied"] is True
    assert window["rerank_score"] == pytest.approx(3.5)
    assert window["rerank_score_normalized"] == pytest.approx(0.970688)
    # 排序主鍵必須明寫為 raw logit —— normalized 永遠不是排序依據。
    assert window["ranking_score_source"] == "rerank_raw_logit"

    tail = items[2]
    assert tail["rerank_applied"] is False
    assert tail["rerank_score"] is None
    assert tail["ranking_score_source"] == "vector_ranking_score"


def test_normalized_is_never_the_ranking_key_for_scored_results():
    """raw 為負而 normalized 接近 0 時，仍必須標示 raw 是排序主鍵。"""
    ce = score_case(
        _rerank_case("c1", latency=1.0, hits=0, misses=2),
        EvidenceMatcher(),
        DEFAULT_METRIC_CONFIG,
    )
    negative = ce.top_results_payload()[1]
    assert negative["rerank_score"] < 0.0
    assert 0.0 < negative["rerank_score_normalized"] < 1.0
    assert negative["ranking_score_source"] == "rerank_raw_logit"


# --------------------------------------------------------------------------
# 4. Migration 0009
# --------------------------------------------------------------------------
def test_migration_0009_is_registered_last_in_the_base_list():
    versions = [v for v, _ in MIGRATIONS]
    assert versions[-1] == "0009_rerank"
    # 全域編號政策：0007（review）/ 0008（deployment）屬其他 DB namespace，刻意不在此。
    assert "0007_review_workflow" not in versions
    assert "0008_deployment" not in versions


def test_migration_0009_adds_both_rerank_columns(tmp_path):
    db = str(tmp_path / "opt.db")
    repo = OptimizerRepository(db)
    assert repo.applied_migrations()[-1] == "0009_rerank"

    conn = sqlite3.connect(db)
    try:
        case_cols = {r[1] for r in conn.execute("PRAGMA table_info(trial_case_results)")}
        trial_cols = {r[1] for r in conn.execute("PRAGMA table_info(retrieval_trials)")}
    finally:
        conn.close()
    assert "rerank_meta_json" in case_cols       # per-case rerank metadata
    assert "rerank_stats_json" in trial_cols     # aggregate rerank telemetry
    # 0004 的欄位必須仍在（0009 是純追加，不取代）。
    assert "retrieval_meta_json" in case_cols


def test_migration_0009_is_idempotent_on_an_existing_db(tmp_path):
    from evaluation.v11.optimizer_repository import run_migrations

    db = str(tmp_path / "opt.db")
    OptimizerRepository(db)
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        assert run_migrations(conn) == []
    finally:
        conn.close()


def test_rerank_columns_round_trip_and_default_to_null(tmp_path):
    from evaluation.v11.run_models import (
        CleanupStatus,
        EvaluationRun,
        RetrievalTrial,
        TrialCaseResult,
        TrialStatus,
    )

    repo = OptimizerRepository(str(tmp_path / "opt.db"))
    repo.upsert_run(
        EvaluationRun(run_id="r1", run_name="n", status="completed", config_json="{}")
    )

    def _trial(tid, ordinal, stats):
        return RetrievalTrial(
            trial_id=tid, run_id="r1", ordinal=ordinal, trial_name=tid,
            status=TrialStatus.COMPLETED, config_json="{}", config_hash=tid,
            cleanup_status=CleanupStatus.CLEANED, rerank_stats_json=stats,
        )

    repo.upsert_trial(_trial("t-rerank", 0, json.dumps({"rerank_case_count": 3})))
    repo.upsert_trial(_trial("t-plain", 1, None))

    trials = {t.trial_id: t for t in repo.list_trials("r1")}
    assert json.loads(trials["t-rerank"].rerank_stats_json)["rerank_case_count"] == 3
    # 未啟用 rerank 的 trial 維持 NULL —— 既有 row 形狀不變。
    assert trials["t-plain"].rerank_stats_json is None

    repo.save_trial_case_results(
        "t-rerank",
        [
            TrialCaseResult(
                trial_id="t-rerank", case_id="c1", case_status="relevant_hit",
                query_type="semantic", answerable=True, latency_ms=10,
                first_relevant_rank=1, top_results_json="[]", evidence_match_json="{}",
                rerank_meta_json=json.dumps({"rerank_window_count": 5}),
            ),
            TrialCaseResult(
                trial_id="t-rerank", case_id="c2", case_status="miss",
                query_type="semantic", answerable=True, latency_ms=10,
                first_relevant_rank=None, top_results_json="[]", evidence_match_json="{}",
            ),
        ],
    )
    rows = {c.case_id: c for c in repo.get_trial_case_results("t-rerank")}
    assert json.loads(rows["c1"].rerank_meta_json)["rerank_window_count"] == 5
    assert rows["c2"].rerank_meta_json is None


# --------------------------------------------------------------------------
# 5. Report 呈現
# --------------------------------------------------------------------------
def _report_with_rerank():
    return {
        "run_id": "r", "run_name": "n", "run_status": "completed",
        "run_config": {}, "git_commit": "abc", "working_tree_dirty": False,
        "corpus_manifest": [], "corpus_hash": "h", "dataset_version": "v",
        "dataset_hash": "dh", "canonical_text_version": "c", "loader_version": "l",
        "identifier_normalization_version": "i", "random_seed": 42,
        "metric_config": {"version": "1.0"},
        "not_computable_metrics": [], "errors": [], "remaining_collections": [],
        "known_limitations": [],
        "trials": [
            {
                "trial_id": "t", "trial_name": "rerank-trial", "ordinal": 0,
                "status": "completed", "config_hash": "ch",
                "config": {
                    "retrieval_config": {
                        "strategy": "dense", "ranking_version": "dense_v1+cross_encoder_v1",
                        "rerank_strategy": "cross_encoder", "rerank_top_n": 5,
                    }
                },
                "collection_name": "optimizer_x", "cleanup_status": "cleaned",
                "point_count": 24, "index_build_time_ms": 1, "retrieval_time_ms": 2,
                "attempt_count": 1, "error": None, "strategy": "dense",
                "chunk_count": 24, "chunk_config_hash": "cc", "chunk_cache_stats": None,
                "rerank_stats": {
                    "rerank_strategy": "cross_encoder",
                    "rerank_fingerprint_hash": "e0dedc0255c2b8a9367c331769664ec01",
                    "rerank_fingerprint": {
                        "model_name": "BAAI/bge-reranker-base",
                        "model_revision": "2cfc18c9415c912f9d8155881c133215df768a70",
                        "num_labels": 1, "max_length": 512,
                    },
                    "scored_activation": "identity", "rerank_batch_size": 1,
                    "rerank_cache_enabled": False,
                    "rerank_cache_hits": 0, "rerank_cache_misses": 10,
                    "rerank_cache_hit_rate": 0.0,
                    "rerank_scored_pair_count": 10, "rerank_tail_pair_count": 2,
                    "rerank_case_count": 2, "cache_schema_version": "v1",
                    "total_rerank_latency_ms": 1800.0, "max_rerank_latency_ms": 1000.0,
                    "latency_note": "cached rerank latency is INFORMATIONAL ONLY.",
                },
                "aggregate_metrics": [
                    {"metric_name": "mean_rerank_latency_ms", "value": 900.0,
                     "computable": True, "sample_count": 2, "not_computable_reason": None},
                    {"metric_name": "p50_rerank_latency_ms", "value": 900.0,
                     "computable": True, "sample_count": 2, "not_computable_reason": None},
                    {"metric_name": "p95_rerank_latency_ms", "value": 990.0,
                     "computable": True, "sample_count": 2, "not_computable_reason": None},
                ],
                "per_query_type_metrics": [],
                "ndcg_summary": {"computable_cases": 0, "non_computable_cases": 0,
                                 "not_computable_reasons": {}},
                "cases": [
                    {
                        "case_id": "c1", "case_status": "relevant_hit",
                        "query_type": "semantic", "answerable": True,
                        "latency_ms": 1200, "first_relevant_rank": 1,
                        "top_results": [
                            {"rank": 1, "filename": "d.md", "section": "S", "page": None,
                             "score": 0.8, "relevance_level": 2, "identifiers": [],
                             "rerank_applied": True, "rerank_score": 3.568463,
                             "rerank_score_normalized": 0.972574,
                             "ranking_score_source": "rerank_raw_logit"},
                            {"rank": 2, "filename": "d.md", "section": "S", "page": None,
                             "score": 0.7, "relevance_level": 0, "identifiers": [],
                             "rerank_applied": False, "rerank_score": None,
                             "rerank_score_normalized": None,
                             "ranking_score_source": "vector_ranking_score"},
                        ],
                        "evidence_match": {"matches": []},
                        "retrieval_meta": None,
                        "rerank_meta": {
                            "rerank_strategy": "cross_encoder", "admitted_count": 7,
                            "rerank_window_count": 5, "rerank_tail_count": 2,
                            "rerank_latency_ms": 1000.0, "rerank_cache_hits": 0,
                            "rerank_cache_misses": 5, "rerank_cache_hit_rate": 0.0,
                            "ranking_score_source": "rerank_raw_logit",
                        },
                        "error": None,
                    }
                ],
            }
        ],
    }


def test_report_renders_rerank_strategy_and_top_n():
    from evaluation.v11.report import render_markdown

    md = render_markdown(_report_with_rerank())
    assert "Rerank: **cross_encoder** top_n=5" in md
    assert "### Cross-Encoder Rerank" in md


def test_report_renders_fingerprint_and_cache_statistics():
    from evaluation.v11.report import render_markdown

    md = render_markdown(_report_with_rerank())
    assert "BAAI/bge-reranker-base" in md
    assert "2cfc18c9415c912f9d8155881c133215df768a70" in md
    assert "Rerank Cache Enabled: **False**" in md
    assert "Misses: 10" in md
    assert "Hit Rate: 0.0000" in md


def test_report_renders_rerank_latency_and_the_honesty_note():
    from evaluation.v11.report import render_markdown

    md = render_markdown(_report_with_rerank())
    assert "Rerank Latency (ms): mean=900.0000 (n=2)" in md
    assert "p95=990.0000 (n=2)" in md
    # cached latency 只能是資訊 —— 這句警語必須出現在報表上。
    assert "INFORMATIONAL ONLY" in md


def test_report_states_raw_logit_is_the_ranking_key():
    from evaluation.v11.report import render_markdown

    md = render_markdown(_report_with_rerank())
    assert "Scored Activation: **identity**" in md
    assert "raw logit" in md
    # per-result provenance：raw 與 normalized 並列，且標明 raw 才是排序主鍵。
    assert "rerank_raw=3.568463 norm=0.972574" in md
    assert "raw is the ranking key" in md
    # tail 明確標示為未計分、保留原順序。
    assert "rerank=tail (not scored; original order preserved)" in md


def test_report_omits_the_whole_rerank_section_when_disabled():
    from evaluation.v11.report import render_markdown

    report = _report_with_rerank()
    t = report["trials"][0]
    t["trial_name"] = "plain-trial"  # 名稱本身不得干擾下方的全文檢查
    t["rerank_stats"] = None
    t["config"]["retrieval_config"] = {"strategy": "dense", "ranking_version": "dense_v1"}
    t["aggregate_metrics"] = []
    t["cases"][0]["rerank_meta"] = None
    for r in t["cases"][0]["top_results"]:
        r.pop("rerank_applied")

    md = render_markdown(report)
    assert "Cross-Encoder Rerank" not in md
    assert "rerank" not in md.lower()
