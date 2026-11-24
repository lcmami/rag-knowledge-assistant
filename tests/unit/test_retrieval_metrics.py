"""Retrieval Metrics 單元測試（Slice 2 §17.3）。"""
from __future__ import annotations

import pytest

from evaluation.v11.case_schema import EvaluationCaseV11, EvidenceSpan
from evaluation.v11.metric_config import MetricConfig
from evaluation.v11.retrieval_metrics import (
    CaseRetrieval,
    IndexChunk,
    RankedResult,
    evaluate_trial,
)


# --- builders ---
def ev(cs=0, ce=100, *, sha="sha-a", ctv="v1", scope="document", section="S", page=None,
       anchors=("A",), eid=None) -> EvidenceSpan:
    return EvidenceSpan(document_sha256=sha, canonical_text_version=ctv, section=section,
                        page=page, offset_scope=scope, char_start=cs, char_end=ce,
                        anchor_text=list(anchors), evidence_id=eid)


def case(cid="c", qt="exact_code", answerable=True, evidence=None, q="E102 是什麼") -> EvaluationCaseV11:
    return EvaluationCaseV11(id=cid, question=q, query_type=qt, answerable=answerable,
                             group_id="g-" + cid, split="test", expected_evidence=evidence or [])


def prov(cs=0, ce=100, *, sha="sha-a", ctv="v1", scope="document", section="S", page=None):
    return {"document_sha256": sha, "canonical_text_version": ctv, "loader_version": "v1",
            "page": page, "section": section, "paragraph_index": 0, "offset_scope": scope,
            "char_start": cs, "char_end": ce}


def rr(rank, *, doc_id="doc-a", sha="sha-a", section="S", page=None, chunk_index=0,
       score=0.9, text="txt", provenance=None, identifiers=None) -> RankedResult:
    return RankedResult(rank=rank, document_id=doc_id, document_sha256=sha, filename="f.md",
                        section=section, page=page, chunk_index=chunk_index, score=score,
                        text=text, provenance_spans=provenance or [],
                        identifiers=identifiers or [], canonical_text_version="v1")


def run(case_retrievals, **kw):
    te = evaluate_trial(case_retrievals, **kw)
    return {(m.metric_name, m.scope, m.query_type): m for m in te.metrics}, te


def run_with_cfg(case_retrievals, cfg: MetricConfig, **kw):
    return run(case_retrievals, metric_config=cfg, **kw)


def m(metrics, name, scope="aggregate", qt=None):
    return metrics[(name, scope, qt)]


# --- Strong / Relevant hit + MRR ---
def test_strong_hit_at_1():
    c = case(evidence=[ev()])
    results = [rr(1, provenance=[prov(0, 100)], text="E102")]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "strong_evidence_hit_at_1").value == 1.0
    assert m(metrics, "relevant_context_hit_at_1").value == 1.0


def test_relevant_at_rank3_not_strong_at_1_and_mrr():
    c = case(evidence=[ev(section="S")])
    results = [
        rr(1, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")], text="n"),
        rr(2, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")], text="n"),
        rr(3, provenance=[prov(500, 600, section="S")], text="context only"),
    ]
    metrics, te = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "strong_evidence_hit_at_1").value == 0.0
    assert m(metrics, "relevant_context_hit_at_1").value == 0.0
    assert m(metrics, "relevant_context_hit_at_3").value == 1.0
    assert te.case_evaluations[0].first_relevant_rank == 3
    assert m(metrics, "mrr").value == pytest.approx(1 / 3, abs=1e-4)


# --- NDCG (trial_index_gold_v1) ---
# provenance 樣板：L2=完整覆蓋、L1=僅 section、L0=錯誤文件
_L2 = lambda: [prov(0, 100, section="S")]
_L1 = lambda: [prov(500, 600, section="S")]
_L0 = lambda: [prov(sha="sha-x", section="Z")]


def uc(provenance, *, chunk_index=0, doc_id="doc-a", sha="sha-a", ctv="v1", text="t", section="S"):
    return IndexChunk(document_id=doc_id, document_sha256=sha, chunk_index=chunk_index, text=text,
                      page=None, section=section, provenance_spans=provenance, identifiers=[],
                      canonical_text_version=ctv)


def _ndcg(te, i=0):
    return te.case_evaluations[i].ndcg


def test_ndcg_missing_high_relevance_chunk_is_penalized():
    c = case(evidence=[ev(0, 100, section="S")])
    universe = [uc(_L2(), chunk_index=0), uc(_L2(), chunk_index=1), uc(_L1(), chunk_index=2)]
    # 只取回 L1（取回集合內部排序無誤），但漏掉兩個 L2 → NDCG 必 < 1
    retrieved = [rr(1, chunk_index=2, provenance=_L1(), text="ctx")]
    metrics, te = run([CaseRetrieval(c, retrieved, 5)], index_universe=universe)
    nd = _ndcg(te)
    assert nd["computable"] is True and 0.0 < nd["value"] < 1.0
    assert nd["relevant_chunk_count"] == 3
    assert m(metrics, "graded_ndcg_at_5").value == pytest.approx(nd["value"])


def test_ndcg_perfect_ranking_is_one():
    c = case(evidence=[ev(0, 100, section="S")])
    universe = [uc(_L2(), chunk_index=0), uc(_L2(), chunk_index=1), uc(_L1(), chunk_index=2)]
    retrieved = [
        rr(1, chunk_index=0, provenance=_L2(), text="a"),
        rr(2, chunk_index=1, provenance=_L2(), text="b"),
        rr(3, chunk_index=2, provenance=_L1(), text="c"),
    ]
    _, te = run([CaseRetrieval(c, retrieved, 5)], index_universe=universe)
    assert _ndcg(te)["value"] == pytest.approx(1.0)


def test_ndcg_wrong_ranking_between_0_and_1():
    c = case(evidence=[ev(0, 100, section="S")])
    universe = [uc(_L2(), chunk_index=0), uc(_L1(), chunk_index=1), uc(_L0(), chunk_index=2)]
    # 取回順序 L0, L1, L2（相關者被壓在後面）→ 0 < NDCG < 1
    retrieved = [
        rr(1, chunk_index=2, sha="sha-x", doc_id="doc-x", provenance=_L0(), text="n"),
        rr(2, chunk_index=1, provenance=_L1(), text="c"),
        rr(3, chunk_index=0, provenance=_L2(), text="a"),
    ]
    _, te = run([CaseRetrieval(c, retrieved, 5)], index_universe=universe)
    assert 0.0 < _ndcg(te)["value"] < 1.0


def test_ndcg_total_miss_is_zero_but_computable():
    c = case(evidence=[ev(0, 100, section="S")])
    universe = [uc(_L2(), chunk_index=0), uc(_L1(), chunk_index=1)]
    retrieved = [rr(1, chunk_index=9, sha="sha-x", doc_id="doc-x", provenance=_L0(), text="n")]
    _, te = run([CaseRetrieval(c, retrieved, 5)], index_universe=universe)
    nd = _ndcg(te)
    assert nd["computable"] is True and nd["value"] == 0.0
    assert nd["dcg"] == 0.0 and nd["idcg"] > 0.0


def test_ndcg_not_computable_no_relevant_chunk_in_index():
    c = case(evidence=[ev(0, 100, section="S")])
    universe = [uc(_L0(), chunk_index=0, sha="sha-x"), uc(_L0(), chunk_index=1, sha="sha-x")]
    retrieved = [rr(1, sha="sha-x", doc_id="doc-x", provenance=_L0(), text="n")]
    _, te = run([CaseRetrieval(c, retrieved, 5)], index_universe=universe)
    nd = _ndcg(te)
    assert nd["computable"] is False
    assert nd["not_computable_reason"] == "no_relevant_chunks_in_trial_index"


def test_ndcg_not_computable_unanswerable():
    c = case(cid="u", qt="unanswerable", answerable=False, evidence=[])
    universe = [uc(_L2(), chunk_index=0)]
    _, te = run([CaseRetrieval(c, [], 5)], index_universe=universe)
    assert _ndcg(te)["not_computable_reason"] == "unanswerable_case"


def test_ndcg_not_computable_version_mismatch():
    c = case(evidence=[ev(0, 100, section="S", ctv="v2")])  # gold 綁 v2
    universe = [uc(_L2(), chunk_index=0, ctv="v1")]  # index 只有 v1
    retrieved = [rr(1, provenance=_L2(), text="a")]
    _, te = run([CaseRetrieval(c, retrieved, 5)], index_universe=universe)
    assert _ndcg(te)["not_computable_reason"] == "version_mismatch"


def test_ndcg_not_computable_anchor_only():
    c = case(evidence=[ev(cs=None, ce=None, anchors=("E102",))])
    universe = [uc(_L1(), chunk_index=0)]
    retrieved = [rr(1, provenance=[prov(0, 100)], text="E102")]
    _, te = run([CaseRetrieval(c, retrieved, 5)], index_universe=universe)
    assert _ndcg(te)["not_computable_reason"] == "insufficient_graded_evidence"


def test_ndcg_missing_index_universe():
    c = case(evidence=[ev(0, 100, section="S")])
    retrieved = [rr(1, provenance=_L2(), text="a")]
    _, te = run([CaseRetrieval(c, retrieved, 5)])  # 不傳 index_universe
    assert _ndcg(te)["not_computable_reason"] == "missing_index_universe"


def test_ndcg_aggregate_averages_only_computable():
    c_ok = case(cid="ok", evidence=[ev(0, 100, section="S")])
    c_un = case(cid="un", qt="unanswerable", answerable=False, evidence=[])
    universe = [uc(_L2(), chunk_index=0), uc(_L1(), chunk_index=1)]
    metrics, te = run([
        CaseRetrieval(c_ok, [rr(1, chunk_index=0, provenance=_L2(), text="a"),
                             rr(2, chunk_index=1, provenance=_L1(), text="b")], 5),
        CaseRetrieval(c_un, [rr(1, text="x")], 5),
    ], index_universe=universe)
    agg = m(metrics, "graded_ndcg_at_5")
    ok_val = te.case_evaluations[0].ndcg["value"]
    assert agg.value == pytest.approx(ok_val)  # 只平均 computable case（不含 unanswerable）
    assert m(metrics, "graded_ndcg_computable_count").value == 1.0
    assert m(metrics, "graded_ndcg_non_computable_count").value == 1.0


# --- Evidence Coverage ---
def test_evidence_coverage_union_no_double_count():
    c = case(evidence=[ev(0, 100, section="S")])
    # 兩個 level2 result（各覆蓋 0.6），provenance span 交疊 → union 覆蓋 = 1.0（非 1.2）
    results = [
        rr(1, chunk_index=0, provenance=[prov(0, 60, section="S")], text="a"),
        rr(2, chunk_index=1, provenance=[prov(40, 100, section="S")], text="b"),
    ]
    metrics, te = run([CaseRetrieval(c, results, 5)])
    assert te.case_evaluations[0].scored[0].relevance_level == 2
    cov = m(metrics, "evidence_coverage_at_5")
    assert cov.computable is True
    assert cov.value == pytest.approx(1.0, abs=1e-6)


def test_evidence_coverage_anchor_only_not_computable():
    c = case(evidence=[ev(cs=None, ce=None, anchors=("E102",))])
    results = [rr(1, provenance=[prov(0, 100)], text="E102")]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    cov = m(metrics, "evidence_coverage_at_5")
    assert cov.computable is False and cov.value is None


# --- Relevant Section Precision@5 ---
def test_relevant_section_precision_denominator_fixed_5():
    c = case(evidence=[ev(section="S")])
    # 只有 3 筆結果，其中 2 筆相關 → precision = 2/5（缺位視為不相關）
    results = [
        rr(1, provenance=[prov(0, 100, section="S")], text="x"),
        rr(2, provenance=[prov(0, 100, section="S")], text="y", chunk_index=1),
        rr(3, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")], text="n"),
    ]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "relevant_section_precision_at_5").value == pytest.approx(0.4)


# --- Exact-code Top-1 ---
def test_exact_code_top1_hit_with_identifier_match():
    c = case(qt="exact_code", evidence=[ev(section="S")], q="E102 是什麼")
    results = [rr(1, provenance=[prov(0, 100, section="S")], identifiers=["E102"], text="E102")]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "exact_code_top1_accuracy").value == 1.0


def test_exact_code_top1_fails_on_identifier_mismatch():
    c = case(qt="exact_code", evidence=[ev(section="S")], q="E102 是什麼")
    # top1 相關（section）但識別碼是 E999 ≠ 查詢 E102 → 判失敗
    results = [rr(1, provenance=[prov(0, 100, section="S")], identifiers=["E999"], text="E999")]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "exact_code_top1_accuracy").value == 0.0


# --- Confusable error ---
def test_confusable_error_wrong_id_ranked_above_relevant():
    c = case(cid="cf", qt="confusable", evidence=[ev(section="S")], q="E102 通訊模組")
    results = [
        rr(1, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")],
           identifiers=["D900"], text="D900"),
        rr(2, provenance=[prov(0, 100, section="S")], identifiers=["E102"], text="E102"),
    ]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "confusable_code_error_count").value == 1.0
    assert m(metrics, "confusable_code_error_rate").value == 1.0


def test_confusable_error_no_relevant_but_other_id():
    c = case(cid="cf2", qt="confusable", evidence=[ev(section="S")], q="E102 通訊模組")
    results = [
        rr(1, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")],
           identifiers=["D900"], text="D900"),
    ]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "confusable_code_error_count").value == 1.0


# --- Unanswerable false positive ---
def test_unanswerable_false_positive():
    c_fp = case(cid="u1", qt="unanswerable", answerable=False, evidence=[])
    c_ok = case(cid="u2", qt="unanswerable", answerable=False, evidence=[])
    metrics, _ = run([
        CaseRetrieval(c_fp, [rr(1, text="something")], 5),
        CaseRetrieval(c_ok, [], 5),
    ])
    assert m(metrics, "unanswerable_false_positive_count").value == 1.0
    assert m(metrics, "unanswerable_false_positive_rate").value == 0.5


# --- Duplicate / near-duplicate ---
def test_exact_duplicate_ratio():
    c = case(evidence=[ev(section="S")])
    results = [
        rr(1, doc_id="doc-a", chunk_index=7, provenance=[prov(0, 100, section="S")], text="A"),
        rr(2, doc_id="doc-a", chunk_index=7, provenance=[prov(0, 100, section="S")], text="A"),
    ]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "duplicate_ratio").value == pytest.approx(0.25)


def test_near_duplicate_ratio_same_section():
    c = case(evidence=[ev(section="S")])
    results = [
        rr(1, doc_id="doc-a", section="S", chunk_index=0, text="感測器通訊異常請檢查線路",
           provenance=[prov(0, 100, section="S")]),
        rr(2, doc_id="doc-a", section="S", chunk_index=1, text="感測器通訊異常請檢查線路",
           provenance=[prov(0, 100, section="S")]),
    ]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "duplicate_ratio").value == pytest.approx(0.25)


def test_different_section_not_near_duplicate():
    c = case(evidence=[ev(section="S")])
    results = [
        rr(1, doc_id="doc-a", section="S1", chunk_index=0, text="相同文字內容",
           provenance=[prov(0, 100, section="S")]),
        rr(2, doc_id="doc-a", section="S2", chunk_index=1, text="相同文字內容",
           provenance=[prov(0, 100, section="S")]),
    ]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    assert m(metrics, "duplicate_ratio").value == 0.0


# --- edge cases ---
def test_empty_dataset_no_crash():
    metrics, te = run([])
    assert te.case_evaluations == []
    assert m(metrics, "mrr").computable is False
    assert m(metrics, "mrr").value is None


def test_zero_answerable_denominator():
    c = case(cid="u", qt="unanswerable", answerable=False, evidence=[])
    metrics, _ = run([CaseRetrieval(c, [], 5)])
    assert m(metrics, "relevant_context_hit_at_1").computable is False
    assert m(metrics, "unanswerable_false_positive_rate").computable is True


def test_per_query_type_aggregation():
    c1 = case(cid="a", qt="exact_code", evidence=[ev(section="S")])
    c2 = case(cid="b", qt="semantic", evidence=[ev(section="S")], q="語意查詢")
    metrics, _ = run([
        CaseRetrieval(c1, [rr(1, provenance=[prov(0, 100, section="S")], text="x")], 5),
        CaseRetrieval(c2, [rr(1, provenance=[prov(0, 100, section="S")], text="y")], 5),
    ])
    assert m(metrics, "case_count", scope="query_type", qt="exact_code").value == 1.0
    assert m(metrics, "case_count", scope="query_type", qt="semantic").value == 1.0
    assert m(metrics, "relevant_context_hit_at_1", scope="query_type", qt="exact_code").value == 1.0


def test_latency_percentiles():
    cases = []
    for i, lat in enumerate([10, 20, 30, 40, 50]):
        c = case(cid=f"c{i}", evidence=[ev(section="S")])
        cases.append(CaseRetrieval(c, [rr(1, provenance=[prov(0, 100, section="S")], text="x")], lat))
    metrics, _ = run(cases)
    assert m(metrics, "mean_retrieval_latency_ms").value == pytest.approx(30.0)
    assert m(metrics, "p50_retrieval_latency_ms").value == pytest.approx(30.0)
    assert m(metrics, "p95_retrieval_latency_ms").value == pytest.approx(48.0)


def test_index_point_count_and_build_time_metrics():
    c = case(evidence=[ev(section="S")])
    metrics, _ = run([CaseRetrieval(c, [rr(1, provenance=[prov(0, 100, section="S")], text="x")], 5)],
                     point_count=10, index_build_time_ms=123)
    assert m(metrics, "index_point_count").value == 10.0
    assert m(metrics, "index_build_time_ms").value == 123.0


# --- v1.1 retrieval metric semantic fixes ---
# recall_at_* 已從 v1.1 retrieval evaluator 移除（與 relevant_context_hit_at_* 重複）。
# Hit@K 與 NDCG 維持原本固定 cutoff 行為：candidate 因為較小 top_k 而無法取得較後順位
# relevant result，會反映為 metric 較差，而不是 metric 不存在或 non-computable。
def test_no_recall_at_metrics_emitted():
    c = case(evidence=[ev()])
    results = [rr(1, provenance=[prov(0, 100)], text="E102")]
    metrics, _ = run([CaseRetrieval(c, results, 5)])
    for k in (1, 3, 5):
        assert (f"recall_at_{k}", "aggregate", None) not in metrics


def test_aggregate_hit_at_k_fixed_cutoff_when_top_k_3():
    """cfg.top_k=3 時：aggregate Hit@1 / Hit@3 / Hit@5 仍固定輸出。

    rank 1 已 relevant → Hit@1 = Hit@3 = Hit@5 = 1（Hit@K 是 top-K 內是否
    存在至少一筆 relevant 的 case-level 指標；rank 1 命中時所有 K >= 1 都為 1）。
    """
    cfg = MetricConfig(top_k=3, ndcg_k=5)
    c = case(evidence=[ev()])
    results = [rr(1, provenance=[prov(0, 100)], text="E102")]
    metrics, _ = run_with_cfg([CaseRetrieval(c, results, 5)], cfg)
    assert m(metrics, "strong_evidence_hit_at_1").value == 1.0
    assert m(metrics, "strong_evidence_hit_at_3").value == 1.0
    assert m(metrics, "strong_evidence_hit_at_5").value == 1.0
    assert m(metrics, "relevant_context_hit_at_1").value == 1.0
    assert m(metrics, "relevant_context_hit_at_3").value == 1.0
    assert m(metrics, "relevant_context_hit_at_5").value == 1.0


def test_aggregate_hit_at_k_zero_when_top_k_3_no_relevant_in_retrieved():
    """cfg.top_k=3、實際 retrieved rank 1~3 全部 irrelevant 時：
    Hit@3 = Hit@5 = 0（Hit@5 與 Hit@3 相同，因為 candidate 沒有 rank 4、5）。

    註：理論上若 index universe 中存在 rank 4 的 relevant chunk，cfg.top_k=3 的
    serving configuration 不會回傳它，因此 Hit@5 自然為 0；不需在 CaseRetrieval
    人工塞 rank 4。
    """
    cfg = MetricConfig(top_k=3, ndcg_k=5)
    c = case(evidence=[ev(section="S")])
    results = [
        rr(1, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")], text="n"),
        rr(2, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")], text="n"),
        rr(3, sha="sha-x", doc_id="doc-x", provenance=[prov(sha="sha-x", section="Z")], text="n"),
    ]
    metrics, _ = run_with_cfg([CaseRetrieval(c, results, 5)], cfg)
    assert m(metrics, "relevant_context_hit_at_3").value == 0.0
    assert m(metrics, "relevant_context_hit_at_5").value == 0.0
    assert m(metrics, "strong_evidence_hit_at_3").value == 0.0
    assert m(metrics, "strong_evidence_hit_at_5").value == 0.0


def test_per_query_type_hit_at_k_fixed_cutoff_when_top_k_3():
    """cfg.top_k=3 時：per-query-type Hit@1 / Hit@5 仍固定輸出。

    兩個 case 的 rank 1 都已 relevant → Hit@1 = Hit@5 = 1（Hit@K 是 case-level
    指標，rank 1 命中時所有 K >= 1 都為 1）。
    """
    cfg = MetricConfig(top_k=3, ndcg_k=5)
    c1 = case(cid="a", qt="exact_code", evidence=[ev(section="S")])
    c2 = case(cid="b", qt="semantic", evidence=[ev(section="S")], q="語意查詢")
    metrics, _ = run_with_cfg([
        CaseRetrieval(c1, [rr(1, provenance=[prov(0, 100, section="S")], text="x")], 5),
        CaseRetrieval(c2, [rr(1, provenance=[prov(0, 100, section="S")], text="y")], 5),
    ], cfg)
    assert m(metrics, "relevant_context_hit_at_1", scope="query_type", qt="exact_code").value == 1.0
    assert m(metrics, "relevant_context_hit_at_5", scope="query_type", qt="exact_code").value == 1.0
    assert m(metrics, "relevant_context_hit_at_5", scope="query_type", qt="semantic").value == 1.0
    # MRR / case_count 行為完全不變
    assert m(metrics, "mrr", scope="query_type", qt="exact_code").value == 1.0
    assert m(metrics, "case_count", scope="query_type", qt="exact_code").value == 1.0


def test_ndcg_unchanged_when_ndcg_k_gt_top_k():
    """cfg.ndcg_k=5、cfg.top_k=3 時：NDCG 仍依原本規則計算（DCG 用實際最多 3 筆，
    IDCG 用固定 NDCG@5 ideal ranking），不因 retrieval depth 較小而標成 non-computable。

    本 case：candidate 完整取回 index universe 中所有 3 個 relevant chunk → NDCG@5 = 1.0。
    用來確認 ndcg_k > top_k 不會自動變 non-computable，且完整取回時仍可得到 NDCG=1。
    """
    cfg = MetricConfig(top_k=3, ndcg_k=5)
    c = case(evidence=[ev(0, 100, section="S")])
    universe = [uc(_L2(), chunk_index=0), uc(_L2(), chunk_index=1), uc(_L1(), chunk_index=2)]
    retrieved = [
        rr(1, chunk_index=0, provenance=_L2(), text="a"),
        rr(2, chunk_index=1, provenance=_L2(), text="b"),
        rr(3, chunk_index=2, provenance=_L1(), text="c"),
    ]
    metrics, _ = run_with_cfg([CaseRetrieval(c, retrieved, 5)], cfg, index_universe=universe)
    ndcg = m(metrics, "graded_ndcg_at_5")
    assert ndcg.computable is True
    assert ndcg.value == pytest.approx(1.0)
    assert ndcg.not_computable_reason is None
    assert m(metrics, "graded_ndcg_computable_count").value == 1.0
    assert m(metrics, "graded_ndcg_non_computable_count").value == 0.0


def test_ndcg_truncation_penalty_when_top_k_3_misses_relevant():
    """cfg.top_k=3、cfg.ndcg_k=5、index universe 有 4 個正 gain relevant chunk，
    candidate 只取回 3 筆（漏掉 1 個）→ graded_ndcg_at_5 computable=True 且 value < 1.0。

    目的：確認較小 serving top_k 漏掉 index universe 中仍存在的 relevant evidence
    會透過固定 NDCG@5 自然受到懲罰，而不是變成 non-computable。
    """
    cfg = MetricConfig(top_k=3, ndcg_k=5)
    c = case(evidence=[ev(0, 100, section="S")])
    # 4 個 L2 relevant chunk（gain=3），candidate 只取回前 3 個 → 漏掉 chunk_index=3
    universe = [
        uc(_L2(), chunk_index=0),
        uc(_L2(), chunk_index=1),
        uc(_L2(), chunk_index=2),
        uc(_L2(), chunk_index=3),
    ]
    retrieved = [
        rr(1, chunk_index=0, provenance=_L2(), text="a"),
        rr(2, chunk_index=1, provenance=_L2(), text="b"),
        rr(3, chunk_index=2, provenance=_L2(), text="c"),
    ]
    metrics, _ = run_with_cfg([CaseRetrieval(c, retrieved, 5)], cfg, index_universe=universe)
    ndcg = m(metrics, "graded_ndcg_at_5")
    assert ndcg.computable is True
    assert ndcg.not_computable_reason is None
    assert 0.0 <= ndcg.value < 1.0
