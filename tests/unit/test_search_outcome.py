"""Slice 5A §七 —— Test 驗收語意、Apply Eligibility、樣本粒度、latency tie-break。"""
from __future__ import annotations

import pytest

from evaluation.v11.run_models import TrialMetric
from evaluation.v11.search import outcome as oc
from evaluation.v11.search.constraints import MetricLookup, evaluate_constraints
from evaluation.v11.search.search_config import (
    DIRECTION_MAXIMIZE,
    DIRECTION_MINIMIZE,
    MetricRef,
    ParetoObjective,
)
from tests.unit.search_helpers import make_search_config

NDCG = MetricRef("graded_ndcg_at_5")
EXACT = MetricRef("exact_code_top1_accuracy")
FP = MetricRef("unanswerable_false_positive_rate")
MRR = MetricRef("mrr")
P95 = MetricRef("p95_retrieval_latency_ms")

OBJECTIVES = [
    ParetoObjective(ref=NDCG, direction=DIRECTION_MAXIMIZE),
    ParetoObjective(ref=EXACT, direction=DIRECTION_MAXIMIZE),
    ParetoObjective(ref=MRR, direction=DIRECTION_MAXIMIZE),
    ParetoObjective(ref=FP, direction=DIRECTION_MINIMIZE),
    ParetoObjective(ref=P95, direction=DIRECTION_MINIMIZE),
]

EXACT_FLOOR = {
    "name": "exact_code_top1_floor",
    "metric": "exact_code_top1_accuracy",
    "operator": ">=",
    "value": 0.75,
    "safety_critical": True,
    "on_not_computable": "fail",
}


def lk(**metrics):
    return MetricLookup([
        TrialMetric(trial_id="t", metric_name=k, scope="aggregate", query_type=None,
                    value=v, sample_count=2, computable=v is not None,
                    not_computable_reason=None if v is not None else "x")
        for k, v in metrics.items()
    ])


# --------------------------------------------------------------------------
# §7.1 絕對門檻失敗 ≠ 相對 baseline 退化
# --------------------------------------------------------------------------
def test_both_violate_same_absolute_constraint_is_failed_not_regressed():
    """baseline 與 recommendation 都 0.5（<0.75）→ constraint failed，但比較為 same。"""
    cfg = make_search_config(hard_constraints=[EXACT_FLOOR])
    baseline = lk(exact_code_top1_accuracy=0.5)
    rec = lk(exact_code_top1_accuracy=0.5)

    outcome = oc.evaluate_test_constraints(cfg.hard_constraints, rec, baseline)
    assert outcome.status == oc.TEST_CONSTRAINT_FAILED
    assert outcome.failed_test_constraints == ["exact_code_top1_floor"]

    cmp_ = oc.compare_to_baseline([EXACT], baseline, rec, OBJECTIVES)
    assert cmp_.status == oc.CMP_SAME, "兩者同分不得被判為相對退化"


def test_constraint_passed_when_recommendation_meets_absolute_threshold():
    cfg = make_search_config(hard_constraints=[EXACT_FLOOR])
    outcome = oc.evaluate_test_constraints(
        cfg.hard_constraints, lk(exact_code_top1_accuracy=1.0), lk(exact_code_top1_accuracy=0.5)
    )
    assert outcome.status == oc.TEST_CONSTRAINT_PASSED
    assert outcome.failed_test_constraints == []


def test_not_evaluated_when_no_test_metrics():
    cfg = make_search_config(hard_constraints=[EXACT_FLOOR])
    assert oc.evaluate_test_constraints(cfg.hard_constraints, None, None).status == (
        oc.TEST_CONSTRAINT_NOT_EVALUATED
    )


# --------------------------------------------------------------------------
# §7.2 baseline comparison
# --------------------------------------------------------------------------
def test_mixed_when_quality_improves_and_latency_regresses():
    """NDCG 改善、latency 退化、其餘持平 → mixed。"""
    baseline = lk(graded_ndcg_at_5=0.7021, exact_code_top1_accuracy=0.5,
                  unanswerable_false_positive_rate=0.0, mrr=0.8,
                  p95_retrieval_latency_ms=47.6)
    rec = lk(graded_ndcg_at_5=0.7847, exact_code_top1_accuracy=0.5,
             unanswerable_false_positive_rate=0.0, mrr=0.8,
             p95_retrieval_latency_ms=67.9)
    cmp_ = oc.compare_to_baseline(
        [NDCG, EXACT, FP, MRR, P95], baseline, rec, OBJECTIVES
    )
    assert cmp_.status == oc.CMP_MIXED
    by = {m.metric: m for m in cmp_.metrics}
    assert by["graded_ndcg_at_5"].comparison == oc.CMP_IMPROVED
    assert by["exact_code_top1_accuracy"].comparison == oc.CMP_SAME
    assert by["unanswerable_false_positive_rate"].comparison == oc.CMP_SAME
    assert by["mrr"].comparison == oc.CMP_SAME
    assert by["p95_retrieval_latency_ms"].comparison == oc.CMP_REGRESSED


def test_minimize_direction_lower_is_improved():
    cmp_ = oc.compare_to_baseline(
        [FP], lk(unanswerable_false_positive_rate=0.5),
        lk(unanswerable_false_positive_rate=0.0), OBJECTIVES,
    )
    assert cmp_.metrics[0].comparison == oc.CMP_IMPROVED
    assert cmp_.status == oc.CMP_IMPROVED


def test_all_same_yields_same():
    cmp_ = oc.compare_to_baseline([MRR], lk(mrr=0.8), lk(mrr=0.8), OBJECTIVES)
    assert cmp_.status == oc.CMP_SAME


def test_all_worse_yields_regressed():
    cmp_ = oc.compare_to_baseline(
        [NDCG, MRR], lk(graded_ndcg_at_5=0.9, mrr=0.9),
        lk(graded_ndcg_at_5=0.5, mrr=0.5), OBJECTIVES,
    )
    assert cmp_.status == oc.CMP_REGRESSED


def test_direction_inferred_when_not_an_objective():
    """未列為 objective 的 metric 仍需正確推斷方向。"""
    assert oc.metric_direction(MetricRef("p95_retrieval_latency_ms")) == DIRECTION_MINIMIZE
    assert oc.metric_direction(MetricRef("duplicate_ratio")) == DIRECTION_MINIMIZE
    assert oc.metric_direction(MetricRef("confusable_code_error_rate")) == DIRECTION_MINIMIZE
    assert oc.metric_direction(MetricRef("recall_at_5")) == DIRECTION_MAXIMIZE


# --------------------------------------------------------------------------
# §7.7 / §7.8 quality tie + latency 穩定性
# --------------------------------------------------------------------------
def _tied_candidates(n=8):
    """n 個品質完全相同、只有 latency 不同的 candidate。"""
    return {
        f"cand{i}": lk(graded_ndcg_at_5=0.9193, exact_code_top1_accuracy=1.0,
                       mrr=1.0, unanswerable_false_positive_rate=0.0,
                       p95_retrieval_latency_ms=40.0 + i)
        for i in range(n)
    }


def test_quality_tie_count_and_unstable_latency_selection():
    sel = oc.analyze_selection("cand0", _tied_candidates(8), OBJECTIVES)
    assert sel.quality_tie_candidate_count == 8
    assert sel.selection_basis == oc.BASIS_LATENCY_TIEBREAK
    assert sel.latency_measurement_repetitions == 1
    assert sel.latency_selection_stable is False
    # §六 要求的英文揭露文字必須照錄，並填入實際平手數。
    assert "All 8 feasible validation candidates were tied" in sel.note
    assert "determined by measured P95 latency" in sel.note
    assert "treated as unstable" in sel.note


def test_latency_reversal_is_stated_in_the_disclosure():
    sel = oc.analyze_selection(
        "cand0", _tied_candidates(8), OBJECTIVES, latency_order_reversed_on_test=True
    )
    assert "reversed relative ordering on the test split" in sel.note


def test_outcome_notes_do_not_collide_when_merged():
    """§二／§三／§六 的說明各自有唯一 key。

    這四個 dict 會被 build_outcome 以 ``**`` 合併成單一 outcome；若共用 "note"
    key，後合併者會靜默蓋掉前者，導致 §六 的 latency tie-break 揭露從報告消失。
    """
    test_outcome = oc.TestConstraintOutcome(
        status=oc.TEST_CONSTRAINT_FAILED, failed_test_constraints=["exact_code_top1_floor"]
    )
    comparison = oc.BaselineComparison(status=oc.CMP_MIXED, note="cmp")
    selection = oc.analyze_selection("cand0", _tied_candidates(8), OBJECTIVES)
    eligibility = _eligibility()

    dicts = [d.to_dict() for d in (test_outcome, comparison, selection, eligibility)]
    keys = [k for d in dicts for k in d]
    assert len(keys) == len(set(keys)), f"outcome key 衝突：{sorted(keys)}"

    merged = {}
    for d in dicts:
        merged.update(d)
    assert merged["selection_note"] == selection.note
    assert merged["test_constraint_note"] == test_outcome.note
    assert merged["baseline_comparison_note"] == comparison.note
    assert merged["apply_eligibility_note"] == eligibility.note


def test_quality_winner_is_marked_stable():
    cands = _tied_candidates(3)
    cands["winner"] = lk(graded_ndcg_at_5=0.99, exact_code_top1_accuracy=1.0, mrr=1.0,
                         unanswerable_false_positive_rate=0.0,
                         p95_retrieval_latency_ms=99.0)
    sel = oc.analyze_selection("winner", cands, OBJECTIVES)
    assert sel.quality_tie_candidate_count == 1
    assert sel.selection_basis == oc.BASIS_QUALITY_RANKING
    assert sel.latency_selection_stable is True


def test_latency_metrics_excluded_from_quality_tie():
    """latency 本身不得被算進 retrieval-quality 平手判定。"""
    sel = oc.analyze_selection("cand0", _tied_candidates(4), OBJECTIVES)
    assert sel.quality_tie_candidate_count == 4


def test_no_recommendation_yields_no_basis():
    sel = oc.analyze_selection(None, {}, OBJECTIVES)
    assert sel.selection_basis == oc.BASIS_NOT_SELECTED
    assert sel.latency_selection_stable is False


# --------------------------------------------------------------------------
# §7.5 apply eligibility
# --------------------------------------------------------------------------
def _eligibility(**over):
    cfg = make_search_config(hard_constraints=[EXACT_FLOOR])
    ce = evaluate_constraints(cfg.hard_constraints, lk(exact_code_top1_accuracy=1.0))
    kwargs = dict(
        recommended_config_hash="winner",
        validation_constraints=[ce],
        selection=oc.analyze_selection("cand0", _tied_candidates(8), OBJECTIVES),
        test_outcome=oc.TestConstraintOutcome(status=oc.TEST_CONSTRAINT_FAILED,
                                              failed_test_constraints=["exact_code_top1_floor"]),
    )
    kwargs.update(over)
    return oc.assess_apply_eligibility(**kwargs)


def test_recommendation_can_exist_while_not_apply_eligible():
    e = _eligibility()
    assert e.apply_eligible is False
    assert e.confidence == "low"
    for reason in (
        oc.REASON_SMALL_VALIDATION_SAMPLE,
        oc.REASON_QUALITY_METRICS_TIED,
        oc.REASON_SINGLE_RUN_LATENCY,
        oc.REASON_TEST_CONSTRAINT_FAILED,
        oc.REASON_DEMO_CORPUS_ONLY,
    ):
        assert reason in e.reasons, f"缺少原因 {reason}"
    assert "不代表取消 recommendation lock" in e.note


def test_demo_corpus_alone_still_blocks_apply():
    """即使沒有其他問題，demo corpus 也足以擋下 apply。"""
    cfg = make_search_config(hard_constraints=[EXACT_FLOOR])
    big = MetricLookup([
        TrialMetric(trial_id="t", metric_name="exact_code_top1_accuracy", scope="aggregate",
                    query_type=None, value=1.0, sample_count=50, computable=True)
    ])
    e = oc.assess_apply_eligibility(
        recommended_config_hash="w",
        validation_constraints=[evaluate_constraints(cfg.hard_constraints, big)],
        selection=oc.analyze_selection("w", {"w": big}, OBJECTIVES),
        test_outcome=oc.TestConstraintOutcome(status=oc.TEST_CONSTRAINT_PASSED),
    )
    assert e.apply_eligible is False
    assert e.reasons == [oc.REASON_DEMO_CORPUS_ONLY]


def test_no_recommendation_is_not_apply_eligible():
    e = _eligibility(recommended_config_hash=None)
    assert e.apply_eligible is False
    assert oc.REASON_NO_RECOMMENDATION in e.reasons


# --------------------------------------------------------------------------
# §7.6 樣本粒度
# --------------------------------------------------------------------------
def test_sample_granularity_exposes_possible_values():
    cfg = make_search_config(hard_constraints=[EXACT_FLOOR])
    g = oc.sample_granularity(
        {
            "validation": {"exact_code": 1, "unanswerable": 2, "semantic": 1, "procedure": 1},
            "test": {"exact_code": 2, "unanswerable": 2, "semantic": 1, "confusable": 2},
        },
        cfg.hard_constraints,
    )
    rows = {(r["split"], r["query_type"]): r for r in g["per_split"]}
    assert rows[("validation", "exact_code")]["n"] == 1
    assert rows[("validation", "exact_code")]["possible_values"] == [0.0, 1.0]
    assert rows[("validation", "unanswerable")]["n"] == 2
    assert rows[("validation", "unanswerable")]["possible_values"] == [0.0, 0.5, 1.0]
    assert rows[("test", "exact_code")]["n"] == 2
    assert rows[("test", "exact_code")]["possible_values"] == [0.0, 0.5, 1.0]

    joined = " ".join(g["notes"])
    assert "effectively requires 2/2 success" in joined
    assert "low statistical confidence" in joined


# --------------------------------------------------------------------------
# §五 identifier-bearing unanswerable 缺口
# --------------------------------------------------------------------------
def test_identifier_gap_is_disclosed_when_validation_has_none():
    cov = oc.identifier_unanswerable_coverage({
        "validation": [
            {"case_id": "salary", "identifiers": []},
            {"case_id": "stock", "identifiers": []},
        ],
        "test": [{"case_id": "price", "identifiers": ["CX500"]}],
    })
    assert cov["validation_has_identifier_bearing_unanswerable"] is False
    assert "do not contain an identifier that exists" in cov["limitation"]
    assert "identifier_vector_floor" in cov["limitation"]
    assert "held-out test group" in cov["required_followup"]


def test_no_gap_reported_when_validation_has_identifier_case():
    cov = oc.identifier_unanswerable_coverage({
        "validation": [{"case_id": "x", "identifiers": ["CX500"]}],
    })
    assert cov["validation_has_identifier_bearing_unanswerable"] is True
    assert cov["required_followup"] == ""
