"""Slice 5 §25.3 —— Hard Constraint Engine。"""
from __future__ import annotations

import pytest

from evaluation.v11.run_models import TrialMetric
from evaluation.v11.search import metric_names as mn
from evaluation.v11.search.constraints import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_WARN,
    MetricLookup,
    compare,
    evaluate_constraint,
    evaluate_constraints,
)
from tests.unit.search_helpers import make_search_config


def metric(name, value, *, scope="aggregate", query_type=None, computable=True,
           reason=None, n=10):
    return TrialMetric(
        trial_id="t", metric_name=name, scope=scope, query_type=query_type,
        value=value, sample_count=n, computable=computable,
        not_computable_reason=reason,
    )


def lookup(*metrics):
    return MetricLookup(metrics)


def constraint(**kw):
    raw = {"metric": "mrr", "operator": ">=", "value": 0.5, "safety_critical": False}
    raw.update(kw)
    return make_search_config(hard_constraints=[raw]).hard_constraints[0]


# --------------------------------------------------------------------------
# 運算子
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "obs,op,req,expected",
    [
        (0.5, ">=", 0.5, True), (0.49, ">=", 0.5, False),
        (0.6, ">", 0.5, True), (0.5, ">", 0.5, False),
        (0.25, "<=", 0.25, True), (0.26, "<=", 0.25, False),
        (0.2, "<", 0.25, True), (0.25, "<", 0.25, False),
        (1.0, "==", 1.0, True), (1.0, "==", 0.9, False),
    ],
)
def test_compare_operators(obs, op, req, expected):
    assert compare(obs, op, req) is expected


def test_full_precision_is_used_not_rounded_display():
    """§14.10：0.7499999 顯示成 0.75 也不得算通過。"""
    c = constraint(metric="exact_code_top1_accuracy", operator=">=", value=0.75)
    r = evaluate_constraint(c, lookup(metric("exact_code_top1_accuracy", 0.7499999999)))
    assert r.passed is False
    assert round(r.observed_value, 4) == 0.75  # 顯示層看起來剛好達標
    assert r.status == STATUS_FAIL


# --------------------------------------------------------------------------
# Absolute
# --------------------------------------------------------------------------
def test_absolute_pass_records_all_fields():
    c = constraint(metric="exact_code_top1_accuracy", operator=">=", value=0.75)
    r = evaluate_constraint(c, lookup(metric("exact_code_top1_accuracy", 0.8, n=4)))
    assert (r.status, r.passed, r.computable) == (STATUS_PASS, True, True)
    assert r.observed_value == 0.8
    assert r.required_value == 0.75
    assert r.baseline_value is None
    assert r.sample_count == 4
    assert r.is_relative is False


def test_unanswerable_fp_ceiling_rejects_regression():
    """Slice 4 的 floor=0.4 FP=0.5 必須被 FP<=0.25 擋下（deterministic fake）。"""
    c = constraint(
        metric="unanswerable_false_positive_rate", operator="<=", value=0.25,
        safety_critical=True, on_not_computable="fail",
    )
    bad = evaluate_constraint(c, lookup(metric("unanswerable_false_positive_rate", 0.5)))
    good = evaluate_constraint(c, lookup(metric("unanswerable_false_positive_rate", 0.0)))
    assert bad.passed is False and bad.status == STATUS_FAIL
    assert good.passed is True and good.status == STATUS_PASS


def test_p95_latency_ceiling():
    c = constraint(metric="p95_retrieval_latency_ms", operator="<=", value=86)
    assert evaluate_constraint(c, lookup(metric("p95_retrieval_latency_ms", 35.8))).passed
    assert not evaluate_constraint(c, lookup(metric("p95_retrieval_latency_ms", 120.0))).passed


# --------------------------------------------------------------------------
# Relative-to-baseline
# --------------------------------------------------------------------------
def test_relative_constraint_uses_baseline_plus_delta():
    c = constraint(
        metric="relevant_context_hit_at_5", scope="query_type", query_type="semantic",
        operator=">=", baseline_delta=-0.2, value=None,
    )
    base = lookup(metric("relevant_context_hit_at_5", 1.0, scope="query_type",
                         query_type="semantic"))
    ok = evaluate_constraint(
        c, lookup(metric("relevant_context_hit_at_5", 0.8, scope="query_type",
                         query_type="semantic")), base)
    bad = evaluate_constraint(
        c, lookup(metric("relevant_context_hit_at_5", 0.79, scope="query_type",
                         query_type="semantic")), base)
    assert ok.passed is True
    assert ok.baseline_value == 1.0 and ok.required_value == pytest.approx(0.8)
    assert ok.is_relative is True
    assert bad.passed is False


def test_relative_constraint_without_baseline_is_not_computable():
    c = constraint(metric="mrr", operator=">=", baseline_delta=-0.1, value=None,
                   on_not_computable="fail")
    r = evaluate_constraint(c, lookup(metric("mrr", 0.9)), None)
    assert r.computable is False and r.passed is False
    assert "baseline" in r.reason


def test_relative_constraint_with_non_computable_baseline_fails():
    c = constraint(metric="mrr", operator=">=", baseline_delta=-0.1, value=None,
                   on_not_computable="fail")
    base = lookup(metric("mrr", None, computable=False, reason="no_answerable_cases"))
    r = evaluate_constraint(c, lookup(metric("mrr", 0.9)), base)
    assert r.passed is False
    assert "baseline metric 不可計算" in r.reason


# --------------------------------------------------------------------------
# 不可計算政策（§14.5 / §14.6）
# --------------------------------------------------------------------------
def test_missing_metric_row_is_not_computable_not_a_pass():
    c = constraint(metric="exact_code_top1_accuracy", operator=">=", value=0.75,
                   safety_critical=True, on_not_computable="fail")
    r = evaluate_constraint(c, lookup())  # 完全沒有這筆 metric
    assert r.computable is False
    assert r.passed is False
    assert mn.REASON_METRIC_ABSENT in r.reason


@pytest.mark.parametrize(
    "policy,status,passed",
    [("fail", STATUS_FAIL, False), ("warn", STATUS_WARN, True), ("skip", STATUS_SKIP, True)],
)
def test_not_computable_policies(policy, status, passed):
    c = constraint(metric="mrr", operator=">=", value=0.5,
                   safety_critical=False, on_not_computable=policy)
    r = evaluate_constraint(
        c, lookup(metric("mrr", None, computable=False, reason="no_answerable_cases"))
    )
    assert (r.status, r.passed) == (status, passed)
    assert r.computable is False


def test_warn_and_skip_do_not_count_as_pass_for_ranking():
    """不可計算者不得在 constraint_pass_count 上取得不應得的優勢。"""
    for policy in ("warn", "skip"):
        c = constraint(metric="mrr", operator=">=", value=0.5,
                       safety_critical=False, on_not_computable=policy)
        r = evaluate_constraint(
            c, lookup(metric("mrr", None, computable=False, reason="x"))
        )
        assert r.passed is True
        assert r.counts_as_pass is False


def test_zero_is_never_substituted_for_not_computable():
    c = constraint(metric="unanswerable_false_positive_rate", operator="<=", value=0.25,
                   safety_critical=True, on_not_computable="fail")
    r = evaluate_constraint(
        c,
        lookup(metric("unanswerable_false_positive_rate", None, computable=False,
                      reason="no_unanswerable_cases")),
    )
    # 若被當成 0，<=0.25 會誤判成通過
    assert r.observed_value is None
    assert r.passed is False


# --------------------------------------------------------------------------
# 整組評估
# --------------------------------------------------------------------------
def test_evaluation_aggregate_feasibility_and_counts():
    cfg = make_search_config(
        hard_constraints=[
            {"name": "a", "metric": "mrr", "operator": ">=", "value": 0.5,
             "safety_critical": False},
            {"name": "b", "metric": "duplicate_ratio", "operator": "<=", "value": 0.1,
             "safety_critical": False},
        ]
    )
    ok = evaluate_constraints(
        cfg.hard_constraints, lookup(metric("mrr", 0.9), metric("duplicate_ratio", 0.0))
    )
    assert ok.feasible is True and ok.pass_count == 2 and ok.fail_count == 0

    bad = evaluate_constraints(
        cfg.hard_constraints, lookup(metric("mrr", 0.1), metric("duplicate_ratio", 0.9))
    )
    assert bad.feasible is False and bad.fail_count == 2
    assert bad.failed_names() == ["a", "b"]
    assert len(bad.rejection_reasons()) == 2


def test_constraint_results_preserve_sample_counts():
    c = constraint(metric="mrr", operator=">=", value=0.1)
    r = evaluate_constraint(c, lookup(metric("mrr", 0.9, n=7)))
    assert r.sample_count == 7
    assert r.to_dict()["sample_count"] == 7
