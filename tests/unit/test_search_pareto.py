"""Slice 5 §25.4 —— Pareto Frontier。"""
from __future__ import annotations

from evaluation.v11.run_models import TrialMetric
from evaluation.v11.search.constraints import MetricLookup
from evaluation.v11.search.pareto import compute_pareto, dominates
from evaluation.v11.search.search_config import (
    DIRECTION_MAXIMIZE,
    DIRECTION_MINIMIZE,
    MetricRef,
    ParetoObjective,
)

MAX_MRR = ParetoObjective(ref=MetricRef("mrr"), direction=DIRECTION_MAXIMIZE)
MIN_FP = ParetoObjective(
    ref=MetricRef("unanswerable_false_positive_rate"), direction=DIRECTION_MINIMIZE
)


def lk(**metrics):
    return MetricLookup(
        [
            TrialMetric(trial_id="t", metric_name=k, scope="aggregate", query_type=None,
                        value=v, sample_count=5, computable=v is not None,
                        not_computable_reason=None if v is not None else "no_computable_sample")
            for k, v in metrics.items()
        ]
    )


# --------------------------------------------------------------------------
# dominance 原語
# --------------------------------------------------------------------------
def test_dominates_strictly_better_on_one_not_worse_on_rest():
    assert dominates([1.0, 1.0], [0.5, 1.0], 1e-9) is True
    assert dominates([1.0, 1.0], [1.0, 1.0], 1e-9) is False  # 完全相同不算 dominate
    assert dominates([1.0, 0.5], [0.5, 1.0], 1e-9) is False  # 互有優劣


def test_dominance_respects_tolerance():
    assert dominates([1.0 + 1e-12, 1.0], [1.0, 1.0], 1e-9) is False
    assert dominates([1.1, 1.0], [1.0, 1.0], 1e-9) is True


# --------------------------------------------------------------------------
# frontier
# --------------------------------------------------------------------------
def test_a_dominates_b_so_only_a_is_member():
    r = compute_pareto(
        {
            "a": lk(mrr=1.0, unanswerable_false_positive_rate=0.0),
            "b": lk(mrr=0.5, unanswerable_false_positive_rate=0.5),
        },
        [MAX_MRR, MIN_FP],
    )
    by = {p.key: p for p in r.points}
    assert by["a"].pareto_member is True and by["a"].pareto_rank == 1
    assert by["b"].pareto_member is False and by["b"].pareto_rank == 2
    assert by["a"].dominates == ["b"]
    assert by["b"].dominated_by == ["a"]


def test_mutually_non_dominant_both_on_frontier():
    r = compute_pareto(
        {
            "a": lk(mrr=1.0, unanswerable_false_positive_rate=0.5),
            "b": lk(mrr=0.5, unanswerable_false_positive_rate=0.0),
        },
        [MAX_MRR, MIN_FP],
    )
    assert {p.key for p in r.frontier} == {"a", "b"}
    assert all(p.dominated_by == [] for p in r.points)


def test_identical_candidates_both_on_frontier():
    r = compute_pareto(
        {"a": lk(mrr=0.7, unanswerable_false_positive_rate=0.1),
         "b": lk(mrr=0.7, unanswerable_false_positive_rate=0.1)},
        [MAX_MRR, MIN_FP],
    )
    assert len(r.frontier) == 2


def test_minimize_direction_is_honoured():
    r = compute_pareto(
        {"low": lk(unanswerable_false_positive_rate=0.0),
         "high": lk(unanswerable_false_positive_rate=0.9)},
        [MIN_FP],
    )
    assert {p.key for p in r.frontier} == {"low"}


def test_result_is_deterministic():
    cands = {
        "c": lk(mrr=0.9, unanswerable_false_positive_rate=0.2),
        "a": lk(mrr=0.9, unanswerable_false_positive_rate=0.1),
        "b": lk(mrr=0.5, unanswerable_false_positive_rate=0.1),
    }
    first = compute_pareto(cands, [MAX_MRR, MIN_FP]).to_dict()
    for _ in range(5):
        assert compute_pareto(cands, [MAX_MRR, MIN_FP]).to_dict() == first


def test_point_order_is_sorted_by_config_hash():
    r = compute_pareto(
        {"zz": lk(mrr=0.5), "aa": lk(mrr=0.5), "mm": lk(mrr=0.5)}, [MAX_MRR]
    )
    assert [p.key for p in r.points] == ["aa", "mm", "zz"]


# --------------------------------------------------------------------------
# 不可計算 objective（§17.2 / §17.3）
# --------------------------------------------------------------------------
def test_exclude_candidate_policy_removes_candidate_not_treats_as_zero():
    r = compute_pareto(
        {"good": lk(mrr=0.5), "missing": lk(mrr=None)},
        [ParetoObjective(ref=MetricRef("mrr"), direction=DIRECTION_MAXIMIZE,
                         on_not_computable="exclude_candidate")],
    )
    by = {p.key: p for p in r.points}
    assert by["missing"].included is False
    assert by["missing"].pareto_member is False
    assert "不可計算" in by["missing"].exclusion_reason
    # 若被當成 0，"good" 仍會是 member —— 但 missing 不得因此被視為最差而「參與」比較
    assert [p.key for p in r.frontier] == ["good"]
    assert r.excluded[0]["config_hash"] == "missing"


def test_ignore_objective_policy_drops_objective_globally():
    r = compute_pareto(
        {"a": lk(mrr=0.9, unanswerable_false_positive_rate=None),
         "b": lk(mrr=0.5, unanswerable_false_positive_rate=0.0)},
        [
            MAX_MRR,
            ParetoObjective(
                ref=MetricRef("unanswerable_false_positive_rate"),
                direction=DIRECTION_MINIMIZE,
                on_not_computable="ignore_objective",
            ),
        ],
    )
    assert [o.ref.metric for o in r.active_objectives] == ["mrr"]
    assert len(r.ignored_objectives) == 1
    assert r.ignored_objectives[0]["affected_candidates"] == ["a"]
    # 只剩 mrr → a dominate b
    assert [p.key for p in r.frontier] == ["a"]


def test_empty_candidate_set_yields_empty_frontier():
    r = compute_pareto({}, [MAX_MRR])
    assert r.points == [] and r.frontier == []


def test_dominated_by_and_dominates_are_persisted():
    r = compute_pareto(
        {"top": lk(mrr=1.0), "mid": lk(mrr=0.6), "low": lk(mrr=0.2)}, [MAX_MRR]
    )
    by = {p.key: p for p in r.points}
    assert by["top"].dominates == ["low", "mid"]
    assert by["low"].dominated_by == ["mid", "top"]
    assert by["top"].pareto_rank == 1
    assert by["mid"].pareto_rank == 2
    assert by["low"].pareto_rank == 3


def test_no_opaque_weighted_score_in_output():
    r = compute_pareto({"a": lk(mrr=0.5)}, [MAX_MRR]).to_dict()
    blob = str(r)
    for banned in ("weighted_score", "total_score", "overall_score", "composite"):
        assert banned not in blob
