"""Slice 5 §25.5 —— Recommendation Policy / lexicographic tie-break / no-feasible。"""
from __future__ import annotations

from evaluation.v11.run_models import TrialMetric
from evaluation.v11.search import complexity as complexity_mod
from evaluation.v11.search.constraints import MetricLookup, evaluate_constraints
from evaluation.v11.search.recommendation import (
    STATUS_NO_FEASIBLE_CANDIDATE,
    STATUS_OK,
    RankableCandidate,
    rank_candidates,
    select_recommendation,
)
from evaluation.v11.search.search_config import (
    DIRECTION_MAXIMIZE,
    DIRECTION_MINIMIZE,
    MetricRef,
    SortKey,
)
from tests.unit.search_helpers import make_search_config

DENSE = {"strategy": "dense", "top_k": 5, "score_threshold": 0.55,
         "candidate_multiplier": 3, "candidate_cap": 50, "ranking_version": "dense_v1",
         "schema_version": "1.0"}
IDA = {**DENSE, "strategy": "identifier_aware", "ranking_version": "identifier_aware_v1",
       "identifier_candidate_cap": 20, "exact_match_boost": 0.2,
       "confusable_penalty": 0.15, "identifier_vector_floor": 0.4}
CUR = {"strategy": "current", "target_chars": 600, "overlap_chars": 100,
       "max_tokens": 480, "prepend_section_title": False,
       "context_before_paragraphs": 0, "context_after_paragraphs": 0,
       "prefix_format_version": "v1", "schema_version": "1.0"}
HEAD = {**CUR, "strategy": "heading", "prepend_section_title": True}

MRR_DESC = SortKey("mrr", DIRECTION_MAXIMIZE, MetricRef("mrr"))
COMPLEXITY_ASC = SortKey("config_complexity", DIRECTION_MINIMIZE)
HASH_ASC = SortKey("config_hash", DIRECTION_MINIMIZE)


def lk(**metrics):
    return MetricLookup(
        [
            TrialMetric(trial_id="t", metric_name=k, scope="aggregate", query_type=None,
                        value=v, sample_count=5, computable=v is not None,
                        not_computable_reason=None if v is not None else "x")
            for k, v in metrics.items()
        ]
    )


def cand(h, *, mrr=None, chunk=None, retrieval=None, ce=None):
    return RankableCandidate(
        config_hash=h, metrics=lk(mrr=mrr), constraint_eval=ce,
        chunk_config=chunk or CUR, retrieval_config=retrieval or DENSE,
    )


# --------------------------------------------------------------------------
# 排序
# --------------------------------------------------------------------------
def test_primary_metric_decides_before_tiebreaks():
    ranked = rank_candidates(
        [cand("zzz", mrr=0.9), cand("aaa", mrr=0.5)], [MRR_DESC, COMPLEXITY_ASC, HASH_ASC]
    )
    assert [r.config_hash for r in ranked] == ["zzz", "aaa"]


def test_complexity_breaks_tie_only_when_metrics_equal():
    """dense+current 比 identifier_aware+heading 簡單。"""
    ranked = rank_candidates(
        [
            cand("bbb", mrr=0.9, chunk=HEAD, retrieval=IDA),
            cand("aaa", mrr=0.9, chunk=CUR, retrieval=DENSE),
        ],
        [MRR_DESC, COMPLEXITY_ASC, HASH_ASC],
    )
    assert ranked[0].config_hash == "aaa"


def test_complexity_never_overrides_quality():
    """較複雜但 mrr 較好者仍應勝出。"""
    ranked = rank_candidates(
        [
            cand("aaa", mrr=0.5, chunk=CUR, retrieval=DENSE),
            cand("bbb", mrr=0.9, chunk=HEAD, retrieval=IDA),
        ],
        [MRR_DESC, COMPLEXITY_ASC, HASH_ASC],
    )
    assert ranked[0].config_hash == "bbb"


def test_config_hash_is_final_deterministic_tiebreak():
    ranked = rank_candidates(
        [cand("ccc", mrr=0.5), cand("aaa", mrr=0.5), cand("bbb", mrr=0.5)],
        [MRR_DESC, COMPLEXITY_ASC, HASH_ASC],
    )
    assert [r.config_hash for r in ranked] == ["aaa", "bbb", "ccc"]


def test_ranking_is_stable_across_input_orderings():
    a, b, c = cand("aaa", mrr=0.5), cand("bbb", mrr=0.9), cand("ccc", mrr=0.5)
    keys = [MRR_DESC, COMPLEXITY_ASC, HASH_ASC]
    expected = [r.config_hash for r in rank_candidates([a, b, c], keys)]
    for order in ([c, b, a], [b, a, c], [c, a, b]):
        assert [r.config_hash for r in rank_candidates(order, keys)] == expected


def test_non_computable_metric_sorts_last_regardless_of_direction():
    for direction in (DIRECTION_MAXIMIZE, DIRECTION_MINIMIZE):
        key = SortKey("mrr", direction, MetricRef("mrr"))
        ranked = rank_candidates(
            [cand("aaa", mrr=None), cand("bbb", mrr=0.5)], [key, HASH_ASC]
        )
        assert ranked[-1].config_hash == "aaa"


def test_constraint_pass_count_sort_key():
    cfg = make_search_config(
        hard_constraints=[
            {"name": "x", "metric": "mrr", "operator": ">=", "value": 0.5,
             "safety_critical": False}
        ]
    )
    good = evaluate_constraints(cfg.hard_constraints, lk(mrr=0.9))
    bad = evaluate_constraints(cfg.hard_constraints, lk(mrr=0.1))
    ranked = rank_candidates(
        [cand("aaa", mrr=0.1, ce=bad), cand("bbb", mrr=0.9, ce=good)],
        [SortKey("constraint_pass_count", DIRECTION_MAXIMIZE), HASH_ASC],
    )
    assert ranked[0].config_hash == "bbb"


# --------------------------------------------------------------------------
# Recommendation 選擇
# --------------------------------------------------------------------------
def test_selects_best_eligible_and_records_tiebreak_trace():
    r = select_recommendation(
        [cand("aaa", mrr=0.5), cand("bbb", mrr=0.9)],
        [MRR_DESC, COMPLEXITY_ASC, HASH_ASC],
    )
    assert r.status == STATUS_OK
    assert r.recommended_config_hash == "bbb"
    assert r.tie_break_trace
    steps = r.tie_break_trace[0]["steps"]
    assert steps[0]["key"] == "mrr" and steps[0]["decided"] is True


def test_no_eligible_candidate_yields_null_recommendation():
    r = select_recommendation([], [MRR_DESC, HASH_ASC])
    assert r.status == STATUS_NO_FEASIBLE_CANDIDATE
    assert r.recommended_config_hash is None
    assert r.ranking == []
    assert "不得從不合格 candidate 中硬選" in r.reason


def test_infeasible_candidates_are_never_recommended():
    cfg = make_search_config(
        hard_constraints=[
            {"name": "x", "metric": "mrr", "operator": ">=", "value": 0.9,
             "safety_critical": False}
        ]
    )
    bad = evaluate_constraints(cfg.hard_constraints, lk(mrr=0.1))
    r = select_recommendation(
        [], [MRR_DESC, HASH_ASC], infeasible=[cand("aaa", mrr=0.1, ce=bad)]
    )
    assert r.recommended_config_hash is None
    assert r.closest_infeasible
    entry = r.closest_infeasible[0]
    assert entry["config_hash"] == "aaa"
    assert entry["recommended"] is False
    assert entry["fail_count"] == 1


def test_closest_infeasible_sorted_by_fewest_failures():
    cfg = make_search_config(
        hard_constraints=[
            {"name": "a", "metric": "mrr", "operator": ">=", "value": 0.9,
             "safety_critical": False},
            {"name": "b", "metric": "duplicate_ratio", "operator": "<=", "value": 0.0,
             "safety_critical": False},
        ]
    )
    one_fail = RankableCandidate(
        "one", lk(mrr=0.95, duplicate_ratio=0.5), evaluate_constraints(
            cfg.hard_constraints, lk(mrr=0.95, duplicate_ratio=0.5)),
        CUR, DENSE)
    two_fail = RankableCandidate(
        "two", lk(mrr=0.1, duplicate_ratio=0.5), evaluate_constraints(
            cfg.hard_constraints, lk(mrr=0.1, duplicate_ratio=0.5)),
        CUR, DENSE)
    r = select_recommendation([], [MRR_DESC, HASH_ASC], infeasible=[two_fail, one_fail])
    assert [e["config_hash"] for e in r.closest_infeasible] == ["one", "two"]


# --------------------------------------------------------------------------
# Complexity 測度
# --------------------------------------------------------------------------
def test_complexity_ordering_matches_documented_rules():
    assert complexity_mod.complexity_vector(CUR, DENSE) < complexity_mod.complexity_vector(CUR, IDA)
    assert complexity_mod.complexity_vector(CUR, DENSE) < complexity_mod.complexity_vector(HEAD, DENSE)
    para = {**CUR, "strategy": "paragraph"}
    assert complexity_mod.complexity_vector(para, DENSE) < complexity_mod.complexity_vector(HEAD, DENSE)


def test_complexity_describe_is_explainable():
    d = complexity_mod.describe(HEAD, IDA)
    assert d["complexity_version"] == complexity_mod.COMPLEXITY_VERSION
    assert set(d["components"]) == {
        "retrieval_strategy_rank", "chunk_strategy_rank",
        "prefix_or_context_rank", "non_default_param_count",
    }
