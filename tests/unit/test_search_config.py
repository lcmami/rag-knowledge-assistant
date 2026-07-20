"""Slice 5 §25.1 —— Search Config schema / 驗證 / hash。"""
from __future__ import annotations

import json
import os

import pytest

from evaluation.v11.hashing import repo_root
from evaluation.v11.search.search_config import (
    SEARCH_CONFIG_SCHEMA_VERSION,
    SearchConfig,
    SearchConfigError,
)
from tests.unit.search_helpers import make_search_config, raw_search_config


# --------------------------------------------------------------------------
# 基本合法性
# --------------------------------------------------------------------------
def test_minimal_config_parses():
    cfg = make_search_config()
    assert cfg.search_mode == "grid"
    assert cfg.schema_version == SEARCH_CONFIG_SCHEMA_VERSION
    # baseline 一律展開成 canonical（current → 明確 600/100/480，不依 .env）
    assert cfg.baseline_chunk_config["target_chars"] == 600
    assert cfg.baseline_chunk_config["overlap_chars"] == 100
    assert cfg.baseline_chunk_config["max_tokens"] == 480
    assert cfg.baseline_retrieval_config["ranking_version"] == "dense_v1"


def test_shipped_slice5_smoke_config_is_valid():
    path = os.path.join(repo_root(), "evaluation", "configs", "slice5-optimizer-smoke.json")
    with open(path, encoding="utf-8") as f:
        cfg = SearchConfig.from_raw(json.load(f))
    assert cfg.search_name == "slice5-optimizer-smoke"
    assert cfg.policy_source == "slice0_baseline_recommendation"
    assert cfg.policy_confidence == "low"
    assert len(cfg.hard_constraints) == 4
    assert len(cfg.pareto_objectives) == 7


# --------------------------------------------------------------------------
# search_mode / budget
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["bayesian", "grid_random", "", None, 5])
def test_unknown_search_mode_rejected(mode):
    with pytest.raises(SearchConfigError, match="search_mode"):
        make_search_config(search_mode=mode)


@pytest.mark.parametrize("budget", [0, -1, -100])
def test_non_positive_max_candidates_rejected(budget):
    with pytest.raises(SearchConfigError, match="max_candidates"):
        make_search_config(max_candidates=budget)


@pytest.mark.parametrize("n", [0, -3])
def test_non_positive_validation_top_n_rejected(n):
    with pytest.raises(SearchConfigError, match="validation_top_n"):
        make_search_config(validation_top_n=n)


def test_boolean_is_not_accepted_as_int_budget():
    with pytest.raises(SearchConfigError):
        make_search_config(max_candidates=True)


# --------------------------------------------------------------------------
# search space 白名單
# --------------------------------------------------------------------------
def test_unknown_search_space_dimension_rejected():
    with pytest.raises(SearchConfigError, match="不是本 Slice 可搜尋的維度"):
        make_search_config(search_space={"retrieval": {"nonsense_knob": [1, 2]}})


@pytest.mark.parametrize(
    "key",
    ["embedding_model", "embedding_dimension", "identifier_normalization_version",
     "canonical_text_version", "ranking_version", "gain_mapping", "ndcg_variant"],
)
def test_forbidden_search_dimensions_rejected(key):
    """§七：固定版本資訊不得作為搜尋參數。"""
    with pytest.raises(SearchConfigError, match="不得出現在搜尋空間"):
        make_search_config(search_space={"retrieval": {key: ["a", "b"]}})


def test_empty_search_space_rejected():
    with pytest.raises(SearchConfigError, match="至少要有一個可搜尋維度"):
        make_search_config(search_space={})


def test_empty_value_list_rejected():
    with pytest.raises(SearchConfigError, match="非空陣列"):
        make_search_config(search_space={"retrieval": {"top_k": []}})


def test_duplicate_values_in_dimension_rejected():
    with pytest.raises(SearchConfigError, match="重複值"):
        make_search_config(search_space={"retrieval": {"top_k": [5, 5]}})


# --------------------------------------------------------------------------
# 路徑安全（§6.9）
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad", ["/etc/passwd", "C:\\Users\\me\\cases.json", "D:/data/x.json", "../../secret.json"]
)
def test_absolute_or_traversal_dataset_path_rejected(bad):
    with pytest.raises(SearchConfigError):
        make_search_config(dataset_path=bad)


def test_empty_corpus_paths_rejected():
    with pytest.raises(SearchConfigError, match="corpus_paths"):
        make_search_config(corpus_paths=[])


# --------------------------------------------------------------------------
# metric 名稱驗證（§14.3）
# --------------------------------------------------------------------------
def test_unknown_metric_name_rejected():
    with pytest.raises(SearchConfigError, match="不存在"):
        make_search_config(
            hard_constraints=[
                {"metric": "totally_made_up_metric", "operator": ">=", "value": 1}
            ]
        )


def test_metric_valid_at_aggregate_but_not_at_query_type_scope_rejected():
    """exact_code_top1_accuracy 只有 aggregate scope 會產生。"""
    with pytest.raises(SearchConfigError, match="不存在"):
        make_search_config(
            hard_constraints=[
                {
                    "metric": "exact_code_top1_accuracy",
                    "scope": "query_type",
                    "query_type": "exact_code",
                    "operator": ">=",
                    "value": 0.5,
                }
            ]
        )


def test_query_type_scope_requires_query_type():
    with pytest.raises(SearchConfigError, match="必須指定 query_type"):
        make_search_config(
            hard_constraints=[
                {"metric": "mrr", "scope": "query_type", "operator": ">=", "value": 0.5}
            ]
        )


def test_aggregate_scope_rejects_query_type():
    with pytest.raises(SearchConfigError, match="不得指定 query_type"):
        make_search_config(
            hard_constraints=[
                {
                    "metric": "mrr",
                    "scope": "aggregate",
                    "query_type": "semantic",
                    "operator": ">=",
                    "value": 0.5,
                }
            ]
        )


def test_invalid_query_type_rejected():
    with pytest.raises(SearchConfigError, match="query_type"):
        make_search_config(
            hard_constraints=[
                {
                    "metric": "mrr",
                    "scope": "query_type",
                    "query_type": "not_a_type",
                    "operator": ">=",
                    "value": 0.5,
                }
            ]
        )


# --------------------------------------------------------------------------
# constraint 結構
# --------------------------------------------------------------------------
def test_constraint_requires_exactly_one_of_value_or_delta():
    with pytest.raises(SearchConfigError, match="只能指定"):
        make_search_config(
            hard_constraints=[
                {"metric": "mrr", "operator": ">=", "value": 1, "baseline_delta": -0.1}
            ]
        )
    with pytest.raises(SearchConfigError, match="只能指定"):
        make_search_config(hard_constraints=[{"metric": "mrr", "operator": ">="}])


@pytest.mark.parametrize("op", [">=", ">", "<=", "<", "=="])
def test_all_supported_operators_accepted(op):
    cfg = make_search_config(
        hard_constraints=[{"metric": "mrr", "operator": op, "value": 0.5}]
    )
    assert cfg.hard_constraints[0].operator == op


def test_unknown_operator_rejected():
    with pytest.raises(SearchConfigError, match="operator"):
        make_search_config(
            hard_constraints=[{"metric": "mrr", "operator": "~=", "value": 0.5}]
        )


def test_safety_critical_constraint_cannot_downgrade_not_computable_policy():
    """§14.7：safety-critical 不可計算時必須 fail。"""
    with pytest.raises(SearchConfigError, match="safety-critical"):
        make_search_config(
            hard_constraints=[
                {
                    "metric": "unanswerable_false_positive_rate",
                    "operator": "<=",
                    "value": 0.25,
                    "safety_critical": True,
                    "on_not_computable": "warn",
                }
            ]
        )


def test_non_safety_constraint_may_warn_or_skip():
    for policy in ("warn", "skip"):
        cfg = make_search_config(
            hard_constraints=[
                {
                    "metric": "mrr",
                    "operator": ">=",
                    "value": 0.5,
                    "safety_critical": False,
                    "on_not_computable": policy,
                }
            ]
        )
        assert cfg.hard_constraints[0].on_not_computable == policy


def test_nan_and_inf_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(SearchConfigError, match="NaN|Inf"):
            make_search_config(
                hard_constraints=[{"metric": "mrr", "operator": ">=", "value": bad}]
            )


def test_duplicate_constraint_names_rejected():
    c = {"name": "dup", "metric": "mrr", "operator": ">=", "value": 0.1}
    with pytest.raises(SearchConfigError, match="name 不得重複"):
        make_search_config(hard_constraints=[c, dict(c, value=0.2)])


# --------------------------------------------------------------------------
# recommendation policy
# --------------------------------------------------------------------------
def test_ranking_must_end_with_config_hash():
    with pytest.raises(SearchConfigError, match="config_hash"):
        make_search_config(
            recommendation_policy={
                "tune_ranking": [{"key": "mrr", "direction": "maximize"}],
                "recommendation_ranking": [{"key": "mrr", "direction": "maximize"}],
            }
        )


def test_duplicate_sort_keys_rejected():
    with pytest.raises(SearchConfigError, match="排序鍵重複"):
        make_search_config(
            recommendation_policy={
                "tune_ranking": [
                    {"key": "mrr", "direction": "maximize"},
                    {"key": "mrr", "direction": "minimize"},
                    {"key": "config_hash", "direction": "minimize"},
                ],
                "recommendation_ranking": [{"key": "config_hash", "direction": "minimize"}],
            }
        )


def test_tune_prefilter_defaults_and_validation():
    cfg = make_search_config()
    pf = cfg.recommendation_policy.tune_prefilter
    assert pf.eliminate_on_failed_trial is True
    assert pf.eliminate_on_computable_safety_violation is True
    # 小樣本保護：預設不因不可計算而在 tune 淘汰
    assert pf.eliminate_on_not_computable is False

    with pytest.raises(SearchConfigError, match="未知欄位"):
        make_search_config(
            recommendation_policy={**dict(raw_search_config()["recommendation_policy"]),
                                   "tune_prefilter": {"nope": True}}
        )


def test_unknown_top_level_field_rejected():
    with pytest.raises(SearchConfigError, match="未知欄位"):
        make_search_config(surprise=1)


def test_unsupported_schema_version_rejected():
    with pytest.raises(SearchConfigError, match="schema_version"):
        make_search_config(schema_version="9.9")


# --------------------------------------------------------------------------
# hash 決定性（§6.7 / §6.8）
# --------------------------------------------------------------------------
def test_hash_is_deterministic_and_key_order_independent():
    a = SearchConfig.from_raw(raw_search_config())
    b = SearchConfig.from_raw(dict(reversed(list(raw_search_config().items()))))
    assert a.compute_hash() == b.compute_hash()


def test_hash_ignores_human_labels_but_covers_semantics():
    base = make_search_config()
    # search_name 不改變語意 → hash 不變
    assert make_search_config(search_name="other").compute_hash() == base.compute_hash()
    # 以下每一項都必須改變 hash（§6.8）
    assert make_search_config(random_seed=7).compute_hash() != base.compute_hash()
    assert make_search_config(max_candidates=9).compute_hash() != base.compute_hash()
    assert make_search_config(validation_top_n=2).compute_hash() != base.compute_hash()
    assert (
        make_search_config(
            baseline={
                "chunk_config": {"strategy": "current"},
                "retrieval_config": {"strategy": "dense", "top_k": 7},
            }
        ).compute_hash()
        != base.compute_hash()
    )
    assert (
        make_search_config(
            search_space={"retrieval": {"top_k": [3, 5]}}
        ).compute_hash()
        != base.compute_hash()
    )
    assert (
        make_search_config(
            hard_constraints=[{"metric": "mrr", "operator": ">=", "value": 0.1}]
        ).compute_hash()
        != base.compute_hash()
    )
    assert (
        make_search_config(
            pareto_objectives=[
                {"metric": "mrr", "scope": "aggregate", "direction": "minimize"}
            ]
        ).compute_hash()
        != base.compute_hash()
    )


def test_sanitized_contains_no_absolute_paths():
    blob = json.dumps(make_search_config().sanitized(), ensure_ascii=False)
    assert "C:\\" not in blob and "D:\\" not in blob
    assert not any(seg.startswith("/") for seg in (blob.split('"')))
