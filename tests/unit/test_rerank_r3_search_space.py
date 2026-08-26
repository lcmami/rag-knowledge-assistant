"""Slice R3（v1.2.1）—— rerank 進入 optimizer search space。

涵蓋三件事：

1. `rerank_strategy` / `rerank_top_n` 可搜尋；`rerank_model` **明確禁止**（比照 embedding_model）。
2. Conditional projection：`rerank_strategy=none` 時移除 `rerank_top_n`，
   duplicate 語義維持 deterministic（比照既有的 dense × IDENTIFIER_ONLY_FIELDS）。
3. 四個 arm 的比較 config 產生的正是四個 arm，且所有 ranking 參數維持文件化預設值。

不載入模型、不連 Qdrant。
"""
from __future__ import annotations

import json
import os

import pytest

from evaluation.v11.search.search_config import (
    ALLOWED_RETRIEVAL_SEARCH_KEYS,
    FORBIDDEN_SEARCH_KEYS,
    SearchConfig,
    SearchConfigError,
)
from evaluation.v11.search.search_space import (
    IDENTIFIER_ONLY_FIELDS,
    RERANK_ONLY_FIELDS,
    generate_candidates,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPARISON_CONFIG = os.path.join(
    ROOT, "evaluation", "configs", "v121-rerank-comparison.json"
)

_VERSIONS = {"canonical_text_version": "v1", "identifier_normalization_version": "v1"}


def _load_comparison_raw() -> dict:
    with open(COMPARISON_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def _base_raw(**space):
    """以**真實的比較 config** 為底，只換掉 search_space。

    刻意不手工拼一份最小 config：SearchConfig 對 constraints / objectives / policy
    都有必填驗證，手工版本會隨 schema 演進而腐爛，而且測到的東西會與正式設定脫節。
    """
    raw = _load_comparison_raw()
    raw["search_name"] = "r3-projection-probe"
    raw["max_candidates"] = 64
    raw["search_space"] = {"retrieval": space}
    return raw


def _generate(**space):
    return generate_candidates(SearchConfig.from_raw(_base_raw(**space)), **_VERSIONS)


def _arm(candidate):
    rc = candidate.retrieval_config
    return (rc["strategy"], rc.get("rerank_strategy", "none"))


# --------------------------------------------------------------------------
# 1. 可搜尋 / 禁止的 key
# --------------------------------------------------------------------------
def test_rerank_strategy_and_top_n_are_searchable():
    assert "rerank_strategy" in ALLOWED_RETRIEVAL_SEARCH_KEYS
    assert "rerank_top_n" in ALLOWED_RETRIEVAL_SEARCH_KEYS


def test_rerank_model_is_forbidden_with_a_reason():
    """換 reranker = 換 fingerprint = 換 baseline，是版本決策不是候選比較。"""
    assert "rerank_model" in FORBIDDEN_SEARCH_KEYS
    assert "fingerprint" in FORBIDDEN_SEARCH_KEYS["rerank_model"]


def test_searching_rerank_model_is_rejected():
    with pytest.raises(SearchConfigError) as exc:
        SearchConfig.from_raw(_base_raw(rerank_model=["BAAI/bge-reranker-large"]))
    assert "rerank_model" in str(exc.value)


def test_embedding_model_remains_forbidden():
    """R3 只新增禁止項，不得放寬既有的。"""
    with pytest.raises(SearchConfigError):
        SearchConfig.from_raw(_base_raw(embedding_model=["x"]))


# --------------------------------------------------------------------------
# 2. Conditional projection
# --------------------------------------------------------------------------
def test_rerank_top_n_is_projected_away_when_rerank_is_off():
    res = _generate(rerank_strategy=["none"], rerank_top_n=[5])
    assert len(res.executable) == 1
    c = res.executable[0]
    assert "rerank_top_n" in c.projected_fields
    assert "rerank_top_n" not in c.retrieval_config
    assert "rerank_strategy" not in c.retrieval_config  # none → 完全不輸出


def test_rerank_fields_survive_when_rerank_is_on():
    res = _generate(rerank_strategy=["cross_encoder"], rerank_top_n=[5])
    c = res.executable[0]
    assert c.projected_fields == ()
    assert c.retrieval_config["rerank_strategy"] == "cross_encoder"
    assert c.retrieval_config["rerank_top_n"] == 5
    assert c.retrieval_config["ranking_version"] == "dense_v1+cross_encoder_v1"


def test_multiple_top_n_values_collapse_to_one_candidate_when_rerank_is_off():
    """projection 後 duplicate 語義必須 deterministic：none × 任意 top_n 收斂成同一個 config。"""
    res = _generate(rerank_strategy=["none"], rerank_top_n=[5, 10, 20])
    assert res.raw_combination_count == 3
    assert len(res.executable) == 1
    assert res.duplicate_count == 2
    assert res.invalid_count == 0  # projection 的重點：不是產生一堆 invalid
    for c in res.rejected:
        assert c.duplicate_of_hash == res.executable[0].config_hash


def test_projection_is_recorded_in_generation_notes():
    res = _generate(rerank_strategy=["none"], rerank_top_n=[5, 10])
    notes = " ".join(res.stats()["notes"])
    assert "rerank_strategy=none" in notes
    assert "rerank_top_n" in notes


def test_rerank_projection_is_orthogonal_to_strategy():
    """rerank 不是第三個 strategy —— projection 對 dense 與 identifier_aware 皆須適用。"""
    res = _generate(
        strategy=["dense", "identifier_aware"],
        rerank_strategy=["none"],
        rerank_top_n=[5],
    )
    assert len(res.executable) == 2
    for c in res.executable:
        assert "rerank_top_n" in c.projected_fields
        assert "rerank_top_n" not in c.retrieval_config


def test_identifier_projection_still_works_alongside_rerank_projection():
    """兩個 projection 同時發生時都要被記錄，不得互相覆蓋。"""
    res = _generate(
        strategy=["dense"],
        identifier_vector_floor=[0.4],
        rerank_strategy=["none"],
        rerank_top_n=[5],
    )
    c = res.executable[0]
    assert "identifier_vector_floor" in c.projected_fields
    assert "rerank_top_n" in c.projected_fields
    notes = " ".join(res.stats()["notes"])
    assert all(f in notes for f in ("identifier_vector_floor", "rerank_top_n"))


def test_field_lists_are_disjoint():
    assert not set(IDENTIFIER_ONLY_FIELDS) & set(RERANK_ONLY_FIELDS)


def test_generation_is_deterministic_across_repeated_runs():
    space = dict(
        strategy=["dense", "identifier_aware"],
        rerank_strategy=["none", "cross_encoder"],
        rerank_top_n=[5],
    )
    a = [c.config_hash for c in _generate(**space).executable]
    b = [c.config_hash for c in _generate(**space).executable]
    assert a == b


# --------------------------------------------------------------------------
# 3. 比較 config 本身
# --------------------------------------------------------------------------
def _comparison_result():
    return generate_candidates(SearchConfig.from_raw(_load_comparison_raw()), **_VERSIONS)


def test_comparison_config_generates_exactly_the_four_required_arms():
    res = _comparison_result()
    assert {_arm(c) for c in res.executable} == {
        ("dense", "none"),
        ("dense", "cross_encoder"),
        ("identifier_aware", "none"),
        ("identifier_aware", "cross_encoder"),
    }
    assert res.invalid_count == 0
    assert res.duplicate_count == 0
    assert len(res.executable) == 4


def test_comparison_config_holds_every_ranking_parameter_at_its_default():
    """R3 只比較 rerank on/off × strategy —— 其他旋鈕一律不得被動到。"""
    for c in _comparison_result().executable:
        rc = c.retrieval_config
        assert rc["top_k"] == 5
        assert rc["score_threshold"] == 0.55
        assert rc["candidate_multiplier"] == 3
        assert rc["candidate_cap"] == 50
        if rc["strategy"] == "identifier_aware":
            assert rc["exact_match_boost"] == 0.2
            assert rc["confusable_penalty"] == 0.15
            assert rc["identifier_vector_floor"] == 0.4
            assert rc["identifier_candidate_cap"] == 20


def test_comparison_config_keeps_rerank_top_n_frozen_at_five():
    """R0.5 由 coverage audit 選出並 FROZEN；R3 不得掃描它。"""
    raw = _load_comparison_raw()
    assert raw["search_space"]["retrieval"]["rerank_top_n"] == [5]
    for c in _comparison_result().executable:
        rc = c.retrieval_config
        if rc.get("rerank_strategy") == "cross_encoder":
            assert rc["rerank_top_n"] == 5


def test_comparison_config_latency_constraint_uses_the_project_slo():
    raw = _load_comparison_raw()
    latency = [c for c in raw["hard_constraints"] if c["metric"] == "p95_retrieval_latency_ms"]
    assert len(latency) == 1
    assert latency[0]["value"] == 86
    assert latency[0]["operator"] == "<="


def test_comparison_config_does_not_search_any_frozen_dimension():
    raw = _load_comparison_raw()
    searched = set(raw["search_space"].get("retrieval", {}))
    # 只允許這三個維度；threshold / boost / penalty / floor / model 一律不得出現。
    assert searched == {"strategy", "rerank_strategy", "rerank_top_n"}
    assert not raw["search_space"].get("chunking")
