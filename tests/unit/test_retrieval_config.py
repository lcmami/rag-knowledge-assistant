"""RetrievalConfig 單元測試（v1.1 Slice 4 §二十一.1）。"""
from __future__ import annotations

import math

import pytest

from evaluation.v11.hashing import config_hash
from evaluation.v11.retrieval.config import RetrievalConfig, RetrievalConfigError


def _h(raw: dict) -> str:
    return config_hash(RetrievalConfig.from_raw(raw).to_canonical_dict())


def test_valid_dense():
    cfg = RetrievalConfig.from_raw({"strategy": "dense", "top_k": 5, "score_threshold": 0.55})
    assert cfg.strategy == "dense"
    assert cfg.ranking_version == "dense_v1"
    d = cfg.to_canonical_dict()
    # dense 不輸出 identifier 欄位（§八.11）。
    assert "exact_match_boost" not in d and "identifier_vector_floor" not in d


def test_none_defaults_to_dense():
    cfg = RetrievalConfig.from_raw(None)
    assert cfg.strategy == "dense"


def test_valid_identifier_aware():
    cfg = RetrievalConfig.from_raw(
        {"strategy": "identifier_aware", "exact_match_boost": 0.2,
         "confusable_penalty": 0.15, "identifier_vector_floor": 0.4}
    )
    assert cfg.strategy == "identifier_aware"
    assert cfg.ranking_version == "identifier_aware_v1"
    d = cfg.to_canonical_dict()
    assert d["exact_match_boost"] == 0.2 and d["identifier_vector_floor"] == 0.4


def test_unknown_strategy_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "hybrid"})


def test_top_k_non_positive_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "dense", "top_k": 0})
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "dense", "top_k": -1})


def test_multiplier_non_positive_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "dense", "candidate_multiplier": 0})


def test_candidate_cap_less_than_top_k_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "dense", "top_k": 10, "candidate_cap": 5})


def test_identifier_candidate_cap_non_positive_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "identifier_aware", "identifier_candidate_cap": 0})


def test_nan_inf_boost_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RetrievalConfigError):
            RetrievalConfig.from_raw({"strategy": "identifier_aware", "exact_match_boost": bad})


def test_nan_inf_penalty_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RetrievalConfigError):
            RetrievalConfig.from_raw({"strategy": "identifier_aware", "confusable_penalty": bad})


def test_nan_inf_identifier_floor_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RetrievalConfigError):
            RetrievalConfig.from_raw({"strategy": "identifier_aware", "identifier_vector_floor": bad})


def test_negative_boost_penalty_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "identifier_aware", "exact_match_boost": -0.1})
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "identifier_aware", "confusable_penalty": -0.1})


def test_dense_with_identifier_fields_rejected():
    # §八.11：dense 不適用 identifier 欄位 → 明確拒絕，不默默忽略。
    for f in ("identifier_candidate_cap", "exact_match_boost", "confusable_penalty",
              "identifier_vector_floor"):
        with pytest.raises(RetrievalConfigError):
            RetrievalConfig.from_raw({"strategy": "dense", f: 1})


def test_unknown_key_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "dense", "bogus": 1})


def test_ranking_version_mismatch_rejected():
    with pytest.raises(RetrievalConfigError):
        RetrievalConfig.from_raw({"strategy": "identifier_aware", "ranking_version": "dense_v1"})
    # 相符則接受。
    RetrievalConfig.from_raw({"strategy": "identifier_aware", "ranking_version": "identifier_aware_v1"})


def test_config_hash_deterministic_and_order_independent():
    a = _h({"strategy": "identifier_aware", "exact_match_boost": 0.2, "confusable_penalty": 0.15})
    b = _h({"confusable_penalty": 0.15, "exact_match_boost": 0.2, "strategy": "identifier_aware"})
    assert a == b


def test_config_hash_changes_on_ranking_params():
    base = {"strategy": "identifier_aware", "top_k": 5, "score_threshold": 0.55,
            "exact_match_boost": 0.2, "confusable_penalty": 0.15, "identifier_vector_floor": 0.4}
    h0 = _h(base)
    assert _h({**base, "exact_match_boost": 0.25}) != h0
    assert _h({**base, "confusable_penalty": 0.2}) != h0
    assert _h({**base, "identifier_vector_floor": 0.45}) != h0
    assert _h({**base, "score_threshold": 0.5}) != h0
    assert _h({**base, "candidate_multiplier": 4}) != h0


def test_dense_and_identifier_aware_hash_differ():
    assert _h({"strategy": "dense"}) != _h({"strategy": "identifier_aware"})


def test_candidate_limit_formula():
    cfg = RetrievalConfig.from_raw(
        {"strategy": "dense", "top_k": 5, "candidate_multiplier": 3, "candidate_cap": 50}
    )
    assert cfg.candidate_limit(5) == 15
    capped = RetrievalConfig.from_raw(
        {"strategy": "dense", "top_k": 5, "candidate_multiplier": 20, "candidate_cap": 50}
    )
    assert capped.candidate_limit(5) == 50
