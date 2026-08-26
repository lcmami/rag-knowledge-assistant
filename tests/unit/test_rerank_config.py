"""RetrievalConfig 的 v1.2.1 rerank 欄位（Slice R1）。

最關鍵的是**回溯相容釘樁**:`rerank_strategy=none` 的 canonical dict 與 config hash
必須與 v1.1 byte-identical，否則所有既有 config_hash、dataset、baseline 與已 locked
的推薦全部失效。
"""
from __future__ import annotations

import pytest

from evaluation.v11.hashing import canonical_json
from evaluation.v11.retrieval.config import (
    CROSS_ENCODER_RANKING_SUFFIX,
    DEFAULT_RERANK_TOP_N,
    DENSE_RANKING_VERSION,
    IDENTIFIER_AWARE_RANKING_VERSION,
    RERANK_CROSS_ENCODER,
    RERANK_NONE,
    RetrievalConfig,
    RetrievalConfigError,
)

# v1.1 在加入 rerank 之前的 canonical dict（逐字釘住，不從程式碼推導）。
_V11_DENSE_CANONICAL = {
    "schema_version": "1.0",
    "strategy": "dense",
    "top_k": 5,
    "score_threshold": 0.55,
    "candidate_multiplier": 3,
    "candidate_cap": 50,
    "ranking_version": "dense_v1",
}
_V11_IDA_CANONICAL = {
    **_V11_DENSE_CANONICAL,
    "strategy": "identifier_aware",
    "ranking_version": "identifier_aware_v1",
    "identifier_candidate_cap": 20,
    "exact_match_boost": 0.2,
    "confusable_penalty": 0.15,
    "identifier_vector_floor": 0.4,
}


# --------------------------------------------------------------------------
# 回溯相容（最重要）
# --------------------------------------------------------------------------
def test_dense_canonical_dict_byte_identical_to_v11():
    cfg = RetrievalConfig.from_raw({"strategy": "dense"})
    assert cfg.to_canonical_dict() == _V11_DENSE_CANONICAL
    assert canonical_json(cfg.to_canonical_dict()) == canonical_json(_V11_DENSE_CANONICAL)


def test_identifier_aware_canonical_dict_byte_identical_to_v11():
    cfg = RetrievalConfig.from_raw({"strategy": "identifier_aware"})
    assert cfg.to_canonical_dict() == _V11_IDA_CANONICAL
    assert canonical_json(cfg.to_canonical_dict()) == canonical_json(_V11_IDA_CANONICAL)


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_explicit_rerank_none_is_indistinguishable_from_omitting_it(strategy):
    """明確寫 rerank_strategy=none 與完全不寫，canonical 輸出必須完全相同。"""
    omitted = RetrievalConfig.from_raw({"strategy": strategy})
    explicit = RetrievalConfig.from_raw(
        {"strategy": strategy, "rerank_strategy": RERANK_NONE}
    )
    assert omitted.to_canonical_dict() == explicit.to_canonical_dict()
    assert "rerank_strategy" not in explicit.to_canonical_dict()
    assert "rerank_top_n" not in explicit.to_canonical_dict()


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_ranking_version_unchanged_when_rerank_disabled(strategy):
    cfg = RetrievalConfig.from_raw({"strategy": strategy})
    expected = (
        DENSE_RANKING_VERSION
        if strategy == "dense"
        else IDENTIFIER_AWARE_RANKING_VERSION
    )
    assert cfg.ranking_version == expected
    assert CROSS_ENCODER_RANKING_SUFFIX not in cfg.ranking_version


# --------------------------------------------------------------------------
# rerank 啟用
# --------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_rerank_is_orthogonal_to_strategy(strategy):
    """rerank 是正交欄位 —— dense 與 identifier_aware 皆可搭配。"""
    cfg = RetrievalConfig.from_raw(
        {"strategy": strategy, "rerank_strategy": RERANK_CROSS_ENCODER}
    )
    assert cfg.is_rerank_enabled
    assert cfg.rerank_top_n == DEFAULT_RERANK_TOP_N
    d = cfg.to_canonical_dict()
    assert d["rerank_strategy"] == RERANK_CROSS_ENCODER
    assert d["rerank_top_n"] == DEFAULT_RERANK_TOP_N
    assert d["ranking_version"].endswith(CROSS_ENCODER_RANKING_SUFFIX)


def test_enabling_rerank_changes_the_config_hash():
    base = RetrievalConfig.from_raw({"strategy": "identifier_aware"})
    rer = RetrievalConfig.from_raw(
        {"strategy": "identifier_aware", "rerank_strategy": RERANK_CROSS_ENCODER}
    )
    assert canonical_json(base.to_canonical_dict()) != canonical_json(
        rer.to_canonical_dict()
    )


def test_rerank_top_n_changes_the_config_hash():
    a = RetrievalConfig.from_raw(
        {"strategy": "dense", "rerank_strategy": RERANK_CROSS_ENCODER, "rerank_top_n": 5}
    )
    b = RetrievalConfig.from_raw(
        {"strategy": "dense", "rerank_strategy": RERANK_CROSS_ENCODER, "rerank_top_n": 10}
    )
    assert canonical_json(a.to_canonical_dict()) != canonical_json(b.to_canonical_dict())


# --------------------------------------------------------------------------
# 嚴格驗證（不得默默忽略）
# --------------------------------------------------------------------------
def test_rerank_field_rejected_when_rerank_disabled():
    with pytest.raises(RetrievalConfigError, match="rerank_strategy=none 不適用"):
        RetrievalConfig.from_raw({"strategy": "dense", "rerank_top_n": 5})


def test_unknown_rerank_strategy_rejected():
    with pytest.raises(RetrievalConfigError, match="不支援的 rerank_strategy"):
        RetrievalConfig.from_raw({"strategy": "dense", "rerank_strategy": "llm_listwise"})


def test_rerank_top_n_must_be_at_least_top_k():
    with pytest.raises(RetrievalConfigError, match=r"rerank_top_n .* 必須 >= top_k"):
        RetrievalConfig.from_raw(
            {
                "strategy": "dense",
                "top_k": 10,
                "rerank_strategy": RERANK_CROSS_ENCODER,
                "rerank_top_n": 5,
            }
        )


def test_rerank_top_n_must_be_positive_int():
    for bad in (0, -1, 2.5, True, "5"):
        with pytest.raises(RetrievalConfigError):
            RetrievalConfig.from_raw(
                {
                    "strategy": "dense",
                    "top_k": 1,
                    "rerank_strategy": RERANK_CROSS_ENCODER,
                    "rerank_top_n": bad,
                }
            )


def test_ranking_version_override_rejected_for_rerank_arm():
    with pytest.raises(RetrievalConfigError, match="ranking_version 必須為"):
        RetrievalConfig.from_raw(
            {
                "strategy": "dense",
                "rerank_strategy": RERANK_CROSS_ENCODER,
                "ranking_version": "dense_v1",  # 少了 rerank 後綴
            }
        )


def test_unknown_field_still_rejected():
    with pytest.raises(RetrievalConfigError, match="未知欄位"):
        RetrievalConfig.from_raw({"strategy": "dense", "rerank_model": "BAAI/x"})
