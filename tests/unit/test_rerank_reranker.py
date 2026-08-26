"""CrossEncoderReranker 的實作常數與呼叫契約（v1.2.1 Slice R1）。

**不載入真實模型** —— 以 spy 取代 `CrossEncoder`，驗證我們**確實**傳了正確的參數。
真實模型的第一次驗證已在 R0 完成（activation semantics gate + batch invariance gate）;
這裡是 **regression coverage**，確保後續改動不會悄悄破壞那兩個前提。
"""
from __future__ import annotations

import pytest

from evaluation.v11.retrieval.reranker import (
    RERANK_BATCH_SIZE,
    SCORED_ACTIVATION,
    CrossEncoderReranker,
    build_reranker,
    NoOpReranker,
)


class SpyCrossEncoder:
    """記錄 predict 收到的參數;不做任何實際推論。"""

    def __init__(self, scores=None):
        self.calls = []
        self._scores = scores

    def predict(self, pairs, **kwargs):
        self.calls.append({"pairs": list(pairs), **kwargs})
        if self._scores is not None:
            return list(self._scores)
        return [float(i) for i in range(len(pairs))]


# --------------------------------------------------------------------------
# R0 gate 的 regression coverage
# --------------------------------------------------------------------------
def test_batch_size_constant_is_one():
    """R0 remediation:padding-induced drift 為 0 且 CPU 上最快。"""
    assert RERANK_BATCH_SIZE == 1


def test_predict_is_always_called_with_batch_size_one():
    spy = SpyCrossEncoder()
    CrossEncoderReranker("m", model=spy).score("q", ["a", "b", "c"])
    assert spy.calls[0]["batch_size"] == RERANK_BATCH_SIZE == 1


def test_predict_is_always_called_with_identity_activation():
    """若沿用 predict() 預設，num_labels=1 會套 Sigmoid，raw-logit 設計會靜默失效。"""
    import torch

    spy = SpyCrossEncoder()
    CrossEncoderReranker("m", model=spy).score("q", ["a"])
    activation = spy.calls[0]["activation_fn"]
    assert activation is not None, "activation_fn 必須顯式傳入，不得沿用預設"
    assert isinstance(activation, torch.nn.Identity)
    assert SCORED_ACTIVATION == "identity"


def test_progress_bar_is_disabled():
    spy = SpyCrossEncoder()
    CrossEncoderReranker("m", model=spy).score("q", ["a"])
    assert spy.calls[0]["show_progress_bar"] is False


# --------------------------------------------------------------------------
# 呼叫契約
# --------------------------------------------------------------------------
def test_pairs_preserve_query_and_input_order():
    spy = SpyCrossEncoder()
    CrossEncoderReranker("m", model=spy).score("the query", ["x", "y", "z"])
    assert spy.calls[0]["pairs"] == [("the query", "x"), ("the query", "y"), ("the query", "z")]


def test_scores_are_returned_in_input_order():
    spy = SpyCrossEncoder(scores=[3.0, 1.0, 2.0])
    out = CrossEncoderReranker("m", model=spy).score("q", ["a", "b", "c"])
    assert out == [3.0, 1.0, 2.0]


def test_empty_input_short_circuits_without_calling_the_model():
    spy = SpyCrossEncoder()
    assert CrossEncoderReranker("m", model=spy).score("q", []) == []
    assert spy.calls == []


def test_length_mismatch_from_model_is_a_loud_failure():
    spy = SpyCrossEncoder(scores=[1.0])
    with pytest.raises(RuntimeError, match="不符"):
        CrossEncoderReranker("m", model=spy).score("q", ["a", "b"])


def test_model_is_lazily_loaded():
    """建構 reranker 不得觸發模型載入（否則 config 驗證階段就會很慢）。"""
    r = CrossEncoderReranker("definitely-not-a-real-model")
    assert r._model is None


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def test_build_reranker_none_returns_noop():
    assert isinstance(build_reranker("none"), NoOpReranker)


def test_build_reranker_cross_encoder_uses_default_model():
    from evaluation.v11.retrieval.config import DEFAULT_RERANK_MODEL

    r = build_reranker("cross_encoder")
    assert isinstance(r, CrossEncoderReranker)
    assert r.model_name == DEFAULT_RERANK_MODEL


def test_build_reranker_rejects_unknown_strategy():
    with pytest.raises(KeyError):
        build_reranker("llm_listwise")
