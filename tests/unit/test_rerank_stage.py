"""Rerank stage 語義（v1.2.1 Slice R1）。

重點在三個**不可退讓**的性質:

1. **H3 安全不變量** —— rerank 無 admission/filtering 權限;非空性一致。
2. **W/T 語義** —— tail 不被丟棄、相對順序保留、`rerank_applied` 標記正確。
3. **raw logit 作排序主鍵** —— 飽和端不得塌成假平手;normalized 不參與排序。

一律使用 `FakeReranker`，**不載入真實模型**（`tests/conftest.py` 的既定規則）。
"""
from __future__ import annotations

import pytest

from evaluation.v11.retrieval.base import RankedRetrievalResult
from evaluation.v11.retrieval.rerank_stage import (
    RerankStageError,
    apply_rerank,
)
from evaluation.v11.retrieval.reranker import NoOpReranker, sigmoid


class FakeReranker:
    """依 text -> score 對照表計分;未列出的 text 給預設值。"""

    def __init__(self, scores: dict[str, float], default: float = 0.0):
        self._scores = scores
        self._default = default
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [self._scores.get(t, self._default) for t in texts]


class DictCache:
    """最小記憶體快取，用來驗證 stage 的 hit/miss 行為。"""

    def __init__(self, seed=None):
        self.store = dict(seed or {})
        self.put_calls = 0

    def get_many(self, query, texts):
        return [self.store.get((query, t)) for t in texts]

    def put_many(self, query, texts, scores):
        self.put_calls += 1
        for t, s in zip(texts, scores):
            self.store[(query, t)] = s


def _result(doc: str, idx: int, vector_score: float, text: str = None, exact=None):
    return RankedRetrievalResult(
        rank=0,
        document_id=doc,
        chunk_index=idx,
        text=text if text is not None else f"{doc}-{idx}",
        page=None,
        section=None,
        filename=doc,
        file_hash=None,
        identifiers=[],
        provenance_spans=[],
        canonical_text_version="v1",
        vector_score=vector_score,
        ranking_score=vector_score,
        exact_identifier_matches=list(exact or []),
        conflicting_identifiers=[],
        exact_match_boost=0.0,
        confusable_penalty=0.0,
        candidate_sources=["dense"],
        admission_reason="dense_threshold",
    )


def _admitted(n: int) -> list[RankedRetrievalResult]:
    return [_result("d", i, 0.9 - i * 0.01) for i in range(n)]


# --------------------------------------------------------------------------
# H3 安全不變量
# --------------------------------------------------------------------------
def test_h3_empty_admitted_yields_empty_output():
    out = apply_rerank(
        [], query="q", reranker=FakeReranker({}), rerank_top_n=5, top_k=5
    )
    assert out["results"] == []
    assert out["admitted_count"] == 0


@pytest.mark.parametrize("n_admitted", [1, 3, 5, 9, 20])
def test_h3_non_empty_admitted_always_yields_non_empty_output(n_admitted):
    """A 非空 => 輸出非空。這是 unanswerable_false_positive_rate 恆等的結構依據。"""
    out = apply_rerank(
        _admitted(n_admitted),
        query="q",
        # 全部給極低分 —— rerank 仍不得把任何候選濾掉。
        reranker=FakeReranker({}, default=-999.0),
        rerank_top_n=5,
        top_k=5,
    )
    assert len(out["results"]) > 0


def test_rerank_has_no_filtering_authority_total_candidates_preserved():
    """window + tail 加總必須等於 admitted 數 —— rerank 不得新增或移除候選。"""
    admitted = _admitted(9)
    out = apply_rerank(
        admitted,
        query="q",
        reranker=FakeReranker({}, default=-50.0),
        rerank_top_n=5,
        top_k=3,
    )
    assert out["window_count"] + out["tail_count"] == out["admitted_count"] == 9
    assert len(out["results"]) == 3  # 只是截斷至 top_k，屬既有行為


# --------------------------------------------------------------------------
# W / T 語義
# --------------------------------------------------------------------------
def test_window_and_tail_split_at_rerank_top_n():
    out = apply_rerank(
        _admitted(9),
        query="q",
        reranker=FakeReranker({}),
        rerank_top_n=5,
        top_k=5,
    )
    assert out["window_count"] == 5
    assert out["tail_count"] == 4


def test_only_window_is_sent_to_the_reranker():
    fake = FakeReranker({})
    apply_rerank(_admitted(9), query="q", reranker=fake, rerank_top_n=5, top_k=5)
    assert len(fake.calls) == 1
    _, texts = fake.calls[0]
    assert texts == [f"d-{i}" for i in range(5)]  # tail d-5..d-8 未送進去


def test_tail_is_not_dropped_and_is_left_untouched():
    """tail 不被移除，且**完全未被 rerank 觸碰**。

    注意:因為 config 強制 ``rerank_top_n >= top_k``，tail 在結構上**永遠不會進入輸出**
    （輸出全部來自 window）。T 的存在只是為了維持「rerank 不移除任何候選」的不變量，
    所以這裡驗證的是 admitted 物件本身的狀態，而不是 results。
    """
    admitted = _admitted(8)
    window_size = 4
    out = apply_rerank(
        admitted,
        query="q",
        reranker=FakeReranker({f"d-{i}": float(i) for i in range(window_size)}),
        rerank_top_n=window_size,
        top_k=window_size,
    )
    assert out["window_count"] == window_size
    assert out["tail_count"] == len(admitted) - window_size

    # window 依 raw score desc 重排
    assert [r.text for r in out["results"]] == ["d-3", "d-2", "d-1", "d-0"]

    # tail 物件仍在原 list、原相對順序，且未被計分
    tail = admitted[window_size:]
    assert [r.text for r in tail] == ["d-4", "d-5", "d-6", "d-7"]
    assert all(r.rerank_score is None for r in tail)
    assert all(r.rerank_applied is False for r in tail)


def test_rerank_applied_flag_marks_window_only():
    admitted = _admitted(8)
    apply_rerank(
        admitted, query="q", reranker=FakeReranker({}), rerank_top_n=3, top_k=3
    )
    assert [r.rerank_applied for r in admitted[:3]] == [True, True, True]
    assert [r.rerank_applied for r in admitted[3:]] == [False] * 5


def test_window_larger_than_admitted_is_safe():
    out = apply_rerank(
        _admitted(2), query="q", reranker=FakeReranker({}), rerank_top_n=5, top_k=5
    )
    assert out["window_count"] == 2
    assert out["tail_count"] == 0
    assert len(out["results"]) == 2


def test_rerank_top_n_below_top_k_is_rejected():
    with pytest.raises(RerankStageError, match=r"rerank_top_n .* 必須 >= top_k"):
        apply_rerank(
            _admitted(5), query="q", reranker=FakeReranker({}), rerank_top_n=3, top_k=5
        )


# --------------------------------------------------------------------------
# raw logit 作排序主鍵
# --------------------------------------------------------------------------
def test_ranking_score_becomes_raw_logit_and_vector_score_is_untouched():
    admitted = [_result("d", 0, 0.91), _result("d", 1, 0.72)]
    out = apply_rerank(
        admitted,
        query="q",
        reranker=FakeReranker({"d-0": -4.5, "d-1": 7.25}),
        rerank_top_n=2,
        top_k=2,
    )
    top = out["results"][0]
    assert top.text == "d-1"
    assert top.rerank_score == 7.25
    assert top.ranking_score == 7.25          # 主鍵 = raw logit
    assert top.vector_score == 0.72           # RAW cosine 未被覆寫
    assert top.rerank_score_normalized == pytest.approx(sigmoid(7.25))


def test_saturated_logits_do_not_collapse_into_a_tie():
    """sigmoid 在飽和端經報表捨入會塌成平手;raw logit 不會。這是採 raw 的理由。"""
    a, b = 8.9, 9.1
    assert round(sigmoid(a), 4) == round(sigmoid(b), 4)  # normalized 塌了
    out = apply_rerank(
        [_result("d", 0, 0.5), _result("d", 1, 0.5)],
        query="q",
        reranker=FakeReranker({"d-0": a, "d-1": b}),
        rerank_top_n=2,
        top_k=2,
    )
    assert [r.text for r in out["results"]] == ["d-1", "d-0"]  # raw 仍可分辨


def test_normalized_score_is_monotonic_but_not_the_sort_key():
    out = apply_rerank(
        [_result("d", i, 0.5) for i in range(3)],
        query="q",
        reranker=FakeReranker({"d-0": -2.0, "d-1": 3.0, "d-2": 0.5}),
        rerank_top_n=3,
        top_k=3,
    )
    raws = [r.rerank_score for r in out["results"]]
    norms = [r.rerank_score_normalized for r in out["results"]]
    assert raws == sorted(raws, reverse=True)
    assert norms == [pytest.approx(sigmoid(v)) for v in raws]


def test_exact_match_breaks_ties_when_raw_scores_are_equal():
    """raw score 完全相等時，落入既有 5 鍵 chain（第 2 鍵 = exact match）。"""
    a = _result("d", 0, 0.5, exact=[])
    b = _result("d", 1, 0.5, exact=["E102"])
    out = apply_rerank(
        [a, b],
        query="q",
        reranker=FakeReranker({}, default=1.0),  # 兩者同分
        rerank_top_n=2,
        top_k=2,
    )
    assert [r.text for r in out["results"]] == ["d-1", "d-0"]


def test_ranks_are_reassigned_from_one():
    out = apply_rerank(
        _admitted(5),
        query="q",
        reranker=FakeReranker({f"d-{i}": float(i) for i in range(5)}),
        rerank_top_n=5,
        top_k=5,
    )
    assert [r.rank for r in out["results"]] == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------
# Determinism / cache
# --------------------------------------------------------------------------
def test_same_input_twice_yields_identical_order_and_scores():
    scores = {f"d-{i}": (i * 7 % 5) - 2.0 for i in range(6)}
    runs = [
        apply_rerank(
            _admitted(6),
            query="q",
            reranker=FakeReranker(scores),
            rerank_top_n=6,
            top_k=6,
        )
        for _ in range(2)
    ]
    assert [r.text for r in runs[0]["results"]] == [r.text for r in runs[1]["results"]]
    assert [r.rerank_score for r in runs[0]["results"]] == [
        r.rerank_score for r in runs[1]["results"]
    ]


def test_cache_hit_avoids_calling_the_reranker():
    cache = DictCache({("q", "d-0"): 5.0, ("q", "d-1"): 1.0})
    fake = FakeReranker({})
    out = apply_rerank(
        _admitted(2), query="q", reranker=fake, rerank_top_n=2, top_k=2, cache=cache
    )
    assert fake.calls == []  # 全部命中，模型未被呼叫
    assert out["cache_hits"] == 2
    assert out["cache_misses"] == 0
    assert out["results"][0].rerank_score == 5.0


def test_partial_cache_only_computes_the_misses_and_stores_them():
    cache = DictCache({("q", "d-0"): 5.0})
    fake = FakeReranker({"d-1": 2.0, "d-2": 3.0})
    out = apply_rerank(
        _admitted(3), query="q", reranker=fake, rerank_top_n=3, top_k=3, cache=cache
    )
    assert fake.calls[0][1] == ["d-1", "d-2"]  # 只算 miss
    assert out["cache_hits"] == 1
    assert out["cache_misses"] == 2
    assert cache.store[("q", "d-1")] == 2.0


def test_noop_reranker_refuses_to_score():
    with pytest.raises(RuntimeError, match="不應被呼叫"):
        NoOpReranker().score("q", ["a"])


def test_sigmoid_does_not_overflow_on_large_magnitudes():
    assert sigmoid(-1000.0) == pytest.approx(0.0)
    assert sigmoid(1000.0) == pytest.approx(1.0)
