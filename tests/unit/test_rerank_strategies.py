"""rerank 接進兩個 strategy 後的行為（v1.2.1 Slice R1）。

沿用 `tests/unit/retrieval_helpers.py` 的 `FakeBackend` —— 不需要 Qdrant，也不載入模型。

最重要的兩件事:
- `rerank_strategy=none` 時，兩個 strategy 的輸出與 v1.1 **完全相同**。
- rerank 啟用時，admission 集合不變、非空性不變（H3）。
"""
from __future__ import annotations

import pytest

from evaluation.v11.retrieval.base import RetrievalRequest, build_retrieval_strategy
from evaluation.v11.retrieval.config import RERANK_CROSS_ENCODER, RetrievalConfig
from evaluation.v11.retrieval.rerank_stage import RerankStageError
from tests.unit.retrieval_helpers import FakeBackend, cand


class FakeReranker:
    def __init__(self, scores, default=0.0):
        self._scores = scores
        self._default = default
        self.call_count = 0

    def score(self, query, texts):
        self.call_count += 1
        return [self._scores.get(t, self._default) for t in texts]


def _candidates(n, *, base=0.95, ids=None):
    return [
        cand(
            "doc",
            i,
            round(base - i * 0.02, 4),
            identifiers=list(ids or []),
            text=f"c{i}",
        )
        for i in range(n)
    ]


def _request(cfg, *, reranker=None, query="E102 警報"):
    return RetrievalRequest(
        query=query,
        query_vector=[0.0] * 4,
        collection_name="c",
        config=cfg,
        reranker=reranker,
    )


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_rerank_none_output_is_unchanged_from_v11(strategy):
    """rerank 未啟用 -> 結果、順序、ranking_version、meta 皆與 v1.1 相同。"""
    cands = _candidates(8)
    backend = FakeBackend(dense=cands, identifier=[])
    cfg = RetrievalConfig.from_raw({"strategy": strategy, "top_k": 3})

    out = build_retrieval_strategy(strategy).retrieve(_request(cfg), backend)

    assert [r.text for r in out.results] == ["c0", "c1", "c2"]
    assert all(r.rerank_score is None for r in out.results)
    assert all(r.rerank_applied is False for r in out.results)
    assert out.ranking_version in ("dense_v1", "identifier_aware_v1")
    # meta 不得出現 rerank 欄位（既有 report / 持久化內容不變）
    assert "rerank_strategy" not in out.meta()
    assert "rerank_latency_ms" not in out.meta()
    # ranking_details 亦不得出現 rerank 欄位
    assert "rerank_score" not in out.results[0].ranking_details()


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_rerank_reorders_within_window(strategy):
    cands = _candidates(8)
    backend = FakeBackend(dense=cands, identifier=[])
    cfg = RetrievalConfig.from_raw(
        {
            "strategy": strategy,
            "top_k": 3,
            "rerank_strategy": RERANK_CROSS_ENCODER,
            "rerank_top_n": 5,
        }
    )
    # c4 原本排最後，rerank 後應躍升第一
    fake = FakeReranker({"c4": 9.0, "c0": 1.0, "c1": 0.5, "c2": 0.2, "c3": -1.0})

    out = build_retrieval_strategy(strategy).retrieve(
        _request(cfg, reranker=fake), backend
    )

    assert [r.text for r in out.results] == ["c4", "c0", "c1"]
    assert out.results[0].rerank_score == 9.0
    assert out.results[0].ranking_score == 9.0
    assert out.results[0].vector_score == cands[4].vector_score  # RAW 未被覆寫
    assert fake.call_count == 1


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_rerank_meta_is_emitted_only_when_enabled(strategy):
    backend = FakeBackend(dense=_candidates(8), identifier=[])
    cfg = RetrievalConfig.from_raw(
        {
            "strategy": strategy,
            "top_k": 3,
            "rerank_strategy": RERANK_CROSS_ENCODER,
            "rerank_top_n": 5,
        }
    )
    out = build_retrieval_strategy(strategy).retrieve(
        _request(cfg, reranker=FakeReranker({})), backend
    )
    meta = out.meta()
    assert meta["rerank_strategy"] == RERANK_CROSS_ENCODER
    assert meta["admitted_count"] == 8
    assert meta["rerank_window_count"] == 5
    assert meta["rerank_tail_count"] == 3
    assert meta["ranking_version"].endswith("+cross_encoder_v1")


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_h3_admission_is_unaffected_by_rerank(strategy):
    """低於 threshold 的候選不得被 rerank 救回;admitted 數與未啟用時相同。"""
    cands = _candidates(4) + [
        cand("doc", 9, 0.10, text="low")
    ]
    backend = FakeBackend(dense=cands, identifier=[])

    plain = RetrievalConfig.from_raw({"strategy": strategy, "top_k": 5})
    out_plain = build_retrieval_strategy(strategy).retrieve(_request(plain), backend)

    cfg = RetrievalConfig.from_raw(
        {
            "strategy": strategy,
            "top_k": 5,
            "rerank_strategy": RERANK_CROSS_ENCODER,
            "rerank_top_n": 5,
        }
    )
    # 給 "low" 極高分 —— 仍不得出現，因為它根本沒通過 admission。
    out_rr = build_retrieval_strategy(strategy).retrieve(
        _request(cfg, reranker=FakeReranker({"low": 99.0})),
        FakeBackend(dense=cands, identifier=[]),
    )

    assert "low" not in [r.text for r in out_plain.results]
    assert "low" not in [r.text for r in out_rr.results]
    assert out_rr.meta()["admitted_count"] == len(out_plain.results)


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_h3_non_emptiness_preserved_under_rerank(strategy):
    backend = FakeBackend(dense=_candidates(3), identifier=[])
    cfg = RetrievalConfig.from_raw(
        {
            "strategy": strategy,
            "top_k": 5,
            "rerank_strategy": RERANK_CROSS_ENCODER,
            "rerank_top_n": 5,
        }
    )
    out = build_retrieval_strategy(strategy).retrieve(
        _request(cfg, reranker=FakeReranker({}, default=-999.0)), backend
    )
    assert len(out.results) == 3  # 非空性不變，即使分數極低


def test_identifier_aware_no_identifier_query_still_reranks():
    """query 無 identifier 時退化為 dense，但 rerank 仍須生效且統計不得遺失。"""
    backend = FakeBackend(dense=_candidates(6), identifier=[])
    cfg = RetrievalConfig.from_raw(
        {
            "strategy": "identifier_aware",
            "top_k": 3,
            "rerank_strategy": RERANK_CROSS_ENCODER,
            "rerank_top_n": 5,
        }
    )
    out = build_retrieval_strategy("identifier_aware").retrieve(
        _request(cfg, reranker=FakeReranker({"c3": 8.0}), query="沒有代碼的問題"),
        backend,
    )
    assert out.query_identifiers == []
    assert out.results[0].text == "c3"
    assert out.meta()["rerank_strategy"] == RERANK_CROSS_ENCODER
    assert out.meta()["rerank_window_count"] == 5


@pytest.mark.parametrize("strategy", ["dense", "identifier_aware"])
def test_missing_reranker_is_a_loud_failure(strategy):
    backend = FakeBackend(dense=_candidates(4), identifier=[])
    cfg = RetrievalConfig.from_raw(
        {"strategy": strategy, "top_k": 3, "rerank_strategy": RERANK_CROSS_ENCODER}
    )
    with pytest.raises(RerankStageError, match="reranker 為 None"):
        build_retrieval_strategy(strategy).retrieve(_request(cfg), backend)
