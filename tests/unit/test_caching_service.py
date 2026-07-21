"""Caching Document Embedding Service 單元測試（v1.1 Slice 3 §十七 / §二十二.9）。"""
from __future__ import annotations

import pytest

from evaluation.v11.embedding_cache import (
    CachingDocumentEmbeddingService,
    EmbeddingCacheRepository,
)
from evaluation.v11.embedding_fingerprint import make_fingerprint


class FakeUnderlying:
    dimension = 4
    model_name = "fake"

    def __init__(self, fail=False):
        self.calls: list[list[str]] = []
        self.fail = fail

    def count_tokens(self, text):
        return len(text)

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("底層 embedding 失敗")
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

    def embed_query(self, q):
        return [9.0, 0.0, 0.0, 0.0]


def _fp():
    return make_fingerprint(provider="st", model_name="fake", model_revision=None,
                            dimension=4, normalize_embeddings=True, pooling_mode="cls")


def _svc(tmp_path, underlying=None, enabled=True, repo=None):
    underlying = underlying or FakeUnderlying()
    repo = repo if repo is not None else EmbeddingCacheRepository(str(tmp_path / "c.embedding-cache.db"))
    return CachingDocumentEmbeddingService(underlying, repo, _fp(), enabled=enabled), underlying, repo


def test_all_miss_first_time(tmp_path):
    svc, under, _ = _svc(tmp_path)
    out = svc.embed_documents(["a", "b", "c"])
    assert out == [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    assert svc.stats.embedding_cache_miss_count == 3
    assert svc.stats.embedding_cache_hit_count == 0
    assert under.calls == [["a", "b", "c"]]


def test_all_hit_second_time(tmp_path):
    svc, under, repo = _svc(tmp_path)
    svc.embed_documents(["a", "b"])
    svc2 = CachingDocumentEmbeddingService(under, repo, _fp(), enabled=True)
    under.calls.clear()
    svc2.embed_documents(["a", "b"])
    assert svc2.stats.embedding_cache_hit_count == 2
    assert under.calls == []  # cache 命中不呼叫底層


def test_partial_hit(tmp_path):
    svc, under, repo = _svc(tmp_path)
    svc.embed_documents(["a"])
    svc2 = CachingDocumentEmbeddingService(under, repo, _fp(), enabled=True)
    under.calls.clear()
    svc2.embed_documents(["a", "b"])
    assert svc2.stats.embedding_cache_hit_count == 1
    assert svc2.stats.embedding_cache_miss_count == 1
    assert under.calls == [["b"]]  # 只 embed miss


def test_batch_dedup_embeds_unique_once(tmp_path):
    svc, under, _ = _svc(tmp_path)
    out = svc.embed_documents(["a", "b", "a", "a"])
    assert under.calls == [["a", "b"]]  # 唯一文字只送一次
    assert out[0] == out[2] == out[3]  # 相同文字結果一致
    assert svc.stats.batch_dedup_savings == 2
    # dedup savings 與 persistent hit 分開：首次全 miss。
    assert svc.stats.embedding_cache_hit_count == 0


def test_output_order_preserved(tmp_path):
    svc, _, _ = _svc(tmp_path)
    texts = ["zzz", "y", "xx", "y", "zzz"]
    out = svc.embed_documents(texts)
    assert out == [[3, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0]]


def test_underlying_failure_writes_nothing(tmp_path):
    under = FakeUnderlying(fail=True)
    svc, _, repo = _svc(tmp_path, underlying=under)
    with pytest.raises(RuntimeError):
        svc.embed_documents(["a", "b"])
    assert repo.stats()["entry_count"] == 0  # 半成品未寫入


def test_cache_write_failure_does_not_fail_trial(tmp_path):
    class BoomRepo(EmbeddingCacheRepository):
        def put_many(self, entries):
            raise RuntimeError("寫入失敗")

    repo = BoomRepo(str(tmp_path / "c.embedding-cache.db"))
    svc = CachingDocumentEmbeddingService(FakeUnderlying(), repo, _fp(), enabled=True)
    out = svc.embed_documents(["a", "b"])  # 不得拋出
    assert len(out) == 2
    assert svc.stats.embedding_cache_write_failure_count == 2


def test_cache_disabled_full_passthrough(tmp_path):
    svc, under, _ = _svc(tmp_path, enabled=False)
    svc.embed_documents(["a", "a"])
    assert under.calls == [["a", "a"]]  # 停用 → 完全退回底層（不 dedup / 不快取）
    assert svc.stats.cache_enabled is False
    assert svc.stats.embedding_model_call_text_count == 2


def test_shared_cache_reused_across_trials(tmp_path):
    repo = EmbeddingCacheRepository(str(tmp_path / "c.embedding-cache.db"))
    under = FakeUnderlying()
    trial_a = CachingDocumentEmbeddingService(under, repo, _fp(), enabled=True)
    trial_a.embed_documents(["shared-chunk"])
    under.calls.clear()
    trial_b = CachingDocumentEmbeddingService(under, repo, _fp(), enabled=True)
    trial_b.embed_documents(["shared-chunk"])
    assert trial_b.stats.embedding_cache_hit_count == 1
    assert under.calls == []


def test_identifier_metadata_change_does_not_invalidate(tmp_path):
    # cache key 只看文字；identifier version 不在 key 中 → 相同文字仍 hit。
    svc, under, repo = _svc(tmp_path)
    svc.embed_documents(["E102 警報說明"])
    svc2 = CachingDocumentEmbeddingService(under, repo, _fp(), enabled=True)
    under.calls.clear()
    svc2.embed_documents(["E102 警報說明"])
    assert svc2.stats.embedding_cache_hit_count == 1


def test_query_embedding_not_cached(tmp_path):
    svc, under, repo = _svc(tmp_path)
    svc.embed_query("問題")
    assert repo.stats()["entry_count"] == 0  # query 不寫 cache


def test_hit_rate_computable_and_none_when_no_lookup(tmp_path):
    svc, _, _ = _svc(tmp_path)
    assert svc.stats.hit_rate is None  # 尚未查詢 → 分母 0 不可計算
    svc.embed_documents(["a"])
    assert svc.stats.hit_rate == 0.0
