"""Slice 4 Identifier-aware Retrieval 整合測試（§二十一.9）。

真實 Qdrant + 本機 BGE，於隔離 ``optimizer_`` collection 驗證：

- identifiers keyword payload index 存在（filtered vector search 可用）。
- Dense strategy 與正式 RetrievalService 排序 / 分數一致（parity；不斷言固定向量分數）。
- Identifier-filtered vector search 找回 E102；E-102 / CX-500 正規化後可 match。
- Candidate union：E102 即使不在 dense 截斷集合仍被 exact search 找回，且排在 D900 之前。
- 無 identifier query 與 dense 等價。
- slice4-identifier-smoke 端到端：兩 strategy、retrieval_meta 持久化、NDCG universe==point_count、
  doc embedding cache 重用、collection 全清理、正式 factory_knowledge 前後一致。

未就緒（無真實 Qdrant）時 skip。
"""
from __future__ import annotations

import json
import os

import pytest

from tests.integration.conftest import (
    drop_collections,
    list_collections,
    prod_point_count,
    unique_token,
)

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, "evaluation", "configs", "slice4-identifier-smoke.json")
PROD_COLLECTION = "factory_knowledge"
CORPUS = ["sample_docs", "evaluation/fixtures"]


@pytest.fixture
def qdrant_url(monkeypatch):
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    try:
        from qdrant_client import QdrantClient

        probe = QdrantClient(url=url)
        probe.get_collections()
        probe.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"真實 Qdrant 未就緒（{url}）：{exc}")
    return url


def _trial_config(name, retrieval):
    from evaluation.v11.run_models import TrialConfig

    return TrialConfig.from_raw(
        {"trial_name": name, "chunk_config": {"strategy": "current"}, "retrieval_config": retrieval},
        0,
    )


DENSE = {"strategy": "dense", "top_k": 5, "score_threshold": 0.55,
         "candidate_multiplier": 3, "candidate_cap": 50}
IDA = {"strategy": "identifier_aware", "top_k": 5, "score_threshold": 0.55,
       "candidate_multiplier": 3, "candidate_cap": 50, "identifier_candidate_cap": 20,
       "exact_match_boost": 0.2, "confusable_penalty": 0.15, "identifier_vector_floor": 0.4}


def _min_rank_with_identifier(results, ident):
    ranks = [r.rank for r in results if ident in (r.identifiers or [])]
    return min(ranks) if ranks else None


def _min_rank_pure(results, ident, exclude):
    """含 ident 但不含 exclude 的結果最前排名（純 confusable chunk，非同時列出兩碼的對照表）。"""
    ranks = [r.rank for r in results
             if ident in (r.identifiers or []) and exclude not in (r.identifiers or [])]
    return min(ranks) if ranks else None


def test_identifier_aware_union_boost_penalty(qdrant_url, tmp_path):
    from evaluation.v11 import hashing
    from evaluation.v11.real_services import RealTrialServiceProvider

    from qdrant_client import QdrantClient as _QC

    probe = _QC(url=qdrant_url)
    before_prod = prod_point_count(probe)

    files = [
        (os.path.basename(rel), abs_path)
        for rel, abs_path in hashing.collect_corpus_files(CORPUS, ROOT)
    ]
    workdir = str(tmp_path / "work")
    cache_db = str(tmp_path / "s4.embedding-cache.db")
    provider = RealTrialServiceProvider.create(qdrant_url, workdir, cache_db_path=cache_db)
    collection = f"optimizer_s4itg_direct_{unique_token()}"
    try:
        build_tc = _trial_config("build", DENSE)
        provider.build_index(collection, build_tc, files)

        store = provider._contexts[collection].store
        client = store.client

        # --- identifiers keyword index 存在（filtered vector search 可用）---
        from qdrant_client import models

        e102_pts = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="identifiers", match=models.MatchValue(value="E102"))]
            ),
            limit=50, with_payload=True,
        )[0]
        assert e102_pts, "E102 chunk 應存在於 trial 索引"
        d900_pts = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="identifiers", match=models.MatchValue(value="D900"))]
            ),
            limit=50, with_payload=True,
        )[0]
        assert d900_pts, "D900 confusable chunk 應存在"

        dense_tc = _trial_config("dense", DENSE)
        ida_tc = _trial_config("ida", IDA)

        # --- Dense parity 與正式 RetrievalService 一致 ---
        from app.services.retrieval_service import RetrievalService

        q = "設備出現 E102 警報時應如何處理？"
        rs = RetrievalService(provider._embedding, store, score_threshold=0.55,
                              candidate_multiplier=3, candidate_cap=50)
        ref = rs.retrieve(q, 5)
        dense_out = provider.retrieve(collection, dense_tc, q, 5)
        assert [(c.document_id, c.chunk_index) for c in ref.results] == \
               [(r.document_id, r.chunk_index) for r in dense_out.results]
        assert [round(c.score, 6) for c in ref.results] == \
               [round(r.score, 6) for r in dense_out.results]

        # --- Identifier-aware：E102 找回並排在「純 D900」confusable chunk 之前 ---
        # 註：同時列出 E102 與 D900 的對照表 chunk 為正確 exact-match（含 E102），不算 confusable。
        ida_out = provider.retrieve(collection, ida_tc, q, 5)
        e_rank = _min_rank_with_identifier(ida_out.results, "E102")
        pure_d900_rank = _min_rank_pure(ida_out.results, "D900", exclude="E102")
        assert e_rank is not None, "identifier-aware 應找回 E102"
        assert e_rank == 1, "boost 後 E102 exact-match 應為 Top-1"
        if pure_d900_rank is not None:
            assert e_rank < pure_d900_rank, "正確 E102 應排在純 D900 confusable chunk 之前"
        # raw vector_score 保留（score 欄位為 raw），ranking metadata 存在。
        assert ida_out.meta["strategy_name"] == "identifier_aware"
        assert "E102" in ida_out.meta["query_identifiers"]

        # --- E-102 / CX-500 正規化 → match ---
        e_dash = provider.retrieve(collection, ida_tc, "E-102 是什麼警報？", 5)
        assert "E102" in e_dash.meta["query_identifiers"]
        assert _min_rank_with_identifier(e_dash.results, "E102") is not None
        cx = provider.retrieve(collection, ida_tc, "CX-500 設備的保養週期是多久？", 5)
        assert "CX500" in cx.meta["query_identifiers"]

        # --- 無 identifier query 與 dense 等價 ---
        q2 = "設備的定期維護包含哪些基本作業？"
        d2 = provider.retrieve(collection, dense_tc, q2, 5)
        i2 = provider.retrieve(collection, ida_tc, q2, 5)
        assert i2.meta["query_identifiers"] == []
        assert [(r.document_id, r.chunk_index) for r in d2.results] == \
               [(r.document_id, r.chunk_index) for r in i2.results]
        assert [r.score for r in d2.results] == [r.score for r in i2.results]
    finally:
        try:
            provider.delete_collection(collection)
        except Exception:  # noqa: BLE001
            pass
        provider.close()
        drop_collections(probe, [collection])  # safety net + 確認真的消失

    # 本測試自身 collection 已清理、正式資料不變（不斷言整個 namespace）。
    assert not probe.collection_exists(collection)
    assert prod_point_count(probe) == before_prod
    assert PROD_COLLECTION in list_collections(probe)
    probe.close()


def test_slice4_smoke_end_to_end(qdrant_url, tmp_path):
    from evaluation.v11 import trial_runner
    from evaluation.v11.optimizer_repository import OptimizerRepository
    from evaluation.v11.real_services import RealTrialServiceProvider
    from evaluation.v11.run_models import RunConfig, RunStatus, TrialStatus

    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    before_prod = prod_point_count(client)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        run_config = RunConfig.from_raw(json.load(f))

    repo = OptimizerRepository(str(tmp_path / "s4.optimizer.db"))
    provider = RealTrialServiceProvider.create(
        qdrant_url, str(tmp_path / "work"), cache_db_path=str(tmp_path / "s4.embedding-cache.db")
    )
    # 唯一 run_id → 唯一 trial collection。
    run_id = f"s4-itg-{unique_token()}"
    own_collections: list[str] = []
    try:
        result = trial_runner.execute_run(
            run_config, repo, provider, output_dir=str(tmp_path / "runs"), run_id=run_id
        )
        own_collections = [x.collection_name for x in repo.list_trials(run_id)]
        # 在 safety-net cleanup 之前記錄 runner 是否已自行清乾淨。
        leftover_after_run = [n for n in own_collections if client.collection_exists(n)]
    finally:
        provider.close()
        drop_collections(client, own_collections)  # safety net

    assert result.run.status == RunStatus.COMPLETED
    trials = {t.trial_name: t for t in repo.list_trials(run_id)}
    assert set(trials) == {"A-dense-baseline", "B-identifier-aware"}
    for t in trials.values():
        assert t.status == TrialStatus.COMPLETED
        assert t.point_count and t.point_count > 0
        metrics = {m.metric_name: m for m in repo.get_trial_metrics(t.trial_id)}
        # NDCG universe == point count（完整 Trial Index Universe，不受 reranking 影響）。
        assert metrics["index_universe_size"].value == float(t.point_count)

    # 兩 Trial chunk_config 皆 current → B doc embedding 全 hit（retrieval 變更不致 cache miss）。
    a_cache = json.loads(trials["A-dense-baseline"].chunk_cache_stats_json)["embedding_cache"]
    b_cache = json.loads(trials["B-identifier-aware"].chunk_cache_stats_json)["embedding_cache"]
    assert a_cache["embedding_cache_miss_count"] > 0
    assert b_cache["embedding_cache_miss_count"] == 0
    assert b_cache["embedding_model_call_text_count"] == 0

    # identifier-aware trial 的 retrieval_meta 持久化（至少一題含 query identifier）。
    b = trials["B-identifier-aware"]
    metas = [json.loads(c.retrieval_meta_json) for c in repo.get_trial_case_results(b.trial_id)
             if c.retrieval_meta_json]
    assert metas and all(m["strategy_name"] == "identifier_aware" for m in metas)
    assert any(m["query_identifiers"] for m in metas)  # 至少一題抽出 identifier

    # 本 Run 的 collection 全清理；正式 factory_knowledge 前後一致。
    assert leftover_after_run == [], f"runner 未清除自身 collection：{leftover_after_run}"
    assert result.remaining_collections == []
    assert prod_point_count(client) == before_prod
    assert PROD_COLLECTION in list_collections(client)
    client.close()
