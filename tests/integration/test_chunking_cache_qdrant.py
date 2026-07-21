"""Slice 3 Chunking + Embedding Cache 整合測試（§二十二.11）。

真實 Qdrant + 本機 BGE，跑 slice3-chunking-smoke（current / paragraph / heading + current
threshold variant）：
- 各 strategy 建索引成功；Index Point Count == Chunk Count。
- 相同 Chunk Text 重用 Embedding（Trial B 相同 chunk_config → document embedding 全 hit，
  底層 model text count 下降到 0）。
- Provenance 可被 EvidenceMatcher 使用（至少一題有 evidence match）。
- NDCG 使用完整 Trial Index Universe（universe size == point count）。
- 所有 Trial collection 清除；Cache 使用暫存路徑。
- 正式 ``factory_knowledge`` 前後快照一致。

不斷言固定向量分數，也不宣稱某 strategy 必然更好。未就緒時 skip。
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
    wait_collection_absent,
)

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, "evaluation", "configs", "slice3-chunking-smoke.json")
PROD_COLLECTION = "factory_knowledge"


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


def test_slice3_smoke_end_to_end(qdrant_url, tmp_path):
    from qdrant_client import QdrantClient

    from evaluation.v11 import trial_runner
    from evaluation.v11.optimizer_repository import OptimizerRepository
    from evaluation.v11.real_services import RealTrialServiceProvider
    from evaluation.v11.run_models import RunConfig, RunStatus, TrialStatus

    client = QdrantClient(url=qdrant_url)
    before_prod = prod_point_count(client)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        run_config = RunConfig.from_raw(json.load(f))

    # 唯一 run_id → 唯一 trial collection 名稱，不沿用前次中斷執行的殘留。
    run_id = f"s3-itg-{unique_token()}"
    repo = OptimizerRepository(str(tmp_path / "s3.optimizer.db"))
    output_dir = str(tmp_path / "runs")
    workdir = str(tmp_path / "work")
    cache_db = str(tmp_path / "s3.embedding-cache.db")  # 暫存 cache 路徑
    provider = RealTrialServiceProvider.create(qdrant_url, workdir, cache_db_path=cache_db)
    own_collections: list[str] = []
    try:
        result = trial_runner.execute_run(
            run_config, repo, provider, output_dir=output_dir, run_id=run_id
        )
        own_collections = [t.collection_name for t in repo.list_trials(run_id)]
        # 在 safety-net cleanup 之前記錄「runner 是否已自行清乾淨」，
        # 否則下面的 finally 會遮蔽真正的 cleanup 缺陷。
        leftover_after_run = [n for n in own_collections if client.collection_exists(n)]
    finally:
        provider.close()
        drop_collections(client, own_collections)  # safety net，不取代上面的斷言

    assert result.run.status == RunStatus.COMPLETED
    trials = {t.trial_name: t for t in repo.list_trials(run_id)}
    assert set(trials) == {"A-current-baseline", "B-current-threshold-variant", "C-paragraph", "D-heading-prefix"}

    for name, t in trials.items():
        assert t.status == TrialStatus.COMPLETED, name
        assert t.point_count and t.point_count > 0
        # Index Point Count == Chunk Count（每個 chunk 一個 point）。
        assert t.point_count == t.chunk_count, name
        assert t.collection_name.startswith("optimizer_")
        stats = json.loads(t.chunk_cache_stats_json)
        assert stats["strategy"] == t.strategy
        assert "embedding_cache" in stats
        # NDCG Universe 大小 == point count（完整 Trial Index Universe）。
        metrics = {m.metric_name: m for m in repo.get_trial_metrics(t.trial_id)}
        assert metrics["index_universe_size"].value == float(t.point_count)

    # 相同 chunk_config（A/B）→ Trial B document embedding 全 hit、底層 model text count == 0。
    a_cache = json.loads(trials["A-current-baseline"].chunk_cache_stats_json)["embedding_cache"]
    b_cache = json.loads(trials["B-current-threshold-variant"].chunk_cache_stats_json)["embedding_cache"]
    assert a_cache["embedding_cache_miss_count"] > 0  # A 首次建立 cache
    assert b_cache["embedding_cache_hit_count"] == b_cache["unique_embedding_text_count"]
    assert b_cache["embedding_cache_miss_count"] == 0
    assert b_cache["embedding_model_call_text_count"] == 0  # B 未呼叫底層 model

    # C/D 為不同 chunk text → 需重算（至少有 miss）。
    c_cache = json.loads(trials["C-paragraph"].chunk_cache_stats_json)["embedding_cache"]
    assert c_cache["embedding_cache_miss_count"] > 0

    # heading fallback（sample corpus 若有無 section 的 TXT → fallback 可見；否則為 0，不強制）。
    d_stats = json.loads(trials["D-heading-prefix"].chunk_cache_stats_json)
    assert d_stats["section_prefix_enabled"] is True
    assert "strategy_fallback_reasons" in d_stats

    # Provenance 可被 EvidenceMatcher 使用（至少一 Trial 有 evidence match）。
    any_evidence = False
    for t in trials.values():
        for c in repo.get_trial_case_results(t.trial_id):
            em = json.loads(c.evidence_match_json)
            if isinstance(em, dict) and em.get("matches"):
                any_evidence = True
    assert any_evidence

    # 所有本 Run 的 Trial collection 由 runner 自行清除、無殘留；
    # 只檢查「本測試自己建立的 collection」，不斷言整個 Qdrant namespace，
    # 否則任何無關殘留都會讓本測試失敗（間歇性根因）。
    assert leftover_after_run == [], f"runner 未清除自身 collection：{leftover_after_run}"
    assert result.remaining_collections == []
    # 正式 factory_knowledge 前後一致且仍存在。
    assert prod_point_count(client) == before_prod
    assert PROD_COLLECTION in list_collections(client)
    client.close()
