"""Trial Runner Qdrant 整合測試（Slice 2 §17.5）。

真實 Qdrant + 本機 BGE，使用唯一隔離 Trial collection：
- 用 tracked demo corpus 建索引、跑 v1.1 dataset、保存 Top-5、EvidenceMatcher 判 relevance。
- 產生 Metrics 與 Report。
- Trial collection 最後清理，無殘留。
- 前後 ``factory_knowledge`` 快照一致（不觸碰正式資料）。

不斷言固定向量分數，也不將特定錯誤排名寫成永久測試條件。
未就緒（無真實 Qdrant / 模型）時 skip 而非失敗。
"""
from __future__ import annotations

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
CONFIG_PATH = os.path.join(ROOT, "evaluation", "configs", "slice2-smoke.json")
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


def test_trial_runner_end_to_end(qdrant_url, tmp_path):
    from evaluation.v11 import trial_runner
    from evaluation.v11.optimizer_repository import OptimizerRepository
    from evaluation.v11.real_services import RealTrialServiceProvider
    from evaluation.v11.run_models import RunConfig, RunStatus, TrialStatus
    import json

    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    before_prod = prod_point_count(client)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        run_config = RunConfig.from_raw(json.load(f))

    repo = OptimizerRepository(str(tmp_path / "smoke.optimizer.db"))
    output_dir = str(tmp_path / "runs")
    workdir = str(tmp_path / "work")
    provider = RealTrialServiceProvider.create(qdrant_url, workdir)
    # 唯一 run_id → 唯一 trial collection，不沿用前次中斷執行的殘留。
    run_id = f"itg-run-{unique_token()}"
    own_collections: list[str] = []
    try:
        result = trial_runner.execute_run(
            run_config, repo, provider, output_dir=output_dir, run_id=run_id
        )
        own_collections = [x.collection_name for x in repo.list_trials(run_id)]
        # 在 safety-net cleanup 之前記錄 runner 是否已自行清乾淨。
        leftover_after_run = [n for n in own_collections if client.collection_exists(n)]
    finally:
        provider.close()
        drop_collections(client, own_collections)  # safety net

    # Run / Trial 完成
    assert result.run.status == RunStatus.COMPLETED
    trials = repo.list_trials(run_id)
    assert len(trials) == 1
    t = trials[0]
    assert t.status == TrialStatus.COMPLETED
    assert t.point_count and t.point_count > 0
    assert t.collection_name.startswith("optimizer_")

    # 建立唯一隔離 collection，跑 dataset，保存 Top-5，產生 Metrics
    case_results = repo.get_trial_case_results(t.trial_id)
    assert len(case_results) == len(run_config_cases(run_config))
    assert any(c.top_results_json and c.top_results_json != "[]" for c in case_results)
    metrics = {m.metric_name: m for m in repo.get_trial_metrics(t.trial_id)}
    assert metrics
    assert {"recall_at_1", "mrr", "exact_code_top1_accuracy", "index_point_count",
            "graded_ndcg_at_5", "index_universe_size"} <= set(metrics)
    # §八.10：NDCG 使用該 Trial 自己的 Index Universe，數量等於 Trial Point Count
    assert metrics["index_universe_size"].value == metrics["index_point_count"].value == float(t.point_count)
    # NDCG 為 trial_index_gold 變體，至少一題可計算
    ndcg_computable = int(metrics["graded_ndcg_computable_count"].value)
    assert ndcg_computable >= 1
    # EvidenceMatcher 有讀到 provenance（至少一題有 relevance 判定的證據理由）
    assert any(c.evidence_match_json and c.evidence_match_json != "{}" for c in case_results)

    # Report 產生
    assert os.path.exists(result.report_json_path)
    assert os.path.exists(result.report_markdown_path)

    # 本 Run 的 Trial collection 最後清理，無殘留。
    # 只檢查本測試自身建立的 collection，不斷言整個 Qdrant namespace。
    assert leftover_after_run == [], f"runner 未清除自身 collection：{leftover_after_run}"
    assert result.remaining_collections == []

    # factory_knowledge 前後快照一致（不觸碰正式資料）
    assert prod_point_count(client) == before_prod
    assert PROD_COLLECTION in list_collections(client)
    client.close()


def run_config_cases(run_config):
    import json

    with open(os.path.join(ROOT, run_config.dataset_path), encoding="utf-8") as f:
        return json.load(f)["cases"]
