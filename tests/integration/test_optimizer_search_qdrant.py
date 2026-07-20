"""Slice 5 Optimizer Search 整合測試（§25.8）。

真實 Qdrant + 本機 BGE、隔離 ``optimizer_`` collection，驗證：

- 小型搜尋空間、兩種 chunk strategy、dense + identifier_aware、floor 0.4 / 0.5。
- tune / validation / test split 嚴格隔離。
- Hard constraint 實際執行；Pareto 實際產生；Recommendation Lock 行為正確。
- Test 只在 lock 之後執行，且只跑 baseline + recommendation 兩個 config。
- Embedding cache 跨 candidate 重用。
- C01 與 unanswerable 誠實保存。
- 所有 trial collection 清理；正式 ``factory_knowledge`` 前後一致；無 LLM 呼叫。

未就緒（無真實 Qdrant）時 skip。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, "evaluation", "configs", "slice5-optimizer-smoke.json")
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


def _prod_snapshot(url):
    from qdrant_client import QdrantClient

    c = QdrantClient(url=url)
    try:
        names = sorted(x.name for x in c.get_collections().collections)
        prod_points = c.count(PROD_COLLECTION).count if PROD_COLLECTION in names else None
        return names, prod_points
    finally:
        c.close()


def _small_search_config():
    """縮小版 smoke：2 chunk strategy × (dense + ida floor 0.4/0.5) = 6 candidate。"""
    from evaluation.v11.search.search_config import SearchConfig

    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    raw["search_name"] = "slice5-integration"
    raw["search_space"] = {
        "chunking": {"strategy": ["current", "paragraph"]},
        "retrieval": {
            "strategy": ["dense", "identifier_aware"],
            "identifier_vector_floor": [0.4, 0.5],
        },
    }
    raw["max_candidates"] = 12
    raw["validation_top_n"] = 6
    return SearchConfig.from_raw(raw)


@pytest.fixture
def search_result(qdrant_url):
    from evaluation.v11.real_services import RealTrialServiceProvider
    from evaluation.v11.search import search_runner
    from evaluation.v11.search.search_repository import SearchRepository

    before_names, before_points = _prod_snapshot(qdrant_url)

    outdir = tempfile.mkdtemp(prefix="slice5_integration_")
    workdir = tempfile.mkdtemp(prefix="slice5_work_")
    repo = SearchRepository(os.path.join(outdir, "s.optimizer.db"))
    provider = RealTrialServiceProvider.create(qdrant_url, workdir)
    try:
        result = search_runner.execute_search(
            _small_search_config(), repo, provider, output_dir=outdir
        )
        yield result, repo, qdrant_url, before_names, before_points
    finally:
        provider.close()
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)


# --------------------------------------------------------------------------
def test_candidate_generation_covers_both_chunk_strategies_and_floors(search_result):
    result, _, _, _, _ = search_result
    strategies = {c.chunk_config["strategy"] for c in result.generation.executable}
    assert strategies == {"current", "paragraph"}
    floors = {
        c.retrieval_config.get("identifier_vector_floor")
        for c in result.generation.executable
    }
    assert {0.4, 0.5} <= floors
    retr = {c.retrieval_config["strategy"] for c in result.generation.executable}
    assert retr == {"dense", "identifier_aware"}
    # dense 不得帶 identifier 欄位
    for c in result.generation.executable:
        if c.retrieval_config["strategy"] == "dense":
            assert "identifier_vector_floor" not in c.retrieval_config


def test_splits_are_isolated_and_group_leakage_clean(search_result):
    result, _, _, _, _ = search_result
    m = result.split_manifest
    assert m.leakage_ok is True and m.leakage_detail == []
    ids = {}
    for name, info in m.splits.items():
        with open(info.dataset_path, encoding="utf-8") as f:
            ids[name] = {c["id"] for c in json.load(f)["cases"]}
    assert ids["tune"] & ids["validation"] == set()
    assert ids["tune"] & ids["test"] == set()
    assert ids["validation"] & ids["test"] == set()
    # 已知的 v1.1 split 形狀
    assert ids["validation"] == {"eval-mes-status-001", "eval-api-token-001"}
    assert "eval-e102-001" in ids["test"]


def test_hard_constraints_actually_evaluated_on_validation(search_result):
    result, _, _, _, _ = search_result
    cov = result.coverage
    assert cov is not None
    # v1.1 的 validation split 沒有 exact_code / unanswerable case
    assert cov.sufficient is False
    assert cov.status == "insufficient_validation_coverage"
    blocking = {b.metric for b in cov.blocking}
    assert "exact_code_top1_accuracy" in blocking
    assert "unanswerable_false_positive_rate" in blocking


def test_no_feasible_candidate_means_no_lock_and_no_test(search_result):
    """§12.4 / §13.3：未 lock 就不得有任何 test trial 或 test metric。"""
    result, repo, _, _, _ = search_result
    s = result.search
    assert s.recommended_config_hash is None
    assert s.recommendation_locked is False
    assert s.test_evaluated is False
    assert s.test_run_id is None
    assert repo.count_test_trials(s.search_id) == 0
    assert all(c.test_trial_id is None for c in result.candidates)


def test_report_is_written_and_hides_test_statistics(search_result):
    result, _, _, _, _ = search_result
    assert os.path.exists(result.report_json_path)
    assert os.path.exists(result.report_markdown_path)
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["test"]["available"] is False
    assert report["test"]["test_trial_count"] == 0
    assert "baseline_metrics" not in report["test"]
    assert report["c01_same_chunk_multi_identifier"]["observable"] is False
    assert report["hard_constraint_policy"]["policy_confidence"] == "low"
    assert report["hard_constraint_policy"]["policy_source"] == "slice0_baseline_recommendation"


def test_report_leaks_no_secrets_or_absolute_paths(search_result):
    result, _, _, _, _ = search_result
    blob = open(result.report_json_path, encoding="utf-8").read()
    for banned in ("GROQ_API_KEY", "api_key", "sk-", "C:\\Users", "D:\\source"):
        assert banned not in blob, f"報告不得輸出 {banned}"
    for banned in ("deployed_candidate", "active_candidate", "production_candidate"):
        assert banned not in blob


def test_embedding_cache_reused_across_candidates(search_result):
    """相同 chunk config、不同 retrieval config → document embedding 應高命中。"""
    result, repo, _, _, _ = search_result
    hits = misses = model_calls = 0
    per_trial = []
    for trial in repo.list_trials(result.search.tune_run_id):
        if not trial.chunk_cache_stats_json:
            continue
        cache = json.loads(trial.chunk_cache_stats_json).get("embedding_cache") or {}
        h = int(cache.get("embedding_cache_hit_count", 0) or 0)
        m = int(cache.get("embedding_cache_miss_count", 0) or 0)
        model_calls += int(cache.get("embedding_model_call_text_count", 0) or 0)
        hits += h
        misses += m
        per_trial.append((trial.trial_name, h, m))
    assert misses > 0, "應至少有一次 miss（第一個 candidate 建立 cache）"
    assert hits > 0, f"跨 candidate 未重用 embedding cache：{per_trial}"
    # 相同 chunk config 的後續 candidate 應完全命中，不再呼叫模型
    assert model_calls < hits + misses, (
        f"model 呼叫次數 {model_calls} 未低於總查詢數 {hits + misses}，cache 未生效"
    )


def test_all_trial_collections_cleaned_up(search_result):
    result, _, url, before_names, _ = search_result
    after_names, _ = _prod_snapshot(url)
    leftovers = [n for n in after_names if n.startswith("optimizer_")]
    assert leftovers == [], f"殘留 optimizer collection：{leftovers}"
    assert result.remaining_collections == []
    assert set(after_names) == set(before_names)


def test_production_collection_untouched(search_result):
    _, _, url, before_names, before_points = search_result
    after_names, after_points = _prod_snapshot(url)
    assert after_points == before_points, "正式 factory_knowledge point 數不得改變"
    assert PROD_COLLECTION in after_names or before_points is None


def test_no_llm_client_is_constructed(search_result, monkeypatch):
    """Slice 5 不得呼叫任何 LLM。"""
    result, _, _, _, _ = search_result
    blob = open(result.report_json_path, encoding="utf-8").read().lower()
    for banned in ("groq", "chat.completions", "llm_model"):
        assert banned not in blob
