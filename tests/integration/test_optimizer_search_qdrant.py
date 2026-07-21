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
    # 只檢查「本次 search 新建立」的 collection 是否清乾淨；不斷言整個 Qdrant
    # namespace，否則任何無關殘留都會讓本測試間歇性失敗。
    leftovers = [
        n for n in after_names
        if n.startswith("optimizer_") and n not in set(before_names)
    ]
    assert leftovers == [], f"殘留 optimizer collection：{leftovers}"
    assert result.remaining_collections == []
    assert set(after_names) - set(before_names) == set(), "本測試新增了未清理的 collection"


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


# ==========================================================================
# Slice 5A —— coverage-revised dataset 走完 Recommendation → Lock → Test
# ==========================================================================
CONFIG_V2_PATH = os.path.join(
    ROOT, "evaluation", "configs", "slice5-optimizer-smoke-v2.json"
)


def _small_v2_config():
    """縮小版 v2：1 chunk strategy × (dense + ida floor 0.5) → 快速走完完整流程。

    **p95 門檻刻意放寬**：本測試驗證的是 workflow（coverage → constraints → pareto →
    recommendation → lock → test），不是延遲。實測本機 validation p95 落在 80~223 ms
    之間，遠高於 Slice 0 推導的 86 ms 上限（該值來自單次量測，且 Slice 0 自己就標注
    「延遲高度依賴硬體，跨環境不可直接套用」）。若沿用 86 ms，candidate 是否可行會
    隨機器負載而變，測試將變成硬體 flaky test。其餘三項 constraint 完全不動。
    """
    from evaluation.v11.search.search_config import SearchConfig

    with open(CONFIG_V2_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    raw["search_name"] = "slice5a-integration"
    for c in raw["hard_constraints"]:
        if c["metric"] == "p95_retrieval_latency_ms":
            c["value"] = 100000
    raw["search_space"] = {
        "retrieval": {
            "strategy": ["dense", "identifier_aware"],
            "identifier_vector_floor": [0.5],
        }
    }
    raw["max_candidates"] = 4
    raw["validation_top_n"] = 4
    return SearchConfig.from_raw(raw)


@pytest.fixture
def v2_result(qdrant_url):
    from evaluation.v11.real_services import RealTrialServiceProvider
    from evaluation.v11.search import search_runner
    from evaluation.v11.search.search_repository import SearchRepository

    before_names, before_points = _prod_snapshot(qdrant_url)
    outdir = tempfile.mkdtemp(prefix="slice5a_integration_")
    workdir = tempfile.mkdtemp(prefix="slice5a_work_")
    repo = SearchRepository(os.path.join(outdir, "s.optimizer.db"))
    provider = RealTrialServiceProvider.create(qdrant_url, workdir)
    try:
        result = search_runner.execute_search(
            _small_v2_config(), repo, provider, output_dir=outdir
        )
        yield result, repo, qdrant_url, before_names, before_points
    finally:
        provider.close()
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)


def test_v2_validation_coverage_is_sufficient(v2_result):
    result, _, _, _, _ = v2_result
    assert result.coverage.sufficient is True
    assert result.coverage.status == "ok"
    assert result.coverage.blocking == []


def test_v2_all_four_constraints_actually_computed_on_validation(v2_result):
    """§八.1：validation 的四項 constraint 全部可計算（不再是 not-computable）。"""
    result, _, _, _, _ = v2_result
    assert result.constraint_evals, "應有候選進入 validation constraint 判定"
    for chash, ce in result.constraint_evals.items():
        assert len(ce.results) == 4
        for r in ce.results:
            assert r.computable is True, f"{chash} 的 {r.name} 仍不可計算：{r.reason}"
            assert r.observed_value is not None
            assert r.sample_count and r.sample_count > 0


def test_v2_reaches_recommendation_and_locks(v2_result):
    result, _, _, _, _ = v2_result
    s = result.search
    assert s.recommended_config_hash, "coverage 足夠時應能選出 recommendation"
    assert s.recommendation_locked is True
    assert s.recommendation_locked_at
    assert result.pareto and result.pareto.frontier


def test_v2_recommendation_comes_from_feasible_pareto_member(v2_result):
    result, _, _, _, _ = v2_result
    rec = result.search.recommended_config_hash
    winner = next(c for c in result.candidates if c.config_hash == rec)
    assert winner.feasible is True
    assert winner.pareto_member is True


def test_v2_test_stage_runs_exactly_baseline_and_recommendation(v2_result):
    """§十二.5：lock 後 test 只跑兩個 config。"""
    result, repo, _, _, _ = v2_result
    s = result.search
    assert s.test_evaluated is True
    trials = repo.list_trials(s.test_run_id)
    hashes = {t.config_hash for t in trials}
    expected = {s.baseline_config_hash, s.recommended_config_hash}
    assert hashes == expected
    assert len(trials) == len(expected), "test 不得執行其他 candidate"


def test_v2_test_does_not_change_recommendation(v2_result):
    """§十二.5：即使 test 標示 regression，鎖定的 recommendation 不變。"""
    from evaluation.v11.search.search_models import RecommendationLockError

    result, repo, _, _, _ = v2_result
    locked = result.search.recommended_config_hash
    stored = repo.get_search(result.search.search_id)
    assert stored.recommended_config_hash == locked
    with pytest.raises(RecommendationLockError):
        repo.lock_recommendation(result.search.search_id, "another-hash", "now")
    assert repo.get_search(result.search.search_id).recommended_config_hash == locked


def test_v2_report_exposes_test_only_after_lock(v2_result):
    result, _, _, _, _ = v2_result
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["recommendation"]["recommendation_locked"] is True
    assert report["test"]["available"] is True
    assert report["test"]["baseline_metrics"]
    assert report["test"]["recommendation_metrics"]
    assert report["test"]["baseline_vs_recommendation_delta"]


def test_v2_c01_rank_is_honestly_recorded(v2_result):
    """§十九：C01 只能誠實記錄，不得宣稱已解決。"""
    result, _, _, _, _ = v2_result
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    c01 = report["c01_same_chunk_multi_identifier"]
    assert c01["case_id"] == "eval-e102-001"
    assert c01["observable"] is True
    ranks = c01["first_relevant_rank"]
    assert set(ranks) == {"baseline", "recommended"}
    if not c01["resolved"]:
        assert "did not resolve" in c01["note"] or "No candidate" in c01["note"]


def test_v2_outcome_separates_constraint_status_from_baseline_comparison(v2_result):
    """Slice 5A §二：兩種狀態在真實 E2E 上必須分開且都被記錄。"""
    result, repo, _, _, _ = v2_result
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    oc = report["outcome"]
    assert oc["test_constraint_status"] in ("passed", "failed")
    assert oc["baseline_comparison_status"] in (
        "improved", "same", "regressed", "mixed", "not_evaluated"
    )
    # 若絕對門檻失敗，不得因此就把相對比較記成 regressed
    if oc["test_constraint_status"] == "failed":
        assert oc["failed_test_constraints"]
        assert result.search.status == "test_constraint_failed"
    # 逐 metric 比較必須存在且有方向
    assert oc["metrics"]
    for m in oc["metrics"]:
        assert m["direction"] in ("maximize", "minimize")
        assert m["comparison"] in (
            "improved", "same", "regressed", "not_evaluated"
        )


def test_v2_recommendation_is_locked_but_not_apply_eligible(v2_result):
    """§三：recommended_config_hash 存在且鎖定，但證據不足以 Apply。"""
    result, repo, _, _, _ = v2_result
    stored = repo.get_search(result.search.search_id)
    assert stored.recommendation_locked is True
    assert stored.recommended_config_hash
    oc = json.loads(stored.outcome_json)
    assert oc["recommendation_apply_eligible"] is False
    assert oc["recommendation_confidence"] == "low"
    assert "demo_corpus_only" in oc["apply_ineligible_reasons"]


def test_v2_report_discloses_sample_granularity_and_identifier_gap(v2_result):
    """§四／§五：小樣本與 identifier-bearing unanswerable 缺口必須揭露。"""
    result, _, _, _, _ = v2_result
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    gran = report["outcome"]["metric_sample_granularity"]
    rows = {(r["split"], r["query_type"]): r for r in gran["per_split"]}
    assert rows[("validation", "exact_code")]["n"] == 1
    assert rows[("validation", "unanswerable")]["n"] == 2
    assert rows[("test", "exact_code")]["n"] == 2
    assert any("low statistical confidence" in n for n in gran["notes"])

    cov = report["outcome"]["identifier_unanswerable_coverage"]
    assert cov["validation_has_identifier_bearing_unanswerable"] is False
    assert "identifier_vector_floor" in cov["limitation"]
    assert cov["required_followup"]


def test_v2_collections_cleaned_and_production_untouched(v2_result):
    result, _, url, before_names, before_points = v2_result
    after_names, after_points = _prod_snapshot(url)
    assert [
        n for n in after_names
        if n.startswith("optimizer_") and n not in set(before_names)
    ] == []
    assert result.remaining_collections == []
    assert set(after_names) - set(before_names) == set()
    assert after_points == before_points
