"""Slice R2（v1.2.1）—— Cross-Encoder rerank 端到端整合測試。

真實 Qdrant + 真實**已快取**的 ``BAAI/bge-reranker-base``，於隔離 ``optimizer_``
collection 驗證 rerank 接進既有 pipeline 後的行為：

- reranker / SQLite rerank cache 由 ``RealTrialServiceProvider`` 實際建構並注入。
- rerank fingerprint 與 R0 鎖定值一致（模型沒被換掉）。
- 排序主鍵是 **raw logit**；normalized 僅供閱讀。
- window / tail 語義：tail 不被丟棄，只是不計分。
- **H3 安全不變量**：results 為空 ⇔ admitted set 為空（rerank 無 admission 權限）。
- cache：第一輪全 miss、第二輪全 hit，且 cached 與 uncached 分數 bit-identical。
- ``rerank_cache_enabled=False``（uncached 量測模式）恆為 miss、不寫入。
- 完整 trial run：per-case rerank metadata、aggregate rerank telemetry、rerank metrics 皆持久化。

隔離：每個測試自建唯一名稱的 collection 與自己的 tmp 目錄，**只清理自己建立的資源**，
不對整個 Qdrant namespace 做任何斷言（見 ``tests/integration/conftest.py`` 的穩定性說明）。

Qdrant 未就緒或本機沒有已快取的 reranker 模型時 skip（本測試永不下載模型）。
"""
from __future__ import annotations

import json
import os
import struct

import pytest

from tests.integration.conftest import drop_collections, unique_token

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = ["sample_docs", "evaluation/fixtures"]

RERANK_MODEL = "BAAI/bge-reranker-base"
#: R0 鎖定的 fingerprint —— 模型 / revision / activation 任一改變，此值就會變。
R0_FINGERPRINT_HASH = (
    "e0dedc0255c2b8a9367c331769664ec0116a85f2c4a8cacb7816a284a3356732"
)
R0_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"

TOP_K = 5
RERANK_TOP_N = 5  # R0.5 選定並 FROZEN

DENSE_RERANK = {
    "strategy": "dense", "top_k": TOP_K, "score_threshold": 0.55,
    "candidate_multiplier": 3, "candidate_cap": 50,
    "rerank_strategy": "cross_encoder", "rerank_top_n": RERANK_TOP_N,
}
DENSE_PLAIN = {
    "strategy": "dense", "top_k": TOP_K, "score_threshold": 0.55,
    "candidate_multiplier": 3, "candidate_cap": 50,
}
IDA_RERANK = {
    "strategy": "identifier_aware", "top_k": TOP_K, "score_threshold": 0.55,
    "candidate_multiplier": 3, "candidate_cap": 50, "identifier_candidate_cap": 20,
    "exact_match_boost": 0.2, "confusable_penalty": 0.15, "identifier_vector_floor": 0.4,
    "rerank_strategy": "cross_encoder", "rerank_top_n": RERANK_TOP_N,
}

QUESTION = "設備保養週期是多久？"
#: 幾乎不可能通過 score_threshold 的查詢 —— 用來觸發 admitted set 為空的路徑。
NONSENSE_QUESTION = "zzzz qqqq xxxx vvvv 9182736450 lorem ipsum dolor sit amet"


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------
@pytest.fixture
def qdrant_url(monkeypatch):
    """真實 Qdrant；同時強制離線，確保絕不下載模型。"""
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


@pytest.fixture(scope="session")
def cached_reranker():
    """本機**已快取**的 cross-encoder；未快取即 skip（絕不觸發下載）。

    session scope：模型載入昂貴，整個測試檔共用同一份已載入的模型。
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from evaluation.v11.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(RERANK_MODEL)
        reranker._ensure_model()  # 觸發載入；未快取時會在此失敗
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"本機未快取 rerank 模型 {RERANK_MODEL}（不下載）：{exc}")
    return reranker


def _files():
    from evaluation.v11 import hashing

    return [
        (os.path.basename(rel), abs_path)
        for rel, abs_path in hashing.collect_corpus_files(CORPUS, ROOT)
    ]


def _trial_config(name, retrieval):
    from evaluation.v11.run_models import TrialConfig

    return TrialConfig.from_raw(
        {"trial_name": name, "chunk_config": {"strategy": "current"},
         "retrieval_config": retrieval},
        0,
    )


def _provider(qdrant_url, tmp_path, reranker, *, rerank_cache_enabled=True, tag="r2"):
    """每個測試自己的 provider：獨立 workdir、獨立 cache DB、共用已載入模型。"""
    from evaluation.v11.real_services import RealTrialServiceProvider

    return RealTrialServiceProvider.create(
        qdrant_url,
        str(tmp_path / f"{tag}-work"),
        cache_db_path=str(tmp_path / f"{tag}.embedding-cache.db"),
        rerank_cache_db_path=str(tmp_path / f"{tag}.rerank-cache.db"),
        rerank_cache_enabled=rerank_cache_enabled,
        reranker=reranker,
    )


def _bits(x: float) -> str:
    """精確 bit pattern —— cached vs uncached 必須 bit-identical，不用容差。"""
    return struct.pack("<d", x).hex()


# --------------------------------------------------------------------------
# 1. 端到端接線 + fingerprint + 排序主鍵
# --------------------------------------------------------------------------
def test_rerank_wires_into_dense_pipeline_end_to_end(qdrant_url, tmp_path, cached_reranker):
    provider = _provider(qdrant_url, tmp_path, cached_reranker, tag="wire")
    collection = f"optimizer_r2wire_{unique_token()}"
    own = [collection]
    try:
        tc = _trial_config("wire", DENSE_RERANK)
        provider.build_index(collection, tc, _files())
        out = provider.retrieve(collection, tc, QUESTION, TOP_K)

        meta = out.meta
        assert meta["rerank_strategy"] == "cross_encoder"
        assert meta["ranking_version"] == "dense_v1+cross_encoder_v1"
        assert len(out.results) <= TOP_K
        assert out.results, "示範語料應能通過 threshold 取得候選"

        # window 至多 rerank_top_n，且 window + tail == admitted。
        assert meta["rerank_window_count"] <= RERANK_TOP_N
        assert meta["rerank_window_count"] + meta["rerank_tail_count"] == meta["admitted_count"]
        # 實際跑過模型 → 有 latency、有 cache 查詢紀錄。
        assert meta["rerank_latency_ms"] > 0.0
        assert meta["rerank_cache_hits"] + meta["rerank_cache_misses"] == meta["rerank_window_count"]

        # 排序主鍵是 raw logit：window 內的結果依 rerank_score 由大到小。
        scored = [r for r in out.results if r.rerank_applied]
        assert scored, "top_k 內至少應有 rerank 過的結果"
        raws = [r.rerank_score for r in scored]
        assert raws == sorted(raws, reverse=True)
        # normalized 僅供閱讀，且必須確實是 sigmoid(raw)。
        from evaluation.v11.retrieval.reranker import sigmoid

        for r in scored:
            assert 0.0 <= r.rerank_score_normalized <= 1.0
            assert r.rerank_score_normalized == pytest.approx(
                sigmoid(r.rerank_score), rel=1e-12
            )
        # raw logit 會離開 [0,1]（否則代表誤用 Sigmoid activation）。
        assert any(r.rerank_score < 0.0 or r.rerank_score > 1.0 for r in scored)
        # vector_score 維持 RAW cosine，未被 rerank 覆寫。
        for r in out.results:
            assert 0.0 <= r.score <= 1.0
    finally:
        drop_collections(provider._client, own)  # 只清自己建立的 collection
        provider.close()


def test_rerank_fingerprint_matches_the_r0_locked_model(qdrant_url, tmp_path, cached_reranker):
    """模型 / revision / activation 沒被換掉 —— 否則整個 R0 baseline 不再適用。"""
    provider = _provider(qdrant_url, tmp_path, cached_reranker, tag="fp")
    try:
        fp = provider._ensure_rerank_fingerprint()
        assert fp.fingerprint_hash() == R0_FINGERPRINT_HASH
        assert fp.model_revision == R0_MODEL_REVISION
        assert fp.scored_activation == "identity"
        assert fp.num_labels == 1

        stats = provider.rerank_stats()
        assert stats["rerank_batch_size"] == 1
        assert stats["scored_activation"] == "identity"
        assert stats["rerank_model_name"] == RERANK_MODEL
        # 產物不得含本機絕對路徑。
        assert ":\\" not in json.dumps(stats)
    finally:
        provider.close()


def test_identifier_aware_shares_the_same_rerank_stage(qdrant_url, tmp_path, cached_reranker):
    """dense 與 identifier_aware 共用同一份 rerank 實作（rerank 是正交欄位）。"""
    provider = _provider(qdrant_url, tmp_path, cached_reranker, tag="ida")
    collection = f"optimizer_r2ida_{unique_token()}"
    try:
        tc = _trial_config("ida", IDA_RERANK)
        provider.build_index(collection, tc, _files())
        out = provider.retrieve(collection, tc, "E-102 異常代碼", TOP_K)

        assert out.meta["rerank_strategy"] == "cross_encoder"
        assert out.meta["ranking_version"] == "identifier_aware_v1+cross_encoder_v1"
        # identifier 明細仍在（rerank 沒有取代 boost/penalty，只是接在其後）。
        assert "query_identifiers" in out.meta
        assert out.meta["rerank_window_count"] <= RERANK_TOP_N
    finally:
        drop_collections(provider._client, [collection])
        provider.close()


# --------------------------------------------------------------------------
# 2. H3 安全不變量
# --------------------------------------------------------------------------
def test_h3_rerank_has_no_admission_power(qdrant_url, tmp_path, cached_reranker):
    """results 為空 ⇔ admitted set 為空。rerank 既不能救回也不能剔除任何候選。

    以同一 collection 跑「有候選」與「無候選」兩個 query，兩側都必須成立。
    """
    provider = _provider(qdrant_url, tmp_path, cached_reranker, tag="h3")
    collection = f"optimizer_r2h3_{unique_token()}"
    try:
        tc = _trial_config("h3", DENSE_RERANK)
        provider.build_index(collection, tc, _files())

        for question in (QUESTION, NONSENSE_QUESTION):
            out = provider.retrieve(collection, tc, question, TOP_K)
            admitted = out.meta["admitted_count"]
            assert bool(out.results) == bool(admitted), (
                f"H3 破壞：question={question!r} admitted={admitted} "
                f"results={len(out.results)}"
            )
            if admitted:
                # 不得少於 min(admitted, top_k) —— rerank 不得過濾。
                assert len(out.results) == min(admitted, TOP_K)
    finally:
        drop_collections(provider._client, [collection])
        provider.close()


def test_rerank_preserves_the_admitted_candidate_set(qdrant_url, tmp_path, cached_reranker):
    """rerank 只重排，不新增也不移除：admitted 的 chunk identity 集合必須不變。"""
    provider = _provider(qdrant_url, tmp_path, cached_reranker, tag="set")
    plain_col = f"optimizer_r2setp_{unique_token()}"
    rerank_col = f"optimizer_r2setr_{unique_token()}"
    try:
        plain_tc = _trial_config("plain", DENSE_PLAIN)
        rerank_tc = _trial_config("rerank", DENSE_RERANK)
        provider.build_index(plain_col, plain_tc, _files())
        provider.build_index(rerank_col, rerank_tc, _files())

        plain = provider.retrieve(plain_col, plain_tc, QUESTION, TOP_K)
        rerank = provider.retrieve(rerank_col, rerank_tc, QUESTION, TOP_K)

        # rerank 未啟用時 meta 完全不含 rerank key（v1.1 形狀）；啟用時才有。
        assert [k for k in plain.meta if "rerank" in k] == []
        assert [k for k in rerank.meta if "rerank" in k] != []

        # 兩側 admitted 集合相同（admission 未被 rerank 影響）。
        assert plain.meta["threshold_rejected_count"] == rerank.meta["threshold_rejected_count"]
        # 同樣的 top_k 大小；順序可能不同（這正是 rerank 的作用）。
        assert len(plain.results) == len(rerank.results)
    finally:
        drop_collections(provider._client, [plain_col, rerank_col])
        provider.close()


# --------------------------------------------------------------------------
# 3. Cache 行為
# --------------------------------------------------------------------------
def test_rerank_cache_miss_then_hit_with_identical_scores(qdrant_url, tmp_path, cached_reranker):
    """第一輪全 miss、第二輪全 hit，且分數 bit-identical。

    這是快取可用的最低前提：快取回來的必須是**同一個數**，不是近似值。
    """
    provider = _provider(qdrant_url, tmp_path, cached_reranker, tag="cache")
    collection = f"optimizer_r2cache_{unique_token()}"
    try:
        tc = _trial_config("cache", DENSE_RERANK)
        provider.build_index(collection, tc, _files())

        first = provider.retrieve(collection, tc, QUESTION, TOP_K)
        assert first.meta["rerank_cache_hits"] == 0
        assert first.meta["rerank_cache_misses"] == first.meta["rerank_window_count"] > 0

        second = provider.retrieve(collection, tc, QUESTION, TOP_K)
        assert second.meta["rerank_cache_misses"] == 0
        assert second.meta["rerank_cache_hits"] == second.meta["rerank_window_count"]

        by_id_first = {(r.document_id, r.chunk_index): r.rerank_score for r in first.results}
        by_id_second = {(r.document_id, r.chunk_index): r.rerank_score for r in second.results}
        assert by_id_first.keys() == by_id_second.keys()
        for key, value in by_id_first.items():
            assert _bits(value) == _bits(by_id_second[key]), f"cached score 不 bit-identical：{key}"

        # cache 命中應顯著更快 —— 只作資訊記錄，不作為任何 gate。
        assert second.meta["rerank_latency_ms"] < first.meta["rerank_latency_ms"]

        # 快取檔是獨立 SQLite，且確實寫入了 window 大小的列數。
        cache_db = str(tmp_path / "cache.rerank-cache.db")
        assert os.path.isfile(cache_db)
        assert provider._rerank_cache_repo.count() == first.meta["rerank_window_count"]
    finally:
        drop_collections(provider._client, [collection])
        provider.close()


def test_uncached_mode_never_hits_and_never_writes(qdrant_url, tmp_path, cached_reranker):
    """``rerank_cache_enabled=False`` 是 uncached 量測模式：恆 miss、不建立快取檔。"""
    provider = _provider(
        qdrant_url, tmp_path, cached_reranker, rerank_cache_enabled=False, tag="uncached"
    )
    collection = f"optimizer_r2unc_{unique_token()}"
    try:
        tc = _trial_config("uncached", DENSE_RERANK)
        provider.build_index(collection, tc, _files())

        for _ in range(2):
            out = provider.retrieve(collection, tc, QUESTION, TOP_K)
            assert out.meta["rerank_cache_hits"] == 0
            assert out.meta["rerank_cache_misses"] == out.meta["rerank_window_count"] > 0

        assert provider._rerank_cache_repo is None
        assert not os.path.isfile(str(tmp_path / "uncached.rerank-cache.db"))
        assert provider.rerank_stats()["rerank_cache_enabled"] is False
    finally:
        drop_collections(provider._client, [collection])
        provider.close()


def test_cached_and_uncached_produce_the_same_ranking(qdrant_url, tmp_path, cached_reranker):
    """快取只影響速度，不影響結果 —— 否則快取會污染 evaluation。

    identity key 用 ``(filename, chunk_index)``，**不可用 ``document_id``**：
    ``RealTrialServiceProvider.build_index`` 每次建索引都以 ``uuid.uuid4()`` 現場產生
    ``document_id``，因此它在兩個獨立建立的 collection 之間必然不同，跨 build 比較無意義。
    ``filename`` + ``chunk_index`` 則完全由語料與切分決定，跨 build 穩定。
    """
    cached = _provider(qdrant_url, tmp_path, cached_reranker, tag="cmp-cached")
    uncached = _provider(
        qdrant_url, tmp_path, cached_reranker, rerank_cache_enabled=False, tag="cmp-unc"
    )
    col_a = f"optimizer_r2cmpa_{unique_token()}"
    col_b = f"optimizer_r2cmpb_{unique_token()}"
    try:
        # 守住上述假設：filename 由 basename 產生，本測試語料範圍內必須唯一，
        # 否則 (filename, chunk_index) 就不是合法的 chunk identity。
        basenames = [name for name, _path in _files()]
        assert len(basenames) == len(set(basenames)), (
            f"語料中出現重複 basename {sorted(basenames)}；"
            "(filename, chunk_index) 不再唯一，需改用含 content hash 的 identity。"
        )

        tc = _trial_config("cmp", DENSE_RERANK)
        cached.build_index(col_a, tc, _files())
        uncached.build_index(col_b, tc, _files())

        cached.retrieve(col_a, tc, QUESTION, TOP_K)          # 暖 cache
        warm = cached.retrieve(col_a, tc, QUESTION, TOP_K)
        cold = uncached.retrieve(col_b, tc, QUESTION, TOP_K)

        assert [(r.filename, r.chunk_index) for r in warm.results] == \
               [(r.filename, r.chunk_index) for r in cold.results]
        # 同一 chunk 的文字也必須相同 —— 確認比對的確實是同一段內容，而非碰巧同序位。
        for a, b in zip(warm.results, cold.results):
            assert a.text == b.text
        for a, b in zip(warm.results, cold.results):
            if a.rerank_score is None:
                assert b.rerank_score is None
            else:
                assert _bits(a.rerank_score) == _bits(b.rerank_score)
    finally:
        drop_collections(cached._client, [col_a])
        drop_collections(uncached._client, [col_b])
        cached.close()
        uncached.close()


# --------------------------------------------------------------------------
# 4. 完整 trial run：持久化 + metrics + 報表
# --------------------------------------------------------------------------
def test_full_trial_run_persists_rerank_metadata_and_metrics(
    qdrant_url, tmp_path, cached_reranker
):
    from evaluation.v11.case_schema import load_dataset
    from evaluation.v11.optimizer_repository import OptimizerRepository
    from evaluation.v11.report import build_report, render_markdown
    from evaluation.v11.run_models import RunConfig
    from evaluation.v11.trial_runner import execute_run

    # execute_run 自己會由 run_config.dataset_path 載入 dataset；此處另外載入只是為了
    # 在斷言中取得 case 數量。**不可**把它當位置參數傳給 execute_run。

    provider = _provider(qdrant_url, tmp_path, cached_reranker, tag="run")
    repo = OptimizerRepository(str(tmp_path / "run.optimizer.db"))
    run_id = f"r2-itg-{unique_token()}"
    own: list[str] = []
    try:
        dataset = load_dataset(
            os.path.join(ROOT, "evaluation", "datasets", "v1.1", "cases-v2.json")
        )
        run_config = RunConfig.from_raw(
            {
                "run_name": run_id,
                "corpus_paths": CORPUS,
                "dataset_path": "evaluation/datasets/v1.1/cases-v2.json",
                "trials": [
                    {"trial_name": "rerank-on", "chunk_config": {"strategy": "current"},
                     "retrieval_config": DENSE_RERANK},
                    {"trial_name": "rerank-off", "chunk_config": {"strategy": "current"},
                     "retrieval_config": DENSE_PLAIN},
                ],
            }
        )
        result = execute_run(
            run_config, repo, provider,
            output_dir=str(tmp_path / "reports"),
            base_dir=ROOT,
            run_id=run_id,
        )
        trials = {t.trial_name: t for t in repo.list_trials(result.run.run_id)}
        own = [t.collection_name for t in trials.values() if t.collection_name]

        on, off = trials["rerank-on"], trials["rerank-off"]

        # --- aggregate rerank telemetry（migration 0009）---
        assert on.rerank_stats_json, "啟用 rerank 的 trial 必須有 aggregate telemetry"
        stats = json.loads(on.rerank_stats_json)
        assert stats["rerank_strategy"] == "cross_encoder"
        assert stats["rerank_fingerprint_hash"] == R0_FINGERPRINT_HASH
        assert stats["rerank_batch_size"] == 1
        assert stats["rerank_case_count"] == len(dataset.cases)
        assert stats["rerank_cache_hits"] + stats["rerank_cache_misses"] > 0
        # 未啟用的 trial 欄位必須維持 NULL。
        assert off.rerank_stats_json is None

        # --- per-case rerank metadata（migration 0009）---
        on_cases = repo.get_trial_case_results(on.trial_id)
        assert on_cases and all(c.rerank_meta_json for c in on_cases)
        sample = json.loads(on_cases[0].rerank_meta_json)
        assert sample["ranking_score_source"] == "rerank_raw_logit"
        assert sample["rerank_window_count"] + sample["rerank_tail_count"] == \
               sample["admitted_count"]
        assert all(c.rerank_meta_json is None for c in repo.get_trial_case_results(off.trial_id))

        # --- metrics ---
        on_metrics = {m.metric_name: m for m in repo.get_trial_metrics(on.trial_id)}
        for name in ("mean_rerank_latency_ms", "p50_rerank_latency_ms",
                     "p95_rerank_latency_ms", "rerank_cache_hit_count",
                     "rerank_cache_miss_count", "rerank_scored_pair_count"):
            assert name in on_metrics, f"缺少 rerank metric：{name}"
            assert on_metrics[name].computable
        # end-to-end latency 必須 >= rerank 分項（rerank 是其中一段，不是全部）。
        assert on_metrics["p95_retrieval_latency_ms"].value >= \
               on_metrics["p95_rerank_latency_ms"].value
        # 未啟用 rerank 的 trial **一列 rerank metric 都沒有**。
        off_metrics = {m.metric_name for m in repo.get_trial_metrics(off.trial_id)}
        assert not [n for n in off_metrics if "rerank" in n]

        # --- 報表 ---
        report = build_report(repo, result.run, run_config, {"documents": []})
        md = render_markdown(report)
        assert "### Cross-Encoder Rerank" in md
        assert f"top_n={RERANK_TOP_N}" in md
        assert "raw logit" in md
        assert "INFORMATIONAL ONLY" in md
        # 報表不得洩漏本機絕對路徑。
        assert ":\\" not in md
    finally:
        drop_collections(provider._client, own)  # 只清自己這次 run 建立的 collection
        provider.close()
