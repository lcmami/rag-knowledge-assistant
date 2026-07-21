"""Slice 3 Trial Runner 整合（Fake）測試（§十九 / §二十二.10）。

以 Fake provider 驗證：明確多 Strategy Trial、Chunking/Cache 統計持久化、Trial A 建 cache /
Trial B 重用、失敗隔離、cleanup、resume、不自動生成 config、不選 best。
"""
from __future__ import annotations

import itertools
import json

from evaluation.v11 import trial_runner
from evaluation.v11.hashing import config_hash
from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.retrieval_metrics import IndexChunk
from evaluation.v11.run_models import RunConfig, RunStatus, TrialStatus
from evaluation.v11.trial_collections import assert_deletable
from evaluation.v11.trial_runner import IndexOutcome, RetrievalOutcome


def _ids():
    c = itertools.count()
    return lambda: f"t{next(c)}"


def _clock():
    c = itertools.count()
    return lambda: f"2026-01-01T00:00:{next(c) % 60:02d}+00:00"


def _write_dataset(tmp_path) -> str:
    ds = {"schema_version": "1.1", "dataset_version": "v-test", "cases": [
        {"id": "un", "question": "售價多少", "query_type": "unanswerable",
         "answerable": False, "group_id": "g", "split": "test", "expected_evidence": []},
    ]}
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(ds), encoding="utf-8")
    return str(p)


class Slice3FakeProvider:
    """模擬 Chunking + persistent Embedding Cache 的 fake（跨 Trial 共用 _embedded）。"""

    def __init__(self, fail_build=None):
        self.collections: set[str] = set()
        self._embedded: set[str] = set()  # 模擬 persistent cache（跨 Trial 共用）
        self.build_calls: list[str] = []
        self._fail_build = fail_build or set()

    def _fake_texts(self, trial_config) -> list[str]:
        # 相同 chunk_config → 相同文字集合（→ 第二個相同 chunk_config 的 Trial 全 hit）。
        sig = config_hash(trial_config.chunk_config)
        return [f"{sig}-chunk-{i}" for i in range(3)]

    def build_index(self, collection_name, trial_config, files) -> IndexOutcome:
        assert_deletable(collection_name)
        self.build_calls.append(trial_config.trial_name)
        if trial_config.trial_name in self._fail_build:
            raise RuntimeError("注入 build 失敗")
        self.collections.add(collection_name)
        texts = self._fake_texts(trial_config)
        hits = sum(1 for t in texts if t in self._embedded)
        miss = len(texts) - hits
        for t in texts:
            self._embedded.add(t)
        strategy = trial_config.chunk_config.get("strategy")
        fallback = 1 if strategy == "heading" else 0
        stats = {
            "stats_version": "v1",
            "strategy": strategy,
            "chunk_config": trial_config.chunk_config,
            "chunk_config_hash": config_hash(trial_config.chunk_config),
            "chunk_count": len(texts),
            "strategy_fallback_count": fallback,
            "strategy_fallback_reasons": (["heading:no_section_metadata->paragraph"] if fallback else []),
            "section_prefix_enabled": bool(trial_config.chunk_config.get("prepend_section_title")),
            "embedding_cache": {
                "cache_enabled": True,
                "document_chunk_count": len(texts),
                "unique_embedding_text_count": len(texts),
                "embedding_cache_hit_count": hits,
                "embedding_cache_miss_count": miss,
                "embedding_model_call_text_count": miss,
                "batch_dedup_savings": 0,
                "embedding_cache_hit_rate": (hits / (hits + miss)) if (hits + miss) else None,
            },
            "embedding_fingerprint_hash": "fp-abc",
            "cache_schema_version": "v1",
        }
        return IndexOutcome(point_count=len(texts), build_time_ms=5, chunk_count=len(texts),
                            strategy=strategy, chunk_config_hash=config_hash(trial_config.chunk_config),
                            stats=stats)

    def index_manifest(self, collection_name) -> list[IndexChunk]:
        return []

    def retrieve(self, collection_name, trial_config, question, top_k) -> RetrievalOutcome:
        return RetrievalOutcome(results=[], retrieval_time_ms=1)

    def collection_exists(self, collection_name) -> bool:
        return collection_name in self.collections

    def delete_collection(self, collection_name) -> None:
        assert_deletable(collection_name)
        self.collections.discard(collection_name)

    def close(self):
        pass


TRIALS = [
    {"trial_name": "A-current", "chunk_config": {"strategy": "current"},
     "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
    {"trial_name": "B-current-thr", "chunk_config": {"strategy": "current"},
     "retrieval_config": {"top_k": 5, "score_threshold": 0.5}},
    {"trial_name": "C-paragraph", "chunk_config": {"strategy": "paragraph", "target_chars": 400, "overlap_chars": 50, "max_tokens": 480},
     "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
    {"trial_name": "D-heading", "chunk_config": {"strategy": "heading", "prepend_section_title": True},
     "retrieval_config": {"top_k": 5, "score_threshold": 0.55}},
]


def _cfg(tmp_path, trials=None, **flags):
    raw = {"schema_version": "1.0", "run_name": "s3", "random_seed": 42,
           "dataset_path": _write_dataset(tmp_path),
           "corpus_paths": ["sample_docs", "evaluation/fixtures"], "trials": trials or TRIALS}
    raw.update(flags)
    return RunConfig.from_raw(raw)


def _run(repo, cfg, provider, tmp_path, run_id="run-s3"):
    return trial_runner.execute_run(cfg, repo, provider, output_dir=str(tmp_path / "runs"),
                                    run_id=run_id, now_fn=_clock(), id_factory=_ids())


def test_multi_strategy_trials_persist_stats(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "o.optimizer.db"))
    provider = Slice3FakeProvider()
    result = _run(repo, _cfg(tmp_path), provider, tmp_path)
    assert result.run.status == RunStatus.COMPLETED
    trials = repo.list_trials("run-s3")
    assert [t.trial_name for t in trials] == ["A-current", "B-current-thr", "C-paragraph", "D-heading"]
    for t in trials:
        assert t.status == TrialStatus.COMPLETED
        assert t.strategy in ("current", "paragraph", "heading")
        assert t.chunk_count == 3
        assert t.chunk_config_hash
        stats = json.loads(t.chunk_cache_stats_json)
        assert "embedding_cache" in stats and "strategy" in stats


def test_trial_a_builds_cache_trial_b_reuses(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "o.optimizer.db"))
    provider = Slice3FakeProvider()
    _run(repo, _cfg(tmp_path), provider, tmp_path)
    by_name = {t.trial_name: t for t in repo.list_trials("run-s3")}
    a = json.loads(by_name["A-current"].chunk_cache_stats_json)["embedding_cache"]
    b = json.loads(by_name["B-current-thr"].chunk_cache_stats_json)["embedding_cache"]
    # A/B chunk_config 相同 → A 全 miss（建 cache）、B 全 hit（重用 document embedding）。
    assert a["embedding_cache_miss_count"] == 3 and a["embedding_cache_hit_count"] == 0
    assert b["embedding_cache_hit_count"] == 3 and b["embedding_cache_miss_count"] == 0
    # C 為不同 chunk_config → 重算（miss）。
    c = json.loads(by_name["C-paragraph"].chunk_cache_stats_json)["embedding_cache"]
    assert c["embedding_cache_miss_count"] == 3


def test_strategy_fallback_persisted(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "o.optimizer.db"))
    _run(repo, _cfg(tmp_path), Slice3FakeProvider(), tmp_path)
    d = {t.trial_name: t for t in repo.list_trials("run-s3")}["D-heading"]
    stats = json.loads(d.chunk_cache_stats_json)
    assert stats["strategy_fallback_count"] == 1
    assert stats["strategy_fallback_reasons"]


def test_trial_failure_isolation_and_cleanup(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "o.optimizer.db"))
    provider = Slice3FakeProvider(fail_build={"C-paragraph"})
    result = _run(repo, _cfg(tmp_path), provider, tmp_path)
    by_name = {t.trial_name: t for t in repo.list_trials("run-s3")}
    assert by_name["C-paragraph"].status == TrialStatus.FAILED
    # 其他 Trial 不受影響。
    assert by_name["A-current"].status == TrialStatus.COMPLETED
    assert by_name["D-heading"].status == TrialStatus.COMPLETED
    assert result.run.status == RunStatus.FAILED  # 有未完成 Trial
    assert provider.collections == set()  # 全部 collection 已清理


def test_resume_reuses_cache_and_skips_completed(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "o.optimizer.db"))
    provider = Slice3FakeProvider(fail_build={"C-paragraph"})
    _run(repo, _cfg(tmp_path), provider, tmp_path)
    # resume（retry failed）：同一 provider，_embedded 仍在 → 重跑 C 為 miss（不同 config），A/B 已完成跳過。
    provider._fail_build = set()
    provider.build_calls.clear()
    result = _run(repo, _cfg(tmp_path, retry_failed=True), provider, tmp_path)
    assert result.run.status == RunStatus.COMPLETED
    assert "C-paragraph" in provider.build_calls
    assert "A-current" not in provider.build_calls  # completed 不重跑


def test_no_auto_config_generation_and_no_best_selection(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "o.optimizer.db"))
    result = _run(repo, _cfg(tmp_path), Slice3FakeProvider(), tmp_path)
    # 只執行明確列出的 4 個 Trial，不自動展開參數矩陣。
    assert len(repo.list_trials("run-s3")) == 4
    # 報告不含任何 best/winner 欄位。
    report = json.loads(open(result.report_json_path, encoding="utf-8").read())
    keys = json.dumps(report, ensure_ascii=False)
    assert "best_trial" not in keys and "winner" not in keys
