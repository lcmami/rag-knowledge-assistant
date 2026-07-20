"""Slice 4 Trial Runner 整合（Fake）測試（§二十一.8）。

以 Fake provider 驗證 runner 對 identifier-aware / dense strategy 的整合：明確兩 strategy、
per-case retrieval metadata 持久化、ranking metadata 進 top_results、doc embedding cache 重用
（retrieval config 變更不致 cache miss）、query embed 每 Trial 執行、failure isolation、cleanup、
resume、不自動生成 config、不選 best。
"""
from __future__ import annotations

import itertools
import json

from evaluation.v11 import trial_runner
from evaluation.v11.hashing import config_hash
from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.retrieval_metrics import IndexChunk, RankedResult
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
        {"id": "e102", "question": "E102 警報如何處理", "query_type": "exact_code",
         "answerable": True, "group_id": "g1", "split": "tune",
         "expected_evidence": [{
             "document_sha256": "sha-a", "canonical_text_version": "v1", "loader_version": "v1",
             "section": "E102", "offset_scope": "document", "char_start": 0, "char_end": 50,
             "anchor_text": ["E102"]}]},
        {"id": "un", "question": "售價多少", "query_type": "unanswerable",
         "answerable": False, "group_id": "g2", "split": "test", "expected_evidence": []},
    ]}
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(ds), encoding="utf-8")
    return str(p)


class Slice4FakeProvider:
    """模擬 doc embedding cache（跨 Trial 共用）+ dense / identifier-aware retrieve。"""

    def __init__(self, fail_build=None):
        self.collections: set[str] = set()
        self._embedded: set[str] = set()          # persistent doc embedding cache（跨 Trial）
        self.query_embed_calls: list[str] = []    # 每次 retrieve 記一次（query embed 不快取）
        self.build_calls: list[str] = []
        self._fail_build = fail_build or set()
        self.last_cache_miss: dict[str, int] = {}

    def build_index(self, collection_name, trial_config, files) -> IndexOutcome:
        assert_deletable(collection_name)
        self.build_calls.append(trial_config.trial_name)
        if trial_config.trial_name in self._fail_build:
            raise RuntimeError("注入 build 失敗")
        self.collections.add(collection_name)
        # doc embedding 只依 chunk_config（不依 retrieval config）→ 同 chunk_config 第二個 Trial 全 hit。
        sig = config_hash(trial_config.chunk_config)
        texts = [f"{sig}-chunk-{i}" for i in range(3)]
        miss = sum(1 for t in texts if t not in self._embedded)
        for t in texts:
            self._embedded.add(t)
        self.last_cache_miss[trial_config.trial_name] = miss
        stats = {
            "stats_version": "v1", "strategy": trial_config.chunk_config.get("strategy"),
            "chunk_config": trial_config.chunk_config,
            "chunk_config_hash": config_hash(trial_config.chunk_config),
            "chunk_config_origin": "explicit_trial_config", "chunk_count": len(texts),
            "embedding_cache": {"cache_enabled": True, "embedding_cache_miss_count": miss,
                                "embedding_cache_hit_count": len(texts) - miss},
        }
        return IndexOutcome(point_count=len(texts), build_time_ms=5, chunk_count=len(texts),
                            strategy=trial_config.chunk_config.get("strategy"),
                            chunk_config_hash=config_hash(trial_config.chunk_config), stats=stats)

    def index_manifest(self, collection_name) -> list[IndexChunk]:
        return []

    def retrieve(self, collection_name, trial_config, question, top_k) -> RetrievalOutcome:
        self.query_embed_calls.append(question)  # 模擬每 Trial 每題算一次 query embedding
        strategy = trial_config.strategy
        if strategy == "identifier_aware" and "E102" in question:
            r = RankedResult(
                rank=1, document_id="doc-a", document_sha256="sha-a", filename="f.md",
                section="E102", page=None, chunk_index=0, score=0.50, text="E102 ...",
                provenance_spans=[{"document_sha256": "sha-a", "canonical_text_version": "v1",
                                   "offset_scope": "document", "char_start": 0, "char_end": 50}],
                identifiers=["E102"], canonical_text_version="v1",
                ranking_score=0.70, exact_identifier_matches=["E102"], conflicting_identifiers=[],
                exact_match_boost=0.2, confusable_penalty=0.0, candidate_sources=["dense", "identifier"],
                admission_reason="exact_identifier_floor",
            )
            meta = {"strategy_name": "identifier_aware", "ranking_version": "identifier_aware_v1",
                    "query_identifiers": ["E102"], "dense_candidate_count": 3,
                    "identifier_candidate_count": 1, "union_candidate_count": 4,
                    "deduplicated_count": 3, "exact_match_candidate_count": 1,
                    "confusable_candidate_count": 1, "threshold_rejected_count": 0,
                    "identifier_floor_rejected_count": 0, "warnings": []}
            return RetrievalOutcome(results=[r], retrieval_time_ms=2, meta=meta)
        # dense（或 unanswerable）→ 無結果 / dense meta。
        meta = {"strategy_name": strategy, "ranking_version": f"{strategy}_v1",
                "query_identifiers": [], "dense_candidate_count": 3, "identifier_candidate_count": 0,
                "union_candidate_count": 3, "deduplicated_count": 3, "exact_match_candidate_count": 0,
                "confusable_candidate_count": 0, "threshold_rejected_count": 0,
                "identifier_floor_rejected_count": 0, "warnings": []}
        return RetrievalOutcome(results=[], retrieval_time_ms=1, meta=meta)

    def collection_exists(self, collection_name) -> bool:
        return collection_name in self.collections

    def delete_collection(self, collection_name) -> None:
        assert_deletable(collection_name)
        self.collections.discard(collection_name)

    def close(self):
        pass


TRIALS = [
    {"trial_name": "A-dense", "chunk_config": {"strategy": "current"},
     "retrieval_config": {"strategy": "dense", "top_k": 5, "score_threshold": 0.55}},
    {"trial_name": "B-identifier-aware", "chunk_config": {"strategy": "current"},
     "retrieval_config": {"strategy": "identifier_aware", "top_k": 5, "score_threshold": 0.55,
                          "exact_match_boost": 0.2, "confusable_penalty": 0.15,
                          "identifier_vector_floor": 0.4}},
]


def _cfg(tmp_path, trials=None, **flags):
    raw = {"schema_version": "1.0", "run_name": "s4", "random_seed": 42,
           "dataset_path": _write_dataset(tmp_path),
           "corpus_paths": ["sample_docs", "evaluation/fixtures"], "trials": trials or TRIALS}
    raw.update(flags)
    return RunConfig.from_raw(raw)


def _run(repo, cfg, provider, tmp_path, run_id="run-s4"):
    return trial_runner.execute_run(cfg, repo, provider, output_dir=str(tmp_path / "runs"),
                                    run_id=run_id, now_fn=_clock(), id_factory=_ids())


def test_two_strategies_same_run(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider()
    res = _run(repo, _cfg(tmp_path), provider, tmp_path)
    assert res.run.status == RunStatus.COMPLETED
    trials = {t.trial_name: t for t in repo.list_trials(res.run.run_id)}
    assert set(trials) == {"A-dense", "B-identifier-aware"}
    for t in trials.values():
        assert t.status == TrialStatus.COMPLETED


def test_identifier_aware_retrieval_meta_persisted(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider()
    res = _run(repo, _cfg(tmp_path), provider, tmp_path)
    b = next(t for t in repo.list_trials(res.run.run_id) if t.trial_name == "B-identifier-aware")
    cases = {c.case_id: c for c in repo.get_trial_case_results(b.trial_id)}
    meta = json.loads(cases["e102"].retrieval_meta_json)
    assert meta["strategy_name"] == "identifier_aware"
    assert meta["query_identifiers"] == ["E102"]
    assert meta["union_candidate_count"] == 4 and meta["deduplicated_count"] == 3
    # ranking metadata 進 top_results（vector_score 保留 + ranking_score）。
    top = json.loads(cases["e102"].top_results_json)
    assert top[0]["score"] == 0.5 and top[0]["ranking_score"] == 0.7
    assert top[0]["candidate_sources"] == ["dense", "identifier"]
    assert top[0]["admission_reason"] == "exact_identifier_floor"


def test_dense_trial_has_no_ranking_details_in_results(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider()
    res = _run(repo, _cfg(tmp_path), provider, tmp_path)
    a = next(t for t in repo.list_trials(res.run.run_id) if t.trial_name == "A-dense")
    cases = repo.get_trial_case_results(a.trial_id)
    # dense trial 無結果（fake），但 dense meta 仍持久化。
    meta = json.loads(cases[0].retrieval_meta_json)
    assert meta["strategy_name"] == "dense" and meta["identifier_candidate_count"] == 0


def test_doc_cache_reused_across_retrieval_config_change(tmp_path):
    # A / B chunk_config 皆 current（僅 retrieval 不同）→ B doc embedding 全 hit（§十八.10）。
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider()
    _run(repo, _cfg(tmp_path), provider, tmp_path)
    assert provider.last_cache_miss["A-dense"] == 3      # 建 cache
    assert provider.last_cache_miss["B-identifier-aware"] == 0  # 重用（retrieval 變更不影響 doc cache）


def test_query_embedding_per_case_each_trial(tmp_path):
    # 2 trials × 2 cases = 4 次 query embed（query embedding 不快取，每題每 Trial 都算）。
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider()
    _run(repo, _cfg(tmp_path), provider, tmp_path)
    assert len(provider.query_embed_calls) == 4


def test_failure_isolation_and_cleanup(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider(fail_build={"A-dense"})
    res = _run(repo, _cfg(tmp_path), provider, tmp_path)
    trials = {t.trial_name: t for t in repo.list_trials(res.run.run_id)}
    assert trials["A-dense"].status == TrialStatus.FAILED
    assert trials["B-identifier-aware"].status == TrialStatus.COMPLETED  # 不受 A 影響
    assert provider.collections == set()  # 全部 collection 清理


def test_only_listed_trials_no_auto_generation(tmp_path):
    # 只執行明確列出的 2 個 Trial；不自動展開 boost/penalty 矩陣、不生成新 config。
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider()
    res = _run(repo, _cfg(tmp_path), provider, tmp_path)
    assert len(repo.list_trials(res.run.run_id)) == 2


def test_resume_completed_not_rerun(tmp_path):
    repo = OptimizerRepository(str(tmp_path / "s4.db"))
    provider = Slice4FakeProvider()
    cfg = _cfg(tmp_path)
    _run(repo, cfg, provider, tmp_path)
    first = list(provider.build_calls)
    # resume：completed trial 不重跑。
    provider2 = Slice4FakeProvider()
    trial_runner.execute_run(cfg, repo, provider2, output_dir=str(tmp_path / "runs"),
                             run_id="run-s4", now_fn=_clock(), id_factory=_ids())
    assert provider2.build_calls == []  # 全 completed → 不重建
    assert len(first) == 2
