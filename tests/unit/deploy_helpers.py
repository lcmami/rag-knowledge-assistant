"""Slice 7 Deployment 測試共用 fixture 與 fake collaborator。

- ``seed_eligible_search``：deterministic、隔離的 **eligible** candidate fixture（§十）——
  locked + apply_eligible + test passed + 全 hash 匹配，只存在於測試，不碰真實 locked search。
- ``seed_current_ineligible_search``：忠實重現目前真實 locked recommendation（apply_eligible=false
  + 五個原因），供「真實 recommendation 必須被拒絕」測試（§九）。
- Fake Alias/Builder/Verifier：讓 apply/rollback/reconcile 的 orchestration 邏輯可在**不需真實
  Qdrant** 下完整單元測試（真實 Qdrant 由 integration 測試涵蓋）。
"""
from __future__ import annotations

import json

from evaluation.v11.deploy import confirmation as conf
from evaluation.v11.deploy.eligibility import ApplyRequestSnapshot
from evaluation.v11.deploy.models import (
    DeploymentBuildManifest,
    DeploymentError,
    ErrorCode,
    ReadinessReport,
)
from evaluation.v11.deploy.naming import assert_safe_to_delete, assert_valid_alias_target
from evaluation.v11.deploy.repository import DeploymentRepository
from evaluation.v11.deploy.service import DeploymentConfig
from evaluation.v11.search.search_models import (
    CandidateStageStatus,
    OptimizerSearch,
    SearchCandidate,
    SearchStatus,
)

CANON = "v1"
IDN = "v1"
FP = "fp_eligible_hash"

ELIGIBLE_CONFIG_HASH = "cfg_eligible_0001"
ELIGIBLE_CORPUS_HASH = "corpus_eligible_0001"
ELIGIBLE_DATASET_HASH = "dataset_eligible_0001"
ELIGIBLE_COMMIT = "commit_eligible_0001"
ELIGIBLE_LOCKED_AT = "2026-07-21T00:00:00+00:00"

INELIGIBLE_REASONS = [
    "small_validation_sample",
    "quality_metrics_tied",
    "selection_driven_by_single_run_latency",
    "test_constraint_failed",
    "demo_corpus_only",
]


def eligible_config_json() -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "trial_name": "eligible",
            "chunk_config": {"strategy": "current"},
            "retrieval_config": {"strategy": "identifier_aware"},
            "collection_options": {},
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def make_repo(tmp_path) -> DeploymentRepository:
    return DeploymentRepository(str(tmp_path / "deploy.deployment.db"))


def seed_eligible_search(
    repo: DeploymentRepository,
    *,
    search_id: str = "search-elig",
    candidate_id: str = "cand-elig",
    config_hash: str = ELIGIBLE_CONFIG_HASH,
    corpus_hash: str = ELIGIBLE_CORPUS_HASH,
    dataset_hash: str = ELIGIBLE_DATASET_HASH,
    code_commit: str = ELIGIBLE_COMMIT,
    locked_at: str = ELIGIBLE_LOCKED_AT,
) -> tuple[OptimizerSearch, SearchCandidate]:
    outcome = {
        "recommendation_apply_eligible": True,
        "recommendation_confidence": "medium",
        "apply_ineligible_reasons": [],
        "test_constraint_status": "passed",
    }
    search = OptimizerSearch(
        search_id=search_id, search_name="eligible-fixture",
        status=SearchStatus.COMPLETED, search_mode="grid",
        search_config_json="{}", search_config_hash="scfg",
        dataset_version="v1.1-fixture", dataset_hash=dataset_hash,
        corpus_hash=corpus_hash, code_commit=code_commit, working_tree_dirty=False,
        recommended_config_hash=config_hash, recommendation_locked=True,
        recommendation_locked_at=locked_at, test_evaluated=True,
        outcome_json=json.dumps(outcome, ensure_ascii=False, sort_keys=True),
    )
    repo.upsert_search(search)
    candidate = SearchCandidate(
        search_id=search_id, candidate_id=candidate_id, ordinal=0,
        config_json=eligible_config_json(), generation_status="valid",
        config_hash=config_hash,
        tune_status=CandidateStageStatus.COMPLETED,
        validation_status=CandidateStageStatus.COMPLETED,
        test_status=CandidateStageStatus.COMPLETED,
        feasible=True,
    )
    repo.upsert_candidate(candidate)
    return search, candidate


def seed_current_ineligible_search(
    repo: DeploymentRepository,
    *,
    search_id: str = "search-real",
    candidate_id: str = "cand-real",
    config_hash: str = "cfg_real_e246697d41",
) -> tuple[OptimizerSearch, SearchCandidate]:
    """忠實重現目前真實 locked recommendation（§九）：apply_eligible=false + 五原因。"""
    outcome = {
        "recommendation_apply_eligible": False,
        "recommendation_confidence": "low",
        "apply_ineligible_reasons": list(INELIGIBLE_REASONS),
        "test_constraint_status": "failed",
    }
    search = OptimizerSearch(
        search_id=search_id, search_name="real-locked",
        status=SearchStatus.TEST_CONSTRAINT_FAILED, search_mode="grid",
        search_config_json="{}", search_config_hash="scfg",
        dataset_version="v1.1-coverage-0002", dataset_hash="dataset_real",
        corpus_hash="corpus_real", code_commit="commit_real", working_tree_dirty=False,
        recommended_config_hash=config_hash, recommendation_locked=True,
        recommendation_locked_at="2026-07-20T00:00:00+00:00", test_evaluated=True,
        outcome_json=json.dumps(outcome, ensure_ascii=False, sort_keys=True),
    )
    repo.upsert_search(search)
    candidate = SearchCandidate(
        search_id=search_id, candidate_id=candidate_id, ordinal=0,
        config_json=eligible_config_json(), generation_status="valid",
        config_hash=config_hash,
        tune_status=CandidateStageStatus.COMPLETED,
        validation_status=CandidateStageStatus.COMPLETED,
        test_status=CandidateStageStatus.COMPLETED,
        feasible=True,
    )
    repo.upsert_candidate(candidate)
    return search, candidate


def eligible_env(
    *,
    admin_enabled: bool = True,
    corpus_hash: str = ELIGIBLE_CORPUS_HASH,
    code_commit: str = ELIGIBLE_COMMIT,
    working_tree_dirty: bool = False,
) -> dict:
    return {
        "code_commit": code_commit,
        "working_tree_dirty": working_tree_dirty,
        "target_corpus_hash": corpus_hash,
        "canonical_text_version": CANON,
        "identifier_normalization_version": IDN,
        "embedding_fingerprint_hash": FP,
    }


def eligible_apply_request(
    *,
    alias: str,
    current_collection: str,
    search_id: str = "search-elig",
    candidate_id: str = "cand-elig",
    config_hash: str = ELIGIBLE_CONFIG_HASH,
    corpus_hash: str = ELIGIBLE_CORPUS_HASH,
    dataset_hash: str = ELIGIBLE_DATASET_HASH,
    code_commit: str = ELIGIBLE_COMMIT,
    locked_at: str = ELIGIBLE_LOCKED_AT,
    idempotency_key: str = "idem-apply-1",
    confirmation: str | None = None,
) -> ApplyRequestSnapshot:
    if confirmation is None:
        confirmation = conf.apply_confirmation(alias, config_hash, current_collection)
    return ApplyRequestSnapshot(
        candidate_id=candidate_id, search_id=search_id, alias_name=alias,
        expected_config_hash=config_hash, expected_dataset_hash=dataset_hash,
        expected_corpus_hash=corpus_hash, expected_code_commit=code_commit,
        expected_current_collection=current_collection, confirmation=confirmation,
        idempotency_key=idempotency_key,
        expected_canonical_text_version=CANON,
        expected_identifier_normalization_version=IDN,
        expected_embedding_fingerprint_hash=FP,
        expected_recommendation_locked_at=locked_at,
    )


def service_config(*, active_alias: str = "factory_knowledge_active") -> DeploymentConfig:
    return DeploymentConfig(
        active_alias=active_alias,
        versioned_prefix="factory_test_deploy_v",
        corpus_paths=("sample_docs",),
        admin_enabled=True,
        report_dir=None,
    )


# --------------------------------------------------------------------------
# Fake collaborators
# --------------------------------------------------------------------------
class FakeAliasManager:
    def __init__(self):
        self.collections: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.readback_override: str | None = None

    def add_collection(self, name: str) -> None:
        self.collections.add(name)

    def _collection_exists(self, name: str) -> bool:
        return name in self.collections

    def get_alias_target(self, alias: str):
        return self.aliases.get(alias)

    def alias_exists(self, alias: str) -> bool:
        return alias in self.aliases

    def create_alias(self, alias: str, collection: str) -> None:
        assert_valid_alias_target(collection)
        if alias in self.aliases:
            raise DeploymentError(ErrorCode.ALIAS_ALREADY_INITIALIZED, "exists")
        if collection not in self.collections:
            raise DeploymentError(ErrorCode.COLLECTION_NOT_FOUND, "missing")
        self.aliases[alias] = collection

    def switch_alias(self, alias: str, new: str, *, expected_current=None):
        assert_valid_alias_target(new)
        cur = self.aliases.get(alias)
        if cur is None:
            raise DeploymentError(ErrorCode.ALIAS_NOT_INITIALIZED, "not init")
        if expected_current is not None and cur != expected_current:
            raise DeploymentError(ErrorCode.STALE_ALIAS_TARGET, "stale")
        if new not in self.collections:
            raise DeploymentError(ErrorCode.COLLECTION_NOT_FOUND, "missing")
        self.aliases[alias] = self.readback_override if self.readback_override is not None else new
        return self.aliases[alias]


class FakeBuilder:
    def __init__(self, alias_mgr: FakeAliasManager, *, fail=False, on_build=None, manifest=None):
        self.alias = alias_mgr
        self.fail = fail
        self.on_build = on_build
        self.manifest = manifest
        self.build_calls = 0
        self.cleanup_calls: list[str] = []

    def build(self, collection_name, candidate_config_json, corpus_paths, *,
              corpus_base_dir=None, expected_config_hash, corpus_hash,
              canonical_text_version, identifier_normalization_version):
        self.build_calls += 1
        if self.fail:
            raise DeploymentError(ErrorCode.BUILD_FAILED, "build boom")
        self.alias.add_collection(collection_name)
        if self.on_build:
            self.on_build()
        return self.manifest or DeploymentBuildManifest(
            collection_name=collection_name, point_count=10, chunk_count=10,
            vector_size=512, distance="Cosine", strategy="identifier_aware",
            config_hash=expected_config_hash, chunk_config_hash="chunk_h",
            corpus_hash=corpus_hash, canonical_text_version=canonical_text_version,
            identifier_normalization_version=identifier_normalization_version,
            embedding_fingerprint_hash=FP, identifier_index=True,
            payload_fields=["document_id", "file_hash", "text", "chunk_index", "identifiers"],
        )

    def cleanup(self, collection_name, *, active_targets=()):
        assert_safe_to_delete(collection_name, active_targets=active_targets)
        self.cleanup_calls.append(collection_name)
        self.alias.collections.discard(collection_name)
        return True


class FakeVerifier:
    def __init__(self, *, ready=True, failures=None):
        self.ready = ready
        self.failures = failures or (["point_count"] if not ready else [])
        self.calls = 0

    def verify(self, manifest, **kwargs):
        self.calls += 1
        return ReadinessReport(
            ready=self.ready, collection_name=manifest.collection_name,
            checks=[{"name": "smoke", "passed": self.ready, "detail": None}],
            failures=list(self.failures), probe={"passed": self.ready},
        )
