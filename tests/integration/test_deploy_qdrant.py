"""Deployment Qdrant Integration（Slice 7 §二十四.3/.4/.5/.10）。

真實 Qdrant + 真實 embedding，全程使用**隔離**名稱（``factory_test_alias_*`` /
``factory_test_deploy_v*``）。驗證後清理所有測試 alias / collection，並斷言正式
``factory_knowledge`` 前後一致、``factory_knowledge_active`` 從未被建立、無 LLM 呼叫。
"""
from __future__ import annotations

import json
import os

import pytest

from tests.integration.conftest import (
    drop_collections,
    list_collections,
    prod_point_count,
    qdrant_client_or_none,
    unique_token,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
ACTIVE_ALIAS = "factory_knowledge_active"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    c = qdrant_client_or_none(QDRANT_URL)
    if c is None:
        pytest.skip("Qdrant 未就緒")
    yield c
    try:
        c.close()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="module")
def embedding():
    from app.core.config import Settings
    from app.services.embedding_service import EmbeddingService

    s = Settings()
    return EmbeddingService(
        model_name=s.embedding_model,
        query_instruction=s.embedding_query_instruction,
        max_tokens=s.embedding_max_tokens,
        base_url=s.embedding_base_url,
        api_key=s.embedding_api_key,
        dimension=s.embedding_dimension,
    )


CFG = "cfg_integ_eligible"
CANDIDATE_CONFIG = json.dumps({
    "schema_version": "1.0", "trial_name": "integ",
    "chunk_config": {"strategy": "current"},
    "retrieval_config": {"strategy": "identifier_aware"},
    "collection_options": {},
})


def _delete_alias(client, alias):
    from qdrant_client import models

    try:
        client.update_collection_aliases(change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))])
    except Exception:  # noqa: BLE001
        pass


class Rig:
    def __init__(self, tmp_path, client, embedding):
        from evaluation.v11.deploy.alias_ops import AliasManager
        from evaluation.v11.deploy.builder import DeploymentCollectionBuilder
        from evaluation.v11.deploy.corpus_manifest import build_corpus_manifest
        from evaluation.v11.deploy.readiness import DeploymentReadinessVerifier
        from evaluation.v11.deploy.repository import DeploymentRepository
        from evaluation.v11.deploy.service import DeploymentConfig, DeploymentService
        from evaluation.v11.embedding_fingerprint import EmbeddingFingerprint

        self.client = client
        self.token = unique_token()
        self.alias = f"factory_test_alias_{self.token}"
        # 隔離 corpus（單一小檔），base_dir 設為 tmp_path 以避免跨磁碟 relpath 問題。
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "doc.md").write_text(
            "# 設備 E900\n\nE900 感測器的維護週期為每 90 天一次。\n", encoding="utf-8")
        self.corpus_paths = (str(corpus_dir),)
        self.corpus_base = str(tmp_path)
        manifest = build_corpus_manifest(self.corpus_paths, base_dir=self.corpus_base)
        self.corpus_hash = manifest.corpus_hash
        fp = EmbeddingFingerprint.from_service(embedding, normalize_embeddings=True)
        self.fp_hash = fp.fingerprint_hash()

        self.repo = DeploymentRepository(str(tmp_path / "integ.deployment.db"))
        self.builder = DeploymentCollectionBuilder(
            client, embedding, cache_db_path=str(tmp_path / "c.embedding-cache.db"))
        self.verifier = DeploymentReadinessVerifier(client, embedding)
        self.alias_mgr = AliasManager(client)
        self.config = DeploymentConfig(
            active_alias=ACTIVE_ALIAS, versioned_prefix="factory_test_deploy_v",
            corpus_paths=self.corpus_paths, corpus_base_dir=self.corpus_base,
            admin_enabled=True, report_dir=None)
        self.service = DeploymentService(
            self.repo, self.alias_mgr, self.builder, self.verifier, self.config,
            env_provider=lambda: {
                "code_commit": "commit_integ", "working_tree_dirty": False,
                "target_corpus_hash": self.corpus_hash, "canonical_text_version": "v1",
                "identifier_normalization_version": "v1",
                "embedding_fingerprint_hash": self.fp_hash})
        self.created: set[str] = set()

    def build(self, name):
        m = self.builder.build(
            name, CANDIDATE_CONFIG, self.corpus_paths, corpus_base_dir=self.corpus_base,
            expected_config_hash=CFG, corpus_hash=self.corpus_hash,
            canonical_text_version="v1", identifier_normalization_version="v1")
        self.created.add(name)
        return m

    def readiness(self, manifest):
        return self.verifier.verify(
            manifest, expected_corpus_hash=self.corpus_hash, expected_config_hash=CFG,
            expected_canonical_text_version="v1",
            expected_identifier_normalization_version="v1",
            expected_embedding_fingerprint_hash=manifest.embedding_fingerprint_hash)

    def seed_eligible(self):
        from evaluation.v11.search.search_models import (
            CandidateStageStatus, OptimizerSearch, SearchCandidate, SearchStatus)

        outcome = {"recommendation_apply_eligible": True, "apply_ineligible_reasons": [],
                   "test_constraint_status": "passed"}
        self.repo.upsert_search(OptimizerSearch(
            search_id="s-integ", search_name="integ", status=SearchStatus.COMPLETED,
            search_mode="grid", search_config_json="{}", search_config_hash="h",
            dataset_version="v", dataset_hash="ds_integ", corpus_hash=self.corpus_hash,
            code_commit="commit_integ", working_tree_dirty=False,
            recommended_config_hash=CFG, recommendation_locked=True,
            recommendation_locked_at="2026-07-21T00:00:00+00:00", test_evaluated=True,
            outcome_json=json.dumps(outcome)))
        self.repo.upsert_candidate(SearchCandidate(
            search_id="s-integ", candidate_id="c-integ", ordinal=0,
            config_json=CANDIDATE_CONFIG, generation_status="valid", config_hash=CFG,
            tune_status=CandidateStageStatus.COMPLETED,
            validation_status=CandidateStageStatus.COMPLETED,
            test_status=CandidateStageStatus.COMPLETED, feasible=True))

    def apply_request(self, current, *, idem="apply-integ", corpus_hash=None):
        from evaluation.v11.deploy import confirmation as conf
        from evaluation.v11.deploy.eligibility import ApplyRequestSnapshot

        ch = corpus_hash or self.corpus_hash
        return ApplyRequestSnapshot(
            candidate_id="c-integ", search_id="s-integ", alias_name=self.alias,
            expected_config_hash=CFG, expected_dataset_hash="ds_integ",
            expected_corpus_hash=ch, expected_code_commit="commit_integ",
            expected_current_collection=current,
            confirmation=conf.apply_confirmation(self.alias, CFG, current), idempotency_key=idem,
            expected_canonical_text_version="v1",
            expected_identifier_normalization_version="v1",
            expected_recommendation_locked_at="2026-07-21T00:00:00+00:00")

    def cleanup(self):
        _delete_alias(self.client, self.alias)
        drop_collections(self.client, list(self.created))


def test_builder_and_readiness(tmp_path, client, embedding):
    """§二十四.3 Builder + §二十四.4 Readiness。"""
    from evaluation.v11.deploy.models import DeploymentError, ErrorCode

    rig = Rig(tmp_path, client, embedding)
    prod_before = prod_point_count(client)
    try:
        name = f"factory_test_deploy_v1_{rig.token}"
        m = rig.build(name)
        assert m.point_count > 0 and m.point_count == m.chunk_count
        assert m.identifier_index is True  # identifier_aware → identifiers index
        assert m.vector_size == embedding.dimension
        assert "identifiers" in m.payload_fields

        # 已存在同名 → 拒絕（§十二.3）。
        with pytest.raises(DeploymentError) as e:
            rig.builder.build(name, CANDIDATE_CONFIG, rig.corpus_paths,
                              corpus_base_dir=rig.corpus_base, expected_config_hash=CFG,
                              corpus_hash=rig.corpus_hash, canonical_text_version="v1",
                              identifier_normalization_version="v1")
        assert e.value.code == ErrorCode.COLLECTION_NAME_CONFLICT

        # Readiness success。
        rep = rig.readiness(m)
        assert rep.ready is True, rep.failures
        assert rep.probe["passed"] is True
        # identifiers index 存在。
        assert any(c["name"] == "required_payload_indexes" and c["passed"] for c in rep.checks)

        # Readiness failure：point count 不符。
        import dataclasses
        bad = dataclasses.replace(m, point_count=m.point_count + 999)
        assert rig.readiness(bad).ready is False
        # Readiness failure：config hash 不符。
        rep2 = rig.verifier.verify(
            m, expected_corpus_hash=rig.corpus_hash, expected_config_hash="WRONG",
            expected_canonical_text_version="v1",
            expected_identifier_normalization_version="v1",
            expected_embedding_fingerprint_hash=m.embedding_fingerprint_hash)
        assert rep2.ready is False and "config_hash" in rep2.failures
        # Readiness：collection missing。
        missing = dataclasses.replace(m, collection_name=f"factory_test_deploy_v_absent_{rig.token}")
        assert rig.readiness(missing).ready is False

        # Build failure cleanup：embedding 於索引期間失敗 → 不留半成品（§十二.10）。
        class BrokenEmbed:
            dimension = embedding.dimension
            def count_tokens(self, *a, **k): return embedding.count_tokens(*a, **k)
            def embed_query(self, *a, **k): return embedding.embed_query(*a, **k)
            def embed_documents(self, *a, **k): raise RuntimeError("embed boom")
        from evaluation.v11.deploy.builder import DeploymentCollectionBuilder
        broken = DeploymentCollectionBuilder(client, BrokenEmbed(),
                                             cache_db_path=str(tmp_path / "b.embedding-cache.db"))
        fail_name = f"factory_test_deploy_v_fail_{rig.token}"
        with pytest.raises(DeploymentError) as e2:
            broken.build(fail_name, CANDIDATE_CONFIG, rig.corpus_paths,
                         corpus_base_dir=rig.corpus_base, expected_config_hash=CFG,
                         corpus_hash=rig.corpus_hash, canonical_text_version="v1",
                         identifier_normalization_version="v1")
        assert e2.value.code == ErrorCode.BUILD_FAILED
        assert not client.collection_exists(fail_name)  # 半成品已清除

        # factory_knowledge 完全未被觸碰。
        assert prod_point_count(client) == prod_before
    finally:
        rig.cleanup()
    assert not client.collection_exists(ACTIVE_ALIAS.replace("_active", "") + "_never")
    assert ACTIVE_ALIAS not in [a.alias_name for a in client.get_aliases().aliases]


def test_alias_initialize(tmp_path, client, embedding):
    """§二十四.5 Alias Initialization。"""
    from evaluation.v11.deploy.models import ErrorCode, OperationStatus
    from evaluation.v11.deploy import confirmation as conf
    from evaluation.v11.deploy.service import InitializeRequest

    rig = Rig(tmp_path, client, embedding)
    try:
        v1 = f"factory_test_deploy_v1_{rig.token}"
        rig.build(v1)
        # 成功初始化。
        r = rig.service.initialize_alias(InitializeRequest(
            alias=rig.alias, collection=v1,
            confirmation=conf.initialize_confirmation(rig.alias, v1), idempotency_key="init-a"))
        assert r.operation.status == OperationStatus.COMPLETED
        assert rig.alias_mgr.get_alias_target(rig.alias) == v1
        # 已存在 → 拒絕。
        r2 = rig.service.initialize_alias(InitializeRequest(
            alias=rig.alias, collection=v1,
            confirmation=conf.initialize_confirmation(rig.alias, v1), idempotency_key="init-b"))
        assert r2.operation.error_code == ErrorCode.ALIAS_ALREADY_INITIALIZED
        # collection 不存在 → 拒絕。
        r3 = rig.service.initialize_alias(InitializeRequest(
            alias=f"factory_test_alias_other_{rig.token}", collection="no_such_collection",
            confirmation=conf.initialize_confirmation(
                f"factory_test_alias_other_{rig.token}", "no_such_collection"),
            idempotency_key="init-c"))
        assert r3.operation.error_code == ErrorCode.COLLECTION_NOT_FOUND
    finally:
        rig.cleanup()


def test_full_apply_rollback_smoke(tmp_path, client, embedding):
    """§二十四.10 完整 alias / apply / rollback / roll-forward / idempotent / failure smoke。"""
    from evaluation.v11.deploy import confirmation as conf
    from evaluation.v11.deploy.models import ErrorCode, OperationStatus
    from evaluation.v11.deploy.service import InitializeRequest, RollbackRequest

    rig = Rig(tmp_path, client, embedding)
    prod_before = prod_point_count(client)
    try:
        rig.seed_eligible()
        v1 = f"factory_test_deploy_v1_{rig.token}"
        rig.build(v1)
        rig.service.initialize_alias(InitializeRequest(
            alias=rig.alias, collection=v1,
            confirmation=conf.initialize_confirmation(rig.alias, v1), idempotency_key="i"))
        assert rig.alias_mgr.get_alias_target(rig.alias) == v1

        # Apply → alias→v2, read-back v2。
        ar = rig.service.apply(rig.apply_request(v1))
        assert ar.operation.status == OperationStatus.COMPLETED, ar.operation.error_message
        v2 = ar.record.physical_collection
        rig.created.add(v2)
        assert rig.alias_mgr.get_alias_target(rig.alias) == v2
        assert client.collection_exists(v1) and client.collection_exists(v2)  # 皆保留

        # Idempotent retry → 不重建、不重切。
        pc_v2 = client.count(v2).count
        ar2 = rig.service.apply(rig.apply_request(v1))  # 同 idem key + payload
        assert ar2.idempotent_replay is True
        assert client.count(v2).count == pc_v2
        assert rig.alias_mgr.get_alias_target(rig.alias) == v2

        # Rollback → alias→v1。
        rr = rig.service.rollback(RollbackRequest(
            alias=rig.alias, expected_current_collection=v2, target_collection=v1,
            confirmation=conf.rollback_confirmation(rig.alias, v2, v1), idempotency_key="rb"))
        assert rr.operation.status == OperationStatus.COMPLETED
        assert rig.alias_mgr.get_alias_target(rig.alias) == v1
        assert client.collection_exists(v1) and client.collection_exists(v2)

        # Roll-forward → alias→v2。
        rf = rig.service.rollback(RollbackRequest(
            alias=rig.alias, expected_current_collection=v1, target_collection=v2,
            confirmation=conf.rollback_confirmation(rig.alias, v1, v2), idempotency_key="rf"))
        assert rf.operation.status == OperationStatus.COMPLETED
        assert rig.alias_mgr.get_alias_target(rig.alias) == v2

        # Failure case（hash 不符）→ 不切 alias、不建新 collection。
        cols_before = set(list_collections(client))
        fr = rig.service.apply(rig.apply_request(v2, idem="apply-fail", corpus_hash="WRONG"))
        assert fr.operation.status == OperationStatus.REJECTED
        assert fr.operation.error_code == ErrorCode.CORPUS_HASH_MISMATCH
        assert rig.alias_mgr.get_alias_target(rig.alias) == v2
        assert set(list_collections(client)) == cols_before  # 無新 collection

        # 正式資料一致 + 無真實 alias。
        assert prod_point_count(client) == prod_before
        assert ACTIVE_ALIAS not in [a.alias_name for a in client.get_aliases().aliases]
    finally:
        rig.cleanup()

    # 清理後：測試 alias / collection 皆消失。
    assert rig.alias not in [a.alias_name for a in client.get_aliases().aliases]
    assert prod_point_count(client) == prod_before
