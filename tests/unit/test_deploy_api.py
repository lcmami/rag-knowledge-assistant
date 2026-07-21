"""Optimizer Admin API 測試（Slice 7 §二十四.9 / §二十 / §二十一）。

以 dependency override 注入「以 fake collaborator 支撐的真實 DeploymentService」，
因此測試真實 router → service 流程，但不需 Qdrant / 模型。
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.api import optimizer_admin as adm
from app.core.config import Settings
from app.dependencies import get_settings
from app.main import create_app
from evaluation.v11.deploy import confirmation as conf
from evaluation.v11.deploy.service import DeploymentConfig, DeploymentService, InitializeRequest
from tests.unit.deploy_helpers import (
    ELIGIBLE_CONFIG_HASH,
    ELIGIBLE_CORPUS_HASH,
    ELIGIBLE_DATASET_HASH,
    ELIGIBLE_COMMIT,
    ELIGIBLE_LOCKED_AT,
    FP,
    FakeAliasManager,
    FakeBuilder,
    FakeVerifier,
    eligible_env,
    make_repo,
    seed_eligible_search,
)

ALIAS = "factory_test_alias_api"
BASE = "factory_test_deploy_v1_apibase"
ACTIVE_ALIAS = "factory_knowledge_active"


def _build_service(tmp_path, *, admin_enabled=True, initialize=True):
    repo = make_repo(tmp_path)
    seed_eligible_search(repo)
    alias = FakeAliasManager()
    builder = FakeBuilder(alias)
    verifier = FakeVerifier(ready=True)
    cfg = DeploymentConfig(active_alias=ACTIVE_ALIAS, versioned_prefix="factory_test_deploy_v",
                           corpus_paths=("sample_docs",), admin_enabled=admin_enabled,
                           report_dir=None)
    service = DeploymentService(repo, alias, builder, verifier, cfg,
                                env_provider=lambda: eligible_env(admin_enabled=admin_enabled))
    if initialize:
        alias.add_collection(BASE)
        service.initialize_alias(InitializeRequest(
            alias=ALIAS, collection=BASE,
            confirmation=conf.initialize_confirmation(ALIAS, BASE),
            idempotency_key="api-init"))
    return service, repo


def _make_client(service, repo, *, admin_enabled=True):
    settings = Settings(optimizer_admin_enabled=admin_enabled, optimizer_active_alias=ACTIVE_ALIAS)
    factory_calls = {"n": 0}

    def _factory():
        factory_calls["n"] += 1
        return service

    os.environ["SKIP_STARTUP_SERVICES"] = "1"
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[adm.get_service_factory] = lambda: _factory
    app.dependency_overrides[adm.get_deployment_repo] = lambda: repo
    client = TestClient(app)
    return client, factory_calls


def _apply_body(**over):
    body = dict(
        search_id="search-elig", expected_config_hash=ELIGIBLE_CONFIG_HASH,
        expected_dataset_hash=ELIGIBLE_DATASET_HASH, expected_corpus_hash=ELIGIBLE_CORPUS_HASH,
        expected_code_commit=ELIGIBLE_COMMIT, expected_current_collection=BASE,
        confirmation=conf.apply_confirmation(ALIAS, ELIGIBLE_CONFIG_HASH, BASE),
        idempotency_key="api-apply-1", alias=ALIAS,
        expected_canonical_text_version="v1", expected_identifier_normalization_version="v1",
        expected_embedding_fingerprint_hash=FP, expected_recommendation_locked_at=ELIGIBLE_LOCKED_AT,
    )
    body.update(over)
    return body


def test_admin_disabled_write_returns_403_without_calling_service(tmp_path):
    service, repo = _build_service(tmp_path, admin_enabled=False, initialize=False)
    client, calls = _make_client(service, repo, admin_enabled=False)
    with client:
        r = client.post(f"/api/optimizer/candidates/cand-elig/apply", json=_apply_body())
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "optimizer_admin_disabled"
    assert calls["n"] == 0  # service 未被建構 / 呼叫


def test_strict_schema_rejects_unknown_field(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        r = client.post("/api/optimizer/candidates/c/apply", json=_apply_body(bogus=1))
    assert r.status_code == 422


def test_api_does_not_accept_eligibility_boolean(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        r = client.post("/api/optimizer/candidates/c/apply", json=_apply_body(apply_eligible=True))
    assert r.status_code == 422  # apply_eligible 為未知欄位 → 拒絕（§二十一.9）


def test_apply_eligible_success(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        r = client.post("/api/optimizer/candidates/cand-elig/apply", json=_apply_body())
    assert r.status_code == 200
    body = r.json()
    assert body["operation"]["status"] == "completed"
    assert body["deployment_record"]["previous_collection"] == BASE


def test_apply_hash_mismatch_409(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        r = client.post("/api/optimizer/candidates/cand-elig/apply",
                        json=_apply_body(expected_corpus_hash="wrong", idempotency_key="k-mism"))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "corpus_hash_mismatch"


def test_apply_confirmation_mismatch_409(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        r = client.post("/api/optimizer/candidates/cand-elig/apply",
                        json=_apply_body(confirmation="WRONG", idempotency_key="k-conf"))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "confirmation_mismatch"


def test_initialize_via_api(tmp_path):
    service, repo = _build_service(tmp_path, initialize=False)
    service.alias.add_collection(BASE)
    client, _ = _make_client(service, repo)
    with client:
        r = client.post("/api/optimizer/alias/initialize", json={
            "alias": ALIAS, "collection": BASE,
            "confirmation": conf.initialize_confirmation(ALIAS, BASE),
            "idempotency_key": "api-init2"})
    assert r.status_code == 200
    assert r.json()["operation"]["status"] == "completed"


def test_rollback_via_api(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        ar = client.post("/api/optimizer/candidates/cand-elig/apply", json=_apply_body())
        v2 = ar.json()["deployment_record"]["physical_collection"]
        r = client.post("/api/optimizer/rollback", json={
            "alias": ALIAS, "expected_current_collection": v2, "target_collection": BASE,
            "confirmation": conf.rollback_confirmation(ALIAS, v2, BASE),
            "idempotency_key": "api-rb"})
    assert r.status_code == 200
    assert r.json()["operation"]["status"] == "completed"


def test_reconcile_via_api(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        r = client.post("/api/optimizer/reconcile", json={"alias": ALIAS, "idempotency_key": "api-rc"})
    assert r.status_code == 200
    assert r.json()["reconciliation"]["resolution"] == "consistent"


def test_read_deployment_and_alias_status_no_private_paths(tmp_path):
    service, repo = _build_service(tmp_path)
    client, _ = _make_client(service, repo)
    with client:
        ar = client.post("/api/optimizer/candidates/cand-elig/apply", json=_apply_body())
        op_id = ar.json()["operation"]["operation_id"]
        gd = client.get(f"/api/optimizer/deployments/{op_id}")
        ld = client.get("/api/optimizer/deployments")
        st = client.get("/api/optimizer/alias/status")
    assert gd.status_code == 200
    assert ld.status_code == 200 and len(ld.json()["operations"]) >= 1
    assert st.status_code == 200
    # 無私人絕對路徑洩漏（§二十.3/.10）。
    for resp in (gd, ld, st):
        text = resp.text
        assert "\\Users\\" not in text and "/home/" not in text and ".deployment.db" not in text


def test_apply_ineligible_via_api_409(tmp_path):
    from tests.unit.deploy_helpers import seed_current_ineligible_search

    service, repo = _build_service(tmp_path)
    seed_current_ineligible_search(repo, search_id="search-real", config_hash="cfg_real")
    # env 與真實 search 相符，使唯一原因為 ineligibility。
    service._env_provider = lambda: {
        "code_commit": "commit_real", "working_tree_dirty": False,
        "target_corpus_hash": "corpus_real", "canonical_text_version": "v1",
        "identifier_normalization_version": "v1", "embedding_fingerprint_hash": None}
    client, _ = _make_client(service, repo)
    body = _apply_body(search_id="search-real", expected_config_hash="cfg_real",
                       expected_dataset_hash="dataset_real", expected_corpus_hash="corpus_real",
                       expected_code_commit="commit_real",
                       confirmation=conf.apply_confirmation(ALIAS, "cfg_real", BASE),
                       idempotency_key="api-inelig",
                       expected_embedding_fingerprint_hash=None,
                       expected_recommendation_locked_at=None)
    with client:
        r = client.post("/api/optimizer/candidates/cand-real/apply", json=body)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "apply_rejected"
    assert "demo_corpus_only" in r.json()["result"]["eligibility"]["reasons"]
