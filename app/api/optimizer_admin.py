"""Optimizer Admin API（Slice 7 §二十 / §二十一）。

單機 local admin 介面（無 RBAC）。**Admin Write API 預設停用**（``optimizer_admin_enabled=False``）：
停用時 write endpoint 一律回 ``optimizer_admin_disabled``，**不建構、不呼叫任何 service**（§二十.2）。

Read endpoint 不洩漏私人路徑（回傳一律經 ``to_public_dict``，§二十.3/.10）；前端不得直接呼叫 Qdrant。
Eligibility 一律由 DB locked outcome 重新讀取，API **不接受**前端傳入的 eligibility boolean（§二十一.9）。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.dependencies import get_settings
from app.schemas.optimizer_admin import (
    AliasInitializeRequest,
    ApplyRequestModel,
    ReconcileRequestModel,
    RollbackRequestModel,
)
from evaluation.v11.deploy.eligibility import ApplyRequestSnapshot
from evaluation.v11.deploy.models import DeploymentError, ErrorCode, OperationStatus
from evaluation.v11.deploy.service import (
    DeploymentResult,
    DeploymentService,
    InitializeRequest,
    ReconcileRequest,
    RollbackRequest,
)

router = APIRouter(prefix="/api/optimizer", tags=["optimizer-admin"])


# --------------------------------------------------------------------------
# Error code → HTTP status（§二十.9 語意一致）
# --------------------------------------------------------------------------
_ERROR_HTTP: dict[str, int] = {
    ErrorCode.OPTIMIZER_ADMIN_DISABLED: 403,
    ErrorCode.UNSAFE_COLLECTION: 403,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.DEPLOYMENT_IN_PROGRESS: 409,
    ErrorCode.STALE_ALIAS_TARGET: 409,
    ErrorCode.STALE_CANDIDATE: 409,
    ErrorCode.CORPUS_HASH_MISMATCH: 409,
    ErrorCode.APPLY_REJECTED: 409,
    ErrorCode.CONFIRMATION_MISMATCH: 409,
    ErrorCode.READINESS_FAILED: 409,
    ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED: 409,
    ErrorCode.MANUAL_RECONCILIATION_REQUIRED: 409,
    ErrorCode.ALIAS_ALREADY_INITIALIZED: 409,
    ErrorCode.ALIAS_NOT_INITIALIZED: 409,
    ErrorCode.COLLECTION_NAME_CONFLICT: 409,
    ErrorCode.COLLECTION_NOT_FOUND: 404,
    ErrorCode.SEARCH_NOT_FOUND: 404,
    ErrorCode.CANDIDATE_NOT_FOUND: 404,
    ErrorCode.BUILD_FAILED: 500,
    ErrorCode.INVALID_REQUEST: 400,
}


def _http_for(code: Optional[str]) -> int:
    return _ERROR_HTTP.get(code or "", 400)


def _error_response(code: str, message: str, *, details: Optional[dict[str, Any]] = None) -> JSONResponse:
    return JSONResponse(
        status_code=_http_for(code),
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


def _result_response(result: DeploymentResult) -> JSONResponse:
    op = result.operation
    if op.status == OperationStatus.COMPLETED:
        return JSONResponse(status_code=200, content=result.to_public_dict())
    # rejected / failed / needs_reconcile → 對應 HTTP status + 明確 error。
    return JSONResponse(
        status_code=_http_for(op.error_code),
        content={
            "error": {
                "code": op.error_code or ErrorCode.APPLY_REJECTED,
                "message": op.error_message or "operation not completed",
                "details": {},
            },
            "result": result.to_public_dict(),
        },
    )


# --------------------------------------------------------------------------
# Providers（測試可 override）
# --------------------------------------------------------------------------
def get_deployment_repo(request: Request):
    """讀取用 DeploymentRepository；DB 檔不存在時回 None（不主動建立，避免 GET 副作用）。"""
    from evaluation.v11.deploy.repository import DeploymentRepository

    settings: Settings = request.app.state.settings
    if not os.path.exists(settings.optimizer_db_path):
        return None
    return DeploymentRepository(settings.optimizer_db_path)


def get_service_factory(request: Request) -> Callable[[], DeploymentService]:
    """回傳建構 DeploymentService 的 factory（惰性；停用時 write endpoint 不會呼叫它）。"""

    def build() -> DeploymentService:
        return _build_real_service(request)

    return build


def _build_real_service(request: Request) -> DeploymentService:
    from evaluation.v11 import hashing
    from evaluation.v11.deploy.alias_ops import AliasManager
    from evaluation.v11.deploy.builder import DeploymentCollectionBuilder
    from evaluation.v11.deploy.corpus_manifest import build_corpus_manifest
    from evaluation.v11.deploy.readiness import DeploymentReadinessVerifier
    from evaluation.v11.deploy.repository import DeploymentRepository
    from evaluation.v11.deploy.service import DeploymentConfig
    from evaluation.v11.embedding_fingerprint import EmbeddingFingerprint

    settings: Settings = request.app.state.settings
    state = request.app.state
    client = state.vector_store.client
    embedding = state.embedding_service
    repo = DeploymentRepository(settings.optimizer_db_path)
    corpus_paths = ("sample_docs", "evaluation/fixtures")
    cache_db = os.path.join("evaluation", "deployments", "deployment.embedding-cache.db")

    def env_provider() -> dict[str, Any]:
        git = hashing.git_state()
        manifest = build_corpus_manifest(corpus_paths)
        fp = EmbeddingFingerprint.from_service(embedding, normalize_embeddings=True)
        return {
            "code_commit": git.get("commit"),
            "working_tree_dirty": git.get("working_tree_dirty"),
            "target_corpus_hash": manifest.corpus_hash,
            "canonical_text_version": manifest.canonical_text_version,
            "identifier_normalization_version": "v1",
            "embedding_fingerprint_hash": fp.fingerprint_hash(),
        }

    config = DeploymentConfig(
        active_alias=settings.optimizer_active_alias,
        versioned_prefix=settings.optimizer_versioned_collection_prefix,
        corpus_paths=corpus_paths,
        admin_enabled=settings.optimizer_admin_enabled,
        report_dir=os.path.join("evaluation", "deployments"),
    )
    builder = DeploymentCollectionBuilder(client, embedding, cache_db_path=cache_db)
    verifier = DeploymentReadinessVerifier(client, embedding)
    return DeploymentService(
        repo, AliasManager(client), builder, verifier, config, env_provider=env_provider
    )


def _require_admin(settings: Settings) -> Optional[JSONResponse]:
    if not settings.optimizer_admin_enabled:
        return _error_response(
            ErrorCode.OPTIMIZER_ADMIN_DISABLED,
            "Optimizer admin write API 未啟用（optimizer_admin_enabled=false）。",
        )
    return None


# --------------------------------------------------------------------------
# Read endpoints
# --------------------------------------------------------------------------
@router.get("/deployments")
def list_deployments(
    request: Request,
    settings: Settings = Depends(get_settings),
    repo=Depends(get_deployment_repo),
) -> dict:
    if repo is None:
        return {"operations": []}
    alias = request.query_params.get("alias")
    ops = repo.list_operations(alias)
    return {"operations": [o.to_public_dict() for o in ops]}


@router.get("/deployments/{operation_id}")
def get_deployment(
    operation_id: str,
    settings: Settings = Depends(get_settings),
    repo=Depends(get_deployment_repo),
):
    if repo is None:
        return _error_response(ErrorCode.SEARCH_NOT_FOUND, "operation not found")
    op = repo.get_operation(operation_id)
    if op is None:
        return _error_response(ErrorCode.SEARCH_NOT_FOUND, "operation not found")
    audit = [
        {"event_type": e.event_type, "created_at": e.created_at}
        for e in repo.list_audit_events(operation_id)
    ]
    return {"operation": op.to_public_dict(), "audit": audit}


@router.get("/alias/status")
def alias_status(
    settings: Settings = Depends(get_settings),
    repo=Depends(get_deployment_repo),
    factory: Callable[[], DeploymentService] = Depends(get_service_factory),
):
    alias = settings.optimizer_active_alias
    db_active = None
    if repo is not None:
        rec = repo.get_active_record(alias)
        db_active = rec.physical_collection if rec else None
    alias_target = None
    try:
        service = factory()
        alias_target = service.alias.get_alias_target(alias)
    except Exception:  # noqa: BLE001 - alias 讀取失敗不得洩漏內部細節
        alias_target = None
    return {
        "alias": alias,
        "alias_target": alias_target,
        "db_active_target": db_active,
        "initialized": alias_target is not None,
    }


# --------------------------------------------------------------------------
# Write endpoints（admin 停用時不建構 / 不呼叫 service）
# --------------------------------------------------------------------------
@router.post("/alias/initialize")
def initialize_alias(
    payload: AliasInitializeRequest,
    settings: Settings = Depends(get_settings),
    factory: Callable[[], DeploymentService] = Depends(get_service_factory),
):
    disabled = _require_admin(settings)
    if disabled is not None:
        return disabled
    try:
        service = factory()
        result = service.initialize_alias(InitializeRequest(
            alias=payload.alias, collection=payload.collection,
            confirmation=payload.confirmation, idempotency_key=payload.idempotency_key,
        ))
    except DeploymentError as exc:
        return _error_response(exc.code, exc.message, details=exc.details)
    return _result_response(result)


@router.post("/candidates/{candidate_id}/apply")
def apply_candidate(
    candidate_id: str,
    payload: ApplyRequestModel,
    settings: Settings = Depends(get_settings),
    factory: Callable[[], DeploymentService] = Depends(get_service_factory),
):
    disabled = _require_admin(settings)
    if disabled is not None:
        return disabled
    snapshot = ApplyRequestSnapshot(
        candidate_id=candidate_id,
        search_id=payload.search_id,
        alias_name=payload.alias or settings.optimizer_active_alias,
        expected_config_hash=payload.expected_config_hash,
        expected_dataset_hash=payload.expected_dataset_hash,
        expected_corpus_hash=payload.expected_corpus_hash,
        expected_code_commit=payload.expected_code_commit,
        expected_current_collection=payload.expected_current_collection,
        confirmation=payload.confirmation,
        idempotency_key=payload.idempotency_key,
        expected_canonical_text_version=payload.expected_canonical_text_version,
        expected_identifier_normalization_version=payload.expected_identifier_normalization_version,
        expected_embedding_fingerprint_hash=payload.expected_embedding_fingerprint_hash,
        expected_recommendation_locked_at=payload.expected_recommendation_locked_at,
    )
    try:
        service = factory()
        result = service.apply(snapshot)
    except DeploymentError as exc:
        return _error_response(exc.code, exc.message, details=exc.details)
    return _result_response(result)


@router.post("/rollback")
def rollback(
    payload: RollbackRequestModel,
    settings: Settings = Depends(get_settings),
    factory: Callable[[], DeploymentService] = Depends(get_service_factory),
):
    disabled = _require_admin(settings)
    if disabled is not None:
        return disabled
    try:
        service = factory()
        result = service.rollback(RollbackRequest(
            alias=payload.alias or settings.optimizer_active_alias,
            expected_current_collection=payload.expected_current_collection,
            target_collection=payload.target_collection,
            confirmation=payload.confirmation,
            idempotency_key=payload.idempotency_key,
        ))
    except DeploymentError as exc:
        return _error_response(exc.code, exc.message, details=exc.details)
    return _result_response(result)


@router.post("/reconcile")
def reconcile(
    payload: ReconcileRequestModel,
    settings: Settings = Depends(get_settings),
    factory: Callable[[], DeploymentService] = Depends(get_service_factory),
):
    disabled = _require_admin(settings)
    if disabled is not None:
        return disabled
    try:
        service = factory()
        result = service.reconcile(ReconcileRequest(
            alias=payload.alias or settings.optimizer_active_alias,
            idempotency_key=payload.idempotency_key,
        ))
    except DeploymentError as exc:
        return _error_response(exc.code, exc.message, details=exc.details)
    return _result_response(result)
