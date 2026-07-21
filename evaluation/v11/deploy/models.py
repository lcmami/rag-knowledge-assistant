"""Deployment 資料模型、狀態常數與穩定 Error Code（Slice 7 §八 / §十五～§十九）。

純資料模型，不依賴 FastAPI、不依賴 Qdrant。對應 Deployment SQLite 的四張表
（``deployment_operations`` / ``deployment_records`` / ``deployment_audit_events`` /
``deployment_locks``，migration ``0008``）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------
# Operation type / status（§十五 / §十六 / §十九）
# --------------------------------------------------------------------------
class OperationType:
    INITIALIZE = "initialize_alias"
    APPLY = "apply"
    ROLLBACK = "rollback"
    RECONCILE = "reconcile"


class OperationStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    #: Eligibility Guard 或前置驗證拒絕（未建立 collection、未切 alias）。
    REJECTED = "rejected"
    FAILED = "failed"
    #: Alias 已切但 DB 後續寫入需人工／自動 reconcile。
    NEEDS_RECONCILE = "needs_reconcile"


# --------------------------------------------------------------------------
# Audit event type（§十八 deployment_audit_events）
# --------------------------------------------------------------------------
class EventType:
    OPERATION_CREATED = "operation_created"
    ELIGIBILITY_REJECTED = "eligibility_rejected"
    BUILD_STARTED = "build_started"
    BUILD_COMPLETED = "build_completed"
    BUILD_FAILED = "build_failed"
    BUILD_CLEANED = "build_cleaned"
    READINESS_PASSED = "readiness_passed"
    READINESS_FAILED = "readiness_failed"
    ALIAS_INITIALIZED = "alias_initialized"
    ALIAS_SWITCHED = "alias_switched"
    ALIAS_READBACK = "alias_readback"
    ROLLBACK_COMPLETED = "rollback_completed"
    DEPLOYMENT_ACTIVATED = "deployment_activated"
    RECONCILE_STARTED = "reconcile_started"
    RECONCILE_RESOLVED = "reconcile_resolved"
    RECONCILE_MANUAL_REQUIRED = "reconcile_manual_required"
    OPERATION_REJECTED = "operation_rejected"
    OPERATION_FAILED = "operation_failed"


# --------------------------------------------------------------------------
# 穩定 Error Code（§二十.8：HTTP 與 report 皆引用同一組字串）
# --------------------------------------------------------------------------
class ErrorCode:
    OPTIMIZER_ADMIN_DISABLED = "optimizer_admin_disabled"
    APPLY_REJECTED = "apply_rejected"
    STALE_CANDIDATE = "stale_candidate"
    STALE_ALIAS_TARGET = "stale_alias_target"
    CORPUS_HASH_MISMATCH = "corpus_hash_mismatch"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    DEPLOYMENT_IN_PROGRESS = "deployment_in_progress"
    READINESS_FAILED = "readiness_failed"
    ROLLBACK_TARGET_NOT_ALLOWED = "rollback_target_not_allowed"
    MANUAL_RECONCILIATION_REQUIRED = "manual_reconciliation_required"
    # 額外（仍為穩定字串）
    CONFIRMATION_MISMATCH = "confirmation_mismatch"
    ALIAS_ALREADY_INITIALIZED = "alias_already_initialized"
    ALIAS_NOT_INITIALIZED = "alias_not_initialized"
    COLLECTION_NOT_FOUND = "collection_not_found"
    COLLECTION_NAME_CONFLICT = "collection_name_conflict"
    BUILD_FAILED = "build_failed"
    SEARCH_NOT_FOUND = "search_not_found"
    CANDIDATE_NOT_FOUND = "candidate_not_found"
    UNSAFE_COLLECTION = "unsafe_collection"
    INVALID_REQUEST = "invalid_request"


class DeploymentError(Exception):
    """帶穩定 error code 的 Deployment 例外。

    ``message`` 不得包含私人絕對路徑或機密（§九.8 / §二十.10）。
    """

    def __init__(self, code: str, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


# --------------------------------------------------------------------------
# 持久化模型
# --------------------------------------------------------------------------
@dataclass
class DeploymentOperation:
    """一次 write 操作的完整生命週期（§十八 deployment_operations）。"""

    operation_id: str
    idempotency_key: str
    operation_type: str
    status: str
    alias_name: str
    request_payload_hash: str
    search_id: Optional[str] = None
    candidate_id: Optional[str] = None
    config_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    corpus_hash: Optional[str] = None
    code_commit: Optional[str] = None
    expected_current_collection: Optional[str] = None
    source_collection: Optional[str] = None
    target_collection: Optional[str] = None
    confirmation_hash: Optional[str] = None
    readiness_json: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    reconcile_required: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        """Read API / report 用（不輸出私人路徑；hash 為安全摘要）。"""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "status": self.status,
            "alias_name": self.alias_name,
            "search_id": self.search_id,
            "candidate_id": self.candidate_id,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "corpus_hash": self.corpus_hash,
            "code_commit": self.code_commit,
            "expected_current_collection": self.expected_current_collection,
            "source_collection": self.source_collection,
            "target_collection": self.target_collection,
            "request_payload_hash": self.request_payload_hash,
            "confirmation_hash": self.confirmation_hash,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "reconcile_required": self.reconcile_required,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class DeploymentRecord:
    """一筆 alias→collection 的部署歷史（§十八 deployment_records）。

    ``active=True`` 表示目前 alias 指向此 physical collection（依 DB 記錄）。
    真實 alias target 以 Qdrant 為準；不一致時需 reconcile（§十九）。
    """

    deployment_id: str
    alias_name: str
    physical_collection: str
    operation_id: str
    active: bool
    previous_collection: Optional[str] = None
    search_id: Optional[str] = None
    candidate_id: Optional[str] = None
    config_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    corpus_hash: Optional[str] = None
    code_commit: Optional[str] = None
    canonical_text_version: Optional[str] = None
    identifier_normalization_version: Optional[str] = None
    embedding_fingerprint_hash: Optional[str] = None
    point_count: Optional[int] = None
    readiness_json: Optional[str] = None
    activated_at: Optional[str] = None
    deactivated_at: Optional[str] = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "alias_name": self.alias_name,
            "physical_collection": self.physical_collection,
            "previous_collection": self.previous_collection,
            "operation_id": self.operation_id,
            "active": self.active,
            "search_id": self.search_id,
            "candidate_id": self.candidate_id,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "corpus_hash": self.corpus_hash,
            "code_commit": self.code_commit,
            "canonical_text_version": self.canonical_text_version,
            "identifier_normalization_version": self.identifier_normalization_version,
            "embedding_fingerprint_hash": self.embedding_fingerprint_hash,
            "point_count": self.point_count,
            "activated_at": self.activated_at,
            "deactivated_at": self.deactivated_at,
        }


@dataclass
class DeploymentAuditEvent:
    """Append-only 稽核事件（§十八.8：published 事件不得被普通 update 覆寫）。"""

    event_id: str
    operation_id: str
    event_type: str
    actor: str
    details_json: str
    created_at: Optional[str] = None


@dataclass
class DeploymentLock:
    """單一 alias 的部署鎖（§十七 concurrent guard）。"""

    alias_name: str
    operation_id: str
    acquired_at: str
    heartbeat_at: str


# --------------------------------------------------------------------------
# 決策 / 結果模型（非持久化，供 workflow 與 report）
# --------------------------------------------------------------------------
@dataclass
class ApplyEligibilityDecision:
    """Eligibility Guard 的結果（§八：不得只回傳單一 Boolean）。"""

    eligible: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_hashes: dict[str, Any] = field(default_factory=dict)
    candidate_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "validated_hashes": dict(self.validated_hashes),
            "candidate_snapshot": dict(self.candidate_snapshot),
        }


@dataclass
class CorpusManifest:
    """Target Active Corpus Manifest（§十一）。

    僅穩定欄位：相對／穩定名稱 + 每檔 SHA-256 + count + 版本。**不含**絕對路徑或 mtime。
    """

    corpus_hash: str
    corpus_hash_version: str
    document_count: int
    chunk_count: Optional[int]
    canonical_text_version: str
    loader_version: str
    documents: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_hash": self.corpus_hash,
            "corpus_hash_version": self.corpus_hash_version,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "canonical_text_version": self.canonical_text_version,
            "loader_version": self.loader_version,
            "documents": list(self.documents),
        }


@dataclass
class DeploymentBuildManifest:
    """``DeploymentCollectionBuilder`` 的輸出（§十二）。"""

    collection_name: str
    point_count: int
    chunk_count: int
    vector_size: int
    distance: str
    strategy: str
    config_hash: str
    chunk_config_hash: str
    corpus_hash: str
    canonical_text_version: str
    identifier_normalization_version: str
    embedding_fingerprint_hash: str
    identifier_index: bool
    payload_fields: list[str] = field(default_factory=list)
    reused_existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "point_count": self.point_count,
            "chunk_count": self.chunk_count,
            "vector_size": self.vector_size,
            "distance": self.distance,
            "strategy": self.strategy,
            "config_hash": self.config_hash,
            "chunk_config_hash": self.chunk_config_hash,
            "corpus_hash": self.corpus_hash,
            "canonical_text_version": self.canonical_text_version,
            "identifier_normalization_version": self.identifier_normalization_version,
            "embedding_fingerprint_hash": self.embedding_fingerprint_hash,
            "identifier_index": self.identifier_index,
            "payload_fields": sorted(set(self.payload_fields)),
            "reused_existing": self.reused_existing,
        }


@dataclass
class ReadinessReport:
    """``DeploymentReadinessVerifier`` 的結果（§十三）。"""

    ready: bool
    collection_name: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    probe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "collection_name": self.collection_name,
            "checks": list(self.checks),
            "failures": list(self.failures),
            "probe": dict(self.probe),
        }
