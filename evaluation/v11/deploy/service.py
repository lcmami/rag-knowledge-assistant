"""Deployment Service：initialize / apply / rollback / reconcile（Slice 7 §十四～§十九）。

協調 Eligibility Guard、Collection Builder、Readiness Verifier、Alias Manager 與
Deployment Repository，串起安全、可稽核、可回滾的 workflow。所有 collaborator 皆可注入，
因此 orchestration 邏輯（順序 / lock / idempotency / stale / cleanup / audit）可在**不需真實
Qdrant** 下以 fake 完整單元測試；integration 測試再以真實 Qdrant 驗證 atomic alias 切換。

安全不變式：
- 不對真實 active alias（``config.active_alias``）執行任何 write，除非 ``allow_production_alias``（§六.5 預設 False）。
- Build / Readiness 失敗（alias 尚未切）→ 清除新建 collection；alias 已切後之 DB 失敗 → **不刪** active collection（§十五.10/.11）。
- 任何路徑都不刪除 previous collection（§十五.7 / §十六.7）。
- Eligibility 一律由 DB locked outcome 重新讀取，不信任 request boolean（§二十一.9）。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from evaluation.v11.deploy import confirmation as conf
from evaluation.v11.deploy import naming, report as report_mod
from evaluation.v11.deploy.eligibility import (
    ApplyEligibilityGuard,
    ApplyRequestSnapshot,
    CurrentEnvironment,
    primary_error_code,
)
from evaluation.v11.deploy.models import (
    ApplyEligibilityDecision,
    DeploymentError,
    DeploymentOperation,
    DeploymentRecord,
    ErrorCode,
    EventType,
    OperationStatus,
    OperationType,
    ReadinessReport,
)
from evaluation.v11.deploy.repository import DeploymentRepository
from evaluation.v11.hashing import canonical_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------
# Request / config / result 模型
# --------------------------------------------------------------------------
@dataclass
class InitializeRequest:
    alias: str
    collection: str
    confirmation: str
    idempotency_key: str


ApplyRequest = ApplyRequestSnapshot  # 別名：apply request 就是 eligibility 的 expected snapshot


@dataclass
class RollbackRequest:
    alias: str
    expected_current_collection: str
    target_collection: str
    confirmation: str
    idempotency_key: str


@dataclass
class ReconcileRequest:
    alias: str
    idempotency_key: str


@dataclass
class DeploymentConfig:
    active_alias: str = naming.DEFAULT_ACTIVE_ALIAS
    versioned_prefix: str = naming.DEFAULT_VERSIONED_PREFIX
    corpus_paths: tuple[str, ...] = ()
    corpus_base_dir: Optional[str] = None
    admin_enabled: bool = False
    report_dir: Optional[str] = None
    allow_production_alias: bool = False
    actor: str = "local-admin"
    canonical_text_version: str = "v1"
    identifier_normalization_version: str = "v1"


@dataclass
class DeploymentResult:
    operation: DeploymentOperation
    decision: Optional[ApplyEligibilityDecision] = None
    build_manifest: Optional[dict[str, Any]] = None
    readiness: Optional[dict[str, Any]] = None
    record: Optional[DeploymentRecord] = None
    reconciliation: Optional[dict[str, Any]] = None
    report_paths: Optional[dict[str, str]] = None
    idempotent_replay: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.to_public_dict(),
            "eligibility": (self.decision.to_dict() if self.decision else None),
            "build_manifest": self.build_manifest,
            "readiness": self.readiness,
            "deployment_record": (self.record.to_public_dict() if self.record else None),
            "reconciliation": self.reconciliation,
            "idempotent_replay": self.idempotent_replay,
        }


class DeploymentService:
    """單機 local admin deployment 協調器（§二十.11）。"""

    def __init__(
        self,
        repo: DeploymentRepository,
        alias_manager,
        builder,
        verifier,
        config: DeploymentConfig,
        *,
        env_provider: Callable[[], dict[str, Any]],
        cache_db_path: Optional[str] = None,
    ):
        self.repo = repo
        self.alias = alias_manager
        self.builder = builder
        self.verifier = verifier
        self.config = config
        self._env_provider = env_provider
        self.guard = ApplyEligibilityGuard()

    # ------------------------------------------------------------------
    # 共用輔助
    # ------------------------------------------------------------------
    def _assert_operable_alias(self, alias: str) -> None:
        """本 Slice 不對真實 active alias 執行任何 write（§六.5 / §四.4）。"""
        if alias == self.config.active_alias and not self.config.allow_production_alias:
            raise DeploymentError(
                ErrorCode.INVALID_REQUEST,
                "本 Slice 不對真實 active alias 執行部署操作；請使用隔離 alias（§六.5）。",
            )

    def _audit(self, operation_id: str, event_type: str, details: dict[str, Any]) -> None:
        self.repo.add_audit_event(
            _new_id("evt"), operation_id, event_type,
            actor=self.config.actor, details=details,
        )

    def _finish(
        self, op: DeploymentOperation, status: str, *, error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        op.status = status
        op.error_code = error_code
        op.error_message = error_message
        op.completed_at = _now()
        self.repo.update_operation(op)

    def _write_report(self, op: DeploymentOperation, **kwargs) -> Optional[dict[str, str]]:
        if not self.config.report_dir:
            return None
        try:
            audit = [
                {"event_type": e.event_type, "created_at": e.created_at}
                for e in self.repo.list_audit_events(op.operation_id)
            ]
            rep = report_mod.build_report(op, audit_summary=audit, **kwargs)
            return report_mod.write_report(self.config.report_dir, op.operation_id, rep)
        except Exception:  # noqa: BLE001 - report 為側產物，失敗不影響操作結果
            return None

    def _current_environment(self, alias: str) -> CurrentEnvironment:
        env = self._env_provider()
        return CurrentEnvironment(
            admin_enabled=self.config.admin_enabled,
            code_commit=env.get("code_commit"),
            working_tree_dirty=env.get("working_tree_dirty"),
            target_corpus_hash=env.get("target_corpus_hash"),
            canonical_text_version=env.get("canonical_text_version"),
            identifier_normalization_version=env.get("identifier_normalization_version"),
            embedding_fingerprint_hash=env.get("embedding_fingerprint_hash"),
            alias_current_target=self.alias.get_alias_target(alias),
        )

    # ------------------------------------------------------------------
    # §十四 initialize_alias
    # ------------------------------------------------------------------
    def initialize_alias(self, request: InitializeRequest) -> DeploymentResult:
        if not self.config.admin_enabled:
            raise DeploymentError(ErrorCode.OPTIMIZER_ADMIN_DISABLED, "admin write API 未啟用。")
        self._assert_operable_alias(request.alias)
        naming.assert_valid_alias_target(request.collection)

        payload_hash = _hash_payload({
            "type": OperationType.INITIALIZE,
            "alias": request.alias,
            "collection": request.collection,
            "confirmation_hash": conf.confirmation_hash(request.confirmation),
        })
        op = DeploymentOperation(
            operation_id=_new_id("op"),
            idempotency_key=request.idempotency_key,
            operation_type=OperationType.INITIALIZE,
            status=OperationStatus.PENDING,
            alias_name=request.alias,
            request_payload_hash=payload_hash,
            expected_current_collection=request.collection,
            target_collection=request.collection,
            confirmation_hash=conf.confirmation_hash(request.confirmation),
        )
        op, created = self.repo.create_operation(op)
        if not created:
            return DeploymentResult(operation=op, idempotent_replay=True)

        self._audit(op.operation_id, EventType.OPERATION_CREATED, {"type": op.operation_type})
        try:
            lock = self.repo.acquire_lock(request.alias, op.operation_id)
        except DeploymentError as exc:
            self._finish(op, OperationStatus.REJECTED, error_code=exc.code, error_message=exc.message)
            self._audit(op.operation_id, EventType.OPERATION_REJECTED, {"error_code": exc.code})
            return DeploymentResult(operation=op)

        try:
            op.status = OperationStatus.RUNNING
            op.started_at = _now()
            self.repo.update_operation(op)

            # §十四.1：alias 已存在不得覆寫。
            if self.alias.alias_exists(request.alias):
                return self._reject(op, ErrorCode.ALIAS_ALREADY_INITIALIZED,
                                    f"alias {request.alias!r} 已存在。")
            # §十四.2：existing collection 必須存在。
            if not self.alias._collection_exists(request.collection):
                return self._reject(op, ErrorCode.COLLECTION_NOT_FOUND,
                                    f"collection {request.collection!r} 不存在。")
            # §十四.3：confirmation 精確匹配。
            expected_conf = conf.initialize_confirmation(request.alias, request.collection)
            if not conf.confirmation_matches(request.confirmation, expected_conf):
                return self._reject(op, ErrorCode.CONFIRMATION_MISMATCH, "confirmation 不符。")

            # §十四.Atomic create alias + read-back。
            self.alias.create_alias(request.alias, request.collection)
            target = self.alias.get_alias_target(request.alias)
            if target != request.collection:
                op.reconcile_required = True
                self._finish(op, OperationStatus.NEEDS_RECONCILE,
                             error_code=ErrorCode.MANUAL_RECONCILIATION_REQUIRED,
                             error_message="alias read-back 與預期不符。")
                self._audit(op.operation_id, EventType.RECONCILE_MANUAL_REQUIRED,
                            {"read_back": target})
                return DeploymentResult(operation=op)

            # §十四.baseline deployment record。
            record = DeploymentRecord(
                deployment_id=_new_id("dep"),
                alias_name=request.alias,
                physical_collection=request.collection,
                previous_collection=None,
                active=True,
                operation_id=op.operation_id,
                activated_at=_now(),
            )
            self.repo.insert_record(record)
            self._audit(op.operation_id, EventType.ALIAS_INITIALIZED,
                        {"alias": request.alias, "collection": request.collection})
            self._finish(op, OperationStatus.COMPLETED)
            paths = self._write_report(op, record=record,
                                       alias_switch={"final": target, "read_back": target})
            return DeploymentResult(operation=op, record=record, report_paths=paths)
        finally:
            self.repo.release_lock(request.alias, op.operation_id)

    # ------------------------------------------------------------------
    # §十五 apply
    # ------------------------------------------------------------------
    def apply(self, request: ApplyRequest) -> DeploymentResult:
        if not self.config.admin_enabled:
            raise DeploymentError(ErrorCode.OPTIMIZER_ADMIN_DISABLED, "admin write API 未啟用。")
        self._assert_operable_alias(request.alias_name)

        payload_hash = _hash_payload({
            "type": OperationType.APPLY,
            "alias": request.alias_name,
            "search_id": request.search_id,
            "candidate_id": request.candidate_id,
            "expected_config_hash": request.expected_config_hash,
            "expected_dataset_hash": request.expected_dataset_hash,
            "expected_corpus_hash": request.expected_corpus_hash,
            "expected_code_commit": request.expected_code_commit,
            "expected_current_collection": request.expected_current_collection,
            "confirmation_hash": conf.confirmation_hash(request.confirmation),
        })
        op = DeploymentOperation(
            operation_id=_new_id("op"),
            idempotency_key=request.idempotency_key,
            operation_type=OperationType.APPLY,
            status=OperationStatus.PENDING,
            alias_name=request.alias_name,
            request_payload_hash=payload_hash,
            search_id=request.search_id,
            candidate_id=request.candidate_id,
            config_hash=request.expected_config_hash,
            dataset_hash=request.expected_dataset_hash,
            corpus_hash=request.expected_corpus_hash,
            code_commit=request.expected_code_commit,
            expected_current_collection=request.expected_current_collection,
            source_collection=request.expected_current_collection,
            confirmation_hash=conf.confirmation_hash(request.confirmation),
        )
        op, created = self.repo.create_operation(op)
        if not created:
            # §十七.1：idempotent replay —— 不重建 collection、不重切 alias。
            readiness = json.loads(op.readiness_json) if op.readiness_json else None
            return DeploymentResult(operation=op, readiness=readiness, idempotent_replay=True)

        self._audit(op.operation_id, EventType.OPERATION_CREATED, {"type": op.operation_type})
        try:
            lock = self.repo.acquire_lock(request.alias_name, op.operation_id)
        except DeploymentError as exc:
            self._finish(op, OperationStatus.REJECTED, error_code=exc.code, error_message=exc.message)
            self._audit(op.operation_id, EventType.OPERATION_REJECTED, {"error_code": exc.code})
            return DeploymentResult(operation=op)

        built_collection: Optional[str] = None
        try:
            op.status = OperationStatus.RUNNING
            op.started_at = _now()
            self.repo.update_operation(op)

            # 1) Eligibility Guard（由 DB locked outcome 重新讀取，§二十一）。
            search = self.repo.get_search(request.search_id)
            candidate = (
                self.repo.get_candidate_by_config_hash(request.search_id, request.expected_config_hash)
                if search else None
            )
            environment = self._current_environment(request.alias_name)
            decision = self.guard.evaluate(
                search=search, candidate=candidate, environment=environment, request=request
            )
            if not decision.eligible:
                code = primary_error_code(decision.reasons)
                self._finish(op, OperationStatus.REJECTED, error_code=code,
                             error_message="apply rejected：" + ",".join(decision.reasons))
                self._audit(op.operation_id, EventType.ELIGIBILITY_REJECTED,
                            {"reasons": decision.reasons})
                paths = self._write_report(op, eligibility=decision.to_dict())
                return DeploymentResult(operation=op, decision=decision, report_paths=paths)

            # 2) 建 target collection 名稱（deterministic / idempotent）。
            target = naming.build_versioned_collection_name(
                request.expected_config_hash, payload_hash[:12],
                prefix=self.config.versioned_prefix,
            )
            op.target_collection = target
            self.repo.update_operation(op)

            # 3) Build（失敗 → cleanup 半成品，alias 未動）。
            self._audit(op.operation_id, EventType.BUILD_STARTED, {"target": target})
            try:
                manifest = self.builder.build(
                    target, candidate.config_json, self.config.corpus_paths,
                    corpus_base_dir=self.config.corpus_base_dir,
                    expected_config_hash=request.expected_config_hash,
                    corpus_hash=request.expected_corpus_hash,
                    canonical_text_version=self.config.canonical_text_version,
                    identifier_normalization_version=self.config.identifier_normalization_version,
                )
                built_collection = target
            except DeploymentError as exc:
                self._audit(op.operation_id, EventType.BUILD_FAILED, {"error_code": exc.code})
                self._finish(op, OperationStatus.FAILED, error_code=exc.code, error_message=exc.message)
                paths = self._write_report(op, eligibility=decision.to_dict())
                return DeploymentResult(operation=op, decision=decision, report_paths=paths)
            self._audit(op.operation_id, EventType.BUILD_COMPLETED, manifest.to_dict())

            # 4) Readiness（失敗 → 不切 alias、cleanup 新建 collection）。
            readiness = self.verifier.verify(
                manifest,
                expected_corpus_hash=request.expected_corpus_hash,
                expected_config_hash=request.expected_config_hash,
                expected_canonical_text_version=self.config.canonical_text_version,
                expected_identifier_normalization_version=self.config.identifier_normalization_version,
                expected_embedding_fingerprint_hash=manifest.embedding_fingerprint_hash,
            )
            op.readiness_json = canonical_json(readiness.to_dict())
            self.repo.update_operation(op)
            if not readiness.ready:
                self._audit(op.operation_id, EventType.READINESS_FAILED,
                            {"failures": readiness.failures})
                self._cleanup_built(built_collection)
                self._finish(op, OperationStatus.FAILED, error_code=ErrorCode.READINESS_FAILED,
                             error_message="readiness failed：" + ",".join(readiness.failures))
                paths = self._write_report(op, eligibility=decision.to_dict(),
                                           build_manifest=manifest.to_dict(),
                                           readiness=readiness.to_dict())
                return DeploymentResult(operation=op, decision=decision,
                                        build_manifest=manifest.to_dict(),
                                        readiness=readiness.to_dict(), report_paths=paths)
            self._audit(op.operation_id, EventType.READINESS_PASSED, {"collection": target})

            # 5) Alias 仍指向 expected current？（build 期間可能被切走，§十五.4）
            current = self.alias.get_alias_target(request.alias_name)
            if current != request.expected_current_collection:
                self._cleanup_built(built_collection)
                self._finish(op, OperationStatus.REJECTED, error_code=ErrorCode.STALE_ALIAS_TARGET,
                             error_message="alias target 在 build 期間變更，拒絕覆蓋。")
                self._audit(op.operation_id, EventType.OPERATION_REJECTED,
                            {"error_code": ErrorCode.STALE_ALIAS_TARGET, "current": current})
                paths = self._write_report(op, eligibility=decision.to_dict(),
                                           build_manifest=manifest.to_dict(),
                                           readiness=readiness.to_dict())
                return DeploymentResult(operation=op, decision=decision,
                                        build_manifest=manifest.to_dict(),
                                        readiness=readiness.to_dict(), report_paths=paths)

            # 6) Atomic alias switch + read-back。
            try:
                read_back = self.alias.switch_alias(
                    request.alias_name, target,
                    expected_current=request.expected_current_collection,
                )
            except DeploymentError as exc:
                # alias 未切（switch 前置檢查失敗）→ 可安全 cleanup 新建 collection（§十五.10）。
                self._cleanup_built(built_collection)
                self._finish(op, OperationStatus.REJECTED, error_code=exc.code, error_message=exc.message)
                self._audit(op.operation_id, EventType.OPERATION_REJECTED, {"error_code": exc.code})
                return DeploymentResult(operation=op, decision=decision,
                                        build_manifest=manifest.to_dict(),
                                        readiness=readiness.to_dict())
            self._audit(op.operation_id, EventType.ALIAS_SWITCHED,
                        {"from": current, "to": target})

            if read_back != target:
                # alias 已切但 read-back 不符 → 需 reconcile；**不刪** collection（§十五.6）。
                op.reconcile_required = True
                self._finish(op, OperationStatus.NEEDS_RECONCILE,
                             error_code=ErrorCode.MANUAL_RECONCILIATION_REQUIRED,
                             error_message="alias read-back 與預期不符，需 reconcile。")
                self._audit(op.operation_id, EventType.RECONCILE_MANUAL_REQUIRED,
                            {"read_back": read_back, "expected": target})
                return DeploymentResult(operation=op, decision=decision,
                                        build_manifest=manifest.to_dict(),
                                        readiness=readiness.to_dict())

            # 7) 持久化 deployment record（alias 已切；DB 失敗 → needs_reconcile，不刪 active）。
            record = DeploymentRecord(
                deployment_id=_new_id("dep"),
                alias_name=request.alias_name,
                physical_collection=target,
                previous_collection=request.expected_current_collection,
                search_id=request.search_id,
                candidate_id=request.candidate_id,
                config_hash=request.expected_config_hash,
                dataset_hash=request.expected_dataset_hash,
                corpus_hash=request.expected_corpus_hash,
                code_commit=request.expected_code_commit,
                canonical_text_version=manifest.canonical_text_version,
                identifier_normalization_version=manifest.identifier_normalization_version,
                embedding_fingerprint_hash=manifest.embedding_fingerprint_hash,
                point_count=manifest.point_count,
                readiness_json=op.readiness_json,
                active=True,
                operation_id=op.operation_id,
                activated_at=_now(),
            )
            try:
                self.repo.switch_active_record(record, _now())
            except Exception as exc:  # noqa: BLE001 - alias 已切，DB 失敗不得刪 active（§十五.11）
                op.reconcile_required = True
                self._finish(op, OperationStatus.NEEDS_RECONCILE,
                             error_code=ErrorCode.MANUAL_RECONCILIATION_REQUIRED,
                             error_message="alias 已切但 deployment record 寫入失敗，需 reconcile。")
                self._audit(op.operation_id, EventType.RECONCILE_MANUAL_REQUIRED,
                            {"reason": "record_write_failed"})
                return DeploymentResult(operation=op, decision=decision,
                                        build_manifest=manifest.to_dict(),
                                        readiness=readiness.to_dict())

            self._audit(op.operation_id, EventType.DEPLOYMENT_ACTIVATED,
                        {"collection": target, "previous": request.expected_current_collection})
            self._finish(op, OperationStatus.COMPLETED)
            paths = self._write_report(
                op, eligibility=decision.to_dict(), build_manifest=manifest.to_dict(),
                readiness=readiness.to_dict(), record=record,
                alias_switch={"from": current, "to": target, "read_back": read_back},
            )
            return DeploymentResult(operation=op, decision=decision,
                                    build_manifest=manifest.to_dict(),
                                    readiness=readiness.to_dict(), record=record,
                                    report_paths=paths)
        finally:
            self.repo.release_lock(request.alias_name, op.operation_id)

    # ------------------------------------------------------------------
    # §十六 rollback
    # ------------------------------------------------------------------
    def rollback(self, request: RollbackRequest) -> DeploymentResult:
        if not self.config.admin_enabled:
            raise DeploymentError(ErrorCode.OPTIMIZER_ADMIN_DISABLED, "admin write API 未啟用。")
        self._assert_operable_alias(request.alias)

        payload_hash = _hash_payload({
            "type": OperationType.ROLLBACK,
            "alias": request.alias,
            "expected_current": request.expected_current_collection,
            "target": request.target_collection,
            "confirmation_hash": conf.confirmation_hash(request.confirmation),
        })
        op = DeploymentOperation(
            operation_id=_new_id("op"),
            idempotency_key=request.idempotency_key,
            operation_type=OperationType.ROLLBACK,
            status=OperationStatus.PENDING,
            alias_name=request.alias,
            request_payload_hash=payload_hash,
            expected_current_collection=request.expected_current_collection,
            source_collection=request.expected_current_collection,
            target_collection=request.target_collection,
            confirmation_hash=conf.confirmation_hash(request.confirmation),
        )
        op, created = self.repo.create_operation(op)
        if not created:
            return DeploymentResult(operation=op, idempotent_replay=True)

        self._audit(op.operation_id, EventType.OPERATION_CREATED, {"type": op.operation_type})
        try:
            lock = self.repo.acquire_lock(request.alias, op.operation_id)
        except DeploymentError as exc:
            self._finish(op, OperationStatus.REJECTED, error_code=exc.code, error_message=exc.message)
            self._audit(op.operation_id, EventType.OPERATION_REJECTED, {"error_code": exc.code})
            return DeploymentResult(operation=op)

        try:
            op.status = OperationStatus.RUNNING
            op.started_at = _now()
            self.repo.update_operation(op)

            current = self.alias.get_alias_target(request.alias)
            # §十六.5：current 與 expected 不符 → 拒絕。
            if current != request.expected_current_collection:
                return self._reject(op, ErrorCode.STALE_ALIAS_TARGET,
                                    f"alias 目前指向 {current!r}，與 expected 不符。")
            # §十六.2：target 必須是本 alias deployment history 中的 known collection。
            known = self.repo.known_collections(request.alias)
            if request.target_collection not in known:
                return self._reject(op, ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED,
                                    "rollback target 不在部署歷史中。")
            # §十六.3：target 不得為 optimizer trial collection。
            if naming.is_optimizer_trial(request.target_collection):
                return self._reject(op, ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED,
                                    "rollback target 不得為 optimizer trial collection。")
            # §十六.4：target 必須存在且曾 readiness 通過。
            if not self.alias._collection_exists(request.target_collection):
                return self._reject(op, ErrorCode.COLLECTION_NOT_FOUND, "rollback target collection 不存在。")
            if not self._has_ready_record(request.alias, request.target_collection):
                return self._reject(op, ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED,
                                    "rollback target 沒有 readiness 通過的部署記錄。")
            # §十六.6：confirmation。
            expected_conf = conf.rollback_confirmation(
                request.alias, current, request.target_collection)
            if not conf.confirmation_matches(request.confirmation, expected_conf):
                return self._reject(op, ErrorCode.CONFIRMATION_MISMATCH, "confirmation 不符。")

            # Atomic switch + read-back。
            try:
                read_back = self.alias.switch_alias(
                    request.alias, request.target_collection, expected_current=current)
            except DeploymentError as exc:
                return self._reject(op, exc.code, exc.message)
            self._audit(op.operation_id, EventType.ALIAS_SWITCHED,
                        {"from": current, "to": request.target_collection})
            if read_back != request.target_collection:
                op.reconcile_required = True
                self._finish(op, OperationStatus.NEEDS_RECONCILE,
                             error_code=ErrorCode.MANUAL_RECONCILIATION_REQUIRED,
                             error_message="rollback read-back 不符，需 reconcile。")
                self._audit(op.operation_id, EventType.RECONCILE_MANUAL_REQUIRED,
                            {"read_back": read_back})
                return DeploymentResult(operation=op)

            prior = self._find_record(request.alias, request.target_collection)
            record = DeploymentRecord(
                deployment_id=_new_id("dep"),
                alias_name=request.alias,
                physical_collection=request.target_collection,
                previous_collection=current,
                search_id=(prior.search_id if prior else None),
                candidate_id=(prior.candidate_id if prior else None),
                config_hash=(prior.config_hash if prior else None),
                dataset_hash=(prior.dataset_hash if prior else None),
                corpus_hash=(prior.corpus_hash if prior else None),
                code_commit=(prior.code_commit if prior else None),
                canonical_text_version=(prior.canonical_text_version if prior else None),
                identifier_normalization_version=(
                    prior.identifier_normalization_version if prior else None),
                embedding_fingerprint_hash=(prior.embedding_fingerprint_hash if prior else None),
                point_count=(prior.point_count if prior else None),
                readiness_json=(prior.readiness_json if prior else None),
                active=True,
                operation_id=op.operation_id,
                activated_at=_now(),
            )
            self.repo.switch_active_record(record, _now())
            self._audit(op.operation_id, EventType.ROLLBACK_COMPLETED,
                        {"from": current, "to": request.target_collection})
            self._finish(op, OperationStatus.COMPLETED)
            paths = self._write_report(
                op, record=record, rollback_target=request.target_collection,
                alias_switch={"from": current, "to": request.target_collection,
                              "read_back": read_back},
            )
            return DeploymentResult(operation=op, record=record, report_paths=paths)
        finally:
            self.repo.release_lock(request.alias, op.operation_id)

    # ------------------------------------------------------------------
    # §十九 reconcile
    # ------------------------------------------------------------------
    def reconcile(self, request: ReconcileRequest) -> DeploymentResult:
        if not self.config.admin_enabled:
            raise DeploymentError(ErrorCode.OPTIMIZER_ADMIN_DISABLED, "admin write API 未啟用。")

        payload_hash = _hash_payload({"type": OperationType.RECONCILE, "alias": request.alias})
        op = DeploymentOperation(
            operation_id=_new_id("op"),
            idempotency_key=request.idempotency_key,
            operation_type=OperationType.RECONCILE,
            status=OperationStatus.RUNNING,
            alias_name=request.alias,
            request_payload_hash=payload_hash,
            started_at=_now(),
        )
        op, created = self.repo.create_operation(op)
        if not created:
            return DeploymentResult(operation=op, idempotent_replay=True)
        self._audit(op.operation_id, EventType.RECONCILE_STARTED, {"alias": request.alias})

        # §十九.1：先讀真實 alias target（不切 alias，不刪 collection）。
        alias_target = self.alias.get_alias_target(request.alias)
        active_record = self.repo.get_active_record(request.alias)
        db_target = active_record.physical_collection if active_record else None
        known = self.repo.known_collections(request.alias)

        # 釋放遺留 lock（其 owning operation 已終態）。
        released_lock = self._release_stale_lock(request.alias)

        result: dict[str, Any] = {
            "alias": request.alias,
            "alias_target": alias_target,
            "db_active_target": db_target,
            "released_stale_lock": released_lock,
        }

        if alias_target is not None and alias_target == db_target:
            result["resolution"] = "consistent"
            self._audit(op.operation_id, EventType.RECONCILE_RESOLVED, result)
            self._finish(op, OperationStatus.COMPLETED)
        elif alias_target is not None and alias_target in known:
            # alias 指向 known collection（可證明）→ 只更新 DB active，不切 alias（§十九.3）。
            rec = self._find_record(request.alias, alias_target)
            ts = _now()
            self.repo.deactivate_all(request.alias, ts)
            if rec:
                self.repo.set_record_active(rec.deployment_id, True, ts)
            result["resolution"] = "db_updated_to_alias"
            self._audit(op.operation_id, EventType.RECONCILE_RESOLVED, result)
            self._finish(op, OperationStatus.COMPLETED)
        else:
            # 無法判定（alias 缺失 / 指向未知 collection）→ 人工（§十九.4/.5，不猜測）。
            op.reconcile_required = True
            result["resolution"] = "manual_reconciliation_required"
            self._audit(op.operation_id, EventType.RECONCILE_MANUAL_REQUIRED, result)
            self._finish(op, OperationStatus.NEEDS_RECONCILE,
                         error_code=ErrorCode.MANUAL_RECONCILIATION_REQUIRED,
                         error_message="alias 與 DB 不一致且無法自動判定。")
        paths = self._write_report(op, reconciliation=result)
        return DeploymentResult(operation=op, reconciliation=result, report_paths=paths)

    # ------------------------------------------------------------------
    # 內部輔助
    # ------------------------------------------------------------------
    def _reject(self, op: DeploymentOperation, code: str, message: str) -> DeploymentResult:
        self._finish(op, OperationStatus.REJECTED, error_code=code, error_message=message)
        self._audit(op.operation_id, EventType.OPERATION_REJECTED, {"error_code": code})
        return DeploymentResult(operation=op)

    def _cleanup_built(self, collection: Optional[str]) -> None:
        if not collection:
            return
        try:
            self.builder.cleanup(collection, active_targets=self.repo.active_targets())
        except Exception:  # noqa: BLE001 - cleanup 不得遮蔽原始結果
            pass

    def _has_ready_record(self, alias: str, collection: str) -> bool:
        for r in self.repo.list_records(alias):
            if r.physical_collection != collection or not r.readiness_json:
                continue
            try:
                if json.loads(r.readiness_json).get("ready") is True:
                    return True
            except (ValueError, TypeError):
                continue
        # baseline initialize record 無 readiness_json，但屬 known & existing → 允許 rollback 回 baseline。
        for r in self.repo.list_records(alias):
            if r.physical_collection == collection and r.previous_collection is None:
                return True
        return False

    def _find_record(self, alias: str, collection: str) -> Optional[DeploymentRecord]:
        matches = [r for r in self.repo.list_records(alias) if r.physical_collection == collection]
        return matches[-1] if matches else None

    def _release_stale_lock(self, alias: str) -> bool:
        lock = self.repo.get_lock(alias)
        if not lock:
            return False
        owner = self.repo.get_operation(lock.operation_id)
        terminal = {OperationStatus.COMPLETED, OperationStatus.FAILED,
                    OperationStatus.REJECTED, OperationStatus.NEEDS_RECONCILE}
        if owner is None or owner.status in terminal:
            self.repo.release_lock(alias, lock.operation_id)
            return True
        return False


def _hash_payload(payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
