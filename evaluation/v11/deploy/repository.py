"""Deployment 持久化（Slice 7 §十八）。

繼承 :class:`~evaluation.v11.search.search_repository.SearchRepository`，因此**同一個
獨立 Optimizer/Deployment SQLite** 內同時擁有 ``optimizer_searches`` /
``optimizer_candidates``（供 Eligibility Guard 讀取，§二十一）與四張 deployment 表。

沿用同一套 migration runner（idempotent、失敗 rollback）；deployment migration ``0008``
接在既有 ``MIGRATIONS`` 之後，但**不修改**既有 search migration 清單（§二十五 向後相容）。

不變式：
- ``deployment_operations.idempotency_key`` 全域 UNIQUE（§十七.4）。
- ``deployment_locks.alias_name`` 為 PRIMARY KEY → 單一 alias 同時只有一個 write（§十七 concurrent guard）。
- Audit event 僅 append（無 update 路徑，§十八.8）。
- 不產生 ``documents`` 表；永不修改正式 ``metadata.db``（§十八.6/.7）。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from evaluation.v11.deploy.models import (
    DeploymentAuditEvent,
    DeploymentError,
    DeploymentLock,
    DeploymentOperation,
    DeploymentRecord,
    ErrorCode,
)
from evaluation.v11.optimizer_repository import MIGRATIONS as _BASE_MIGRATIONS
from evaluation.v11.search.search_repository import SearchRepository

# --------------------------------------------------------------------------
# Migration 0008：Deployment 最小表集（§十八）
# --------------------------------------------------------------------------
_MIGRATION_0008 = [
    """
    CREATE TABLE IF NOT EXISTS deployment_operations (
        operation_id                TEXT PRIMARY KEY,
        idempotency_key             TEXT NOT NULL UNIQUE,
        operation_type              TEXT NOT NULL,
        status                      TEXT NOT NULL,
        alias_name                  TEXT NOT NULL,
        search_id                   TEXT,
        candidate_id                TEXT,
        config_hash                 TEXT,
        dataset_hash                TEXT,
        corpus_hash                 TEXT,
        code_commit                 TEXT,
        expected_current_collection TEXT,
        source_collection           TEXT,
        target_collection           TEXT,
        request_payload_hash        TEXT NOT NULL,
        confirmation_hash           TEXT,
        readiness_json              TEXT,
        error_code                  TEXT,
        error_message               TEXT,
        created_at                  TEXT,
        started_at                  TEXT,
        completed_at                TEXT,
        reconcile_required          INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_deploy_ops_alias ON deployment_operations(alias_name)",
    """
    CREATE TABLE IF NOT EXISTS deployment_records (
        deployment_id                     TEXT PRIMARY KEY,
        alias_name                        TEXT NOT NULL,
        physical_collection               TEXT NOT NULL,
        previous_collection               TEXT,
        search_id                         TEXT,
        candidate_id                      TEXT,
        config_hash                       TEXT,
        dataset_hash                      TEXT,
        corpus_hash                       TEXT,
        code_commit                       TEXT,
        canonical_text_version            TEXT,
        identifier_normalization_version  TEXT,
        embedding_fingerprint_hash        TEXT,
        point_count                       INTEGER,
        readiness_json                    TEXT,
        activated_at                      TEXT,
        deactivated_at                    TEXT,
        active                            INTEGER NOT NULL DEFAULT 0,
        operation_id                      TEXT NOT NULL,
        FOREIGN KEY(operation_id) REFERENCES deployment_operations(operation_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_deploy_records_alias ON deployment_records(alias_name)",
    "CREATE INDEX IF NOT EXISTS idx_deploy_records_active ON deployment_records(alias_name, active)",
    """
    CREATE TABLE IF NOT EXISTS deployment_audit_events (
        event_id      TEXT PRIMARY KEY,
        operation_id  TEXT NOT NULL,
        event_type    TEXT NOT NULL,
        actor         TEXT NOT NULL,
        details_json  TEXT NOT NULL DEFAULT '{}',
        created_at    TEXT NOT NULL,
        FOREIGN KEY(operation_id) REFERENCES deployment_operations(operation_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_deploy_audit_op ON deployment_audit_events(operation_id)",
    """
    CREATE TABLE IF NOT EXISTS deployment_locks (
        alias_name    TEXT PRIMARY KEY,
        operation_id  TEXT NOT NULL,
        acquired_at   TEXT NOT NULL,
        heartbeat_at  TEXT NOT NULL
    )
    """,
]

# --------------------------------------------------------------------------
# Deployment DB migration list = base search migrations (0001..0006) + 0008.
#
# Global migration numbering policy (approved decision):
#   0007 belongs to the separate review database migration namespace.
#   Gaps are intentional; migration IDs are globally unique and need not be
#   contiguous per database. The runner (`run_migrations`) applies pending
#   versions in list order and only skips versions already recorded in
#   `schema_migrations`; it makes NO assumption that IDs are contiguous, so
#   the absence of 0007 here is never treated as a "missing migration".
#
# This Deployment DB therefore applies: 0001,0002,0003,0004,0005,0006,0008.
# --------------------------------------------------------------------------
DEPLOY_MIGRATIONS: list[tuple[str, list[str]]] = list(_BASE_MIGRATIONS) + [
    ("0008_deployment", _MIGRATION_0008),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeploymentRepository(SearchRepository):
    """Deployment CRUD + idempotency + transactional lock（同一 DB 內可讀 search/candidate）。"""

    def __init__(self, db_path: str, migrations: list[tuple[str, list[str]]] | None = None):
        super().__init__(db_path, migrations=migrations or DEPLOY_MIGRATIONS)

    def table_names(self) -> list[str]:
        conn = self._connect()
        try:
            return sorted(
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            )
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Operations（含 idempotency，§十七）
    # ----------------------------------------------------------------------
    _OP_COLUMNS = (
        "operation_id", "idempotency_key", "operation_type", "status", "alias_name",
        "search_id", "candidate_id", "config_hash", "dataset_hash", "corpus_hash",
        "code_commit", "expected_current_collection", "source_collection",
        "target_collection", "request_payload_hash", "confirmation_hash",
        "readiness_json", "error_code", "error_message", "created_at", "started_at",
        "completed_at", "reconcile_required",
    )

    def _op_values(self, op: DeploymentOperation) -> tuple:
        return (
            op.operation_id, op.idempotency_key, op.operation_type, op.status,
            op.alias_name, op.search_id, op.candidate_id, op.config_hash,
            op.dataset_hash, op.corpus_hash, op.code_commit,
            op.expected_current_collection, op.source_collection, op.target_collection,
            op.request_payload_hash, op.confirmation_hash, op.readiness_json,
            op.error_code, op.error_message, op.created_at, op.started_at,
            op.completed_at, 1 if op.reconcile_required else 0,
        )

    def create_operation(self, op: DeploymentOperation) -> tuple[DeploymentOperation, bool]:
        """建立 operation，強制 idempotency（§十七）。

        回傳 ``(operation, created)``：
        - 相同 idempotency_key + 相同 request_payload_hash → 回傳既有 operation，``created=False``。
        - 相同 idempotency_key + 不同 payload → :class:`DeploymentError`(``idempotency_conflict``)。
        - 全新 → 寫入並回傳 ``created=True``。
        """
        existing = self.get_operation_by_idempotency_key(op.idempotency_key)
        if existing is not None:
            if existing.request_payload_hash != op.request_payload_hash:
                raise DeploymentError(
                    ErrorCode.IDEMPOTENCY_CONFLICT,
                    "相同 idempotency_key 但 request payload 不同，拒絕（§十七）。",
                )
            return existing, False

        if op.created_at is None:
            op.created_at = _now()
        placeholders = ",".join("?" for _ in self._OP_COLUMNS)
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT INTO deployment_operations ({','.join(self._OP_COLUMNS)}) "
                f"VALUES ({placeholders})",
                self._op_values(op),
            )
        except sqlite3.IntegrityError:
            # 競態下另一個交易先插入了同 key：改走 idempotent 讀取路徑。
            other = self.get_operation_by_idempotency_key(op.idempotency_key)
            if other is not None:
                if other.request_payload_hash != op.request_payload_hash:
                    raise DeploymentError(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        "相同 idempotency_key 但 request payload 不同，拒絕（§十七）。",
                    )
                return other, False
            raise
        finally:
            conn.close()
        return op, True

    def update_operation(self, op: DeploymentOperation) -> None:
        """更新既有 operation 的可變欄位（不改 operation_id / idempotency_key / created_at）。"""
        updatable = [c for c in self._OP_COLUMNS if c not in ("operation_id", "created_at")]
        assignments = ",".join(f"{c}=?" for c in updatable)
        values = tuple(
            getattr_map(op)[c] for c in updatable
        ) + (op.operation_id,)
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE deployment_operations SET {assignments} WHERE operation_id=?",
                values,
            )
        finally:
            conn.close()

    def get_operation(self, operation_id: str) -> Optional[DeploymentOperation]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM deployment_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_operation(row) if row else None

    def get_operation_by_idempotency_key(self, key: str) -> Optional[DeploymentOperation]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM deployment_operations WHERE idempotency_key=?", (key,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_operation(row) if row else None

    def list_operations(self, alias_name: Optional[str] = None) -> list[DeploymentOperation]:
        conn = self._connect()
        try:
            if alias_name is None:
                rows = conn.execute(
                    "SELECT * FROM deployment_operations ORDER BY created_at, operation_id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM deployment_operations WHERE alias_name=? "
                    "ORDER BY created_at, operation_id",
                    (alias_name,),
                ).fetchall()
        finally:
            conn.close()
        return [self._row_to_operation(r) for r in rows]

    def list_running_operations(self) -> list[DeploymentOperation]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM deployment_operations WHERE status IN ('pending','running') "
                "ORDER BY created_at, operation_id"
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_operation(r) for r in rows]

    @staticmethod
    def _row_to_operation(row: sqlite3.Row) -> DeploymentOperation:
        return DeploymentOperation(
            operation_id=row["operation_id"],
            idempotency_key=row["idempotency_key"],
            operation_type=row["operation_type"],
            status=row["status"],
            alias_name=row["alias_name"],
            search_id=row["search_id"],
            candidate_id=row["candidate_id"],
            config_hash=row["config_hash"],
            dataset_hash=row["dataset_hash"],
            corpus_hash=row["corpus_hash"],
            code_commit=row["code_commit"],
            expected_current_collection=row["expected_current_collection"],
            source_collection=row["source_collection"],
            target_collection=row["target_collection"],
            request_payload_hash=row["request_payload_hash"],
            confirmation_hash=row["confirmation_hash"],
            readiness_json=row["readiness_json"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            reconcile_required=bool(row["reconcile_required"]),
        )

    # ----------------------------------------------------------------------
    # Records（§十八 deployment_records）
    # ----------------------------------------------------------------------
    _REC_COLUMNS = (
        "deployment_id", "alias_name", "physical_collection", "previous_collection",
        "search_id", "candidate_id", "config_hash", "dataset_hash", "corpus_hash",
        "code_commit", "canonical_text_version", "identifier_normalization_version",
        "embedding_fingerprint_hash", "point_count", "readiness_json", "activated_at",
        "deactivated_at", "active", "operation_id",
    )

    def insert_record(self, record: DeploymentRecord) -> None:
        values = (
            record.deployment_id, record.alias_name, record.physical_collection,
            record.previous_collection, record.search_id, record.candidate_id,
            record.config_hash, record.dataset_hash, record.corpus_hash,
            record.code_commit, record.canonical_text_version,
            record.identifier_normalization_version, record.embedding_fingerprint_hash,
            record.point_count, record.readiness_json, record.activated_at,
            record.deactivated_at, 1 if record.active else 0, record.operation_id,
        )
        placeholders = ",".join("?" for _ in self._REC_COLUMNS)
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT INTO deployment_records ({','.join(self._REC_COLUMNS)}) "
                f"VALUES ({placeholders})",
                values,
            )
        finally:
            conn.close()

    def switch_active_record(
        self, record: DeploymentRecord, deactivated_at: str
    ) -> None:
        """原子地：把該 alias 現有 active 記錄設為 inactive，並插入新的 active 記錄。

        單一交易完成，確保「一個 alias 最多一筆 active」不變式（§十八.10）。
        """
        values = (
            record.deployment_id, record.alias_name, record.physical_collection,
            record.previous_collection, record.search_id, record.candidate_id,
            record.config_hash, record.dataset_hash, record.corpus_hash,
            record.code_commit, record.canonical_text_version,
            record.identifier_normalization_version, record.embedding_fingerprint_hash,
            record.point_count, record.readiness_json, record.activated_at,
            record.deactivated_at, 1 if record.active else 0, record.operation_id,
        )
        placeholders = ",".join("?" for _ in self._REC_COLUMNS)
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "UPDATE deployment_records SET active=0, deactivated_at=? "
                "WHERE alias_name=? AND active=1",
                (deactivated_at, record.alias_name),
            )
            conn.execute(
                f"INSERT INTO deployment_records ({','.join(self._REC_COLUMNS)}) "
                f"VALUES ({placeholders})",
                values,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def get_active_record(self, alias_name: str) -> Optional[DeploymentRecord]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM deployment_records WHERE alias_name=? AND active=1 "
                "ORDER BY activated_at DESC, deployment_id DESC LIMIT 1",
                (alias_name,),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_record(row) if row else None

    def list_records(self, alias_name: str) -> list[DeploymentRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM deployment_records WHERE alias_name=? "
                "ORDER BY activated_at, deployment_id",
                (alias_name,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_record(r) for r in rows]

    def known_collections(self, alias_name: str) -> set[str]:
        """該 alias 部署歷史中出現過的 physical collection（rollback known-target guard，§十六.2）。"""
        return {r.physical_collection for r in self.list_records(alias_name)}

    def active_targets(self) -> tuple[str, ...]:
        """所有 alias 目前 active 的 physical collection（cleanup 防護，§十二.11）。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT physical_collection FROM deployment_records WHERE active=1"
            ).fetchall()
        finally:
            conn.close()
        return tuple(r[0] for r in rows)

    def set_record_active(self, deployment_id: str, active: bool, ts: str) -> None:
        conn = self._connect()
        try:
            if active:
                conn.execute(
                    "UPDATE deployment_records SET active=1, deactivated_at=NULL "
                    "WHERE deployment_id=?",
                    (deployment_id,),
                )
            else:
                conn.execute(
                    "UPDATE deployment_records SET active=0, deactivated_at=? "
                    "WHERE deployment_id=?",
                    (ts, deployment_id),
                )
        finally:
            conn.close()

    def deactivate_all(self, alias_name: str, ts: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE deployment_records SET active=0, deactivated_at=? "
                "WHERE alias_name=? AND active=1",
                (ts, alias_name),
            )
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DeploymentRecord:
        return DeploymentRecord(
            deployment_id=row["deployment_id"],
            alias_name=row["alias_name"],
            physical_collection=row["physical_collection"],
            previous_collection=row["previous_collection"],
            search_id=row["search_id"],
            candidate_id=row["candidate_id"],
            config_hash=row["config_hash"],
            dataset_hash=row["dataset_hash"],
            corpus_hash=row["corpus_hash"],
            code_commit=row["code_commit"],
            canonical_text_version=row["canonical_text_version"],
            identifier_normalization_version=row["identifier_normalization_version"],
            embedding_fingerprint_hash=row["embedding_fingerprint_hash"],
            point_count=row["point_count"],
            readiness_json=row["readiness_json"],
            activated_at=row["activated_at"],
            deactivated_at=row["deactivated_at"],
            active=bool(row["active"]),
            operation_id=row["operation_id"],
        )

    # ----------------------------------------------------------------------
    # Audit events（append-only，§十八.8）
    # ----------------------------------------------------------------------
    def add_audit_event(
        self,
        event_id: str,
        operation_id: str,
        event_type: str,
        *,
        actor: str,
        details: Optional[dict[str, Any]] = None,
    ) -> DeploymentAuditEvent:
        ev = DeploymentAuditEvent(
            event_id=event_id,
            operation_id=operation_id,
            event_type=event_type,
            actor=actor,
            details_json=json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
            created_at=_now(),
        )
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO deployment_audit_events "
                "(event_id, operation_id, event_type, actor, details_json, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (ev.event_id, ev.operation_id, ev.event_type, ev.actor,
                 ev.details_json, ev.created_at),
            )
        finally:
            conn.close()
        return ev

    def list_audit_events(self, operation_id: str) -> list[DeploymentAuditEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM deployment_audit_events WHERE operation_id=? "
                "ORDER BY created_at, event_id",
                (operation_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            DeploymentAuditEvent(
                event_id=r["event_id"],
                operation_id=r["operation_id"],
                event_type=r["event_type"],
                actor=r["actor"],
                details_json=r["details_json"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ----------------------------------------------------------------------
    # Locks（transactional concurrent guard，§十七）
    # ----------------------------------------------------------------------
    def acquire_lock(self, alias_name: str, operation_id: str) -> DeploymentLock:
        """交易式取得 alias 部署鎖（§十七.1）。

        alias_name 為 PRIMARY KEY：第二個併發 write 會 UNIQUE 衝突 →
        :class:`DeploymentError`(``deployment_in_progress``)（§十七.2）。
        """
        ts = _now()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO deployment_locks (alias_name, operation_id, acquired_at, heartbeat_at) "
                "VALUES (?,?,?,?)",
                (alias_name, operation_id, ts, ts),
            )
        except sqlite3.IntegrityError as exc:
            raise DeploymentError(
                ErrorCode.DEPLOYMENT_IN_PROGRESS,
                f"alias {alias_name!r} 已有進行中的部署操作（§十七）。",
            ) from exc
        finally:
            conn.close()
        return DeploymentLock(alias_name, operation_id, ts, ts)

    def release_lock(self, alias_name: str, operation_id: str) -> None:
        """釋放鎖；僅刪除自己持有的鎖（operation_id 相符）。"""
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM deployment_locks WHERE alias_name=? AND operation_id=?",
                (alias_name, operation_id),
            )
        finally:
            conn.close()

    def get_lock(self, alias_name: str) -> Optional[DeploymentLock]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM deployment_locks WHERE alias_name=?", (alias_name,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return DeploymentLock(
            alias_name=row["alias_name"],
            operation_id=row["operation_id"],
            acquired_at=row["acquired_at"],
            heartbeat_at=row["heartbeat_at"],
        )

    def heartbeat_lock(self, alias_name: str, operation_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE deployment_locks SET heartbeat_at=? WHERE alias_name=? AND operation_id=?",
                (_now(), alias_name, operation_id),
            )
        finally:
            conn.close()

    def steal_stale_lock(self, alias_name: str, expected_operation_id: str, new_operation_id: str) -> bool:
        """安全接管 stale lock（§十七.4/.5）：僅當現有鎖確實屬 ``expected_operation_id`` 時改寫。

        呼叫端須先以 heartbeat + timeout 規則判定該鎖為 stale，再傳入其 operation_id。
        回傳是否成功接管。
        """
        ts = _now()
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE deployment_locks SET operation_id=?, acquired_at=?, heartbeat_at=? "
                "WHERE alias_name=? AND operation_id=?",
                (new_operation_id, ts, ts, alias_name, expected_operation_id),
            )
            return cur.rowcount > 0
        finally:
            conn.close()


def getattr_map(op: DeploymentOperation) -> dict[str, Any]:
    """DeploymentOperation → column dict（供 update_operation 動態組 SQL）。"""
    return {
        "operation_id": op.operation_id,
        "idempotency_key": op.idempotency_key,
        "operation_type": op.operation_type,
        "status": op.status,
        "alias_name": op.alias_name,
        "search_id": op.search_id,
        "candidate_id": op.candidate_id,
        "config_hash": op.config_hash,
        "dataset_hash": op.dataset_hash,
        "corpus_hash": op.corpus_hash,
        "code_commit": op.code_commit,
        "expected_current_collection": op.expected_current_collection,
        "source_collection": op.source_collection,
        "target_collection": op.target_collection,
        "request_payload_hash": op.request_payload_hash,
        "confirmation_hash": op.confirmation_hash,
        "readiness_json": op.readiness_json,
        "error_code": op.error_code,
        "error_message": op.error_message,
        "started_at": op.started_at,
        "completed_at": op.completed_at,
        "reconcile_required": 1 if op.reconcile_required else 0,
    }
