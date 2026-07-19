"""Optimizer SQLite Repository + 輕量 Migration Runner（Slice 1 §八）。

Optimizer 資料**不得**與正式文件 Registry（``storage/metadata.db`` 的 ``documents``
表）混在同一職責。本模組為獨立、可注入路徑的 SQLite 儲存，Slice 1 只建立最小基礎：

- ``schema_migrations``：版本化 migration 追蹤。
- ``evaluation_cases``：v1.1 evaluation cases。

特性（§八）：
- Repository 接受 DB path 注入（單元測試用 ``tmp_path``）。
- Migration 可重複執行（idempotent）。
- Migration 失敗自動 rollback（不留半套 schema）。
- JSON Dataset ↔ SQLite import / export，且語意等價。
- 不引入 Alembic：使用小型、版本化 migration runner。

**不得**把 v1.1 tables 加入正式 ``metadata.db``；本 repo 一律使用獨立 DB 檔。
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from evaluation.v11.case_schema import (
    SCHEMA_VERSION,
    EvaluationCaseV11,
    EvaluationDatasetV11,
    parse_dataset,
)


class MigrationError(Exception):
    """Migration 執行失敗（已 rollback）。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# 版本化 migration 定義（新增規則往後追加，不改既有）
# --------------------------------------------------------------------------
_MIGRATION_0001 = [
    """
    CREATE TABLE IF NOT EXISTS evaluation_cases (
        case_id                TEXT PRIMARY KEY,
        schema_version         TEXT NOT NULL,
        dataset_version        TEXT NOT NULL,
        question               TEXT NOT NULL,
        query_type             TEXT NOT NULL,
        answerable             INTEGER NOT NULL,
        group_id               TEXT NOT NULL,
        split                  TEXT NOT NULL,
        expected_evidence_json TEXT NOT NULL,
        status                 TEXT NOT NULL,
        created_at             TEXT NOT NULL,
        updated_at             TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_eval_cases_dataset ON evaluation_cases(dataset_version)",
    "CREATE INDEX IF NOT EXISTS idx_eval_cases_group ON evaluation_cases(group_id)",
]

# (version, [statements])
MIGRATIONS: list[tuple[str, list[str]]] = [
    ("0001_initial", _MIGRATION_0001),
]

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""


def run_migrations(
    conn: sqlite3.Connection,
    migrations: list[tuple[str, list[str]]] | None = None,
) -> list[str]:
    """套用尚未執行的 migration；idempotent，失敗即 rollback。回傳本次新套用版本。"""
    migrations = MIGRATIONS if migrations is None else migrations

    # schema_migrations 本身以 IF NOT EXISTS 建立（idempotent）。
    conn.execute(_SCHEMA_MIGRATIONS_DDL)
    conn.commit()

    applied = {
        row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }

    newly_applied: list[str] = []
    for version, statements in migrations:
        if version in applied:
            continue
        try:
            conn.execute("BEGIN")
            for stmt in statements:
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, _now()),
            )
            conn.execute("COMMIT")
        except Exception as exc:  # noqa: BLE001 - 任一 migration 失敗即整批回滾
            conn.execute("ROLLBACK")
            raise MigrationError(
                f"migration {version!r} 失敗，已 rollback：{exc}"
            ) from exc
        newly_applied.append(version)

    return newly_applied


class OptimizerRepository:
    """獨立於正式 registry 的 Optimizer SQLite 儲存。"""

    def __init__(self, db_path: str, migrations: list[tuple[str, list[str]]] | None = None):
        self.db_path = db_path
        self._migrations = migrations
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None：自行以 BEGIN/COMMIT/ROLLBACK 控制交易，確保 DDL 也能回滾。
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            run_migrations(conn, self._migrations)
        finally:
            conn.close()

    def applied_migrations(self) -> list[str]:
        conn = self._connect()
        try:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
            ]
        finally:
            conn.close()

    # --- 匯入 / 匯出 ---
    def import_dataset(self, dataset: EvaluationDatasetV11) -> int:
        """匯入一份 v1.1 dataset（upsert by case_id）。回傳寫入案例數。"""
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            for case in dataset.cases:
                self._upsert_case(conn, dataset.dataset_version, case)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return len(dataset.cases)

    def _upsert_case(
        self, conn: sqlite3.Connection, dataset_version: str, case: EvaluationCaseV11
    ) -> None:
        ts = _now()
        evidence_json = json.dumps(
            [e.to_dict() for e in case.expected_evidence], ensure_ascii=False, sort_keys=True
        )
        conn.execute(
            """
            INSERT INTO evaluation_cases
                (case_id, schema_version, dataset_version, question, query_type,
                 answerable, group_id, split, expected_evidence_json, status,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                schema_version=excluded.schema_version,
                dataset_version=excluded.dataset_version,
                question=excluded.question,
                query_type=excluded.query_type,
                answerable=excluded.answerable,
                group_id=excluded.group_id,
                split=excluded.split,
                expected_evidence_json=excluded.expected_evidence_json,
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (
                case.id,
                case.schema_version,
                dataset_version,
                case.question,
                case.query_type,
                1 if case.answerable else 0,
                case.group_id,
                case.split,
                evidence_json,
                case.status,
                ts,
                ts,
            ),
        )

    def export_dataset(self, dataset_version: str) -> EvaluationDatasetV11:
        """由 SQLite 匯出為合法 v1.1 dataset（與 import 語意等價）。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM evaluation_cases WHERE dataset_version=? ORDER BY case_id",
                (dataset_version,),
            ).fetchall()
        finally:
            conn.close()

        raw = {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": dataset_version,
            "cases": [self._row_to_case_dict(r) for r in rows],
        }
        # 透過 parse_dataset 再驗證一次，確保匯出即合法。
        return parse_dataset(raw)

    @staticmethod
    def _row_to_case_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["case_id"],
            "schema_version": row["schema_version"],
            "dataset_version": row["dataset_version"],
            "question": row["question"],
            "query_type": row["query_type"],
            "answerable": bool(row["answerable"]),
            "group_id": row["group_id"],
            "split": row["split"],
            "status": row["status"],
            "expected_evidence": json.loads(row["expected_evidence_json"]),
        }

    def list_case_ids(self) -> list[str]:
        conn = self._connect()
        try:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT case_id FROM evaluation_cases ORDER BY case_id"
                ).fetchall()
            ]
        finally:
            conn.close()
