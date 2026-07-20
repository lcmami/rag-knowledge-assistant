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
from evaluation.v11.run_models import (
    CleanupStatus,
    EvaluationRun,
    RetrievalTrial,
    TrialCaseResult,
    TrialMetric,
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

# --------------------------------------------------------------------------
# Slice 2（§七）：Run / Trial / Metric / Case Result 執行模型表。
# 均在**獨立 Optimizer DB**，永不加入正式 metadata.db。
# --------------------------------------------------------------------------
_MIGRATION_0002 = [
    """
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        run_id                            TEXT PRIMARY KEY,
        status                            TEXT NOT NULL,
        run_name                          TEXT NOT NULL,
        dataset_version                   TEXT,
        dataset_hash                      TEXT,
        corpus_hash                       TEXT,
        code_commit                       TEXT,
        working_tree_dirty                INTEGER,
        canonical_text_version            TEXT,
        loader_version                    TEXT,
        identifier_normalization_version  TEXT,
        random_seed                       INTEGER NOT NULL,
        config_json                       TEXT,
        created_at                        TEXT,
        started_at                        TEXT,
        completed_at                      TEXT,
        error                             TEXT,
        report_json_path                  TEXT,
        report_markdown_path              TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval_trials (
        trial_id             TEXT PRIMARY KEY,
        run_id               TEXT NOT NULL,
        ordinal              INTEGER NOT NULL,
        trial_name           TEXT NOT NULL,
        status               TEXT NOT NULL,
        config_json          TEXT NOT NULL,
        config_hash          TEXT NOT NULL,
        collection_name      TEXT,
        started_at           TEXT,
        completed_at         TEXT,
        index_build_time_ms  INTEGER,
        retrieval_time_ms    INTEGER,
        point_count          INTEGER,
        cleanup_status       TEXT NOT NULL,
        error                TEXT,
        attempt_count        INTEGER NOT NULL DEFAULT 0,
        UNIQUE(run_id, ordinal),
        UNIQUE(run_id, config_hash),
        FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trials_run ON retrieval_trials(run_id)",
    """
    CREATE TABLE IF NOT EXISTS trial_metrics (
        trial_id               TEXT NOT NULL,
        metric_name            TEXT NOT NULL,
        scope                  TEXT NOT NULL,
        query_type             TEXT NOT NULL DEFAULT '',
        value                  REAL,
        numerator              REAL,
        denominator            REAL,
        sample_count           INTEGER,
        computable             INTEGER NOT NULL,
        not_computable_reason  TEXT,
        PRIMARY KEY(trial_id, scope, query_type, metric_name),
        FOREIGN KEY(trial_id) REFERENCES retrieval_trials(trial_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trial_case_results (
        trial_id             TEXT NOT NULL,
        case_id              TEXT NOT NULL,
        case_status          TEXT NOT NULL,
        query_type           TEXT,
        answerable           INTEGER NOT NULL,
        latency_ms           INTEGER,
        first_relevant_rank  INTEGER,
        top_results_json     TEXT NOT NULL,
        evidence_match_json  TEXT NOT NULL,
        error                TEXT,
        PRIMARY KEY(trial_id, case_id),
        FOREIGN KEY(trial_id) REFERENCES retrieval_trials(trial_id) ON DELETE CASCADE
    )
    """,
]

# --------------------------------------------------------------------------
# Slice 3（§十八 / §二十三）：Chunking + Embedding Cache 統計欄位（additive）。
# 以獨立 migration 追加欄位；舊 DB 只跑一次（migration runner 依版本 idempotent）。
# 均在**獨立 Optimizer DB**，永不加入正式 metadata.db。
# --------------------------------------------------------------------------
_MIGRATION_0003 = [
    "ALTER TABLE retrieval_trials ADD COLUMN strategy TEXT",
    "ALTER TABLE retrieval_trials ADD COLUMN chunk_config_hash TEXT",
    "ALTER TABLE retrieval_trials ADD COLUMN chunk_count INTEGER",
    "ALTER TABLE retrieval_trials ADD COLUMN chunk_cache_stats_json TEXT",
]

# --------------------------------------------------------------------------
# Slice 4（§十八.4 / §十九）：identifier-aware 每題 retrieval 明細（additive）。
# 純追加欄位；舊 case result row 缺此欄位以 NULL 退化。獨立 Optimizer DB，不動正式 metadata.db。
# --------------------------------------------------------------------------
_MIGRATION_0004 = [
    "ALTER TABLE trial_case_results ADD COLUMN retrieval_meta_json TEXT",
]

# --------------------------------------------------------------------------
# Slice 5（§十一）：Optimizer Search / Candidate 持久化。
#
# 兩張新表，與 Slice 2 的 evaluation_runs / retrieval_trials **並存**：一次 search 會
# 為每個 split（tune / validation / test）各開一個 evaluation_run，candidate 則以
# config_hash 對應到各 run 內的 trial。這樣可完整重用既有 runner 的 resume / retry /
# fail-isolation / cleanup，而不需修改 Slice 2–4 任何程式。
#
# 關鍵不變式：
# - UNIQUE(search_id, ordinal)：candidate 序號唯一，resume 不產生重複 candidate。
# - UNIQUE(search_id, config_hash)：可執行 candidate 的設定唯一（非可執行者 hash 存 NULL，
#   SQLite 的 UNIQUE 允許多個 NULL，故 invalid / duplicate / not_sampled 可並存）。
# - test_trial_id 只有在 recommendation_locked=1 之後才允許寫入（§12.4，由
#   SearchRepository 於寫入前檢查）。
#
# 均在**獨立 Optimizer DB**，永不加入正式 metadata.db。
# --------------------------------------------------------------------------
_MIGRATION_0005 = [
    """
    CREATE TABLE IF NOT EXISTS optimizer_searches (
        search_id                 TEXT PRIMARY KEY,
        search_name               TEXT NOT NULL,
        status                    TEXT NOT NULL,
        search_mode               TEXT NOT NULL,
        search_config_json        TEXT NOT NULL,
        search_config_hash        TEXT NOT NULL,
        dataset_version           TEXT,
        dataset_hash              TEXT,
        corpus_hash               TEXT,
        code_commit               TEXT,
        working_tree_dirty        INTEGER,
        random_seed               INTEGER NOT NULL DEFAULT 42,
        max_candidates            INTEGER NOT NULL DEFAULT 32,
        validation_top_n          INTEGER NOT NULL DEFAULT 8,
        baseline_config_hash      TEXT,
        recommended_config_hash   TEXT,
        recommendation_locked     INTEGER NOT NULL DEFAULT 0,
        recommendation_locked_at  TEXT,
        test_evaluated            INTEGER NOT NULL DEFAULT 0,
        created_at                TEXT,
        started_at                TEXT,
        completed_at              TEXT,
        error                     TEXT,
        report_json_path          TEXT,
        report_markdown_path      TEXT,
        tune_run_id               TEXT,
        validation_run_id         TEXT,
        test_run_id               TEXT,
        generation_stats_json     TEXT,
        candidate_manifest_json   TEXT,
        split_manifest_json       TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS optimizer_candidates (
        search_id                    TEXT NOT NULL,
        candidate_id                 TEXT PRIMARY KEY,
        ordinal                      INTEGER NOT NULL,
        config_json                  TEXT NOT NULL,
        config_hash                  TEXT,
        generation_status            TEXT NOT NULL,
        generation_rejection_reason  TEXT,
        duplicate_of_hash            TEXT,
        is_baseline                  INTEGER NOT NULL DEFAULT 0,
        tune_trial_id                TEXT,
        validation_trial_id          TEXT,
        test_trial_id                TEXT,
        tune_status                  TEXT NOT NULL,
        validation_status            TEXT NOT NULL,
        test_status                  TEXT NOT NULL,
        tune_rank                    INTEGER,
        constraint_results_json      TEXT,
        feasible                     INTEGER,
        pareto_member                INTEGER,
        pareto_rank                  INTEGER,
        recommendation_rank          INTEGER,
        rejection_reasons_json       TEXT,
        UNIQUE(search_id, ordinal),
        UNIQUE(search_id, config_hash),
        FOREIGN KEY(search_id) REFERENCES optimizer_searches(search_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_candidates_search ON optimizer_candidates(search_id)",
]

# (version, [statements])
MIGRATIONS: list[tuple[str, list[str]]] = [
    ("0001_initial", _MIGRATION_0001),
    ("0002_runs_trials", _MIGRATION_0002),
    ("0003_chunk_cache_stats", _MIGRATION_0003),
    ("0004_retrieval_meta", _MIGRATION_0004),
    ("0005_optimizer_search", _MIGRATION_0005),
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
        # §七 4：開啟 Foreign Key（SQLite 預設關閉，需每連線設定）。
        conn.execute("PRAGMA foreign_keys=ON")
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

    # ----------------------------------------------------------------------
    # Slice 2：Run CRUD
    # ----------------------------------------------------------------------
    @staticmethod
    def _bool_to_int(value: bool | None) -> int | None:
        return None if value is None else (1 if value else 0)

    @staticmethod
    def _int_to_bool(value: int | None) -> bool | None:
        return None if value is None else bool(value)

    _RUN_COLUMNS = (
        "run_id", "status", "run_name", "dataset_version", "dataset_hash",
        "corpus_hash", "code_commit", "working_tree_dirty", "canonical_text_version",
        "loader_version", "identifier_normalization_version", "random_seed",
        "config_json", "created_at", "started_at", "completed_at", "error",
        "report_json_path", "report_markdown_path",
    )

    def upsert_run(self, run: EvaluationRun) -> None:
        """建立或更新一個 Run（依 run_id upsert；重複 run_id 不產生重複 Run）。"""
        values = (
            run.run_id, run.status, run.run_name, run.dataset_version, run.dataset_hash,
            run.corpus_hash, run.code_commit, self._bool_to_int(run.working_tree_dirty),
            run.canonical_text_version, run.loader_version,
            run.identifier_normalization_version, run.random_seed, run.config_json,
            run.created_at, run.started_at, run.completed_at, run.error,
            run.report_json_path, run.report_markdown_path,
        )
        placeholders = ",".join("?" for _ in self._RUN_COLUMNS)
        updates = ",".join(f"{c}=excluded.{c}" for c in self._RUN_COLUMNS if c != "run_id")
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT INTO evaluation_runs ({','.join(self._RUN_COLUMNS)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(run_id) DO UPDATE SET {updates}",
                values,
            )
        finally:
            conn.close()

    def get_run(self, run_id: str) -> EvaluationRun | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_run(row) if row else None

    def _row_to_run(self, row: sqlite3.Row) -> EvaluationRun:
        return EvaluationRun(
            run_id=row["run_id"],
            status=row["status"],
            run_name=row["run_name"],
            dataset_version=row["dataset_version"],
            dataset_hash=row["dataset_hash"],
            corpus_hash=row["corpus_hash"],
            code_commit=row["code_commit"],
            working_tree_dirty=self._int_to_bool(row["working_tree_dirty"]),
            canonical_text_version=row["canonical_text_version"],
            loader_version=row["loader_version"],
            identifier_normalization_version=row["identifier_normalization_version"],
            random_seed=row["random_seed"],
            config_json=row["config_json"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            report_json_path=row["report_json_path"],
            report_markdown_path=row["report_markdown_path"],
        )

    def delete_run(self, run_id: str) -> None:
        """刪除 Run；FK ON DELETE CASCADE 會連帶刪 trials / metrics / case results。"""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM evaluation_runs WHERE run_id=?", (run_id,))
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Trial CRUD
    # ----------------------------------------------------------------------
    _TRIAL_COLUMNS = (
        "trial_id", "run_id", "ordinal", "trial_name", "status", "config_json",
        "config_hash", "collection_name", "started_at", "completed_at",
        "index_build_time_ms", "retrieval_time_ms", "point_count", "cleanup_status",
        "error", "attempt_count",
        "strategy", "chunk_config_hash", "chunk_count", "chunk_cache_stats_json",
    )

    def upsert_trial(self, trial: RetrievalTrial) -> None:
        """建立或更新一個 Trial（依 trial_id upsert）。"""
        values = (
            trial.trial_id, trial.run_id, trial.ordinal, trial.trial_name, trial.status,
            trial.config_json, trial.config_hash, trial.collection_name,
            trial.started_at, trial.completed_at, trial.index_build_time_ms,
            trial.retrieval_time_ms, trial.point_count, trial.cleanup_status,
            trial.error, trial.attempt_count,
            trial.strategy, trial.chunk_config_hash, trial.chunk_count,
            trial.chunk_cache_stats_json,
        )
        placeholders = ",".join("?" for _ in self._TRIAL_COLUMNS)
        updates = ",".join(f"{c}=excluded.{c}" for c in self._TRIAL_COLUMNS if c != "trial_id")
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT INTO retrieval_trials ({','.join(self._TRIAL_COLUMNS)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(trial_id) DO UPDATE SET {updates}",
                values,
            )
        finally:
            conn.close()

    def _row_to_trial(self, row: sqlite3.Row) -> RetrievalTrial:
        return RetrievalTrial(
            trial_id=row["trial_id"],
            run_id=row["run_id"],
            ordinal=row["ordinal"],
            trial_name=row["trial_name"],
            status=row["status"],
            config_json=row["config_json"],
            config_hash=row["config_hash"],
            collection_name=row["collection_name"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            index_build_time_ms=row["index_build_time_ms"],
            retrieval_time_ms=row["retrieval_time_ms"],
            point_count=row["point_count"],
            cleanup_status=row["cleanup_status"],
            error=row["error"],
            attempt_count=row["attempt_count"],
            strategy=self._row_get(row, "strategy"),
            chunk_config_hash=self._row_get(row, "chunk_config_hash"),
            chunk_count=self._row_get(row, "chunk_count"),
            chunk_cache_stats_json=self._row_get(row, "chunk_cache_stats_json"),
        )

    @staticmethod
    def _row_get(row: sqlite3.Row, key: str):
        """安全讀取欄位（舊 row 缺 Slice 3 欄位時回 None，向後相容）。"""
        try:
            return row[key]
        except (IndexError, KeyError):
            return None

    def get_trial(self, trial_id: str) -> RetrievalTrial | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM retrieval_trials WHERE trial_id=?", (trial_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_trial(row) if row else None

    def get_trial_by_ordinal(self, run_id: str, ordinal: int) -> RetrievalTrial | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM retrieval_trials WHERE run_id=? AND ordinal=?",
                (run_id, ordinal),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_trial(row) if row else None

    def get_trial_by_config_hash(self, run_id: str, config_hash: str) -> RetrievalTrial | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM retrieval_trials WHERE run_id=? AND config_hash=?",
                (run_id, config_hash),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_trial(row) if row else None

    def list_trials(self, run_id: str) -> list[RetrievalTrial]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM retrieval_trials WHERE run_id=? ORDER BY ordinal",
                (run_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_trial(r) for r in rows]

    def list_running_trials(self) -> list[RetrievalTrial]:
        """所有標記為 running 的 Trial（供 resume 前偵測遺留中斷）。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM retrieval_trials WHERE status='running' ORDER BY run_id, ordinal"
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_trial(r) for r in rows]

    # ----------------------------------------------------------------------
    # Trial Metrics / Case Results（delete-then-insert，idempotent per trial）
    # ----------------------------------------------------------------------
    def save_trial_metrics(self, trial_id: str, metrics: list[TrialMetric]) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM trial_metrics WHERE trial_id=?", (trial_id,))
            for m in metrics:
                conn.execute(
                    """
                    INSERT INTO trial_metrics
                        (trial_id, metric_name, scope, query_type, value, numerator,
                         denominator, sample_count, computable, not_computable_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trial_id, m.metric_name, m.scope, m.query_type or "",
                        m.value, m.numerator, m.denominator, m.sample_count,
                        1 if m.computable else 0, m.not_computable_reason,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def get_trial_metrics(self, trial_id: str) -> list[TrialMetric]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM trial_metrics WHERE trial_id=? ORDER BY scope, query_type, metric_name",
                (trial_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            TrialMetric(
                trial_id=r["trial_id"],
                metric_name=r["metric_name"],
                scope=r["scope"],
                query_type=(r["query_type"] or None),
                value=r["value"],
                numerator=r["numerator"],
                denominator=r["denominator"],
                sample_count=r["sample_count"],
                computable=bool(r["computable"]),
                not_computable_reason=r["not_computable_reason"],
            )
            for r in rows
        ]

    def save_trial_case_results(
        self, trial_id: str, results: list[TrialCaseResult]
    ) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM trial_case_results WHERE trial_id=?", (trial_id,))
            for c in results:
                conn.execute(
                    """
                    INSERT INTO trial_case_results
                        (trial_id, case_id, case_status, query_type, answerable,
                         latency_ms, first_relevant_rank, top_results_json,
                         evidence_match_json, error, retrieval_meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trial_id, c.case_id, c.case_status, c.query_type,
                        1 if c.answerable else 0, c.latency_ms, c.first_relevant_rank,
                        c.top_results_json, c.evidence_match_json, c.error,
                        c.retrieval_meta_json,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def get_trial_case_results(self, trial_id: str) -> list[TrialCaseResult]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM trial_case_results WHERE trial_id=? ORDER BY case_id",
                (trial_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            TrialCaseResult(
                trial_id=r["trial_id"],
                case_id=r["case_id"],
                case_status=r["case_status"],
                query_type=r["query_type"],
                answerable=bool(r["answerable"]),
                latency_ms=r["latency_ms"],
                first_relevant_rank=r["first_relevant_rank"],
                top_results_json=r["top_results_json"],
                evidence_match_json=r["evidence_match_json"],
                error=r["error"],
                retrieval_meta_json=(
                    r["retrieval_meta_json"] if "retrieval_meta_json" in r.keys() else None
                ),
            )
            for r in rows
        ]
