"""v1.2 Agent Run 最小 Audit（Slice 5A §八）。

**獨立** SQLite Agent DB，擁有**自己的** migration namespace（``0001_agent_runs``）——
不與 optimizer / search / review / deploy DB 共用 registry，也不建立跨 SQLite DB 的 FoK。
外部 trial id（來自 optimizer DB）只是文字參照。DB 內唯一 FK 為 ``agent_rounds →
agent_runs``。

安全（§六 / §八）：
- **不**保存 API Key / Authorization header / 完整 Prompt / 未裁切私人文件原文。
- 預設只存 response **hash** 與 parsed proposal；只有 debug 模式才存**去敏**後的 raw response。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from evaluation.v11.optimizer_repository import run_migrations

_RAW_RESPONSE_CAP = 4000

_MIGRATION_0001 = [
    """
    CREATE TABLE IF NOT EXISTS agent_runs (
        agent_run_id             TEXT PRIMARY KEY,
        run_name                 TEXT NOT NULL,
        status                   TEXT NOT NULL,
        stop_reason              TEXT,
        baseline_config_hash     TEXT NOT NULL,
        final_parent_config_hash TEXT NOT NULL,
        planner_type             TEXT,
        provider                 TEXT,
        model                    TEXT,
        prompt_version           TEXT,
        llm_calls                INTEGER NOT NULL DEFAULT 0,
        candidates_proposed      INTEGER NOT NULL DEFAULT 0,
        candidates_promoted      INTEGER NOT NULL DEFAULT 0,
        candidates_rejected      INTEGER NOT NULL DEFAULT 0,
        probe_count              INTEGER NOT NULL DEFAULT 0,
        recommendation_eligibility TEXT NOT NULL,
        baseline_metric_summary_json TEXT,
        final_metric_summary_json    TEXT,
        error                    TEXT,
        created_at               TEXT NOT NULL,
        updated_at               TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_rounds (
        agent_run_id          TEXT NOT NULL,
        round_number          INTEGER NOT NULL,
        parent_config_hash    TEXT,
        candidate_config_hash TEXT,
        parent_trial_id       TEXT,
        candidate_trial_id    TEXT,
        context_hash          TEXT,
        response_hash         TEXT,
        parsed_proposal_json  TEXT,
        validation_result_json TEXT,
        probe_comparison_json TEXT,
        executed_config_hash  TEXT,
        tune_trial_id         TEXT,
        validation_trial_id   TEXT,
        metrics_before_json   TEXT,
        metrics_after_json    TEXT,
        constraint_result_json TEXT,
        promotion_decision    TEXT,
        promotion_reason      TEXT,
        candidate_accepted    INTEGER,
        stop_decision         TEXT,
        provider              TEXT,
        model                 TEXT,
        token_usage_json      TEXT,
        raw_response_redacted TEXT,
        error                 TEXT,
        created_at            TEXT NOT NULL,
        PRIMARY KEY (agent_run_id, round_number),
        FOREIGN KEY (agent_run_id) REFERENCES agent_runs(agent_run_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_rounds_run ON agent_rounds(agent_run_id)",
]

AGENT_MIGRATIONS: list[tuple[str, list[str]]] = [
    ("0001_agent_runs", _MIGRATION_0001),
]

# 去敏樣式：API key / bearer token / Authorization header / sk- 前綴金鑰。
_SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*:\s*)(bearer\s+)?[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[A-Za-z0-9._\-]+"),
    re.compile(r"\bsk-[A-Za-z0-9]{8,}\b"),
]
_REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """移除疑似 API key / Authorization header（debug raw response 存檔前使用）。"""
    out = text
    out = _SECRET_PATTERNS[0].sub(lambda m: (m.group(1) or "") + _REDACTED, out)
    out = _SECRET_PATTERNS[1].sub(lambda m: (m.group(1) or "") + _REDACTED, out)
    out = _SECRET_PATTERNS[2].sub(_REDACTED, out)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Optional[str]) -> Any:
    return json.loads(value) if value else None


class AgentRepository:
    """獨立 Agent SQLite 儲存（own migration namespace）。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        try:
            run_migrations(conn, AGENT_MIGRATIONS)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

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

    # ------------------------------------------------------------------
    def write_run_summary(
        self,
        result,
        *,
        raw_responses: Optional[dict[int, str]] = None,
        debug_store_raw: bool = False,
    ) -> None:
        """落庫一次 Agent Run 的摘要與每輪 audit（idempotent：INSERT OR REPLACE）。

        ``raw_responses`` / ``debug_store_raw`` 只在 debug 模式使用：存檔前一律 ``redact_secrets``。
        預設（``debug_store_raw=False``）**不**保存任何 raw response。
        """
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_runs (
                    agent_run_id, run_name, status, stop_reason,
                    baseline_config_hash, final_parent_config_hash,
                    planner_type, provider, model, prompt_version,
                    llm_calls, candidates_proposed, candidates_promoted,
                    candidates_rejected, probe_count, recommendation_eligibility,
                    baseline_metric_summary_json, final_metric_summary_json,
                    error, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result.agent_run_id,
                    result.run_name,
                    result.status,
                    result.stop_reason,
                    result.baseline_config_hash,
                    result.final_parent_config_hash,
                    result.planner_type,
                    result.provider,
                    result.model,
                    result.prompt_version,
                    result.llm_calls,
                    result.candidates_proposed,
                    result.candidates_promoted,
                    result.candidates_rejected,
                    result.probe_count,
                    result.recommendation_eligibility,
                    _dumps(result.baseline_metric_summary),
                    _dumps(result.final_metric_summary),
                    result.error,
                    now,
                    now,
                ),
            )
            conn.execute("DELETE FROM agent_rounds WHERE agent_run_id = ?", (result.agent_run_id,))
            for rec in result.rounds:
                raw_redacted = None
                if debug_store_raw and raw_responses and rec.round_number in raw_responses:
                    raw_redacted = redact_secrets(raw_responses[rec.round_number])[:_RAW_RESPONSE_CAP]
                conn.execute(
                    """
                    INSERT INTO agent_rounds (
                        agent_run_id, round_number, parent_config_hash, candidate_config_hash,
                        parent_trial_id, candidate_trial_id, context_hash, response_hash,
                        parsed_proposal_json, validation_result_json, probe_comparison_json,
                        executed_config_hash, tune_trial_id, validation_trial_id,
                        metrics_before_json, metrics_after_json, constraint_result_json,
                        promotion_decision, promotion_reason, candidate_accepted, stop_decision,
                        provider, model, token_usage_json, raw_response_redacted, error, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        result.agent_run_id,
                        rec.round_number,
                        rec.parent_config_hash,
                        rec.candidate_config_hash,
                        rec.parent_trial_id,
                        rec.candidate_trial_id,
                        rec.context_hash,
                        rec.response_hash,
                        _dumps(rec.proposal),
                        _dumps(rec.validation_result),
                        _dumps(rec.probe_comparison),
                        rec.executed_config_hash,
                        rec.tune_trial_id,
                        rec.validation_trial_id,
                        _dumps(rec.metrics_before),
                        _dumps(rec.metrics_after),
                        _dumps(rec.constraint_result),
                        rec.promotion_decision,
                        rec.promotion_reason,
                        1 if rec.candidate_accepted else 0,
                        rec.stop_decision,
                        rec.provider,
                        rec.model,
                        _dumps(rec.token_usage),
                        raw_redacted,
                        rec.error,
                        now,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def get_run(self, agent_run_id: str) -> Optional[dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE agent_run_id = ?", (agent_run_id,)
            ).fetchone()
            if row is None:
                return None
            out = dict(row)
            out["baseline_metric_summary"] = _loads(out.pop("baseline_metric_summary_json"))
            out["final_metric_summary"] = _loads(out.pop("final_metric_summary_json"))
            return out
        finally:
            conn.close()

    def list_rounds(self, agent_run_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM agent_rounds WHERE agent_run_id = ? ORDER BY round_number",
                (agent_run_id,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                d["proposal"] = _loads(d.pop("parsed_proposal_json"))
                d["validation_result"] = _loads(d.pop("validation_result_json"))
                d["probe_comparison"] = _loads(d.pop("probe_comparison_json"))
                d["metrics_before"] = _loads(d.pop("metrics_before_json"))
                d["metrics_after"] = _loads(d.pop("metrics_after_json"))
                d["constraint_result"] = _loads(d.pop("constraint_result_json"))
                d["token_usage"] = _loads(d.pop("token_usage_json"))
                out.append(d)
            return out
        finally:
            conn.close()
