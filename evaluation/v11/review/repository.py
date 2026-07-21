"""Review SQLite Repository（Slice 6 §九）。

沿用 :func:`evaluation.v11.optimizer_repository.run_migrations` 的小型 migration
runner（idempotent、失敗 rollback），但使用**獨立、可注入的 Review DB 路徑**。

規則（§九）：
- Migration idempotent；失敗即 rollback，不留半套 schema。
- 每連線開啟 Foreign Key（SQLite 預設關閉）。
- Session 刪除時，其 Candidate / Revision / Event 一併 CASCADE 刪除
  （Candidate Revision 不得孤立）。
- **不產生 ``documents`` 表**，也絕不修改正式 ``storage/metadata.db``。
- JSON 欄位 round-trip 語意等價。
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from evaluation.v11.optimizer_repository import run_migrations
from evaluation.v11.review.models import (
    CANDIDATE_PUBLISHED,
    CandidateRevision,
    QuestionCandidate,
    ReviewEvent,
    ReviewSession,
    ReviewStateError,
    new_id,
    now_iso,
)

# --------------------------------------------------------------------------
# Migration 0007：Review Workflow 最小表集
# --------------------------------------------------------------------------
_MIGRATION_0007 = [
    """
    CREATE TABLE IF NOT EXISTS review_sessions (
        session_id                TEXT PRIMARY KEY,
        session_name              TEXT NOT NULL,
        status                    TEXT NOT NULL,
        base_dataset_path         TEXT NOT NULL,
        base_dataset_version      TEXT NOT NULL,
        base_dataset_hash         TEXT NOT NULL,
        corpus_hash               TEXT NOT NULL,
        canonical_text_version    TEXT,
        loader_version            TEXT,
        generator_type            TEXT,
        generator_model           TEXT,
        generator_config_json     TEXT NOT NULL DEFAULT '{}',
        created_at                TEXT NOT NULL,
        updated_at                TEXT NOT NULL,
        completed_at              TEXT,
        error                     TEXT,
        published_dataset_path    TEXT,
        published_dataset_version TEXT,
        published_dataset_hash    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS question_candidates (
        candidate_id             TEXT PRIMARY KEY,
        session_id               TEXT NOT NULL,
        proposed_question        TEXT NOT NULL,
        proposed_query_type      TEXT NOT NULL,
        proposed_answerable      INTEGER NOT NULL,
        status                   TEXT NOT NULL,
        current_revision         INTEGER NOT NULL DEFAULT 0,
        case_id                  TEXT,
        source_document_sha256   TEXT,
        source_filename          TEXT,
        source_page              INTEGER,
        source_section           TEXT,
        source_segment_ids_json  TEXT NOT NULL DEFAULT '[]',
        suggested_group_id       TEXT,
        suggested_split          TEXT,
        suggested_anchor_text_json TEXT NOT NULL DEFAULT '[]',
        generator_metadata_json  TEXT NOT NULL DEFAULT '{}',
        unanswerable_confirmed   INTEGER NOT NULL DEFAULT 0,
        created_at               TEXT NOT NULL,
        updated_at               TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES review_sessions(session_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_candidates_session ON question_candidates(session_id)",
    """
    CREATE TABLE IF NOT EXISTS candidate_revisions (
        candidate_id     TEXT NOT NULL,
        revision_number  INTEGER NOT NULL,
        question         TEXT NOT NULL,
        query_type       TEXT NOT NULL,
        answerable       INTEGER NOT NULL,
        group_id         TEXT,
        split            TEXT,
        evidence_json    TEXT NOT NULL DEFAULT '[]',
        anchor_text_json TEXT NOT NULL DEFAULT '[]',
        editor           TEXT NOT NULL,
        change_reason    TEXT,
        created_at       TEXT NOT NULL,
        PRIMARY KEY(candidate_id, revision_number),
        FOREIGN KEY(candidate_id) REFERENCES question_candidates(candidate_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_events (
        event_id        TEXT PRIMARY KEY,
        session_id      TEXT NOT NULL,
        event_type      TEXT NOT NULL,
        candidate_id    TEXT,
        revision_number INTEGER,
        actor           TEXT NOT NULL,
        detail_json     TEXT NOT NULL DEFAULT '{}',
        created_at      TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES review_sessions(session_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_session ON review_events(session_id)",
]

REVIEW_MIGRATIONS: list[tuple[str, list[str]]] = [
    ("0007_review_workflow", _MIGRATION_0007),
]


def _to_bool(value: Any) -> bool:
    return bool(value)


class ReviewRepository:
    """Review Workflow 的獨立 SQLite 儲存（路徑可注入；測試用 ``tmp_path``）。"""

    def __init__(self, db_path: str, migrations: list[tuple[str, list[str]]] | None = None):
        self.db_path = db_path
        self._migrations = REVIEW_MIGRATIONS if migrations is None else migrations
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
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

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def create_session(self, session: ReviewSession) -> ReviewSession:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO review_sessions
                    (session_id, session_name, status, base_dataset_path,
                     base_dataset_version, base_dataset_hash, corpus_hash,
                     canonical_text_version, loader_version, generator_type,
                     generator_model, generator_config_json, created_at, updated_at,
                     completed_at, error, published_dataset_path,
                     published_dataset_version, published_dataset_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session.session_id, session.session_name, session.status,
                    session.base_dataset_path, session.base_dataset_version,
                    session.base_dataset_hash, session.corpus_hash,
                    session.canonical_text_version, session.loader_version,
                    session.generator_type, session.generator_model,
                    session.generator_config_json, session.created_at,
                    session.updated_at, session.completed_at, session.error,
                    session.published_dataset_path, session.published_dataset_version,
                    session.published_dataset_hash,
                ),
            )
        finally:
            conn.close()
        return session

    def get_session(self, session_id: str) -> ReviewSession | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_session(row) if row else None

    def require_session(self, session_id: str) -> ReviewSession:
        session = self.get_session(session_id)
        if session is None:
            raise ReviewStateError(f"找不到 session：{session_id!r}")
        return session

    def list_sessions(self) -> list[ReviewSession]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM review_sessions ORDER BY created_at, session_id"
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_session(r) for r in rows]

    def update_session(self, session: ReviewSession) -> ReviewSession:
        session.updated_at = now_iso()
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE review_sessions SET
                    session_name=?, status=?, canonical_text_version=?, loader_version=?,
                    generator_type=?, generator_model=?, generator_config_json=?,
                    updated_at=?, completed_at=?, error=?, published_dataset_path=?,
                    published_dataset_version=?, published_dataset_hash=?
                WHERE session_id=?
                """,
                (
                    session.session_name, session.status, session.canonical_text_version,
                    session.loader_version, session.generator_type, session.generator_model,
                    session.generator_config_json, session.updated_at, session.completed_at,
                    session.error, session.published_dataset_path,
                    session.published_dataset_version, session.published_dataset_hash,
                    session.session_id,
                ),
            )
        finally:
            conn.close()
        return session

    def delete_session(self, session_id: str) -> None:
        """刪除 Session；其 Candidate / Revision / Event 一併 CASCADE 刪除。"""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM review_sessions WHERE session_id=?", (session_id,))
        finally:
            conn.close()

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> ReviewSession:
        return ReviewSession(
            session_id=row["session_id"],
            session_name=row["session_name"],
            status=row["status"],
            base_dataset_path=row["base_dataset_path"],
            base_dataset_version=row["base_dataset_version"],
            base_dataset_hash=row["base_dataset_hash"],
            corpus_hash=row["corpus_hash"],
            canonical_text_version=row["canonical_text_version"],
            loader_version=row["loader_version"],
            generator_type=row["generator_type"],
            generator_model=row["generator_model"],
            generator_config_json=row["generator_config_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            published_dataset_path=row["published_dataset_path"],
            published_dataset_version=row["published_dataset_version"],
            published_dataset_hash=row["published_dataset_hash"],
        )

    # ------------------------------------------------------------------
    # Candidate
    # ------------------------------------------------------------------
    def create_candidate(self, candidate: QuestionCandidate) -> QuestionCandidate:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO question_candidates
                    (candidate_id, session_id, proposed_question, proposed_query_type,
                     proposed_answerable, status, current_revision, case_id,
                     source_document_sha256, source_filename, source_page, source_section,
                     source_segment_ids_json, suggested_group_id, suggested_split,
                     suggested_anchor_text_json, generator_metadata_json,
                     unanswerable_confirmed, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    candidate.candidate_id, candidate.session_id,
                    candidate.proposed_question, candidate.proposed_query_type,
                    int(candidate.proposed_answerable), candidate.status,
                    candidate.current_revision, candidate.case_id,
                    candidate.source_document_sha256,
                    candidate.source_filename, candidate.source_page,
                    candidate.source_section, candidate.source_segment_ids_json,
                    candidate.suggested_group_id, candidate.suggested_split,
                    candidate.suggested_anchor_text_json,
                    candidate.generator_metadata_json,
                    int(candidate.unanswerable_confirmed),
                    candidate.created_at, candidate.updated_at,
                ),
            )
        finally:
            conn.close()
        return candidate

    def get_candidate(self, candidate_id: str) -> QuestionCandidate | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM question_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_candidate(row) if row else None

    def require_candidate(self, candidate_id: str) -> QuestionCandidate:
        candidate = self.get_candidate(candidate_id)
        if candidate is None:
            raise ReviewStateError(f"找不到 candidate：{candidate_id!r}")
        return candidate

    def list_candidates(
        self, session_id: str, status: str | None = None
    ) -> list[QuestionCandidate]:
        conn = self._connect()
        try:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM question_candidates WHERE session_id=? "
                    "ORDER BY created_at, candidate_id",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM question_candidates WHERE session_id=? AND status=? "
                    "ORDER BY created_at, candidate_id",
                    (session_id, status),
                ).fetchall()
        finally:
            conn.close()
        return [self._row_to_candidate(r) for r in rows]

    def update_candidate(self, candidate: QuestionCandidate) -> QuestionCandidate:
        candidate.updated_at = now_iso()
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE question_candidates SET
                    proposed_question=?, proposed_query_type=?, proposed_answerable=?,
                    status=?, current_revision=?, case_id=?, suggested_group_id=?, suggested_split=?,
                    suggested_anchor_text_json=?, unanswerable_confirmed=?, updated_at=?
                WHERE candidate_id=?
                """,
                (
                    candidate.proposed_question, candidate.proposed_query_type,
                    int(candidate.proposed_answerable), candidate.status,
                    candidate.current_revision, candidate.case_id, candidate.suggested_group_id,
                    candidate.suggested_split, candidate.suggested_anchor_text_json,
                    int(candidate.unanswerable_confirmed), candidate.updated_at,
                    candidate.candidate_id,
                ),
            )
        finally:
            conn.close()
        return candidate

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> QuestionCandidate:
        return QuestionCandidate(
            candidate_id=row["candidate_id"],
            session_id=row["session_id"],
            proposed_question=row["proposed_question"],
            proposed_query_type=row["proposed_query_type"],
            proposed_answerable=_to_bool(row["proposed_answerable"]),
            status=row["status"],
            current_revision=row["current_revision"],
            case_id=row["case_id"],
            source_document_sha256=row["source_document_sha256"],
            source_filename=row["source_filename"],
            source_page=row["source_page"],
            source_section=row["source_section"],
            source_segment_ids_json=row["source_segment_ids_json"],
            suggested_group_id=row["suggested_group_id"],
            suggested_split=row["suggested_split"],
            suggested_anchor_text_json=row["suggested_anchor_text_json"],
            generator_metadata_json=row["generator_metadata_json"],
            unanswerable_confirmed=_to_bool(row["unanswerable_confirmed"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Revision
    # ------------------------------------------------------------------
    def add_revision(self, revision: CandidateRevision) -> CandidateRevision:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO candidate_revisions
                    (candidate_id, revision_number, question, query_type, answerable,
                     group_id, split, evidence_json, anchor_text_json, editor,
                     change_reason, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    revision.candidate_id, revision.revision_number, revision.question,
                    revision.query_type, int(revision.answerable), revision.group_id,
                    revision.split, revision.evidence_json, revision.anchor_text_json,
                    revision.editor, revision.change_reason, revision.created_at,
                ),
            )
        finally:
            conn.close()
        return revision

    def list_revisions(self, candidate_id: str) -> list[CandidateRevision]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM candidate_revisions WHERE candidate_id=? "
                "ORDER BY revision_number",
                (candidate_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            CandidateRevision(
                candidate_id=r["candidate_id"],
                revision_number=r["revision_number"],
                question=r["question"],
                query_type=r["query_type"],
                answerable=_to_bool(r["answerable"]),
                group_id=r["group_id"],
                split=r["split"],
                evidence_json=r["evidence_json"],
                anchor_text_json=r["anchor_text_json"],
                editor=r["editor"],
                change_reason=r["change_reason"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def latest_revision(self, candidate_id: str) -> CandidateRevision | None:
        revisions = self.list_revisions(candidate_id)
        return revisions[-1] if revisions else None

    def count_revisions(self, session_id: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM candidate_revisions r
                JOIN question_candidates c ON c.candidate_id = r.candidate_id
                WHERE c.session_id = ?
                """,
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0])

    # ------------------------------------------------------------------
    # Audit Event
    # ------------------------------------------------------------------
    def add_event(
        self,
        session_id: str,
        event_type: str,
        *,
        candidate_id: str | None = None,
        revision_number: int | None = None,
        actor: str,
        detail: dict[str, Any] | None = None,
    ) -> ReviewEvent:
        event = ReviewEvent(
            event_id=new_id("evt"),
            session_id=session_id,
            event_type=event_type,
            candidate_id=candidate_id,
            revision_number=revision_number,
            actor=actor,
            detail_json=json.dumps(detail or {}, ensure_ascii=False, sort_keys=True),
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO review_events
                    (event_id, session_id, event_type, candidate_id, revision_number,
                     actor, detail_json, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    event.event_id, event.session_id, event.event_type,
                    event.candidate_id, event.revision_number, event.actor,
                    event.detail_json, event.created_at,
                ),
            )
        finally:
            conn.close()
        return event

    def list_events(self, session_id: str) -> list[ReviewEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM review_events WHERE session_id=? ORDER BY created_at, event_id",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            ReviewEvent(
                event_id=r["event_id"],
                session_id=r["session_id"],
                event_type=r["event_type"],
                candidate_id=r["candidate_id"],
                revision_number=r["revision_number"],
                actor=r["actor"],
                detail_json=r["detail_json"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def mark_candidates_published(self, candidate_ids: list[str]) -> None:
        """發布後將 Candidate 轉為 ``published``（終態）。"""
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            for cid in candidate_ids:
                conn.execute(
                    "UPDATE question_candidates SET status=?, updated_at=? WHERE candidate_id=?",
                    (CANDIDATE_PUBLISHED, now_iso(), cid),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
