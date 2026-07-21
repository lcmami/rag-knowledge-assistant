"""Slice 6 §二十三 — Review Repository 測試。

Migration idempotent / rollback、Session / Candidate CRUD、Revision 連續、
Audit Event、Foreign Key、Published 不可修改、Cancelled 不可發布、JSON round-trip、
且**不產生 documents Table**。全部使用 ``tmp_path``。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from evaluation.v11.optimizer_repository import MigrationError
from evaluation.v11.review.models import (
    CANDIDATE_ACCEPTED,
    CANDIDATE_PROPOSED,
    CANDIDATE_PUBLISHED,
    SESSION_CANCELLED,
    SESSION_DRAFT,
    SESSION_PUBLISHED,
    CandidateRevision,
    QuestionCandidate,
    ReviewSession,
    ReviewStateError,
    new_id,
)
from evaluation.v11.review.repository import REVIEW_MIGRATIONS, ReviewRepository


def _repo(tmp_path) -> ReviewRepository:
    return ReviewRepository(str(tmp_path / "r.review.db"))


def _session(**kwargs) -> ReviewSession:
    defaults = dict(
        session_id=new_id("rs"),
        session_name="s",
        status=SESSION_DRAFT,
        base_dataset_path="base.json",
        base_dataset_version="v1",
        base_dataset_hash="h" * 8,
        corpus_hash="c" * 8,
    )
    defaults.update(kwargs)
    return ReviewSession(**defaults)


def _candidate(session_id: str, **kwargs) -> QuestionCandidate:
    defaults = dict(
        candidate_id=new_id("qc"),
        session_id=session_id,
        proposed_question="Q?",
        proposed_query_type="semantic",
        proposed_answerable=True,
    )
    defaults.update(kwargs)
    return QuestionCandidate(**defaults)


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------
def test_migration_creates_review_tables(tmp_path):
    repo = _repo(tmp_path)
    tables = repo.table_names()
    assert {
        "review_sessions",
        "question_candidates",
        "candidate_revisions",
        "review_events",
    } <= set(tables)


def test_migration_does_not_create_documents_table(tmp_path):
    """§九.8：Review DB 不得產生正式 registry 的 documents 表。"""
    repo = _repo(tmp_path)
    assert "documents" not in repo.table_names()


def test_migration_is_idempotent(tmp_path):
    path = str(tmp_path / "r.review.db")
    first = ReviewRepository(path).applied_migrations()
    second = ReviewRepository(path).applied_migrations()  # 重新開啟不得重複套用
    assert first == second == ["0007_review_workflow"]


def test_migration_failure_rolls_back(tmp_path):
    """失敗的 migration 不得留下半套 schema（§九.2）。"""
    broken = [
        (
            "0099_broken",
            [
                "CREATE TABLE IF NOT EXISTS half_created (x TEXT)",
                "THIS IS NOT VALID SQL",
            ],
        )
    ]
    with pytest.raises(MigrationError):
        ReviewRepository(str(tmp_path / "broken.db"), migrations=broken)

    conn = sqlite3.connect(str(tmp_path / "broken.db"))
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "half_created" not in names


# --------------------------------------------------------------------------
# Session / Candidate CRUD
# --------------------------------------------------------------------------
def test_session_crud_round_trip(tmp_path):
    repo = _repo(tmp_path)
    session = repo.create_session(_session(session_name="alpha"))
    loaded = repo.get_session(session.session_id)
    assert loaded.session_name == "alpha"
    assert loaded.status == SESSION_DRAFT

    loaded.status = SESSION_PUBLISHED
    repo.update_session(loaded)
    assert repo.get_session(session.session_id).status == SESSION_PUBLISHED
    assert [s.session_id for s in repo.list_sessions()] == [session.session_id]


def test_candidate_crud_and_json_round_trip(tmp_path):
    repo = _repo(tmp_path)
    session = repo.create_session(_session())
    segments = ["seg-1", "seg-2"]
    meta = {"generator": "fake", "model": "m"}
    candidate = repo.create_candidate(
        _candidate(
            session.session_id,
            source_segment_ids_json=json.dumps(segments, ensure_ascii=False),
            generator_metadata_json=json.dumps(meta, ensure_ascii=False, sort_keys=True),
        )
    )
    loaded = repo.get_candidate(candidate.candidate_id)
    # JSON round-trip 語意等價（§九.10）
    assert json.loads(loaded.source_segment_ids_json) == segments
    assert json.loads(loaded.generator_metadata_json) == meta
    assert loaded.status == CANDIDATE_PROPOSED

    loaded.status = CANDIDATE_ACCEPTED
    repo.update_candidate(loaded)
    assert repo.get_candidate(candidate.candidate_id).status == CANDIDATE_ACCEPTED
    assert [c.candidate_id for c in repo.list_candidates(session.session_id)] == [
        candidate.candidate_id
    ]
    assert repo.list_candidates(session.session_id, status=CANDIDATE_PROPOSED) == []


def test_require_missing_raises(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ReviewStateError):
        repo.require_session("nope")
    with pytest.raises(ReviewStateError):
        repo.require_candidate("nope")


# --------------------------------------------------------------------------
# Revision / Event
# --------------------------------------------------------------------------
def test_revisions_are_sequential_and_listed_in_order(tmp_path):
    repo = _repo(tmp_path)
    session = repo.create_session(_session())
    candidate = repo.create_candidate(_candidate(session.session_id))
    for n in (1, 2, 3):
        repo.add_revision(
            CandidateRevision(
                candidate_id=candidate.candidate_id,
                revision_number=n,
                question=f"q{n}",
                query_type="semantic",
                answerable=True,
            )
        )
    revisions = repo.list_revisions(candidate.candidate_id)
    assert [r.revision_number for r in revisions] == [1, 2, 3]
    assert repo.latest_revision(candidate.candidate_id).question == "q3"
    assert repo.count_revisions(session.session_id) == 3


def test_duplicate_revision_number_rejected(tmp_path):
    repo = _repo(tmp_path)
    session = repo.create_session(_session())
    candidate = repo.create_candidate(_candidate(session.session_id))
    rev = CandidateRevision(
        candidate_id=candidate.candidate_id,
        revision_number=1,
        question="q",
        query_type="semantic",
        answerable=True,
    )
    repo.add_revision(rev)
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_revision(rev)


def test_audit_events_recorded_and_ordered(tmp_path):
    repo = _repo(tmp_path)
    session = repo.create_session(_session())
    repo.add_event(session.session_id, "candidate_created", actor="local_user")
    repo.add_event(
        session.session_id, "candidate_accepted", actor="local_user", detail={"k": "v"}
    )
    events = repo.list_events(session.session_id)
    assert [e.event_type for e in events] == ["candidate_created", "candidate_accepted"]
    assert json.loads(events[1].detail_json) == {"k": "v"}


# --------------------------------------------------------------------------
# Foreign Key / CASCADE
# --------------------------------------------------------------------------
def test_foreign_key_rejects_orphan_candidate(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_candidate(_candidate("no-such-session"))


def test_deleting_session_cascades_candidates_revisions_events(tmp_path):
    """§九.5：Candidate Revision 不得孤立。"""
    repo = _repo(tmp_path)
    session = repo.create_session(_session())
    candidate = repo.create_candidate(_candidate(session.session_id))
    repo.add_revision(
        CandidateRevision(
            candidate_id=candidate.candidate_id,
            revision_number=1,
            question="q",
            query_type="semantic",
            answerable=True,
        )
    )
    repo.add_event(session.session_id, "candidate_created", actor="local_user")

    repo.delete_session(session.session_id)

    assert repo.get_session(session.session_id) is None
    assert repo.get_candidate(candidate.candidate_id) is None
    assert repo.list_revisions(candidate.candidate_id) == []
    assert repo.list_events(session.session_id) == []


# --------------------------------------------------------------------------
# 終態規則
# --------------------------------------------------------------------------
def test_published_session_is_not_mutable(tmp_path):
    session = _session(status=SESSION_PUBLISHED)
    with pytest.raises(ReviewStateError):
        session.assert_mutable()


def test_cancelled_session_is_not_mutable(tmp_path):
    session = _session(status=SESSION_CANCELLED)
    with pytest.raises(ReviewStateError):
        session.assert_mutable()


def test_published_candidate_is_not_mutable(tmp_path):
    candidate = _candidate("s", status=CANDIDATE_PUBLISHED)
    with pytest.raises(ReviewStateError):
        candidate.assert_mutable()


def test_mark_candidates_published(tmp_path):
    repo = _repo(tmp_path)
    session = repo.create_session(_session())
    candidate = repo.create_candidate(
        _candidate(session.session_id, status=CANDIDATE_ACCEPTED)
    )
    repo.mark_candidates_published([candidate.candidate_id])
    assert repo.get_candidate(candidate.candidate_id).status == CANDIDATE_PUBLISHED
