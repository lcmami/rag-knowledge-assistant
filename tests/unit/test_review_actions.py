"""Slice 6 §二十三 — Review Action 測試（Accept / Edit / Reject / Revision / Audit）。

驗證人工操作的狀態機、Revision 歷史、Audit Trail，以及「**沒有批次自動接受**」
與「未經 Accept 不可發布」等不變量。
"""
from __future__ import annotations

import inspect
import json

import pytest

from evaluation.v11.review import service as service_module
from evaluation.v11.review.generator import FakeCandidateGenerator
from evaluation.v11.review.models import (
    CANDIDATE_ACCEPTED,
    CANDIDATE_PROPOSED,
    CANDIDATE_PUBLISHED,
    CANDIDATE_REJECTED,
    EVENT_CANDIDATE_ACCEPTED,
    EVENT_CANDIDATE_CREATED,
    EVENT_CANDIDATE_EDITED,
    EVENT_CANDIDATE_REJECTED,
    SESSION_DRAFT,
    SESSION_REVIEWING,
    ReviewStateError,
)
from evaluation.v11.review.service import ReviewService
from tests.unit.review_helpers import (
    evidence_for,
    make_base_dataset,
    make_corpus,
    make_service,
)


@pytest.fixture
def ctx(tmp_path):
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    base = make_base_dataset(tmp_path, corpus)
    session = service.create_session(session_name="s", base_dataset_path=base)
    doc = corpus.by_filename("equipment_alarm_manual.md")
    return service, corpus, session, doc, base


def _one_candidate(service, session, doc):
    outcomes = service.generate_candidates(
        session.session_id,
        FakeCandidateGenerator(),
        document_sha256=doc.document_sha256,
        requested_count=1,
    )
    return outcomes[0].target_id


# --------------------------------------------------------------------------
# 建立與初始狀態
# --------------------------------------------------------------------------
def test_generated_candidates_start_proposed_and_move_session_to_reviewing(ctx):
    service, _corpus, session, doc, _base = ctx
    assert session.status == SESSION_DRAFT
    cid = _one_candidate(service, session, doc)
    assert service.repo.require_candidate(cid).status == CANDIDATE_PROPOSED
    assert service.get_session(session.session_id).status == SESSION_REVIEWING


def test_manual_candidate_also_starts_proposed(ctx):
    """§十二：無 Provider 時仍可手動新增，且同樣從 proposed 開始。"""
    service, _corpus, session, doc, _base = ctx
    outcome = service.add_candidate_manual(
        session.session_id,
        question="CX-500 多久換濾芯？",
        query_type="exact_code",
        answerable=True,
        source_document_sha256=doc.document_sha256,
        source_segment_ids=[doc.segments[0].segment_id],
        suggested_group_id="cx500",
        suggested_split="validation",
    )
    assert outcome.new_state["status"] == CANDIDATE_PROPOSED


def test_manual_candidate_rejects_bad_query_type_and_split(ctx):
    service, _corpus, session, _doc, _base = ctx
    with pytest.raises(ReviewStateError):
        service.add_candidate_manual(
            session.session_id, question="q", query_type="nope", answerable=True
        )
    with pytest.raises(ReviewStateError):
        service.add_candidate_manual(
            session.session_id,
            question="q",
            query_type="semantic",
            answerable=True,
            suggested_split="train",
        )


# --------------------------------------------------------------------------
# Accept / Reject
# --------------------------------------------------------------------------
def test_accept_transitions_and_records_event(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    outcome = service.accept_candidate(cid, group_id="g1", split="validation")

    assert outcome.old_state["status"] == CANDIDATE_PROPOSED
    assert outcome.new_state["status"] == CANDIDATE_ACCEPTED
    assert outcome.revision == 1
    assert outcome.validation is not None  # Write Operation 需回傳 validation（§十三）

    types = [e.event_type for e in service.repo.list_events(session.session_id)]
    assert EVENT_CANDIDATE_CREATED in types
    assert EVENT_CANDIDATE_ACCEPTED in types


def test_reject_transitions_and_records_event(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    outcome = service.reject_candidate(cid, reason="低品質")
    assert outcome.new_state["status"] == CANDIDATE_REJECTED
    assert EVENT_CANDIDATE_REJECTED in [
        e.event_type for e in service.repo.list_events(session.session_id)
    ]


def test_rejected_candidate_can_be_reaccepted_but_published_cannot_change(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    service.reject_candidate(cid)
    service.accept_candidate(cid, group_id="g1", split="validation")
    assert service.repo.require_candidate(cid).status == CANDIDATE_ACCEPTED

    service.repo.mark_candidates_published([cid])
    with pytest.raises(ReviewStateError):
        service.accept_candidate(cid, group_id="g1", split="validation")


# --------------------------------------------------------------------------
# Edit / Revision
# --------------------------------------------------------------------------
def test_edit_creates_new_revision_without_overwriting_history(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    original = service.repo.require_candidate(cid).proposed_question

    service.edit_candidate(cid, question="第一次修改？", change_reason="r1")
    service.edit_candidate(cid, question="第二次修改？", change_reason="r2")

    revisions = service.repo.list_revisions(cid)
    assert [r.revision_number for r in revisions] == [1, 2]
    assert revisions[0].question == "第一次修改？"
    assert revisions[1].question == "第二次修改？"
    # 原始提案仍保留在 candidate 上，未被覆寫
    assert service.repo.require_candidate(cid).proposed_question == original
    assert EVENT_CANDIDATE_EDITED in [
        e.event_type for e in service.repo.list_events(session.session_id)
    ]


def test_edit_inherits_unchanged_fields_from_previous_revision(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    service.edit_candidate(cid, group_id="g-keep", split="test", change_reason="set")
    service.edit_candidate(cid, question="只改問題？", change_reason="q only")

    latest = service.repo.latest_revision(cid)
    assert latest.question == "只改問題？"
    assert latest.group_id == "g-keep"
    assert latest.split == "test"


def test_published_candidate_cannot_be_edited(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    service.accept_candidate(cid, group_id="g1", split="validation")
    service.repo.mark_candidates_published([cid])
    with pytest.raises(ReviewStateError):
        service.edit_candidate(cid, question="不該成功")


def test_edit_rejects_invalid_values(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    with pytest.raises(ReviewStateError):
        service.edit_candidate(cid, query_type="bogus")
    with pytest.raises(ReviewStateError):
        service.edit_candidate(cid, split="train")
    with pytest.raises(ReviewStateError):
        service.edit_candidate(cid, question="   ")


# --------------------------------------------------------------------------
# 「沒有批次自動接受」
# --------------------------------------------------------------------------
def test_no_batch_auto_accept_api_exists():
    """§十三：不得在無確認的狀況下批次接受全部 Candidate。"""
    names = [n for n, _ in inspect.getmembers(ReviewService, inspect.isfunction)]
    forbidden = [
        n
        for n in names
        if ("accept" in n and ("all" in n or "batch" in n or "auto" in n))
    ]
    assert forbidden == []
    # accept_candidate 只接受單一 candidate id
    sig = inspect.signature(ReviewService.accept_candidate)
    assert "candidate_id" in sig.parameters
    assert not any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )


def test_service_does_not_import_search_or_optimizer_results():
    """§六.3：Session 不得讀取 Search Test Metrics。"""
    source = inspect.getsource(service_module)
    for forbidden in ("search_runner", "search_report", "trial_runner", "search_repository"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# 終態 Session
# --------------------------------------------------------------------------
def test_cancelled_session_blocks_further_writes(ctx):
    service, _corpus, session, doc, _base = ctx
    cid = _one_candidate(service, session, doc)
    service.cancel_session(session.session_id)
    with pytest.raises(ReviewStateError):
        service.accept_candidate(cid, group_id="g", split="validation")
    with pytest.raises(ReviewStateError):
        service.add_candidate_manual(
            session.session_id, question="q", query_type="semantic", answerable=True
        )
