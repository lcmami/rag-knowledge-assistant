"""Slice 6 §二十三 — Evidence Annotation 測試。

Offset / Anchor / Version 驗證、多 Span、多文件、Unanswerable 規則，
以及 identifier-bearing unanswerable 偵測。
"""
from __future__ import annotations

import pytest

from evaluation.v11.review.coverage import is_identifier_bearing
from evaluation.v11.review.evidence_annotation import (
    EvidenceValidationError,
    find_anchor_candidates,
    parse_evidence_input,
    validate_evidence,
)
from evaluation.v11.review.generator import FakeCandidateGenerator
from tests.unit.review_helpers import (
    evidence_for,
    make_base_dataset,
    make_corpus,
    make_service,
    segment_containing,
)


@pytest.fixture
def corpus(tmp_path):
    return make_corpus(tmp_path)


def _validate(corpus, raw_list, *, answerable=True, confirmed=False):
    spans = [parse_evidence_input(r) for r in raw_list]
    return validate_evidence(
        spans, corpus, answerable=answerable, unanswerable_confirmed=confirmed
    )


def _codes(result):
    return {i.code for i in result.issues}


# --------------------------------------------------------------------------
# 合法 / 非法 Offset
# --------------------------------------------------------------------------
def test_valid_offset_passes(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    result = _validate(corpus, [ev])
    assert result.valid, result.issues
    assert result.snippets  # Report 只顯示短 snippet


def test_offset_outside_segment_is_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev["char_end"] = ev["char_start"] + 100000
    assert "offset_outside_segment" in _codes(_validate(corpus, [ev]))


def test_negative_and_inverted_offsets_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    negative = dict(ev, char_start=-1, char_end=5)
    assert "negative_offset" in _codes(_validate(corpus, [negative]))

    inverted = dict(ev, char_start=50, char_end=50)
    assert "empty_or_inverted_range" in _codes(_validate(corpus, [inverted]))


def test_missing_offset_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev.pop("char_start")
    ev.pop("char_end")
    assert "missing_offset" in _codes(_validate(corpus, [ev]))


# --------------------------------------------------------------------------
# 文件 / 版本
# --------------------------------------------------------------------------
def test_wrong_document_sha_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev["document_sha256"] = "f" * 64
    assert "unknown_document_sha256" in _codes(_validate(corpus, [ev]))


def test_wrong_canonical_version_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev["canonical_text_version"] = "v999"
    assert "canonical_version_mismatch" in _codes(_validate(corpus, [ev]))


def test_wrong_loader_version_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev["loader_version"] = "v999"
    assert "loader_version_mismatch" in _codes(_validate(corpus, [ev]))


def test_filename_mismatch_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev["filename"] = "other.md"
    assert "filename_mismatch" in _codes(_validate(corpus, [ev]))


# --------------------------------------------------------------------------
# Anchor
# --------------------------------------------------------------------------
def test_anchor_missing_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev["anchor_text"] = []
    assert "missing_anchor" in _codes(_validate(corpus, [ev]))


def test_anchor_not_in_segment_rejected(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    ev["anchor_text"] = ["這段文字不存在於文件"]
    assert "anchor_not_found" in _codes(_validate(corpus, [ev]))


def test_anchor_elsewhere_in_same_segment_is_accepted(corpus):
    """Anchor 可存在於 evidence 範圍或對應 segment（§十四.2）。"""
    seg = segment_containing(corpus, "equipment_alarm_manual.md", "E102")
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    other = seg.normalized_text.split()[-1]
    ev["anchor_text"] = [other]
    assert _validate(corpus, [ev]).valid


# --------------------------------------------------------------------------
# 多 Span / 多文件
# --------------------------------------------------------------------------
def test_multiple_spans_same_document(corpus):
    a = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    b = evidence_for(corpus, "equipment_alarm_manual.md", "CX-500")
    result = _validate(corpus, [a, b])
    assert result.valid, result.issues
    assert len(result.snippets) == 2


def test_multiple_documents(corpus):
    a = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    b = evidence_for(corpus, "api_integration_spec.md", "API Token")
    result = _validate(corpus, [a, b])
    assert result.valid, result.issues


def test_one_invalid_span_fails_whole_case(corpus):
    good = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    bad = dict(good, char_start=-5)
    assert not _validate(corpus, [good, bad]).valid


# --------------------------------------------------------------------------
# Answerable / Unanswerable 規則
# --------------------------------------------------------------------------
def test_answerable_without_evidence_is_invalid(corpus):
    assert "answerable_without_evidence" in _codes(_validate(corpus, [], answerable=True))


def test_unanswerable_with_evidence_is_invalid(corpus):
    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    codes = _codes(_validate(corpus, [ev], answerable=False, confirmed=True))
    assert "unanswerable_with_evidence" in codes


def test_unanswerable_without_human_confirmation_is_invalid(corpus):
    codes = _codes(_validate(corpus, [], answerable=False, confirmed=False))
    assert "unanswerable_not_confirmed" in codes


def test_unanswerable_confirmed_and_empty_is_valid(corpus):
    assert _validate(corpus, [], answerable=False, confirmed=True).valid


def test_identifier_bearing_unanswerable_detection():
    """§十五.5：含識別碼的 unanswerable 需可被標示。"""
    assert is_identifier_bearing("CX-500 的售價是多少？")
    assert is_identifier_bearing("E102 的保固期限？")
    assert not is_identifier_bearing("這台機器很貴嗎？")


# --------------------------------------------------------------------------
# Service 層：確認 unanswerable 需人工操作
# --------------------------------------------------------------------------
def test_confirm_unanswerable_requires_answerable_false(tmp_path):
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    session = service.create_session(
        session_name="s", base_dataset_path=make_base_dataset(tmp_path, corpus)
    )
    doc = corpus.by_filename("equipment_alarm_manual.md")
    outcome = service.add_candidate_manual(
        session.session_id,
        question="CX-500 的售價是多少？",
        query_type="unanswerable",
        answerable=True,  # 先設為 answerable
        source_document_sha256=doc.document_sha256,
    )
    cid = outcome.target_id
    from evaluation.v11.review.models import ReviewStateError

    with pytest.raises(ReviewStateError):
        service.confirm_unanswerable(cid)

    service.edit_candidate(cid, answerable=False, change_reason="確認文件未記載")
    result = service.confirm_unanswerable(cid)
    assert result.new_state["unanswerable_confirmed"] is True
    assert service.validate_candidate_evidence(cid).valid


# --------------------------------------------------------------------------
# 定位輔助
# --------------------------------------------------------------------------
def test_find_anchor_candidates_returns_unconfirmed_positions(corpus):
    matches = find_anchor_candidates(corpus, "E102")
    assert matches
    for m in matches:
        payload = m.to_dict()
        assert payload["confirmed"] is False  # 搜尋結果需人工確認
        assert payload["char_end"] > payload["char_start"]


def test_find_anchor_offsets_are_usable_as_evidence(corpus):
    """搜尋結果的 offset 必須真的能通過驗證（但仍需人工採用）。"""
    doc = corpus.by_filename("equipment_alarm_manual.md")
    match = find_anchor_candidates(corpus, "E102", document_sha256=doc.document_sha256)[0]
    ev = {
        "document_sha256": match.document_sha256,
        "filename": match.filename,
        "canonical_text_version": doc.canonical_text_version,
        "loader_version": doc.loader_version,
        "page": match.page,
        "section": match.section,
        "offset_scope": match.offset_scope,
        "char_start": match.char_start,
        "char_end": match.char_end,
        "anchor_text": ["E102"],
    }
    assert _validate(corpus, [ev]).valid


def test_find_anchor_rejects_empty_anchor(corpus):
    with pytest.raises(EvidenceValidationError):
        find_anchor_candidates(corpus, "")
