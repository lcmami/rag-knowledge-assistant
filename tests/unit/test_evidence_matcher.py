"""EvidenceMatcher 單元測試（v1.1 Slice 1 §十三 EvidenceMatcher）。"""
from __future__ import annotations

import pytest

from app.models.canonical import ProvenanceSpan
from evaluation.v11.case_schema import EvidenceSpan
from evaluation.v11.evidence_matcher import (
    RELEVANCE_CONTEXT,
    RELEVANCE_COVERS,
    RELEVANCE_IRRELEVANT,
    EvidenceMatcher,
)


def prov(**over) -> ProvenanceSpan:
    base = dict(
        document_sha256="sha-a",
        canonical_text_version="v1",
        loader_version="v1",
        page=None,
        section="E102 感測器通訊異常",
        paragraph_index=0,
        offset_scope="document",
        char_start=0,
        char_end=100,
    )
    base.update(over)
    return ProvenanceSpan(**base)


def ev(**over) -> EvidenceSpan:
    base = dict(
        document_sha256="sha-a",
        canonical_text_version="v1",
        loader_version="v1",
        section="E102 感測器通訊異常",
        offset_scope="document",
        char_start=0,
        char_end=100,
        anchor_text=["E102"],
    )
    base.update(over)
    return EvidenceSpan(**base)


def test_exact_offset_overlap_is_level2():
    m = EvidenceMatcher(offset_coverage_threshold=0.5)
    r = m.match([ev()], [prov()], chunk_text="E102 ...")
    assert r.relevance_level == RELEVANCE_COVERS
    assert "offset_overlap" in r.match_reasons
    assert r.coverage == pytest.approx(1.0)


def test_partial_overlap_below_threshold_is_context():
    m = EvidenceMatcher(offset_coverage_threshold=0.5)
    # evidence [0,100]、provenance [90,120] → overlap 10 → coverage 0.1
    r = m.match(
        [ev(section=None)],
        [prov(section=None, char_start=90, char_end=120)],
        chunk_text="no anchor here",
    )
    assert r.relevance_level == RELEVANCE_CONTEXT
    assert "partial_offset_overlap" in r.match_reasons
    assert r.coverage == pytest.approx(0.1)


def test_section_match_without_offset_is_context():
    m = EvidenceMatcher()
    r = m.match(
        [ev(char_start=None, char_end=None, anchor_text=["缺"])],
        [prov()],
        chunk_text="不含 anchor",
    )
    assert r.relevance_level == RELEVANCE_CONTEXT
    assert "section_match" in r.match_reasons


def test_page_match_is_context():
    m = EvidenceMatcher()
    e = ev(section=None, offset_scope="page", page=2, char_start=None, char_end=None, anchor_text=["缺"])
    p = prov(section=None, offset_scope="page", page=2, char_start=0, char_end=50)
    r = m.match([e], [p], chunk_text="不含 anchor")
    assert r.relevance_level == RELEVANCE_CONTEXT
    assert "page_match" in r.match_reasons


def test_anchor_fallback_strong_is_level2():
    # 無 offset，但 anchor 全命中且 section 相符 → 視為強 evidence（Level 2）
    m = EvidenceMatcher()
    e = ev(char_start=None, char_end=None, anchor_text=["E102", "感測器"])
    r = m.match([e], [prov()], chunk_text="E102 表示感測器通訊異常")
    assert r.relevance_level == RELEVANCE_COVERS
    assert "anchor_match" in r.match_reasons


def test_anchor_fallback_without_location_is_context():
    m = EvidenceMatcher()
    e = ev(section=None, char_start=None, char_end=None, anchor_text=["E102"])
    r = m.match([e], [prov(section=None)], chunk_text="E102 出現於此")
    assert r.relevance_level == RELEVANCE_CONTEXT
    assert "anchor_match" in r.match_reasons


def test_wrong_document_is_irrelevant():
    m = EvidenceMatcher()
    r = m.match([ev()], [prov(document_sha256="sha-OTHER")], chunk_text="E102")
    assert r.relevance_level == RELEVANCE_IRRELEVANT
    assert "wrong_document" in r.match_reasons


def test_version_mismatch_not_silent():
    m = EvidenceMatcher()
    # 同文件但 canonical 版本不符：offset 不得被靜默採信
    r = m.match(
        [ev(section=None, anchor_text=["缺"])],
        [prov(section=None, canonical_text_version="v2")],
        chunk_text="不含 anchor",
    )
    assert r.version_compatible is False
    assert "version_mismatch" in r.match_reasons
    # 未達其他條件 → 不會因 offset 而誤判為相關
    assert r.relevance_level == RELEVANCE_IRRELEVANT


def test_invalid_offset_uses_anchor_fallback():
    m = EvidenceMatcher()
    e = ev(char_start=None, char_end=None, anchor_text=["E102", "感測器"])
    r = m.match([e], [prov()], chunk_text="E102 感測器")
    assert r.relevance_level == RELEVANCE_COVERS  # anchor 全命中 + section 相符


def test_multiple_spans_one_matches():
    m = EvidenceMatcher(offset_coverage_threshold=0.5)
    good = ev(evidence_id="e-good")
    bad = ev(evidence_id="e-bad", document_sha256="sha-OTHER")
    r = m.match([bad, good], [prov()], chunk_text="E102")
    assert r.relevance_level == RELEVANCE_COVERS
    assert "e-good" in r.matched_evidence_ids
    assert "e-bad" not in r.matched_evidence_ids


def test_configurable_coverage_threshold():
    e = ev(section=None, anchor_text=["缺"])
    p = prov(section=None, char_start=0, char_end=60)  # coverage 0.6
    low = EvidenceMatcher(offset_coverage_threshold=0.5).match([e], [p], chunk_text="x")
    high = EvidenceMatcher(offset_coverage_threshold=0.8).match([e], [p], chunk_text="x")
    assert low.relevance_level == RELEVANCE_COVERS
    assert high.relevance_level == RELEVANCE_CONTEXT


def test_empty_evidence_is_irrelevant():
    m = EvidenceMatcher()
    r = m.match([], [prov()], chunk_text="E102")
    assert r.relevance_level == RELEVANCE_IRRELEVANT
