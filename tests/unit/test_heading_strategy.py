"""Heading-aware Strategy 單元測試（v1.1 Slice 3 §十一 / §二十二.5）。"""
from __future__ import annotations

from tests.unit.chunking_helpers import run_strategy

MULTI_SECTION = [
    ("設備警報", None, "設備警報", "heading"),
    ("警報段落一敘述。", None, "設備警報", "paragraph"),
    ("警報段落二敘述。", None, "設備警報", "paragraph"),
    ("維護保養", None, "維護保養", "heading"),
    ("維護段落敘述。", None, "維護保養", "paragraph"),
]


def test_groups_by_section():
    out = run_strategy(MULTI_SECTION, {"strategy": "heading", "target_chars": 500, "overlap_chars": 0, "max_tokens": 999})
    assert {c.section for c in out.chunks} == {"設備警報", "維護保養"}


def test_multiple_chunks_within_long_section():
    els = [("大段", None, "S1", "heading")] + [(f"段落{i}內容約十餘字填充敘述。", None, "S1", "paragraph") for i in range(6)]
    out = run_strategy(els, {"strategy": "heading", "target_chars": 40, "overlap_chars": 0, "max_tokens": 999})
    s1 = [c for c in out.chunks if c.section == "S1"]
    assert len(s1) > 1


def test_does_not_cross_section():
    out = run_strategy(MULTI_SECTION, {"strategy": "heading", "target_chars": 500, "overlap_chars": 0, "max_tokens": 999})
    for c in out.chunks:
        assert len({s.section for s in c.provenance_spans}) == 1


def test_prefix_on_off():
    on = run_strategy(MULTI_SECTION, {"strategy": "heading", "target_chars": 500, "overlap_chars": 0,
                                      "max_tokens": 999, "prepend_section_title": True})
    off = run_strategy(MULTI_SECTION, {"strategy": "heading", "target_chars": 500, "overlap_chars": 0,
                                       "max_tokens": 999, "prepend_section_title": False})
    assert on.chunks[0].text.startswith("[SECTION] 設備警報\n")
    assert not off.chunks[0].text.startswith("[SECTION]")
    # prefix 改變 text → embedding 會 miss，但 provenance / section 不變。
    assert on.chunks[0].section == off.chunks[0].section


def test_missing_section_falls_back_to_paragraph_and_records():
    # TXT-like：無 section（None）。啟用 prefix 也不得加入 "None"/空白/虛構標題。
    els = [("無標題段落一。", None, None, "paragraph"), ("無標題段落二。", None, None, "paragraph")]
    out = run_strategy(els, {"strategy": "heading", "target_chars": 500, "overlap_chars": 0,
                             "max_tokens": 999, "prepend_section_title": True}, document_type="txt")
    assert any("no_section_metadata" in r for r in out.fallback_reasons)
    assert out.fallback_count >= 1
    for c in out.chunks:
        assert not c.text.startswith("[SECTION]")
        assert "None" not in c.text.split("\n")[0]


def test_no_fabricated_heading_when_section_none():
    els = [("純內文段落敘述。", None, None, "paragraph")]
    out = run_strategy(els, {"strategy": "heading", "target_chars": 500, "overlap_chars": 0,
                             "max_tokens": 999, "prepend_section_title": True}, document_type="txt")
    assert out.chunks[0].section is None
    assert not out.chunks[0].text.startswith("[SECTION]")


def test_deterministic():
    a = run_strategy(MULTI_SECTION, {"strategy": "heading", "target_chars": 40, "overlap_chars": 8, "max_tokens": 999})
    b = run_strategy(MULTI_SECTION, {"strategy": "heading", "target_chars": 40, "overlap_chars": 8, "max_tokens": 999})
    assert [c.text for c in a.chunks] == [c.text for c in b.chunks]
