"""Paragraph-aware Strategy 單元測試（v1.1 Slice 3 §十 / §二十二.4）。"""
from __future__ import annotations

from tests.unit.chunking_helpers import core_spans, run_strategy


def test_whole_paragraphs_preferred():
    els = [("短句一。", None, "S1", "paragraph"), ("短句二。", None, "S1", "paragraph")]
    out = run_strategy(els, {"strategy": "paragraph", "target_chars": 100, "overlap_chars": 0, "max_tokens": 999})
    # 兩短段在 target 內 → 合併為一個 chunk（多 span）。
    assert len(out.chunks) == 1
    assert len(core_spans(out.chunks[0])) == 2


def test_multi_segment_multi_span():
    els = [(f"段落{i}內容敘述。", None, "S1", "paragraph") for i in range(3)]
    out = run_strategy(els, {"strategy": "paragraph", "target_chars": 200, "overlap_chars": 0, "max_tokens": 999})
    assert len(out.chunks) == 1
    spans = core_spans(out.chunks[0])
    assert len(spans) == 3
    assert len({s.paragraph_index for s in spans}) == 3


def test_oversized_segment_falls_back_to_fixed():
    els = [("超" * 300, None, "S1", "paragraph")]
    out = run_strategy(els, {"strategy": "paragraph", "target_chars": 50, "overlap_chars": 0, "max_tokens": 999})
    assert len(out.chunks) > 1  # 單段超限 → fixed split 多個
    assert out.fallback_count >= 1
    assert any("oversized_segment" in r for r in out.fallback_reasons)
    for c in out.chunks:
        assert len(c.text) <= 50


def test_whole_segment_overlap_same_canonical_span():
    els = [(f"段{i}的完整敘述內容約二十字左右填充。", None, "S1", "paragraph") for i in range(4)]
    out = run_strategy(els, {"strategy": "paragraph", "target_chars": 45, "overlap_chars": 20, "max_tokens": 999})
    # overlap 重複出現的段落，其 provenance 仍指向相同 canonical span。
    span_by_para: dict[int, tuple[int, int]] = {}
    for c in out.chunks:
        for s in core_spans(c):
            key = s.paragraph_index
            rng = (s.char_start, s.char_end)
            if key in span_by_para:
                assert span_by_para[key] == rng, "同段 overlap 的 canonical span 不一致"
            span_by_para[key] = rng
    assert len(out.chunks) > len(els) // 2  # 確有 overlap 產生的重複


def test_does_not_cross_page():
    els = [
        ("第一頁段落內容。", 1, "S", "paragraph"),
        ("第二頁段落內容。", 2, "S", "paragraph"),
    ]
    out = run_strategy(els, {"strategy": "paragraph", "target_chars": 500, "overlap_chars": 0, "max_tokens": 999},
                       document_type="pdf")
    for c in out.chunks:
        pages = {s.page for s in c.provenance_spans}
        assert len(pages) == 1


def test_does_not_cross_section_by_default():
    els = [
        ("甲段內容。", None, "S1", "paragraph"),
        ("乙段內容。", None, "S2", "paragraph"),
    ]
    out = run_strategy(els, {"strategy": "paragraph", "target_chars": 500, "overlap_chars": 0, "max_tokens": 999})
    for c in out.chunks:
        sections = {s.section for s in c.provenance_spans}
        assert len(sections) == 1
    assert {c.section for c in out.chunks} == {"S1", "S2"}


def test_deterministic():
    els = [(f"段落{i}。", None, "S1", "paragraph") for i in range(5)]
    a = run_strategy(els, {"strategy": "paragraph", "target_chars": 12, "overlap_chars": 4, "max_tokens": 999})
    b = run_strategy(els, {"strategy": "paragraph", "target_chars": 12, "overlap_chars": 4, "max_tokens": 999})
    assert [c.text for c in a.chunks] == [c.text for c in b.chunks]
    assert [[(s.char_start, s.char_end) for s in c.provenance_spans] for c in a.chunks] == \
           [[(s.char_start, s.char_end) for s in c.provenance_spans] for c in b.chunks]
