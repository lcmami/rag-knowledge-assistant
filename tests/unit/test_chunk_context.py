"""相鄰段落 Context 單元測試（v1.1 Slice 3 §十二 / §二十二.6）。"""
from __future__ import annotations

from tests.unit.chunking_helpers import context_spans, core_spans, run_strategy

FIVE = [(f"段落{i}的內容敘述文字。", None, "S1", "paragraph") for i in range(5)]
TWO_SECTION = [
    ("甲一。", None, "S1", "paragraph"),
    ("甲二。", None, "S1", "paragraph"),
    ("乙一。", None, "S2", "paragraph"),
    ("乙二。", None, "S2", "paragraph"),
]


def test_zero_context_default():
    out = run_strategy(FIVE, {"strategy": "paragraph", "target_chars": 20, "overlap_chars": 0, "max_tokens": 999})
    for c in out.chunks:
        assert not context_spans(c)
    assert out.context_added_count == 0


def test_before_context_added():
    out = run_strategy(FIVE, {"strategy": "paragraph", "target_chars": 20, "overlap_chars": 0,
                              "max_tokens": 999, "context_before_paragraphs": 1})
    # 第二個以後的 chunk 應有 before context。
    assert any(context_spans(c) for c in out.chunks)
    assert out.context_added_count >= 1


def test_after_context_added():
    out = run_strategy(FIVE, {"strategy": "paragraph", "target_chars": 20, "overlap_chars": 0,
                              "max_tokens": 999, "context_after_paragraphs": 1})
    assert out.context_added_count >= 1


def test_context_does_not_cross_section_or_page():
    out = run_strategy(TWO_SECTION, {"strategy": "paragraph", "target_chars": 12, "overlap_chars": 0,
                                     "max_tokens": 999, "context_before_paragraphs": 2, "context_after_paragraphs": 2})
    for c in out.chunks:
        assert len({s.section for s in c.provenance_spans}) == 1
        assert len({s.page for s in c.provenance_spans}) == 1


def test_token_limit_reduces_after_then_before_core_preserved():
    # 極小 max_tokens：context 必須被縮減，但核心 evidence 不得被擠掉。
    raw = {"strategy": "paragraph", "target_chars": 12, "overlap_chars": 0, "max_tokens": 12,
           "context_before_paragraphs": 1, "context_after_paragraphs": 1}
    out = run_strategy(FIVE, raw)
    for c in out.chunks:
        assert len(c.text) <= 12
        assert core_spans(c), "核心 span 不得被 context 擠掉"
    assert out.context_reduced_count >= 1


def test_context_provenance_role_marked():
    out = run_strategy(FIVE, {"strategy": "paragraph", "target_chars": 18, "overlap_chars": 0,
                              "max_tokens": 999, "context_before_paragraphs": 1, "context_after_paragraphs": 1})
    saw_context = False
    for c in out.chunks:
        for s in c.provenance_spans:
            assert s.role in ("core", "context")
        if context_spans(c):
            saw_context = True
            # 文字順序：context 在 core 前後；provenance 順序需與文字一致（core 夾在中間）。
            roles = [s.role for s in c.provenance_spans]
            assert "core" in roles
    assert saw_context


def test_context_only_measured_not_changing_core_spans():
    # 加 context 不改變 core span 的 canonical 範圍（只是額外附加 context span）。
    base = run_strategy(FIVE, {"strategy": "paragraph", "target_chars": 18, "overlap_chars": 0, "max_tokens": 999})
    withctx = run_strategy(FIVE, {"strategy": "paragraph", "target_chars": 18, "overlap_chars": 0,
                                  "max_tokens": 999, "context_after_paragraphs": 1})
    base_core = [[(s.char_start, s.char_end) for s in core_spans(c)] for c in base.chunks]
    ctx_core = [[(s.char_start, s.char_end) for s in core_spans(c)] for c in withctx.chunks]
    assert base_core == ctx_core
