"""Fixed Window Strategy 單元測試（v1.1 Slice 3 §九 / §二十二.3）。"""
from __future__ import annotations

from tests.unit.chunking_helpers import context_spans, core_spans, run_strategy


def _one_para(text, section="S1"):
    return [(text, None, section, "paragraph")]


def test_windows_target_and_overlap_exact():
    # 100 字元、無標點、無換行 → 純由 target/overlap 決定窗口。
    out = run_strategy(_one_para("甲" * 100), {"strategy": "fixed", "target_chars": 40,
                                              "overlap_chars": 10, "max_tokens": 100000})
    spans = [core_spans(c)[0] for c in out.chunks]
    ranges = [(s.char_start, s.char_end) for s in spans]
    assert ranges == [(0, 40), (30, 70), (60, 100)]
    # overlap 區確實重疊。
    assert out.chunks[0].text[30:40] == out.chunks[1].text[0:10]


def test_prefers_paragraph_boundary_and_provenance_excludes_separator():
    # 兩段：AAAA / BBBB，group_text = 'AAAA\n\nBBBB'。窗口不得把分隔符 '\n\n' 算進 provenance。
    els = [("AAAA", None, "S1", "paragraph"), ("BBBB", None, "S1", "paragraph")]
    out = run_strategy(els, {"strategy": "fixed", "target_chars": 6, "overlap_chars": 0,
                             "max_tokens": 100000})
    first = out.chunks[0]
    # 第一窗優先在段落邊界切；provenance 只涵蓋 'AAAA'（char 0..4），不含分隔符。
    cs = core_spans(first)
    assert (cs[0].char_start, cs[0].char_end) == (0, 4)
    # 第一窗的 core provenance 不涵蓋分隔符 '\n\n'（char 4..6），故只有一個 span。
    assert len(cs) == 1


def test_long_text_hard_split_by_tokens():
    out = run_strategy(_one_para("乙" * 200), {"strategy": "fixed", "target_chars": 500,
                                              "overlap_chars": 0, "max_tokens": 30})
    for c in out.chunks:
        assert len(c.text) <= 30  # count_tokens=len → token 硬限制即字元硬限制


def test_max_token_hard_limit_respected_with_prefix_budget():
    out = run_strategy(_one_para("丙" * 200), {"strategy": "fixed", "target_chars": 80,
                                              "overlap_chars": 0, "max_tokens": 40,
                                              "prepend_section_title": True})
    for c in out.chunks:
        # prefix 佔用預算後，最終 text（含 prefix）仍 <= max_tokens。
        assert len(c.text) <= 40


def test_no_infinite_loop_high_overlap():
    # overlap 幾乎等於 target（仍 < target，合法）→ 必須終止且不產生空 chunk。
    out = run_strategy(_one_para("丁" * 120), {"strategy": "fixed", "target_chars": 40,
                                              "overlap_chars": 39, "max_tokens": 100000})
    assert 0 < len(out.chunks) < 200
    assert all(c.text.strip() for c in out.chunks)


def test_no_empty_chunks_and_stable_indices():
    out = run_strategy(_one_para("戊" * 100), {"strategy": "fixed", "target_chars": 40,
                                              "overlap_chars": 10, "max_tokens": 100000})
    assert [c.chunk_index for c in out.chunks] == list(range(len(out.chunks)))
    assert all(c.text.strip() for c in out.chunks)


def test_section_prefix_does_not_fake_canonical_offset():
    out = run_strategy(_one_para("庚辛壬癸甲乙丙丁"), {"strategy": "fixed", "target_chars": 100,
                                                  "overlap_chars": 0, "max_tokens": 100000,
                                                  "prepend_section_title": True})
    c = out.chunks[0]
    assert c.text.startswith("[SECTION] S1\n")
    cs = core_spans(c)
    # 沒有任何 span 涵蓋 prefix；core span 總涵蓋長度 == 核心文字長度（不含 prefix）。
    covered = sum(s.char_end - s.char_start for s in cs)
    assert covered == len("庚辛壬癸甲乙丙丁")
    assert not context_spans(c)  # 未設定 context


def test_deterministic():
    a = run_strategy(_one_para("子" * 90), {"strategy": "fixed", "target_chars": 30, "overlap_chars": 5, "max_tokens": 999})
    b = run_strategy(_one_para("子" * 90), {"strategy": "fixed", "target_chars": 30, "overlap_chars": 5, "max_tokens": 999})
    assert [c.text for c in a.chunks] == [c.text for c in b.chunks]
