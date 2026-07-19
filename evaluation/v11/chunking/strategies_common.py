"""新策略（fixed / paragraph / heading）共用的 canonical 切分原語（v1.1 Slice 3 §九/§十/§十一）。

全部在 **normalized canonical text** 座標上運作：``group_text`` 為某 (page, section) 群組
segments 以 ``"\\n\\n"`` 連接的文字，其 group-local offset 與 canonical char offset 差一個固定
base（``group[0].char_start``），故 provenance 可精確映射（§九.8：只涵蓋實際進入 chunk 的範圍）。
"""
from __future__ import annotations

from app.models.canonical import CanonicalSegment
from evaluation.v11.chunking.base import (
    _JOIN,
    ChunkPlan,
    _SpanContrib,
    segment_span,
)
from evaluation.v11.chunking.config import ChunkingConfig

# 句尾符號（與正式 TextSplitter 一致）。
_SENTENCE_ENDINGS = "。！？；!?;"


# --------------------------------------------------------------------------
# group_text / 座標
# --------------------------------------------------------------------------
def group_text_and_ranges(
    group: list[CanonicalSegment],
) -> tuple[str, list[tuple[int, int, CanonicalSegment]], int]:
    """回傳 ``(group_text, [(local_start, local_end, seg)], base)``。

    ``group_text[local_start:local_end] == seg.normalized_text``；
    canonical offset = ``base + local``。
    """
    base = group[0].char_start
    group_text = _JOIN.join(s.normalized_text for s in group)
    ranges = [(s.char_start - base, s.char_end - base, s) for s in group]
    return group_text, ranges, base


def covered_indices(
    ranges: list[tuple[int, int, CanonicalSegment]], ws: int, we: int
) -> tuple[int, int]:
    """window ``[ws, we)`` 覆蓋到的 segment group-local 索引範圍（first, last）。"""
    idxs = [k for k, (s0, s1, _seg) in enumerate(ranges) if min(we, s1) > max(ws, s0)]
    if not idxs:
        return 0, 0
    return idxs[0], idxs[-1]


def spans_for_window(
    group_text: str,
    ranges: list[tuple[int, int, CanonicalSegment]],
    ws: int,
    we: int,
    base: int,
) -> list[_SpanContrib]:
    """window ``[ws, we)`` 落在各 segment 的**實際覆蓋** canonical 子範圍（§九.8）。"""
    spans: list[_SpanContrib] = []
    for s0, s1, seg in ranges:
        cs = max(ws, s0)
        ce = min(we, s1)
        if ce <= cs:
            continue
        covered = group_text[cs:ce]
        if not covered.strip():
            continue
        spans.append(
            _SpanContrib(
                char_start=base + cs,
                char_end=base + ce,
                page=seg.page,
                section=seg.section,
                paragraph_index=seg.paragraph_index,
                offset_scope=seg.offset_scope,
                text=covered,
            )
        )
    return spans


def context_spans(
    group: list[CanonicalSegment],
    first_idx: int,
    last_idx: int,
    before_n: int,
    after_n: int,
) -> tuple[list[_SpanContrib], list[_SpanContrib]]:
    """核心 segment 範圍 [first, last] 前後的相鄰整段作為 context（不跨 group → 不跨 page/section）。

    before 由**遠到近**、after 由**近到遠**（供 build_chunks 依 §十二.7 由遠端縮減）。
    """
    before = [
        segment_span(group[k]) for k in range(max(0, first_idx - before_n), first_idx)
    ]
    after = [
        segment_span(group[k])
        for k in range(last_idx + 1, min(len(group), last_idx + after_n + 1))
    ]
    return before, after


# --------------------------------------------------------------------------
# fixed-window 切分（邊界優先 + token 硬限制 + 無限迴圈防護）
# --------------------------------------------------------------------------
def _boundary(text: str, i: int, hard_end: int) -> int:
    """在 ``[i, hard_end)`` 內優先於合理邊界切分（§九.4）：段落 > 換行 > 句尾；否則硬切。"""
    seg = text[i:hard_end]
    p = seg.rfind("\n\n")
    if p > 0:
        return i + p + 2
    nl = seg.rfind("\n")
    if nl > 0:
        return i + nl + 1
    for k in range(len(seg) - 1, 0, -1):
        if seg[k] in _SENTENCE_ENDINGS:
            return i + k + 1
    return hard_end


def _shrink_to_tokens(text: str, start: int, end: int, max_tokens: int, count) -> int:
    """把 ``[start, end)`` 縮到 token <= max_tokens 的最大前綴（§九.3 硬限制）。"""
    if count(text[start:end]) <= max_tokens:
        return end
    lo, hi = start + 1, end
    best = start + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(text[start:mid]) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def fixed_windows(
    text: str, target_chars: int, overlap_chars: int, max_tokens: int, count
) -> list[tuple[int, int]]:
    """對單一文字做 fixed-window 切分，回傳 ``[(start, end)]`` group-local 範圍。

    - 依 target_chars 建窗口；優先合理邊界；同時遵守 max_tokens 硬限制。
    - configurable overlap；overlap 不得使起點不前進（§九.6 無限迴圈防護）。
    """
    windows: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    guard = 0
    while i < n:
        guard += 1
        if guard > n + 16:  # 理論上每步至少前進 1 字元；此為額外保險
            break
        hard_end = min(n, i + target_chars)
        end = hard_end if hard_end >= n else _boundary(text, i, hard_end)
        end = _shrink_to_tokens(text, i, end, max_tokens, count)
        if end <= i:
            end = min(n, i + 1)
        windows.append((i, end))
        if end >= n:
            break
        nxt = end - overlap_chars
        if nxt <= i:  # overlap 會使起點不前進 → 放棄本次 overlap，確保前進
            nxt = end
        i = nxt
    return windows


# --------------------------------------------------------------------------
# paragraph-aware 組合（paragraph / heading 共用；§十 / §十一.2）
# --------------------------------------------------------------------------
def paragraph_plans(
    group: list[CanonicalSegment],
    cfg: ChunkingConfig,
    count,
    effective_max: int,
) -> tuple[list[ChunkPlan], list[str]]:
    """在單一 (page, section) 群組內以「完整段落優先」組合 chunk（§十）。

    - 盡量不切斷段落地合併整段，直到超過 target_chars 或 effective_max token。
    - 單一段落超過上限 → 退回安全的 fixed split（記錄 fallback）。
    - overlap 優先重用前一 chunk 尾端的完整段落；重複段落 provenance 仍指向相同 canonical span。
    """
    plans: list[ChunkPlan] = []
    reasons: list[str] = []
    m = len(group)
    i = 0
    guard = 0
    while i < m:
        guard += 1
        if guard > m + 16:
            break
        seg_i = group[i]

        # 單段超過上限 → fixed fallback（§十.3）。
        if len(seg_i.normalized_text) > cfg.target_chars or count(seg_i.normalized_text) > effective_max:
            reasons.append("paragraph:oversized_segment->fixed_split")
            before, after = context_spans(
                group, i, i, cfg.context_before_paragraphs, cfg.context_after_paragraphs
            )
            for ws, we in fixed_windows(
                seg_i.normalized_text, cfg.target_chars, cfg.overlap_chars, effective_max, count
            ):
                core_text = seg_i.normalized_text[ws:we]
                if not core_text.strip():
                    continue
                sc = _SpanContrib(
                    char_start=seg_i.char_start + ws,
                    char_end=seg_i.char_start + we,
                    page=seg_i.page,
                    section=seg_i.section,
                    paragraph_index=seg_i.paragraph_index,
                    offset_scope=seg_i.offset_scope,
                    text=core_text,
                )
                plans.append(
                    ChunkPlan(
                        page=seg_i.page,
                        section=seg_i.section,
                        core_text=core_text,
                        core_spans=[sc],
                        before_context=list(before),
                        after_context=list(after),
                    )
                )
            i += 1
            continue

        # 完整段落優先合併。
        j = i
        combined: list[CanonicalSegment] = []
        while j < m:
            cand = combined + [group[j]]
            text = _JOIN.join(s.normalized_text for s in cand)
            if combined and (len(text) > cfg.target_chars or count(text) > effective_max):
                break
            combined = cand
            j += 1
        last = j - 1
        core_text = _JOIN.join(s.normalized_text for s in combined)
        core_spans = [segment_span(s) for s in combined]
        before, after = context_spans(
            group, i, last, cfg.context_before_paragraphs, cfg.context_after_paragraphs
        )
        plans.append(
            ChunkPlan(
                page=group[i].page,
                section=group[i].section,
                core_text=core_text,
                core_spans=core_spans,
                before_context=before,
                after_context=after,
            )
        )
        if j >= m:
            break
        # overlap：由尾端回退累積完整段落 <= overlap_chars（§十.4/5 deterministic）。
        ov = j
        acc = 0
        while ov > i + 1 and acc < cfg.overlap_chars:
            acc += len(group[ov - 1].normalized_text)
            ov -= 1
        i = ov if ov > i else j
    return plans, reasons
