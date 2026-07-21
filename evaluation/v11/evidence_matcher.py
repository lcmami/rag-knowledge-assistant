"""EvidenceMatcher（Slice 1 §九）。

判定「一個 Retrieved Chunk」相對「一個 Evaluation Case 的 Gold Evidence Spans」的
relevance level，輸出可解釋的比對理由。**純函式、可單元測試**，不依賴 Qdrant / 模型。

輸入：
- ``evidence_spans``：case 的 Gold Evidence（:class:`EvidenceSpan`）。
- ``provenance_spans``：chunk 的 provenance（:class:`ProvenanceSpan`，或具相同屬性者）。
- ``chunk_text``：chunk 文字（供 anchor fallback）。

輸出 :class:`MatchResult`：
- ``relevance_level``：2 / 1 / 0。
- ``matched_evidence_ids``：達 level>=1 的 evidence id。
- ``match_reasons``：命中的比對理由集合。
- ``coverage``：最佳 evidence 的 offset overlap 覆蓋率（無 offset 時為 anchor 覆蓋率）。
- ``version_compatible``：是否有 canonical_text_version 不符（不符不得靜默視為有效 offset）。

relevance 概念（§九）：
- Level 2：offset overlap 覆蓋率達門檻（version 相容）；或 offset 不可用時 anchor 全命中
  且 section/page 相符（強 evidence fallback）。
- Level 1：同正確 page/section 或 anchor 命中，提供上下文但未完整覆蓋。
- Level 0：不相關 / 錯誤文件 / 無有效 evidence。

offset coverage threshold 為**可配置參數**，測試須明確傳入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# 預設 offset 覆蓋門檻；呼叫端 / 測試可覆寫，不寫死為不可改常數。
DEFAULT_OFFSET_COVERAGE_THRESHOLD = 0.5

RELEVANCE_IRRELEVANT = 0
RELEVANCE_CONTEXT = 1
RELEVANCE_COVERS = 2


@dataclass
class MatchResult:
    relevance_level: int
    matched_evidence_ids: list[str] = field(default_factory=list)
    match_reasons: list[str] = field(default_factory=list)
    coverage: float = 0.0
    version_compatible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance_level": self.relevance_level,
            "matched_evidence_ids": list(self.matched_evidence_ids),
            "match_reasons": list(self.match_reasons),
            "coverage": self.coverage,
            "version_compatible": self.version_compatible,
        }


def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


class EvidenceMatcher:
    def __init__(
        self,
        offset_coverage_threshold: float = DEFAULT_OFFSET_COVERAGE_THRESHOLD,
    ):
        if not 0.0 < offset_coverage_threshold <= 1.0:
            raise ValueError("offset_coverage_threshold 必須落在 (0, 1]。")
        self.offset_coverage_threshold = offset_coverage_threshold

    # --- 單一 evidence vs chunk 的比對 ---
    def _match_one(
        self, evidence, provenance_spans: list, chunk_text: str
    ) -> tuple[int, list[str], float, bool]:
        reasons: list[str] = []

        same_doc = [
            p for p in provenance_spans if p.document_sha256 == evidence.document_sha256
        ]
        if not same_doc:
            # 錯誤文件 → Level 0
            return RELEVANCE_IRRELEVANT, ["wrong_document"], 0.0, True

        # version 相容性：同文件的 provenance 是否有相符 canonical_text_version。
        version_ok = any(
            p.canonical_text_version == evidence.canonical_text_version for p in same_doc
        )
        version_compatible = version_ok
        if not version_ok:
            reasons.append("version_mismatch")

        # --- offset overlap（需 version 相容且雙方皆有 offset、scope 相符）---
        best_coverage = 0.0
        ev_has_offset = (
            getattr(evidence, "char_start", None) is not None
            and getattr(evidence, "char_end", None) is not None
        )
        if version_ok and ev_has_offset:
            ev_len = max(1, evidence.char_end - evidence.char_start)
            for p in same_doc:
                if p.char_start is None or p.char_end is None:
                    continue
                if evidence.offset_scope is not None and p.offset_scope != evidence.offset_scope:
                    continue
                # page scope 下需同一頁才可比 offset
                if evidence.offset_scope == "page" and evidence.page is not None:
                    if p.page != evidence.page:
                        continue
                ov = _overlap_len(
                    evidence.char_start, evidence.char_end, p.char_start, p.char_end
                )
                if ov > 0:
                    best_coverage = max(best_coverage, ov / ev_len)

        # --- section / page match（metadata，不依賴 canonical version）---
        page_match = evidence.page is not None and any(
            p.page == evidence.page for p in same_doc
        )
        section_match = evidence.section is not None and any(
            p.section == evidence.section for p in same_doc
        )

        # --- anchor fallback（§九 10：無 offset 時使用 anchor）---
        anchors = list(getattr(evidence, "anchor_text", []) or [])
        matched_anchors = [a for a in anchors if a and a in chunk_text]
        anchor_coverage = (len(matched_anchors) / len(anchors)) if anchors else 0.0
        anchor_all = bool(anchors) and len(matched_anchors) == len(anchors)
        anchor_any = bool(matched_anchors)

        # --- 判級 ---
        # Level 2：version 相容且 offset 覆蓋達門檻
        if version_ok and best_coverage >= self.offset_coverage_threshold:
            reasons.append("offset_overlap")
            if section_match:
                reasons.append("section_match")
            if page_match:
                reasons.append("page_match")
            return RELEVANCE_COVERS, reasons, best_coverage, version_compatible

        # Level 2（fallback）：offset 不可用（無 offset 或 version 不符）時，
        # anchor 全命中且 section/page 相符 → 視為強 evidence
        offset_unusable = (not ev_has_offset) or (not version_ok)
        if offset_unusable and anchor_all and (section_match or page_match):
            reasons.append("anchor_match")
            if section_match:
                reasons.append("section_match")
            if page_match:
                reasons.append("page_match")
            return RELEVANCE_COVERS, reasons, anchor_coverage, version_compatible

        # Level 1：同正確 section / page，或 anchor 命中，提供上下文
        if section_match or page_match:
            if section_match:
                reasons.append("section_match")
            if page_match:
                reasons.append("page_match")
            if anchor_any:
                reasons.append("anchor_match")
            cov = best_coverage if best_coverage > 0 else anchor_coverage
            return RELEVANCE_CONTEXT, reasons, cov, version_compatible
        if anchor_any:
            reasons.append("anchor_match")
            return RELEVANCE_CONTEXT, reasons, anchor_coverage, version_compatible

        # 有部分 offset overlap 但未達門檻 → 仍算上下文
        if best_coverage > 0:
            reasons.append("partial_offset_overlap")
            return RELEVANCE_CONTEXT, reasons, best_coverage, version_compatible

        return RELEVANCE_IRRELEVANT, reasons or ["no_match"], 0.0, version_compatible

    def match(
        self,
        evidence_spans: Iterable,
        provenance_spans: Iterable,
        chunk_text: str = "",
    ) -> MatchResult:
        """比對 chunk 對整個 case 的多個 evidence span（§九 7 multiple evidence）。"""
        evidence_list = list(evidence_spans)
        prov_list = list(provenance_spans)

        if not evidence_list:
            return MatchResult(relevance_level=RELEVANCE_IRRELEVANT)

        best_level = RELEVANCE_IRRELEVANT
        best_coverage = 0.0
        matched_ids: list[str] = []
        reasons: set[str] = set()
        version_compatible = True

        for idx, ev in enumerate(evidence_list):
            level, ev_reasons, coverage, ver_ok = self._match_one(
                ev, prov_list, chunk_text
            )
            reasons.update(ev_reasons)
            if not ver_ok:
                version_compatible = False
            if level >= RELEVANCE_CONTEXT:
                ev_id = getattr(ev, "evidence_id", None) or f"evidence-{idx}"
                matched_ids.append(ev_id)
            if level > best_level or (level == best_level and coverage > best_coverage):
                best_level = level
                best_coverage = coverage

        return MatchResult(
            relevance_level=best_level,
            matched_evidence_ids=matched_ids,
            match_reasons=sorted(reasons),
            coverage=best_coverage,
            version_compatible=version_compatible,
        )
