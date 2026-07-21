"""Canonical Evidence Annotation 與驗證（Slice 6 §十四／§十五）。

Evidence 綁**原始文件的 canonical 座標**（document + page/section + offset + anchor），
不綁 chunk id。本模組負責：

1. 驗證 Offset 真的指向 Canonical Text（且不超出所屬 Canonical Segment）。
2. 驗證 Anchor 確實存在於 Evidence 範圍或對應 Segment。
3. 驗證 ``canonical_text_version`` / ``loader_version`` 與 corpus 相容。
4. 支援同題多 Evidence Span、多文件來源。
5. 提供**定位輔助**文字搜尋：只回傳候選位置，**不自動偽造 Offset**，
   仍需人工確認（§十四.10～§十四.12）。

Unanswerable 規則（§十五）：``answerable=false`` 不得附 Expected Evidence，且必須有
人工 ``unanswerable_confirmed``；Generator 的建議不等於確認。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.v11.case_schema import ALLOWED_OFFSET_SCOPES, EvidenceSpan
from evaluation.v11.review.corpus import CanonicalCorpus, CorpusError

#: Report 只顯示短 Snippet，不顯示全文（§十四.13）。
SNIPPET_MAX_CHARS = 120


class EvidenceValidationError(Exception):
    """Evidence 標註不合法。"""


@dataclass
class EvidenceIssue:
    code: str
    message: str
    evidence_index: int | None = None


@dataclass
class EvidenceValidationResult:
    valid: bool
    issues: list[EvidenceIssue] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [
                {"code": i.code, "message": i.message, "evidence_index": i.evidence_index}
                for i in self.issues
            ],
            "snippets": list(self.snippets),
        }


def _snippet(text: str) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= SNIPPET_MAX_CHARS:
        return one_line
    return one_line[:SNIPPET_MAX_CHARS] + "…"


def parse_evidence_input(raw: dict[str, Any]) -> EvidenceSpan:
    """由 CLI / API 的 dict 建立 :class:`EvidenceSpan`（僅結構檢查，不做 corpus 對位）。"""
    if not isinstance(raw, dict):
        raise EvidenceValidationError("evidence 必須是物件。")
    sha = raw.get("document_sha256")
    if not isinstance(sha, str) or not sha.strip():
        raise EvidenceValidationError("evidence 缺少 document_sha256。")
    ctv = raw.get("canonical_text_version")
    if not isinstance(ctv, str) or not ctv.strip():
        raise EvidenceValidationError("evidence 缺少 canonical_text_version。")
    scope = raw.get("offset_scope")
    if scope not in ALLOWED_OFFSET_SCOPES:
        raise EvidenceValidationError(
            f"非法 offset_scope={scope!r}；允許 page / document 或省略。"
        )
    anchors = raw.get("anchor_text", []) or []
    if not isinstance(anchors, list) or not all(isinstance(a, str) for a in anchors):
        raise EvidenceValidationError("anchor_text 必須是字串陣列。")
    return EvidenceSpan(
        document_sha256=sha,
        canonical_text_version=ctv,
        loader_version=raw.get("loader_version"),
        filename=raw.get("filename"),
        page=raw.get("page"),
        section=raw.get("section"),
        paragraph_index=raw.get("paragraph_index"),
        offset_scope=scope,
        char_start=raw.get("char_start"),
        char_end=raw.get("char_end"),
        anchor_text=list(anchors),
        evidence_id=raw.get("evidence_id"),
    )


def _validate_one(
    evidence: EvidenceSpan, corpus: CanonicalCorpus, index: int
) -> tuple[list[EvidenceIssue], str | None]:
    issues: list[EvidenceIssue] = []

    # 1. 來源文件必須存在於 corpus（§十四.6）。
    if not corpus.has_sha(evidence.document_sha256):
        issues.append(
            EvidenceIssue(
                "unknown_document_sha256",
                f"evidence[{index}] 的 document_sha256 不存在於 corpus。",
                index,
            )
        )
        return issues, None

    doc = corpus.by_sha(evidence.document_sha256)

    # 2. 版本相容（§十四.5）。
    if evidence.canonical_text_version != doc.canonical_text_version:
        issues.append(
            EvidenceIssue(
                "canonical_version_mismatch",
                f"evidence[{index}] canonical_text_version="
                f"{evidence.canonical_text_version!r} 與 corpus {doc.canonical_text_version!r} 不符。",
                index,
            )
        )
    if evidence.loader_version is not None and evidence.loader_version != doc.loader_version:
        issues.append(
            EvidenceIssue(
                "loader_version_mismatch",
                f"evidence[{index}] loader_version={evidence.loader_version!r} "
                f"與 corpus {doc.loader_version!r} 不符。",
                index,
            )
        )
    if evidence.filename is not None and evidence.filename != doc.filename:
        issues.append(
            EvidenceIssue(
                "filename_mismatch",
                f"evidence[{index}] filename={evidence.filename!r} 與 corpus "
                f"{doc.filename!r} 不符。",
                index,
            )
        )

    # 3. Offset 基本規則（§十四.7／§十四.8）。
    cs, ce = evidence.char_start, evidence.char_end
    if cs is None or ce is None:
        issues.append(
            EvidenceIssue(
                "missing_offset",
                f"evidence[{index}] 必須提供 char_start / char_end（人工確認的 offset）。",
                index,
            )
        )
        return issues, None
    if cs < 0:
        issues.append(EvidenceIssue("negative_offset", f"evidence[{index}] char_start < 0。", index))
    if ce <= cs:
        issues.append(
            EvidenceIssue(
                "empty_or_inverted_range",
                f"evidence[{index}] 需滿足 char_end > char_start。",
                index,
            )
        )
    if issues and any(i.code in ("negative_offset", "empty_or_inverted_range") for i in issues):
        return issues, None

    # 4. Offset 必須落在所屬 Canonical Segment 之內（§十四.1／§十四.9）。
    segments = [
        s
        for s in doc.segments
        if (evidence.page is None or s.page == evidence.page)
    ]
    if evidence.offset_scope is not None:
        segments = [s for s in segments if s.offset_scope == evidence.offset_scope]
    covering = [s for s in segments if s.char_start <= cs and ce <= s.char_end]
    if not covering:
        issues.append(
            EvidenceIssue(
                "offset_outside_segment",
                f"evidence[{index}] 範圍 [{cs}, {ce}) 未完整落在任一 Canonical Segment 內"
                f"（page={evidence.page!r}, scope={evidence.offset_scope!r}）。",
                index,
            )
        )
        return issues, None

    segment = covering[0]
    if evidence.section is not None and segment.section != evidence.section:
        issues.append(
            EvidenceIssue(
                "section_mismatch",
                f"evidence[{index}] section={evidence.section!r} 與 segment "
                f"{segment.section!r} 不符。",
                index,
            )
        )
    if (
        evidence.paragraph_index is not None
        and segment.paragraph_index != evidence.paragraph_index
    ):
        issues.append(
            EvidenceIssue(
                "paragraph_index_mismatch",
                f"evidence[{index}] paragraph_index={evidence.paragraph_index} 與 segment "
                f"{segment.paragraph_index} 不符。",
                index,
            )
        )

    # 5. Anchor 必須存在於 Evidence 範圍或對應 Segment（§十四.2）。
    local_start = cs - segment.char_start
    local_end = ce - segment.char_start
    evidence_text = segment.normalized_text[local_start:local_end]
    if not evidence.anchor_text:
        issues.append(
            EvidenceIssue(
                "missing_anchor",
                f"evidence[{index}] 缺少 anchor_text（answerable case 的必要 fallback）。",
                index,
            )
        )
    for anchor in evidence.anchor_text:
        if anchor and anchor not in evidence_text and anchor not in segment.normalized_text:
            issues.append(
                EvidenceIssue(
                    "anchor_not_found",
                    f"evidence[{index}] anchor {anchor!r} 不存在於 evidence 範圍或對應 segment。",
                    index,
                )
            )

    return issues, _snippet(evidence_text)


def validate_evidence(
    evidence_spans: list[EvidenceSpan],
    corpus: CanonicalCorpus,
    *,
    answerable: bool,
    unanswerable_confirmed: bool = False,
) -> EvidenceValidationResult:
    """驗證一個 Case 的所有 Evidence Span（支援多 Span、多文件）。"""
    issues: list[EvidenceIssue] = []
    snippets: list[str] = []

    if not answerable:
        # §十五.1：unanswerable 不得附 Expected Evidence。
        if evidence_spans:
            issues.append(
                EvidenceIssue(
                    "unanswerable_with_evidence",
                    "answerable=false 不得附 expected evidence。",
                )
            )
        # §十五.2／§十五.3：必須經人工確認。
        if not unanswerable_confirmed:
            issues.append(
                EvidenceIssue(
                    "unanswerable_not_confirmed",
                    "answerable=false 必須經人工確認（unanswerable_confirmed=true）；"
                    "Generator 的建議不等於確認。",
                )
            )
        return EvidenceValidationResult(valid=not issues, issues=issues)

    # answerable：至少一個 Evidence Span。
    if not evidence_spans:
        issues.append(
            EvidenceIssue(
                "answerable_without_evidence",
                "answerable=true 至少需要一個經人工確認的 evidence span。",
            )
        )
        return EvidenceValidationResult(valid=False, issues=issues)

    for i, ev in enumerate(evidence_spans):
        one_issues, snippet = _validate_one(ev, corpus, i)
        issues.extend(one_issues)
        if snippet:
            snippets.append(snippet)

    return EvidenceValidationResult(valid=not issues, issues=issues, snippets=snippets)


# --------------------------------------------------------------------------
# 定位輔助（只提供候選，不自動採用）
# --------------------------------------------------------------------------
@dataclass
class AnchorMatch:
    """Anchor 文字搜尋的候選位置；**需人工確認**後才可作為 Evidence（§十四.12）。"""

    document_sha256: str
    filename: str
    segment_id: str
    page: int | None
    section: str | None
    paragraph_index: int
    offset_scope: str
    char_start: int
    char_end: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_sha256": self.document_sha256,
            "filename": self.filename,
            "segment_id": self.segment_id,
            "page": self.page,
            "section": self.section,
            "paragraph_index": self.paragraph_index,
            "offset_scope": self.offset_scope,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "snippet": self.snippet,
            "confirmed": False,  # 搜尋結果一律未確認
        }


def find_anchor_candidates(
    corpus: CanonicalCorpus,
    anchor: str,
    *,
    document_sha256: str | None = None,
    limit: int = 20,
) -> list[AnchorMatch]:
    """在 canonical text 中搜尋 anchor，回傳候選位置（不修改任何狀態）。"""
    if not anchor:
        raise EvidenceValidationError("anchor 不得為空。")

    if document_sha256 is not None:
        try:
            docs = [corpus.by_sha(document_sha256)]
        except CorpusError as exc:
            raise EvidenceValidationError(str(exc)) from exc
    else:
        docs = sorted(corpus.documents.values(), key=lambda d: d.filename)

    matches: list[AnchorMatch] = []
    for doc in docs:
        for seg in doc.segments:
            start = seg.normalized_text.find(anchor)
            while start != -1 and len(matches) < limit:
                end = start + len(anchor)
                matches.append(
                    AnchorMatch(
                        document_sha256=doc.document_sha256,
                        filename=doc.filename,
                        segment_id=seg.segment_id,
                        page=seg.page,
                        section=seg.section,
                        paragraph_index=seg.paragraph_index,
                        offset_scope=seg.offset_scope,
                        char_start=seg.char_start + start,
                        char_end=seg.char_start + end,
                        snippet=_snippet(seg.normalized_text[start:end]),
                    )
                )
                start = seg.normalized_text.find(anchor, start + 1)
            if len(matches) >= limit:
                return matches
    return matches
