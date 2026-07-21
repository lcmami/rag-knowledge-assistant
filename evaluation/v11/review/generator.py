"""Candidate Question Generator 介面與 Fake 實作（Slice 6 §十）。

Generator 的**唯一**輸入是 Canonical Source 與生成限制；它**不得**取得 Test
Metrics、Recommendation Result、Candidate Ranking 或 Production Secrets（§十）。
這個邊界由 :class:`GenerationRequest` 的欄位集合強制：請求物件本身沒有任何
承載評測結果的欄位，因此 Generator 在型別上就拿不到這些資訊。

Generator 只「提出」候選；產出的 :class:`GeneratedQuestionCandidate` 不含狀態，
一律由 Service 以 ``proposed`` 落庫（§七.1／§十一.14）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from evaluation.v11.case_schema import ALLOWED_QUERY_TYPES


class GeneratorError(Exception):
    """Candidate 生成失敗（含 JSON 解析失敗、Provider 失敗／Timeout）。"""


@dataclass(frozen=True)
class SourceSegment:
    """提供給 Generator 的單一 Canonical Segment（唯讀）。"""

    document_sha256: str
    filename: str
    segment_id: str
    canonical_text_version: str
    loader_version: str | None
    page: int | None
    section: str | None
    paragraph_index: int
    text: str


@dataclass(frozen=True)
class GenerationRequest:
    """Generator 的完整輸入（§十）。

    刻意**只**包含 Canonical Source 與生成限制；沒有 metrics / recommendation /
    ranking 欄位，Generator 因此無從得知任何評測結果。
    """

    segments: tuple[SourceSegment, ...]
    allowed_query_types: tuple[str, ...]
    requested_count: int
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.requested_count < 1:
            raise GeneratorError("requested_count 必須 >= 1。")
        if not self.segments:
            raise GeneratorError("至少需要一個 Canonical Segment 作為來源。")
        unknown = set(self.allowed_query_types) - ALLOWED_QUERY_TYPES
        if unknown:
            raise GeneratorError(
                f"不支援的 query_type：{sorted(unknown)}；允許 {sorted(ALLOWED_QUERY_TYPES)}。"
            )


@dataclass
class GeneratedQuestionCandidate:
    """Generator 提出的候選（尚未落庫，也還不是 Case）。

    ``*_suggestion`` 皆為建議值，不是正式值：正式 group／split／answerable／
    evidence 一律由人工於 Review 流程確認（§七.8／§十五.4）。
    Generator **不得**產生正式 offset，也不得宣稱 Evidence 已驗證（§十一.6／§十一.7）。
    """

    question: str
    query_type: str
    answerable_suggestion: bool
    source_document_sha256: str
    source_filename: str
    source_segment_ids: list[str] = field(default_factory=list)
    source_page: int | None = None
    source_section: str | None = None
    anchor_text_suggestions: list[str] = field(default_factory=list)
    group_id_suggestion: str | None = None
    split_suggestion: str | None = None
    rationale: str | None = None
    generator_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.question or not self.question.strip():
            raise GeneratorError("候選問題不得為空。")
        if self.query_type not in ALLOWED_QUERY_TYPES:
            raise GeneratorError(
                f"非法 query_type={self.query_type!r}；允許 {sorted(ALLOWED_QUERY_TYPES)}。"
            )
        if not isinstance(self.answerable_suggestion, bool):
            raise GeneratorError("answerable_suggestion 必須是布林值。")
        if not self.source_document_sha256:
            raise GeneratorError("候選必須保存來源文件 sha256，不能只保存問題文字。")


class CandidateQuestionGenerator(Protocol):
    """Candidate Generator 介面。"""

    #: 供 Session 記錄（例如 ``fake`` / ``llm``）。
    generator_type: str

    def generate(self, request: GenerationRequest) -> list[GeneratedQuestionCandidate]:
        """依 Canonical Segment 提出候選問題；不得回傳已 accept 的 Case。"""
        ...


class FakeCandidateGenerator:
    """Deterministic Fake Generator（測試與離線 Smoke 用；不呼叫任何 Provider）。

    對相同 :class:`GenerationRequest` 恆產生相同輸出，便於重現與斷言。
    """

    generator_type = "fake"

    def __init__(self, model: str = "fake-model-v1"):
        self.model = model

    def generate(self, request: GenerationRequest) -> list[GeneratedQuestionCandidate]:
        out: list[GeneratedQuestionCandidate] = []
        segments = list(request.segments)
        for i in range(request.requested_count):
            seg = segments[i % len(segments)]
            query_type = request.allowed_query_types[i % len(request.allowed_query_types)]
            answerable = query_type != "unanswerable"
            snippet = seg.text.strip().replace("\n", " ")[:24]
            out.append(
                GeneratedQuestionCandidate(
                    question=f"[fake-{i + 1}] 關於「{snippet}」的問題？",
                    query_type=query_type,
                    answerable_suggestion=answerable,
                    source_document_sha256=seg.document_sha256,
                    source_filename=seg.filename,
                    source_segment_ids=[seg.segment_id],
                    source_page=seg.page,
                    source_section=seg.section,
                    anchor_text_suggestions=[snippet] if snippet else [],
                    group_id_suggestion=f"fake-group-{i + 1}",
                    split_suggestion=None,
                    rationale="fake deterministic generator",
                    generator_metadata={"generator": self.generator_type, "model": self.model},
                )
            )
        for c in out:
            c.validate()
        return out
