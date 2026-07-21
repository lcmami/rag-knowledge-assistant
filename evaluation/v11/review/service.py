"""ReviewService — Review Workflow 的唯一核心（Slice 6 §六～§十九）。

CLI 與（未來的）任何介面都必須經由本 Service，確保狀態機與稽核規則只有一份實作。

核心不變量：
- 任何 Candidate（LLM 或手動）建立時一律 ``proposed``（§七.1／§七.2）。
- ``accepted`` 只能來自明確的單一人工操作；**沒有批次自動接受**（§十三）。
- 每次 Edit 產生新的、遞增的 Revision，不覆寫歷史（§八.1／§八.5）。
- ``published`` Candidate 與終態 Session 不可修改（§六.5／§七.5）。
- Session 不得讀取 Search Test Metrics（§六.3）——本模組不 import 任何
  search／optimizer 執行結果模組。
- 所有 Write Operation 回傳「修改對象／舊狀態／新狀態／Revision／Validation」（§十三）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from evaluation.v11 import hashing
from evaluation.v11.case_schema import ALLOWED_QUERY_TYPES, ALLOWED_SPLITS, parse_dataset
from evaluation.v11.review.corpus import CanonicalCorpus
from evaluation.v11.review.coverage import CoverageReport, build_coverage
from evaluation.v11.review.evidence_annotation import (
    EvidenceValidationError,
    EvidenceValidationResult,
    parse_evidence_input,
    validate_evidence,
)
from evaluation.v11.review.generator import (
    CandidateQuestionGenerator,
    GenerationRequest,
    GeneratorError,
    SourceSegment,
)
from evaluation.v11.review.models import (
    CANDIDATE_ACCEPTED,
    CANDIDATE_PROPOSED,
    CANDIDATE_PUBLISHED,
    CANDIDATE_REJECTED,
    DEFAULT_EDITOR,
    EVENT_CANDIDATE_ACCEPTED,
    EVENT_CANDIDATE_CREATED,
    EVENT_CANDIDATE_EDITED,
    EVENT_CANDIDATE_PUBLISHED,
    EVENT_CANDIDATE_REJECTED,
    EVENT_EVIDENCE_ADDED,
    EVENT_EVIDENCE_REMOVED,
    EVENT_SESSION_PUBLISHED,
    EVENT_UNANSWERABLE_CONFIRMED,
    SESSION_DRAFT,
    SESSION_PUBLISHED,
    SESSION_READY,
    SESSION_REVIEWING,
    CandidateRevision,
    QuestionCandidate,
    ReviewSession,
    ReviewStateError,
    assert_candidate_transition,
    new_id,
    now_iso,
)
from evaluation.v11.review.publisher import (
    EligibilityResult,
    PublishResult,
    check_eligibility,
    load_base_dataset,
    publish_dataset,
)
from evaluation.v11.review.repository import ReviewRepository


@dataclass
class WriteOutcome:
    """所有 Write Operation 的統一回傳（§十三）。"""

    target: str
    target_id: str
    old_state: dict[str, Any] = field(default_factory=dict)
    new_state: dict[str, Any] = field(default_factory=dict)
    revision: int | None = None
    validation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_id": self.target_id,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "revision": self.revision,
            "validation": self.validation,
        }


class ReviewService:
    """Review Workflow 服務。

    ``corpus`` 為 canonical corpus 快照（只讀）；``repository`` 為獨立 Review DB。
    本 Service **不**建立 Qdrant collection、不呼叫 Embedding、不執行 Optimizer Search。
    """

    def __init__(self, repository: ReviewRepository, corpus: CanonicalCorpus):
        self.repo = repository
        self.corpus = corpus

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def create_session(
        self,
        *,
        session_name: str,
        base_dataset_path: str,
        generator_type: str | None = None,
        generator_model: str | None = None,
        generator_config: dict[str, Any] | None = None,
    ) -> ReviewSession:
        """建立 Session 並固定 Base Dataset / Corpus Hash（§六.1）。"""
        base_raw = load_base_dataset(base_dataset_path)
        parsed = parse_dataset(base_raw)  # base dataset 必須本身合法
        canonical_versions = {d.canonical_text_version for d in self.corpus.documents.values()}
        loader_versions = {d.loader_version for d in self.corpus.documents.values()}

        session = ReviewSession(
            session_id=new_id("rs"),
            session_name=session_name,
            status=SESSION_DRAFT,
            base_dataset_path=base_dataset_path,
            base_dataset_version=parsed.dataset_version,
            base_dataset_hash=hashing.dataset_hash(base_raw),
            corpus_hash=self.corpus.corpus_hash,
            canonical_text_version=sorted(canonical_versions)[0] if canonical_versions else None,
            loader_version=sorted(loader_versions)[0] if loader_versions else None,
            generator_type=generator_type,
            generator_model=generator_model,
            generator_config_json=json.dumps(
                generator_config or {}, ensure_ascii=False, sort_keys=True
            ),
        )
        return self.repo.create_session(session)

    def get_session(self, session_id: str) -> ReviewSession:
        return self.repo.require_session(session_id)

    def list_sessions(self) -> list[ReviewSession]:
        return self.repo.list_sessions()

    def cancel_session(self, session_id: str, *, actor: str = DEFAULT_EDITOR) -> WriteOutcome:
        from evaluation.v11.review.models import SESSION_CANCELLED

        session = self.repo.require_session(session_id)
        session.assert_mutable()
        old = session.status
        session.status = SESSION_CANCELLED
        session.completed_at = now_iso()
        self.repo.update_session(session)
        return WriteOutcome(
            target="session",
            target_id=session_id,
            old_state={"status": old},
            new_state={"status": session.status},
        )

    def _touch_reviewing(self, session: ReviewSession) -> None:
        if session.status == SESSION_DRAFT:
            session.status = SESSION_REVIEWING
            self.repo.update_session(session)

    # ------------------------------------------------------------------
    # Candidate 建立
    # ------------------------------------------------------------------
    def _create_candidate(
        self,
        session: ReviewSession,
        *,
        question: str,
        query_type: str,
        answerable: bool,
        source_document_sha256: str | None,
        source_filename: str | None,
        source_segment_ids: list[str],
        source_page: int | None,
        source_section: str | None,
        suggested_group_id: str | None,
        suggested_split: str | None,
        suggested_anchor_text: list[str],
        generator_metadata: dict[str, Any],
        actor: str,
    ) -> QuestionCandidate:
        if query_type not in ALLOWED_QUERY_TYPES:
            raise ReviewStateError(
                f"非法 query_type={query_type!r}；允許 {sorted(ALLOWED_QUERY_TYPES)}。"
            )
        if not question or not question.strip():
            raise ReviewStateError("問題不得為空。")
        if suggested_split is not None and suggested_split not in ALLOWED_SPLITS:
            raise ReviewStateError(
                f"非法 split={suggested_split!r}；允許 {sorted(ALLOWED_SPLITS)}。"
            )

        candidate = QuestionCandidate(
            candidate_id=new_id("qc"),
            session_id=session.session_id,
            proposed_question=question.strip(),
            proposed_query_type=query_type,
            proposed_answerable=answerable,
            # §七.1／§七.2：一律從 proposed 開始，LLM 不得直接產生 accepted。
            status=CANDIDATE_PROPOSED,
            current_revision=0,
            source_document_sha256=source_document_sha256,
            source_filename=source_filename,
            source_page=source_page,
            source_section=source_section,
            source_segment_ids_json=json.dumps(source_segment_ids, ensure_ascii=False),
            suggested_group_id=suggested_group_id,
            suggested_split=suggested_split,
            suggested_anchor_text_json=json.dumps(suggested_anchor_text, ensure_ascii=False),
            generator_metadata_json=json.dumps(
                generator_metadata, ensure_ascii=False, sort_keys=True
            ),
        )
        self.repo.create_candidate(candidate)
        self.repo.add_event(
            session.session_id,
            EVENT_CANDIDATE_CREATED,
            candidate_id=candidate.candidate_id,
            actor=actor,
            detail={"query_type": query_type, "answerable": answerable},
        )
        return candidate

    def add_candidate_manual(
        self,
        session_id: str,
        *,
        question: str,
        query_type: str,
        answerable: bool,
        source_document_sha256: str | None = None,
        source_filename: str | None = None,
        source_segment_ids: list[str] | None = None,
        suggested_group_id: str | None = None,
        suggested_split: str | None = None,
        actor: str = DEFAULT_EDITOR,
    ) -> WriteOutcome:
        """手動新增 Candidate（§十二）：無 Provider 時系統仍可使用。"""
        session = self.repo.require_session(session_id)
        session.assert_mutable()

        page = section = None
        segment_ids = list(source_segment_ids or [])
        if source_document_sha256 and segment_ids:
            seg = self.corpus.segment(source_document_sha256, segment_ids[0])
            page, section = seg.page, seg.section
        if source_document_sha256 and source_filename is None:
            source_filename = self.corpus.by_sha(source_document_sha256).filename

        candidate = self._create_candidate(
            session,
            question=question,
            query_type=query_type,
            answerable=answerable,
            source_document_sha256=source_document_sha256,
            source_filename=source_filename,
            source_segment_ids=segment_ids,
            source_page=page,
            source_section=section,
            suggested_group_id=suggested_group_id,
            suggested_split=suggested_split,
            suggested_anchor_text=[],
            generator_metadata={"generator": "manual"},
            actor=actor,
        )
        self._touch_reviewing(session)
        return WriteOutcome(
            target="candidate",
            target_id=candidate.candidate_id,
            old_state={},
            new_state={"status": candidate.status},
            revision=candidate.current_revision,
        )

    def generate_candidates(
        self,
        session_id: str,
        generator: CandidateQuestionGenerator,
        *,
        document_sha256: str,
        segment_ids: list[str] | None = None,
        requested_count: int = 3,
        allowed_query_types: tuple[str, ...] | None = None,
        constraints: dict[str, Any] | None = None,
        actor: str = DEFAULT_EDITOR,
    ) -> list[WriteOutcome]:
        """以 Generator 提出候選；全部以 ``proposed`` 落庫。

        Generator 失敗（含 JSON 解析、Provider Timeout）時**不落庫任何 Candidate**
        （§十一.10）：先取得完整、通過驗證的清單，再一次寫入。
        """
        session = self.repo.require_session(session_id)
        session.assert_mutable()

        doc = self.corpus.by_sha(document_sha256)
        wanted = set(segment_ids) if segment_ids else None
        segments = tuple(
            SourceSegment(
                document_sha256=doc.document_sha256,
                filename=doc.filename,
                segment_id=s.segment_id,
                canonical_text_version=doc.canonical_text_version,
                loader_version=doc.loader_version,
                page=s.page,
                section=s.section,
                paragraph_index=s.paragraph_index,
                text=s.normalized_text,
            )
            for s in doc.segments
            if wanted is None or s.segment_id in wanted
        )
        if not segments:
            raise ReviewStateError(f"文件 {doc.filename!r} 沒有符合的 segment。")

        request = GenerationRequest(
            segments=segments,
            allowed_query_types=allowed_query_types or tuple(sorted(ALLOWED_QUERY_TYPES)),
            requested_count=requested_count,
            constraints=constraints or {},
        )
        generated = generator.generate(request)  # 失敗即拋 GeneratorError，尚未寫入任何列
        if not generated:
            raise GeneratorError("Generator 未提出任何候選。")

        outcomes: list[WriteOutcome] = []
        for g in generated:
            candidate = self._create_candidate(
                session,
                question=g.question,
                query_type=g.query_type,
                answerable=g.answerable_suggestion,
                source_document_sha256=g.source_document_sha256,
                source_filename=g.source_filename,
                source_segment_ids=list(g.source_segment_ids),
                source_page=g.source_page,
                source_section=g.source_section,
                suggested_group_id=g.group_id_suggestion,
                suggested_split=None,  # split 一律由人工決定
                suggested_anchor_text=list(g.anchor_text_suggestions),
                generator_metadata=dict(g.generator_metadata),
                actor=actor,
            )
            outcomes.append(
                WriteOutcome(
                    target="candidate",
                    target_id=candidate.candidate_id,
                    old_state={},
                    new_state={"status": candidate.status},
                    revision=candidate.current_revision,
                )
            )
        self._touch_reviewing(session)
        return outcomes

    # ------------------------------------------------------------------
    # Revision / Edit
    # ------------------------------------------------------------------
    def _next_revision(
        self,
        candidate: QuestionCandidate,
        *,
        question: str | None = None,
        query_type: str | None = None,
        answerable: bool | None = None,
        group_id: str | None = None,
        split: str | None = None,
        evidence_json: str | None = None,
        editor: str,
        change_reason: str | None,
    ) -> CandidateRevision:
        previous = self.repo.latest_revision(candidate.candidate_id)
        base_question = previous.question if previous else candidate.proposed_question
        base_qt = previous.query_type if previous else candidate.proposed_query_type
        base_ans = previous.answerable if previous else candidate.proposed_answerable
        base_group = previous.group_id if previous else candidate.suggested_group_id
        base_split = previous.split if previous else candidate.suggested_split
        base_evidence = previous.evidence_json if previous else "[]"
        base_anchor = (
            previous.anchor_text_json if previous else candidate.suggested_anchor_text_json
        )

        revision = CandidateRevision(
            candidate_id=candidate.candidate_id,
            revision_number=candidate.current_revision + 1,  # §八.5：必須遞增
            question=base_question if question is None else question,
            query_type=base_qt if query_type is None else query_type,
            answerable=base_ans if answerable is None else answerable,
            group_id=base_group if group_id is None else group_id,
            split=base_split if split is None else split,
            evidence_json=base_evidence if evidence_json is None else evidence_json,
            anchor_text_json=base_anchor,
            editor=editor,
            change_reason=change_reason,
        )
        self.repo.add_revision(revision)
        candidate.current_revision = revision.revision_number
        self.repo.update_candidate(candidate)
        return revision

    def edit_candidate(
        self,
        candidate_id: str,
        *,
        question: str | None = None,
        query_type: str | None = None,
        answerable: bool | None = None,
        group_id: str | None = None,
        split: str | None = None,
        case_id: str | None = None,
        change_reason: str | None = None,
        actor: str = DEFAULT_EDITOR,
    ) -> WriteOutcome:
        """人工修改 Candidate；必定產生新的 Revision（不覆寫歷史）。"""
        candidate = self.repo.require_candidate(candidate_id)
        candidate.assert_mutable()  # published 不可再 edit
        session = self.repo.require_session(candidate.session_id)
        session.assert_mutable()

        if query_type is not None and query_type not in ALLOWED_QUERY_TYPES:
            raise ReviewStateError(
                f"非法 query_type={query_type!r}；允許 {sorted(ALLOWED_QUERY_TYPES)}。"
            )
        if split is not None and split not in ALLOWED_SPLITS:
            raise ReviewStateError(f"非法 split={split!r}；允許 {sorted(ALLOWED_SPLITS)}。")
        if question is not None and not question.strip():
            raise ReviewStateError("問題不得為空。")

        old_revision = candidate.current_revision
        if case_id is not None:
            candidate.case_id = case_id
        revision = self._next_revision(
            candidate,
            question=question.strip() if question is not None else None,
            query_type=query_type,
            answerable=answerable,
            group_id=group_id,
            split=split,
            editor=actor,
            change_reason=change_reason,
        )
        self.repo.add_event(
            candidate.session_id,
            EVENT_CANDIDATE_EDITED,
            candidate_id=candidate_id,
            revision_number=revision.revision_number,
            actor=actor,
            detail={"change_reason": change_reason},
        )
        return WriteOutcome(
            target="candidate",
            target_id=candidate_id,
            old_state={"revision": old_revision, "status": candidate.status},
            new_state={"revision": revision.revision_number, "status": candidate.status},
            revision=revision.revision_number,
        )

    # ------------------------------------------------------------------
    # Accept / Reject（單筆、明確人工操作）
    # ------------------------------------------------------------------
    def accept_candidate(
        self,
        candidate_id: str,
        *,
        group_id: str | None = None,
        split: str | None = None,
        case_id: str | None = None,
        change_reason: str | None = None,
        actor: str = DEFAULT_EDITOR,
    ) -> WriteOutcome:
        """明確的單筆人工接受（§七.3）。**沒有批次自動接受 API。**"""
        candidate = self.repo.require_candidate(candidate_id)
        candidate.assert_mutable()
        session = self.repo.require_session(candidate.session_id)
        session.assert_mutable()

        old = candidate.status
        assert_candidate_transition(old, CANDIDATE_ACCEPTED)

        if split is not None and split not in ALLOWED_SPLITS:
            raise ReviewStateError(f"非法 split={split!r}；允許 {sorted(ALLOWED_SPLITS)}。")
        if case_id is not None:
            candidate.case_id = case_id

        revision = self._next_revision(
            candidate,
            group_id=group_id,
            split=split,
            editor=actor,
            change_reason=change_reason or "accepted by human reviewer",
        )
        candidate.status = CANDIDATE_ACCEPTED
        self.repo.update_candidate(candidate)
        self.repo.add_event(
            candidate.session_id,
            EVENT_CANDIDATE_ACCEPTED,
            candidate_id=candidate_id,
            revision_number=revision.revision_number,
            actor=actor,
        )
        validation = self.validate_candidate_evidence(candidate_id)
        return WriteOutcome(
            target="candidate",
            target_id=candidate_id,
            old_state={"status": old},
            new_state={"status": candidate.status},
            revision=revision.revision_number,
            validation=validation.to_dict(),
        )

    def reject_candidate(
        self, candidate_id: str, *, reason: str | None = None, actor: str = DEFAULT_EDITOR
    ) -> WriteOutcome:
        candidate = self.repo.require_candidate(candidate_id)
        candidate.assert_mutable()
        session = self.repo.require_session(candidate.session_id)
        session.assert_mutable()

        old = candidate.status
        assert_candidate_transition(old, CANDIDATE_REJECTED)
        candidate.status = CANDIDATE_REJECTED
        self.repo.update_candidate(candidate)
        self.repo.add_event(
            candidate.session_id,
            EVENT_CANDIDATE_REJECTED,
            candidate_id=candidate_id,
            actor=actor,
            detail={"reason": reason},
        )
        return WriteOutcome(
            target="candidate",
            target_id=candidate_id,
            old_state={"status": old},
            new_state={"status": candidate.status},
            revision=candidate.current_revision,
        )

    def confirm_unanswerable(
        self, candidate_id: str, *, actor: str = DEFAULT_EDITOR
    ) -> WriteOutcome:
        """人工確認 Unanswerable（§十五.3）；Generator 的建議不等於確認。"""
        candidate = self.repo.require_candidate(candidate_id)
        candidate.assert_mutable()
        revision = self.repo.latest_revision(candidate_id)
        answerable = revision.answerable if revision else candidate.proposed_answerable
        if answerable:
            raise ReviewStateError(
                f"candidate {candidate_id!r} 目前為 answerable，不可確認為 unanswerable；"
                "請先 edit 將 answerable 設為 false。"
            )
        old = candidate.unanswerable_confirmed
        candidate.unanswerable_confirmed = True
        self.repo.update_candidate(candidate)
        self.repo.add_event(
            candidate.session_id,
            EVENT_UNANSWERABLE_CONFIRMED,
            candidate_id=candidate_id,
            actor=actor,
        )
        return WriteOutcome(
            target="candidate",
            target_id=candidate_id,
            old_state={"unanswerable_confirmed": old},
            new_state={"unanswerable_confirmed": True},
            revision=candidate.current_revision,
        )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    def add_evidence(
        self,
        candidate_id: str,
        evidence: dict[str, Any],
        *,
        actor: str = DEFAULT_EDITOR,
        change_reason: str | None = None,
    ) -> WriteOutcome:
        """新增一個人工確認的 Evidence Span（產生 Revision 與 Audit Event）。"""
        candidate = self.repo.require_candidate(candidate_id)
        candidate.assert_mutable()
        session = self.repo.require_session(candidate.session_id)
        session.assert_mutable()

        span = parse_evidence_input(evidence)
        previous = self.repo.latest_revision(candidate_id)
        spans = json.loads(previous.evidence_json) if previous else []
        spans.append(span.to_dict())

        revision = self._next_revision(
            candidate,
            evidence_json=json.dumps(spans, ensure_ascii=False, sort_keys=True),
            editor=actor,
            change_reason=change_reason or "evidence added",
        )
        self.repo.add_event(
            candidate.session_id,
            EVENT_EVIDENCE_ADDED,
            candidate_id=candidate_id,
            revision_number=revision.revision_number,
            actor=actor,
            detail={"document_sha256": span.document_sha256},
        )
        validation = self.validate_candidate_evidence(candidate_id)
        return WriteOutcome(
            target="candidate_evidence",
            target_id=candidate_id,
            old_state={"evidence_count": len(spans) - 1},
            new_state={"evidence_count": len(spans)},
            revision=revision.revision_number,
            validation=validation.to_dict(),
        )

    def remove_evidence(
        self, candidate_id: str, index: int, *, actor: str = DEFAULT_EDITOR
    ) -> WriteOutcome:
        candidate = self.repo.require_candidate(candidate_id)
        candidate.assert_mutable()
        previous = self.repo.latest_revision(candidate_id)
        spans = json.loads(previous.evidence_json) if previous else []
        if not 0 <= index < len(spans):
            raise EvidenceValidationError(
                f"evidence index {index} 超出範圍（目前 {len(spans)} 筆）。"
            )
        removed = spans.pop(index)
        revision = self._next_revision(
            candidate,
            evidence_json=json.dumps(spans, ensure_ascii=False, sort_keys=True),
            editor=actor,
            change_reason="evidence removed",
        )
        self.repo.add_event(
            candidate.session_id,
            EVENT_EVIDENCE_REMOVED,
            candidate_id=candidate_id,
            revision_number=revision.revision_number,
            actor=actor,
            detail={"document_sha256": removed.get("document_sha256")},
        )
        validation = self.validate_candidate_evidence(candidate_id)
        return WriteOutcome(
            target="candidate_evidence",
            target_id=candidate_id,
            old_state={"evidence_count": len(spans) + 1},
            new_state={"evidence_count": len(spans)},
            revision=revision.revision_number,
            validation=validation.to_dict(),
        )

    def validate_candidate_evidence(self, candidate_id: str) -> EvidenceValidationResult:
        candidate = self.repo.require_candidate(candidate_id)
        revision = self.repo.latest_revision(candidate_id)
        answerable = revision.answerable if revision else candidate.proposed_answerable
        spans_raw = json.loads(revision.evidence_json) if revision else []
        spans = [parse_evidence_input(s) for s in spans_raw]
        return validate_evidence(
            spans,
            self.corpus,
            answerable=answerable,
            unanswerable_confirmed=candidate.unanswerable_confirmed,
        )

    # ------------------------------------------------------------------
    # Coverage / Eligibility / Publish
    # ------------------------------------------------------------------
    def _accepted_with_revisions(
        self, session_id: str
    ) -> tuple[list[QuestionCandidate], dict[str, CandidateRevision | None]]:
        candidates = self.repo.list_candidates(session_id)
        revisions = {
            c.candidate_id: self.repo.latest_revision(c.candidate_id) for c in candidates
        }
        return candidates, revisions

    def coverage(self, session_id: str, *, include_accepted: bool = True) -> CoverageReport:
        """Base Dataset（＋Accepted Candidate）的 Coverage；不修改任何資料。"""
        from evaluation.v11.review.publisher import candidate_to_case

        session = self.repo.require_session(session_id)
        base = parse_dataset(load_base_dataset(session.base_dataset_path))
        cases = list(base.cases)
        if include_accepted:
            candidates, revisions = self._accepted_with_revisions(session_id)
            for c in candidates:
                if c.status != CANDIDATE_ACCEPTED:
                    continue
                try:
                    cases.append(candidate_to_case(c, revisions.get(c.candidate_id)))
                except Exception:  # noqa: BLE001 - 未完成的候選不納入 coverage
                    continue
        return build_coverage(cases)

    def check_publish_eligibility(
        self, session_id: str, *, new_dataset_version: str, output_path: str
    ) -> EligibilityResult:
        session = self.repo.require_session(session_id)
        candidates, revisions = self._accepted_with_revisions(session_id)
        return check_eligibility(
            session,
            candidates,
            revisions,
            self.corpus,
            load_base_dataset(session.base_dataset_path),
            new_dataset_version=new_dataset_version,
            output_path=output_path,
        )

    def validate_session(
        self, session_id: str, *, new_dataset_version: str, output_path: str
    ) -> EligibilityResult:
        """驗證並在通過時把 Session 標記為 ``ready_to_publish``（§十九）。"""
        session = self.repo.require_session(session_id)
        session.assert_mutable()
        result = self.check_publish_eligibility(
            session_id, new_dataset_version=new_dataset_version, output_path=output_path
        )
        if result.eligible:
            session.status = SESSION_READY
            self.repo.update_session(session)
        return result

    def publish(
        self,
        session_id: str,
        *,
        new_dataset_version: str,
        output_path: str,
        actor: str = DEFAULT_EDITOR,
    ) -> PublishResult:
        """發布新的 Versioned Dataset；成功後 Candidate / Session 轉 ``published``。"""
        session = self.repo.require_session(session_id)
        session.assert_mutable()
        candidates, revisions = self._accepted_with_revisions(session_id)

        result = publish_dataset(
            session,
            candidates,
            revisions,
            self.corpus,
            new_dataset_version=new_dataset_version,
            output_path=output_path,
        )

        self.repo.mark_candidates_published(result.accepted_candidate_ids)
        for cid in result.accepted_candidate_ids:
            self.repo.add_event(
                session_id,
                EVENT_CANDIDATE_PUBLISHED,
                candidate_id=cid,
                actor=actor,
                detail={"dataset_version": result.dataset_version},
            )

        session.status = SESSION_PUBLISHED
        session.completed_at = now_iso()
        session.published_dataset_path = result.output_path
        session.published_dataset_version = result.dataset_version
        session.published_dataset_hash = result.dataset_hash
        self.repo.update_session(session)
        self.repo.add_event(
            session_id,
            EVENT_SESSION_PUBLISHED,
            actor=actor,
            detail={
                "dataset_version": result.dataset_version,
                "dataset_hash": result.dataset_hash,
                "case_count": result.case_count,
            },
        )
        return result
