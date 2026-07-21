"""Review Session / Candidate / Revision / Audit Event 資料模型（Slice 6 §六～§八）。

純資料結構與狀態機規則，不依賴 SQLite、LLM 或 Qdrant，便於單元測試。

狀態機（§六／§七）：

- Session：``draft`` → ``reviewing`` → ``ready_to_publish`` → ``published``；
  任一非終態可轉 ``cancelled`` / ``failed``。``published`` / ``cancelled`` 為終態。
- Candidate：``proposed`` → ``accepted`` | ``rejected``；``accepted`` → ``published``。
  ``published`` 為終態（不可再修改）；``rejected`` 不得進入 Dataset。

**LLM 不得直接產生 ``accepted``**：所有 Candidate 建立時一律為 ``proposed``。
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# --------------------------------------------------------------------------
# 狀態常數
# --------------------------------------------------------------------------
SESSION_DRAFT = "draft"
SESSION_REVIEWING = "reviewing"
SESSION_READY = "ready_to_publish"
SESSION_PUBLISHED = "published"
SESSION_CANCELLED = "cancelled"
SESSION_FAILED = "failed"

SESSION_STATUSES = (
    SESSION_DRAFT,
    SESSION_REVIEWING,
    SESSION_READY,
    SESSION_PUBLISHED,
    SESSION_CANCELLED,
    SESSION_FAILED,
)
#: 終態：不得再修改或再次發布。
SESSION_TERMINAL = (SESSION_PUBLISHED, SESSION_CANCELLED)

CANDIDATE_PROPOSED = "proposed"
CANDIDATE_ACCEPTED = "accepted"
CANDIDATE_REJECTED = "rejected"
CANDIDATE_PUBLISHED = "published"

CANDIDATE_STATUSES = (
    CANDIDATE_PROPOSED,
    CANDIDATE_ACCEPTED,
    CANDIDATE_REJECTED,
    CANDIDATE_PUBLISHED,
)

# --------------------------------------------------------------------------
# Audit Event 種類（§八）
# --------------------------------------------------------------------------
EVENT_CANDIDATE_CREATED = "candidate_created"
EVENT_CANDIDATE_EDITED = "candidate_edited"
EVENT_CANDIDATE_ACCEPTED = "candidate_accepted"
EVENT_CANDIDATE_REJECTED = "candidate_rejected"
EVENT_EVIDENCE_ADDED = "evidence_added"
EVENT_EVIDENCE_REMOVED = "evidence_removed"
EVENT_UNANSWERABLE_CONFIRMED = "unanswerable_confirmed"
EVENT_CANDIDATE_PUBLISHED = "candidate_published"
EVENT_SESSION_PUBLISHED = "session_published"

#: Slice 6 不做真實登入；CLI 操作者固定記為此值（§八.3）。
DEFAULT_EDITOR = "local_user"


class ReviewStateError(Exception):
    """違反 Session / Candidate 狀態機規則。"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------
@dataclass
class ReviewSession:
    """一次人工審核工作階段（§六）。

    ``base_dataset_hash`` / ``corpus_hash`` 於建立時固定；發布前會重新計算比對，
    任一改變即拒絕發布（§六.2），避免在漂移的基礎上發布 Dataset。
    """

    session_id: str
    session_name: str
    status: str
    base_dataset_path: str
    base_dataset_version: str
    base_dataset_hash: str
    corpus_hash: str
    canonical_text_version: str | None = None
    loader_version: str | None = None
    generator_type: str | None = None
    generator_model: str | None = None
    generator_config_json: str = "{}"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    completed_at: str | None = None
    error: str | None = None
    published_dataset_path: str | None = None
    published_dataset_version: str | None = None
    published_dataset_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_terminal(self) -> bool:
        return self.status in SESSION_TERMINAL

    def assert_mutable(self) -> None:
        """Published / Cancelled Session 不得再修改（§六.5／§六.6）。"""
        if self.is_terminal:
            raise ReviewStateError(
                f"session {self.session_id!r} 狀態為 {self.status!r}（終態），不得再修改。"
            )


# --------------------------------------------------------------------------
# Candidate
# --------------------------------------------------------------------------
@dataclass
class QuestionCandidate:
    """一個候選問題（§七）。

    Generator 建議的 ``suggested_*`` 欄位只是建議，不是正式值；正式值由人工於
    Revision 中確認（§七.8）。
    """

    candidate_id: str
    session_id: str
    proposed_question: str
    proposed_query_type: str
    proposed_answerable: bool
    status: str = CANDIDATE_PROPOSED
    current_revision: int = 0
    #: 發布為 Case 時使用的 id；未指定時退回 ``candidate_id``（恆唯一、可追溯）。
    case_id: str | None = None
    source_document_sha256: str | None = None
    source_filename: str | None = None
    source_page: int | None = None
    source_section: str | None = None
    source_segment_ids_json: str = "[]"
    suggested_group_id: str | None = None
    suggested_split: str | None = None
    suggested_anchor_text_json: str = "[]"
    generator_metadata_json: str = "{}"
    #: 人工確認 unanswerable（§十五）；answerable 候選恆為 False。
    unanswerable_confirmed: bool = False
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    @property
    def effective_case_id(self) -> str:
        return self.case_id or self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_mutable(self) -> None:
        """``published`` Candidate 不可修改（§七.5）。"""
        if self.status == CANDIDATE_PUBLISHED:
            raise ReviewStateError(
                f"candidate {self.candidate_id!r} 已 published，不得再修改。"
            )


@dataclass
class CandidateRevision:
    """Candidate 的一次修訂快照（§八）；Edit 不得覆寫而不留歷史。"""

    candidate_id: str
    revision_number: int
    question: str
    query_type: str
    answerable: bool
    group_id: str | None = None
    split: str | None = None
    evidence_json: str = "[]"
    anchor_text_json: str = "[]"
    editor: str = DEFAULT_EDITOR
    change_reason: str | None = None
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewEvent:
    """Audit Trail 事件（§八）；只保存必要資料，不保存 API Key。"""

    event_id: str
    session_id: str
    event_type: str
    candidate_id: str | None = None
    revision_number: int | None = None
    actor: str = DEFAULT_EDITOR
    detail_json: str = "{}"
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 狀態轉移規則
# --------------------------------------------------------------------------
#: Candidate 允許的狀態轉移。
_CANDIDATE_TRANSITIONS: dict[str, tuple[str, ...]] = {
    CANDIDATE_PROPOSED: (CANDIDATE_ACCEPTED, CANDIDATE_REJECTED),
    # 人工可改變心意：accepted ↔ rejected，直到 published 為止。
    CANDIDATE_ACCEPTED: (CANDIDATE_REJECTED, CANDIDATE_PUBLISHED),
    CANDIDATE_REJECTED: (CANDIDATE_ACCEPTED,),
    CANDIDATE_PUBLISHED: (),
}


def assert_candidate_transition(current: str, target: str) -> None:
    allowed = _CANDIDATE_TRANSITIONS.get(current, ())
    if target not in allowed:
        raise ReviewStateError(
            f"不允許的 candidate 狀態轉移：{current!r} → {target!r}"
            f"（允許：{list(allowed)}）。"
        )
