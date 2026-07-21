"""Versioned Dataset Publisher（Slice 6 §十八／§十九）。

把「Base Dataset ＋ Accepted Candidate」發布成**新的**版本化 Dataset 檔。

不可違反的規則：
- **舊 Dataset 永不覆寫**：Output Path 已存在即拒絕（§十八.1／§十八.2）。
- **Atomic Publish**：先寫 Temp File 再 ``os.replace``；失敗不留半檔（§十八.3）。
- 只發布 ``accepted`` Candidate；``proposed`` / ``rejected`` 一律排除（§十八.5／§十八.6）。
- 發布前重新驗證 Base Hash / Corpus Hash / Session / Candidate / Evidence /
  Group Leakage / Split / Schema（§十八.4）。
- **不**自動修改任何 Search Config、**不**啟動 Optimizer Search、**不**自動 Commit
  （§十八.11～§十八.13）。
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from evaluation.v11 import hashing
from evaluation.v11.case_schema import (
    ALLOWED_SPLITS,
    SCHEMA_VERSION,
    CaseSchemaError,
    EvaluationCaseV11,
    parse_dataset,
)
from evaluation.v11.review.corpus import CanonicalCorpus
from evaluation.v11.review.coverage import build_coverage
from evaluation.v11.review.evidence_annotation import parse_evidence_input, validate_evidence
from evaluation.v11.review.models import (
    CANDIDATE_ACCEPTED,
    SESSION_CANCELLED,
    SESSION_PUBLISHED,
    CandidateRevision,
    QuestionCandidate,
    ReviewSession,
    now_iso,
)


class PublishError(Exception):
    """發布被拒絕（含 eligibility 未通過、輸出已存在、驗證失敗）。"""


@dataclass
class EligibilityResult:
    """Publish Eligibility 檢查結果（§十九）。

    Coverage Gap 只是 warning，不得把 Schema／Evidence／Leakage error 降級。
    """

    eligible: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def load_base_dataset(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def candidate_to_case(
    candidate: QuestionCandidate, revision: CandidateRevision | None
) -> EvaluationCaseV11:
    """把 Accepted Candidate ＋ 其最新 Revision 轉為正式 Case。

    正式值一律取自人工 Revision；Generator 的 ``suggested_*`` 只在人工未覆寫時作為
    起始值，且 ``split`` 必須由人工明確給定（不接受 Generator 指派）。
    """
    if revision is None:
        raise PublishError(
            f"candidate {candidate.candidate_id!r} 沒有任何 revision，無法發布。"
        )
    evidence_raw = json.loads(revision.evidence_json or "[]")
    evidence = [parse_evidence_input(e) for e in evidence_raw]
    return EvaluationCaseV11(
        id=candidate.effective_case_id,
        question=revision.question,
        query_type=revision.query_type,
        answerable=revision.answerable,
        group_id=revision.group_id or "",
        split=revision.split or "",
        expected_evidence=evidence,
        schema_version=SCHEMA_VERSION,
        status="accepted",
    )


def check_eligibility(
    session: ReviewSession,
    candidates: list[QuestionCandidate],
    revisions: dict[str, CandidateRevision | None],
    corpus: CanonicalCorpus,
    base_dataset: dict[str, Any],
    *,
    new_dataset_version: str,
    output_path: str,
) -> EligibilityResult:
    """§十九：全部條件成立才可 ``ready_to_publish``。"""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Session 狀態。
    if session.status == SESSION_PUBLISHED:
        errors.append("session 已 published，不得再次發布。")
    if session.status == SESSION_CANCELLED:
        errors.append("session 已 cancelled，不得發布。")

    # 2. Base Dataset / Corpus Hash 未變（§十九.6／§十九.7）。
    current_base_hash = hashing.dataset_hash(base_dataset)
    if current_base_hash != session.base_dataset_hash:
        errors.append(
            "base dataset hash 已改變（session 建立後 base dataset 被修改），"
            "不得在漂移的基礎上發布。"
        )
    if corpus.corpus_hash != session.corpus_hash:
        errors.append("corpus hash 已改變，不得發布。")

    # 3. 至少一個 Accepted Candidate（§十九.1）。
    accepted = [c for c in candidates if c.status == CANDIDATE_ACCEPTED]
    if not accepted:
        errors.append("沒有任何 accepted candidate，無可發布內容。")

    # 4. Dataset Version 未被使用 / Output Path 不存在（§十九.8／§十九.9）。
    if base_dataset.get("dataset_version") == new_dataset_version:
        errors.append(f"dataset_version {new_dataset_version!r} 與 base 相同，必須是新版本。")
    if os.path.exists(output_path):
        errors.append(f"output path 已存在，拒絕覆寫：{output_path}")

    # 5. 每個 Accepted Candidate 的 Evidence / 人工確認 / split / group。
    for c in accepted:
        rev = revisions.get(c.candidate_id)
        if rev is None:
            errors.append(f"candidate {c.candidate_id!r} 沒有 revision。")
            continue
        if not rev.split or rev.split not in ALLOWED_SPLITS:
            errors.append(
                f"candidate {c.candidate_id!r} 的 split={rev.split!r} 非法；"
                f"必須由人工指定為 {sorted(ALLOWED_SPLITS)} 之一。"
            )
        if not rev.group_id:
            errors.append(f"candidate {c.candidate_id!r} 缺少 group_id。")

        try:
            evidence = [parse_evidence_input(e) for e in json.loads(rev.evidence_json or "[]")]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"candidate {c.candidate_id!r} evidence 無法解析：{exc}")
            continue

        result = validate_evidence(
            evidence,
            corpus,
            answerable=rev.answerable,
            unanswerable_confirmed=c.unanswerable_confirmed,
        )
        if not result.valid:
            for issue in result.issues:
                errors.append(f"candidate {c.candidate_id!r} evidence：{issue.message}")

    # 6. 合併後的 Schema / Group Leakage / Split 驗證（§十九.4／§十九.5）。
    if not errors:
        merged = build_merged_dataset(
            base_dataset, accepted, revisions, session, new_dataset_version
        )
        try:
            parsed = parse_dataset(merged)
        except CaseSchemaError as exc:
            errors.append(f"合併後 dataset schema 驗證失敗：{exc}")
        else:
            coverage = build_coverage(parsed.cases)
            if coverage.summary.group_leakage:
                errors.append(
                    f"group leakage：{coverage.summary.group_leakage} 橫跨多個 split。"
                )
            warnings.extend(f"coverage gap：{g.code} — {g.message}" for g in coverage.gaps)

    return EligibilityResult(eligible=not errors, errors=errors, warnings=warnings)


def build_merged_dataset(
    base_dataset: dict[str, Any],
    accepted: list[QuestionCandidate],
    revisions: dict[str, CandidateRevision | None],
    session: ReviewSession,
    new_dataset_version: str,
    *,
    published_at: str | None = None,
) -> dict[str, Any]:
    """組出新的 Dataset dict（不寫檔）。只包含 Base ＋ Accepted Candidate。"""
    cases = [dict(c) for c in base_dataset.get("cases", [])]
    accepted_ids: list[str] = []
    for c in sorted(accepted, key=lambda x: x.candidate_id):
        case = candidate_to_case(c, revisions.get(c.candidate_id))
        cases.append(case.to_dict())
        accepted_ids.append(c.candidate_id)

    return {
        "schema_version": base_dataset.get("schema_version", SCHEMA_VERSION),
        "dataset_version": new_dataset_version,
        # §十八.9：Notes 保存可追溯的發布來源。
        "notes": {
            "base_dataset_version": session.base_dataset_version,
            "base_dataset_hash": session.base_dataset_hash,
            "review_session_id": session.session_id,
            "accepted_candidate_ids": accepted_ids,
            "corpus_hash": session.corpus_hash,
            "published_at": published_at or now_iso(),
            "previous_notes": base_dataset.get("notes"),
        },
        "cases": cases,
    }


@dataclass
class PublishResult:
    output_path: str
    dataset_version: str
    dataset_hash: str
    base_dataset_hash: str
    accepted_candidate_ids: list[str]
    case_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "base_dataset_hash": self.base_dataset_hash,
            "accepted_candidate_ids": list(self.accepted_candidate_ids),
            "case_count": self.case_count,
            "warnings": list(self.warnings),
        }


def _atomic_write_json(payload: dict[str, Any], output_path: str) -> None:
    """Temp File ＋ Atomic Rename；任何失敗都不留下半個檔案（§十八.3）。"""
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".publish-", suffix=".json.tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    except Exception:
        # 失敗即清除 temp，不留半檔。
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def publish_dataset(
    session: ReviewSession,
    candidates: list[QuestionCandidate],
    revisions: dict[str, CandidateRevision | None],
    corpus: CanonicalCorpus,
    *,
    new_dataset_version: str,
    output_path: str,
) -> PublishResult:
    """執行發布；eligibility 未通過即拋 :class:`PublishError`，不寫任何檔案。"""
    base_dataset = load_base_dataset(session.base_dataset_path)

    eligibility = check_eligibility(
        session,
        candidates,
        revisions,
        corpus,
        base_dataset,
        new_dataset_version=new_dataset_version,
        output_path=output_path,
    )
    if not eligibility.eligible:
        raise PublishError(
            "publish eligibility 未通過：\n- " + "\n- ".join(eligibility.errors)
        )

    accepted = [c for c in candidates if c.status == CANDIDATE_ACCEPTED]
    merged = build_merged_dataset(
        base_dataset, accepted, revisions, session, new_dataset_version
    )
    # 最後一道 schema 驗證（寫檔前）。
    parse_dataset(merged)

    _atomic_write_json(merged, output_path)

    return PublishResult(
        output_path=output_path,
        dataset_version=new_dataset_version,
        dataset_hash=hashing.dataset_hash(merged),
        base_dataset_hash=session.base_dataset_hash,
        accepted_candidate_ids=sorted(c.candidate_id for c in accepted),
        case_count=len(merged["cases"]),
        warnings=list(eligibility.warnings),
    )
