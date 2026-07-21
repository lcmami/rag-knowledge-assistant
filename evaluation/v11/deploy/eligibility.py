"""Apply Eligibility Guard（Slice 7 §八 / §九 / §二十一）。

獨立、可單元測試的純判定：從**已持久化、已 Lock** 的 Search Outcome + Candidate + 目前環境 +
Apply Request 決定 candidate 是否可套用。不信任 request 內任何 ``apply_eligible=true``（§二十一.9）——
eligibility 一律由 DB 中的 locked outcome 重新讀取。

回傳 :class:`~evaluation.v11.deploy.models.ApplyEligibilityDecision`（eligible / reasons /
warnings / validated_hashes / candidate_snapshot），**不得**只回傳單一 Boolean（§八）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.v11.deploy.confirmation import apply_confirmation, confirmation_matches
from evaluation.v11.deploy.models import ApplyEligibilityDecision, ErrorCode
from evaluation.v11.search.search_models import (
    CandidateStageStatus,
    OptimizerSearch,
    SearchCandidate,
    SearchStatus,
)


# -- Guard reason codes（穩定字串；§二十四.1 各項對應）--
class Reason:
    RECOMMENDATION_NOT_LOCKED = "recommendation_not_locked"
    CANDIDATE_NOT_RECOMMENDATION = "candidate_not_recommendation"
    RECOMMENDED_CONFIG_HASH_MISMATCH = "recommended_config_hash_mismatch"
    NOT_APPLY_ELIGIBLE = "recommendation_not_apply_eligible"
    TEST_CONSTRAINT_NOT_PASSED = "test_constraint_not_passed"
    WORKING_TREE_DIRTY = "working_tree_dirty"
    SEARCH_NOT_DEPLOYABLE = "search_not_deployable_status"
    CANDIDATE_STAGE_INCOMPLETE = "candidate_stage_incomplete"
    CANDIDATE_CONFIG_HASH_MISMATCH = "candidate_config_hash_mismatch"
    DATASET_HASH_MISMATCH = "dataset_hash_mismatch"
    CORPUS_HASH_MISMATCH = "corpus_hash_mismatch"
    CODE_COMMIT_MISMATCH = "code_commit_mismatch"
    CANONICAL_VERSION_MISMATCH = "canonical_text_version_mismatch"
    IDENTIFIER_VERSION_MISMATCH = "identifier_normalization_version_mismatch"
    EMBEDDING_FINGERPRINT_MISMATCH = "embedding_fingerprint_mismatch"
    STALE_CANDIDATE = "stale_candidate"
    STALE_ALIAS_TARGET = "stale_alias_target"
    CONFIRMATION_MISMATCH = "confirmation_mismatch"
    ADMIN_DISABLED = "admin_disabled"
    DEMO_CORPUS_TARGET_MISMATCH = "demo_corpus_target_mismatch"


#: reason → 對外穩定 error code（§二十.8）。未列者一律 apply_rejected。
_REASON_TO_ERROR_CODE = {
    Reason.STALE_ALIAS_TARGET: ErrorCode.STALE_ALIAS_TARGET,
    Reason.STALE_CANDIDATE: ErrorCode.STALE_CANDIDATE,
    Reason.CORPUS_HASH_MISMATCH: ErrorCode.CORPUS_HASH_MISMATCH,
    Reason.DEMO_CORPUS_TARGET_MISMATCH: ErrorCode.CORPUS_HASH_MISMATCH,
    Reason.CONFIRMATION_MISMATCH: ErrorCode.CONFIRMATION_MISMATCH,
    Reason.ADMIN_DISABLED: ErrorCode.OPTIMIZER_ADMIN_DISABLED,
}


def primary_error_code(reasons: list[str]) -> str:
    """由 reasons 推導對外 error code：admin_disabled 最優先，其餘取第一個有映射者，否則 apply_rejected。"""
    if Reason.ADMIN_DISABLED in reasons:
        return ErrorCode.OPTIMIZER_ADMIN_DISABLED
    for r in reasons:
        if r in _REASON_TO_ERROR_CODE:
            return _REASON_TO_ERROR_CODE[r]
    return ErrorCode.APPLY_REJECTED


@dataclass
class CurrentEnvironment:
    """Apply 當下的真實環境（由 service 於 Apply 前即時取得，§二十一）。"""

    admin_enabled: bool
    code_commit: Optional[str]
    working_tree_dirty: Optional[bool]
    target_corpus_hash: Optional[str]
    canonical_text_version: Optional[str]
    identifier_normalization_version: Optional[str]
    embedding_fingerprint_hash: Optional[str]
    alias_current_target: Optional[str]


@dataclass
class ApplyRequestSnapshot:
    """使用者提出 Apply 時所帶的 expected snapshot（僅供防 stale / TOCTOU，§二十一.3）。"""

    candidate_id: str
    search_id: str
    alias_name: str
    expected_config_hash: str
    expected_dataset_hash: str
    expected_corpus_hash: str
    expected_code_commit: str
    expected_current_collection: str
    confirmation: str
    idempotency_key: str
    expected_canonical_text_version: Optional[str] = None
    expected_identifier_normalization_version: Optional[str] = None
    expected_embedding_fingerprint_hash: Optional[str] = None
    expected_recommendation_locked_at: Optional[str] = None


def _outcome_dict(search: OptimizerSearch) -> dict[str, Any]:
    if not search.outcome_json:
        return {}
    try:
        return json.loads(search.outcome_json)
    except (ValueError, TypeError):
        return {}


def evaluate_apply_eligibility(
    *,
    search: Optional[OptimizerSearch],
    candidate: Optional[SearchCandidate],
    environment: CurrentEnvironment,
    request: ApplyRequestSnapshot,
) -> ApplyEligibilityDecision:
    """核心判定（§八 二十條拒絕條件）。任一不符即記錄 reason；不做短路，盡量回報全部原因。"""
    reasons: list[str] = []
    warnings: list[str] = []
    validated: dict[str, Any] = {}

    # §八.19：Admin Write API 未啟用（防禦性；主要 gate 在 router）。
    if not environment.admin_enabled:
        reasons.append(Reason.ADMIN_DISABLED)

    if search is None or candidate is None:
        # 找不到來源即無法驗證任何 hash；直接回報 stale / not found 類。
        reasons.append(Reason.STALE_CANDIDATE)
        return ApplyEligibilityDecision(
            eligible=False,
            reasons=_dedup(reasons),
            warnings=warnings,
            validated_hashes=validated,
            candidate_snapshot={
                "search_id": request.search_id,
                "candidate_id": request.candidate_id,
                "found": False,
            },
        )

    outcome = _outcome_dict(search)

    # §八.1：Recommendation 未 Lock。
    if not search.recommendation_locked or not search.recommended_config_hash:
        reasons.append(Reason.RECOMMENDATION_NOT_LOCKED)

    # §八.7：Search Status 必須為可部署完成狀態。
    if search.status != SearchStatus.COMPLETED:
        reasons.append(Reason.SEARCH_NOT_DEPLOYABLE)

    # §八.2：Candidate 必須就是 Locked Recommendation。
    if search.recommended_config_hash and candidate.config_hash != search.recommended_config_hash:
        reasons.append(Reason.CANDIDATE_NOT_RECOMMENDATION)

    # §八.3：request expected_config_hash 必須等於 locked recommended_config_hash。
    if request.expected_config_hash != (search.recommended_config_hash or ""):
        reasons.append(Reason.RECOMMENDED_CONFIG_HASH_MISMATCH)
    else:
        validated["recommended_config_hash"] = search.recommended_config_hash

    # §八.9：Candidate Config Hash 必須等於 request expected。
    if candidate.config_hash != request.expected_config_hash:
        reasons.append(Reason.CANDIDATE_CONFIG_HASH_MISMATCH)
    else:
        validated["config_hash"] = candidate.config_hash

    # §八.4：recommendation_apply_eligible 必須為 true（由 DB outcome 讀取，§二十一.9）。
    if outcome.get("recommendation_apply_eligible") is not True:
        reasons.append(Reason.NOT_APPLY_ELIGIBLE)
        # §九：保留已存在的 apply_ineligible_reasons（small_validation_sample 等）。
        for r in outcome.get("apply_ineligible_reasons", []) or []:
            if r not in reasons:
                reasons.append(r)

    # §八.5：test_constraint_status 必須為 passed。
    if outcome.get("test_constraint_status") != "passed":
        reasons.append(Reason.TEST_CONSTRAINT_NOT_PASSED)

    # §八.6：working_tree_dirty。
    if environment.working_tree_dirty:
        reasons.append(Reason.WORKING_TREE_DIRTY)

    # §八.8：Candidate 各階段必須完成。
    if not (
        candidate.tune_status == CandidateStageStatus.COMPLETED
        and candidate.validation_status == CandidateStageStatus.COMPLETED
        and candidate.test_status == CandidateStageStatus.COMPLETED
    ):
        reasons.append(Reason.CANDIDATE_STAGE_INCOMPLETE)

    # §八.10：Dataset Hash。
    if request.expected_dataset_hash != (search.dataset_hash or ""):
        reasons.append(Reason.DATASET_HASH_MISMATCH)
    else:
        validated["dataset_hash"] = search.dataset_hash

    # §八.11 / §十一：Corpus Hash —— candidate(search) 與 request expected 與 target corpus 三者一致。
    corpus_ok = True
    if request.expected_corpus_hash != (search.corpus_hash or ""):
        reasons.append(Reason.CORPUS_HASH_MISMATCH)
        corpus_ok = False
    if (
        environment.target_corpus_hash is not None
        and request.expected_corpus_hash != environment.target_corpus_hash
    ):
        # §八.20：candidate 來自 demo corpus，但 target 是不同 live corpus。
        reasons.append(Reason.DEMO_CORPUS_TARGET_MISMATCH)
        corpus_ok = False
    if corpus_ok:
        validated["corpus_hash"] = search.corpus_hash

    # §八.12：Git Commit（request expected 與 search 與 env 一致）。
    if request.expected_code_commit != (search.code_commit or ""):
        reasons.append(Reason.CODE_COMMIT_MISMATCH)
    elif environment.code_commit is not None and environment.code_commit != search.code_commit:
        reasons.append(Reason.CODE_COMMIT_MISMATCH)
    else:
        validated["code_commit"] = search.code_commit

    # §八.13：Canonical Version。
    if (
        request.expected_canonical_text_version is not None
        and environment.canonical_text_version is not None
        and request.expected_canonical_text_version != environment.canonical_text_version
    ):
        reasons.append(Reason.CANONICAL_VERSION_MISMATCH)
    elif environment.canonical_text_version is not None:
        validated["canonical_text_version"] = environment.canonical_text_version

    # §八.14：Identifier Normalization Version。
    if (
        request.expected_identifier_normalization_version is not None
        and environment.identifier_normalization_version is not None
        and request.expected_identifier_normalization_version
        != environment.identifier_normalization_version
    ):
        reasons.append(Reason.IDENTIFIER_VERSION_MISMATCH)
    elif environment.identifier_normalization_version is not None:
        validated["identifier_normalization_version"] = (
            environment.identifier_normalization_version
        )

    # §八.15：Embedding Fingerprint。
    if (
        request.expected_embedding_fingerprint_hash is not None
        and environment.embedding_fingerprint_hash is not None
        and request.expected_embedding_fingerprint_hash
        != environment.embedding_fingerprint_hash
    ):
        reasons.append(Reason.EMBEDDING_FINGERPRINT_MISMATCH)
    elif environment.embedding_fingerprint_hash is not None:
        validated["embedding_fingerprint_hash"] = environment.embedding_fingerprint_hash

    # §八.16：Candidate Stale —— locked_at 與使用者所見不符（recommendation 已被重新鎖定/取代）。
    if (
        request.expected_recommendation_locked_at is not None
        and search.recommendation_locked_at is not None
        and request.expected_recommendation_locked_at != search.recommendation_locked_at
    ):
        reasons.append(Reason.STALE_CANDIDATE)

    # §八.17：Alias Current Target 與 expected 不符。
    if (
        environment.alias_current_target is not None
        and environment.alias_current_target != request.expected_current_collection
    ):
        reasons.append(Reason.STALE_ALIAS_TARGET)

    # §八.18：Explicit Confirmation。
    expected_conf = apply_confirmation(
        request.alias_name,
        request.expected_config_hash,
        request.expected_current_collection,
    )
    if not confirmation_matches(request.confirmation, expected_conf):
        reasons.append(Reason.CONFIRMATION_MISMATCH)

    reasons = _dedup(reasons)
    snapshot = {
        "search_id": search.search_id,
        "candidate_id": candidate.candidate_id,
        "found": True,
        "ordinal": candidate.ordinal,
        "config_hash": candidate.config_hash,
        "recommended_config_hash": search.recommended_config_hash,
        "recommendation_locked": search.recommendation_locked,
        "recommendation_locked_at": search.recommendation_locked_at,
        "search_status": search.status,
        "tune_status": candidate.tune_status,
        "validation_status": candidate.validation_status,
        "test_status": candidate.test_status,
    }
    return ApplyEligibilityDecision(
        eligible=len(reasons) == 0,
        reasons=reasons,
        warnings=warnings,
        validated_hashes=validated,
        candidate_snapshot=snapshot,
    )


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class ApplyEligibilityGuard:
    """薄封裝：供 service / API 以物件方式呼叫（邏輯仍在純函式）。"""

    def evaluate(
        self,
        *,
        search: Optional[OptimizerSearch],
        candidate: Optional[SearchCandidate],
        environment: CurrentEnvironment,
        request: ApplyRequestSnapshot,
    ) -> ApplyEligibilityDecision:
        return evaluate_apply_eligibility(
            search=search, candidate=candidate, environment=environment, request=request
        )
