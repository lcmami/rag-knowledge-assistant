"""ApplyEligibilityGuard 測試（Slice 7 §二十四.1）。"""
from __future__ import annotations

import dataclasses
import json

import pytest

from evaluation.v11.deploy.confirmation import apply_confirmation
from evaluation.v11.deploy.eligibility import (
    CurrentEnvironment,
    Reason,
    evaluate_apply_eligibility,
)
from evaluation.v11.search.search_models import (
    CandidateStageStatus,
    OptimizerSearch,
    SearchCandidate,
    SearchStatus,
)
from tests.unit.deploy_helpers import (
    CANON,
    ELIGIBLE_COMMIT,
    ELIGIBLE_CONFIG_HASH,
    ELIGIBLE_CORPUS_HASH,
    ELIGIBLE_DATASET_HASH,
    ELIGIBLE_LOCKED_AT,
    FP,
    IDN,
    INELIGIBLE_REASONS,
    eligible_config_json,
)

ALIAS = "factory_test_alias_x"
CURRENT = "factory_test_deploy_v1_abc"


def _search(**over) -> OptimizerSearch:
    base = dict(
        search_id="s", search_name="e", status=SearchStatus.COMPLETED, search_mode="grid",
        search_config_json="{}", search_config_hash="h", dataset_hash=ELIGIBLE_DATASET_HASH,
        corpus_hash=ELIGIBLE_CORPUS_HASH, code_commit=ELIGIBLE_COMMIT, working_tree_dirty=False,
        recommended_config_hash=ELIGIBLE_CONFIG_HASH, recommendation_locked=True,
        recommendation_locked_at=ELIGIBLE_LOCKED_AT, test_evaluated=True,
        outcome_json=json.dumps({
            "recommendation_apply_eligible": True, "apply_ineligible_reasons": [],
            "test_constraint_status": "passed"}),
    )
    base.update(over)
    return OptimizerSearch(**base)


def _candidate(**over) -> SearchCandidate:
    base = dict(
        search_id="s", candidate_id="c", ordinal=0, config_json=eligible_config_json(),
        generation_status="valid", config_hash=ELIGIBLE_CONFIG_HASH,
        tune_status=CandidateStageStatus.COMPLETED,
        validation_status=CandidateStageStatus.COMPLETED,
        test_status=CandidateStageStatus.COMPLETED, feasible=True,
    )
    base.update(over)
    return SearchCandidate(**base)


def _env(**over) -> CurrentEnvironment:
    base = dict(
        admin_enabled=True, code_commit=ELIGIBLE_COMMIT, working_tree_dirty=False,
        target_corpus_hash=ELIGIBLE_CORPUS_HASH, canonical_text_version=CANON,
        identifier_normalization_version=IDN, embedding_fingerprint_hash=FP,
        alias_current_target=CURRENT,
    )
    base.update(over)
    return CurrentEnvironment(**base)


def _request(**over):
    from evaluation.v11.deploy.eligibility import ApplyRequestSnapshot

    base = dict(
        candidate_id="c", search_id="s", alias_name=ALIAS,
        expected_config_hash=ELIGIBLE_CONFIG_HASH, expected_dataset_hash=ELIGIBLE_DATASET_HASH,
        expected_corpus_hash=ELIGIBLE_CORPUS_HASH, expected_code_commit=ELIGIBLE_COMMIT,
        expected_current_collection=CURRENT,
        confirmation=apply_confirmation(ALIAS, ELIGIBLE_CONFIG_HASH, CURRENT),
        idempotency_key="k",
        expected_canonical_text_version=CANON,
        expected_identifier_normalization_version=IDN,
        expected_embedding_fingerprint_hash=FP,
        expected_recommendation_locked_at=ELIGIBLE_LOCKED_AT,
    )
    base.update(over)
    return ApplyRequestSnapshot(**base)


def _eval(search=None, candidate=None, env=None, request=None):
    return evaluate_apply_eligibility(
        search=search or _search(), candidate=candidate or _candidate(),
        environment=env or _env(), request=request or _request(),
    )


def test_eligible_locked_candidate_passes():
    d = _eval()
    assert d.eligible is True
    assert d.reasons == []
    assert d.validated_hashes["config_hash"] == ELIGIBLE_CONFIG_HASH
    assert d.validated_hashes["corpus_hash"] == ELIGIBLE_CORPUS_HASH
    assert d.candidate_snapshot["found"] is True
    # 不是單一 boolean（§八）。
    assert set(d.to_dict()) >= {"eligible", "reasons", "warnings", "validated_hashes", "candidate_snapshot"}


def test_recommendation_not_locked():
    d = _eval(search=_search(recommendation_locked=False, recommendation_locked_at=None))
    assert d.eligible is False
    assert Reason.RECOMMENDATION_NOT_LOCKED in d.reasons


def test_candidate_not_recommendation():
    d = _eval(candidate=_candidate(config_hash="other_hash"))
    assert d.eligible is False
    # candidate 與 locked recommendation 不同 → 至少觸發 candidate/config 不符。
    assert Reason.CANDIDATE_NOT_RECOMMENDATION in d.reasons


def test_apply_eligible_false_preserves_reasons():
    outcome = {
        "recommendation_apply_eligible": False,
        "apply_ineligible_reasons": list(INELIGIBLE_REASONS),
        "test_constraint_status": "failed",
    }
    d = _eval(search=_search(outcome_json=json.dumps(outcome)))
    assert d.eligible is False
    assert Reason.NOT_APPLY_ELIGIBLE in d.reasons
    for r in INELIGIBLE_REASONS:
        assert r in d.reasons


def test_test_constraint_failed():
    outcome = {"recommendation_apply_eligible": True, "apply_ineligible_reasons": [],
               "test_constraint_status": "failed"}
    d = _eval(search=_search(outcome_json=json.dumps(outcome)))
    assert Reason.TEST_CONSTRAINT_NOT_PASSED in d.reasons
    assert d.eligible is False


def test_dirty_git():
    d = _eval(env=_env(working_tree_dirty=True))
    assert Reason.WORKING_TREE_DIRTY in d.reasons


def test_config_hash_mismatch():
    d = _eval(request=_request(expected_config_hash="wrong"))
    assert Reason.RECOMMENDED_CONFIG_HASH_MISMATCH in d.reasons


def test_dataset_hash_mismatch():
    d = _eval(request=_request(expected_dataset_hash="wrong"))
    assert Reason.DATASET_HASH_MISMATCH in d.reasons


def test_corpus_hash_mismatch():
    d = _eval(request=_request(expected_corpus_hash="wrong"),
              env=_env(target_corpus_hash="wrong_also"))
    assert Reason.CORPUS_HASH_MISMATCH in d.reasons


def test_demo_corpus_target_mismatch():
    # request corpus 與 search 相符，但 target live corpus 不同（§八.20）。
    d = _eval(env=_env(target_corpus_hash="different_live_corpus"))
    assert Reason.DEMO_CORPUS_TARGET_MISMATCH in d.reasons
    assert d.eligible is False


def test_code_commit_mismatch():
    d = _eval(request=_request(expected_code_commit="wrong"))
    assert Reason.CODE_COMMIT_MISMATCH in d.reasons


def test_canonical_version_mismatch():
    d = _eval(env=_env(canonical_text_version="v2"))
    assert Reason.CANONICAL_VERSION_MISMATCH in d.reasons


def test_identifier_version_mismatch():
    d = _eval(env=_env(identifier_normalization_version="v2"))
    assert Reason.IDENTIFIER_VERSION_MISMATCH in d.reasons


def test_embedding_fingerprint_mismatch():
    d = _eval(env=_env(embedding_fingerprint_hash="other_fp"))
    assert Reason.EMBEDDING_FINGERPRINT_MISMATCH in d.reasons


def test_alias_target_stale():
    d = _eval(env=_env(alias_current_target="someone_else"))
    assert Reason.STALE_ALIAS_TARGET in d.reasons


def test_stale_candidate_locked_at_changed():
    d = _eval(request=_request(expected_recommendation_locked_at="2000-01-01T00:00:00+00:00"))
    assert Reason.STALE_CANDIDATE in d.reasons


def test_confirmation_wrong():
    d = _eval(request=_request(confirmation="APPLY wrong"))
    assert Reason.CONFIRMATION_MISMATCH in d.reasons


def test_admin_disabled():
    d = _eval(env=_env(admin_enabled=False))
    assert Reason.ADMIN_DISABLED in d.reasons
    assert d.eligible is False


def test_search_not_deployable_status():
    d = _eval(search=_search(status=SearchStatus.TEST_REGRESSION))
    assert Reason.SEARCH_NOT_DEPLOYABLE in d.reasons


def test_candidate_stage_incomplete():
    d = _eval(candidate=_candidate(test_status=CandidateStageStatus.LOCKED_OUT))
    assert Reason.CANDIDATE_STAGE_INCOMPLETE in d.reasons


def test_missing_search_or_candidate_is_stale():
    d = evaluate_apply_eligibility(search=None, candidate=None, environment=_env(), request=_request())
    assert d.eligible is False
    assert Reason.STALE_CANDIDATE in d.reasons
    assert d.candidate_snapshot["found"] is False


def test_current_real_recommendation_rejected_with_full_reasons():
    """目前真實 recommendation（apply_eligible=false）必須被拒絕且五原因完整（§九）。"""
    outcome = {
        "recommendation_apply_eligible": False,
        "apply_ineligible_reasons": list(INELIGIBLE_REASONS),
        "test_constraint_status": "failed",
    }
    search = _search(status=SearchStatus.TEST_CONSTRAINT_FAILED, outcome_json=json.dumps(outcome))
    d = _eval(search=search)
    assert d.eligible is False
    for r in INELIGIBLE_REASONS:
        assert r in d.reasons
    assert Reason.NOT_APPLY_ELIGIBLE in d.reasons
    assert Reason.TEST_CONSTRAINT_NOT_PASSED in d.reasons
    assert Reason.SEARCH_NOT_DEPLOYABLE in d.reasons
