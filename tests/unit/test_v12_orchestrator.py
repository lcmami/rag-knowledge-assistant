"""Slice 3：有界 Orchestrator 端到端測試（Fake LLM + Fake Executor）。

此層負責 Promotion / Stop / Metrics 的決定性（Fake Planner 只保證 proposal/probe）。
"""
from __future__ import annotations

import json
import os

import pytest

from evaluation.v12.agent.orchestrator import (
    AgentOrchestrator,
    AgentState,
    SplitPaths,
    StopReason,
)
from evaluation.v12.agent.planner import FakeAgentPlanner
from evaluation.v12.agent.run_config import AgentRunConfig
from tests.unit.v12_agent_fakes import (
    FakeTrialExecutor,
    default_metric_fn,
    make_proposal,
    sample_segment,
    sample_tune_failure,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")
SPLITS = SplitPaths(tune="tune.json", validation="validation.json", test="test.json")


def _rc(**budget) -> AgentRunConfig:
    with open(SMOKE_CONFIG, encoding="utf-8") as f:
        raw = json.load(f)
    raw["budget"].update(budget)
    return AgentRunConfig.from_raw(raw)


def _executor(**kw) -> FakeTrialExecutor:
    kw.setdefault("segments", (sample_segment(),))
    kw.setdefault("tune_failures", (sample_tune_failure(),))
    return FakeTrialExecutor(**kw)


def _orch(rc, planner, executor):
    return AgentOrchestrator(rc, planner, executor, id_factory=lambda: "run-fixed")


# --------------------------------------------------------------------------
def test_improving_candidate_promoted():
    rc = _rc(max_rounds=1)
    ex = _executor()
    res = _orch(rc, FakeAgentPlanner(), ex).run(splits=SPLITS)
    assert res.candidates_promoted == 1
    assert res.candidates_rejected == 0
    assert res.final_parent_config_hash != res.baseline_config_hash
    assert res.rounds[0].promotion_decision == "promoted"
    assert res.status == AgentState.COMPLETED
    assert res.recommendation_eligibility == "not_evaluated"
    # 所有 collection 最終清乾淨。
    assert res.residual_collections == []
    assert ex.cleaned
    assert not any("factory_knowledge" in c for c in ex.cleaned)


def test_regressing_candidate_rejected_parent_preserved():
    rc = _rc(max_rounds=1)
    planner = FakeAgentPlanner(script=[make_proposal(top_k=4)])  # exact 0.45 < parent 0.5
    res = _orch(rc, planner, _executor()).run(splits=SPLITS)
    assert res.candidates_rejected == 1
    assert res.candidates_promoted == 0
    assert res.final_parent_config_hash == res.baseline_config_hash
    assert "validation_gate_failed" in res.rounds[0].promotion_reason


def test_constraint_failure_rejected():
    def ufp_fail(trial, split):
        m = default_metric_fn(trial, split)
        if trial.top_k == 6:
            m["unanswerable_false_positive_rate"] = 0.9  # > 0.5 absolute ceiling
        return m

    rc = _rc(max_rounds=1)
    res = _orch(rc, FakeAgentPlanner(), _executor(metric_fn=ufp_fail)).run(splits=SPLITS)
    assert res.candidates_rejected == 1
    assert res.final_parent_config_hash == res.baseline_config_hash
    assert "unanswerable_false_positive_ceiling" in res.rounds[0].promotion_reason


def test_duplicate_proposal_stops_without_rerun():
    def ufp_fail(trial, split):
        m = default_metric_fn(trial, split)
        if trial.top_k == 6:
            m["unanswerable_false_positive_rate"] = 0.9
        return m

    rc = _rc(max_rounds=3, no_improvement_rounds=5, max_consecutive_rejections=5)
    planner = FakeAgentPlanner(script=[make_proposal(top_k=6), make_proposal(top_k=6)])
    res = _orch(rc, planner, _executor(metric_fn=ufp_fail)).run(splits=SPLITS)
    assert res.stop_reason == StopReason.DUPLICATE_PROPOSAL
    assert res.status == AgentState.STOPPED
    assert res.candidates_proposed == 1  # 第二次重複不重跑


def test_two_rejections_stop():
    rc = _rc(max_rounds=5, no_improvement_rounds=5, max_consecutive_rejections=2)
    planner = FakeAgentPlanner(script=[make_proposal(top_k=4), make_proposal(top_k=3)])
    res = _orch(rc, planner, _executor()).run(splits=SPLITS)
    assert res.stop_reason == StopReason.MAX_CONSECUTIVE_REJECTIONS
    assert res.candidates_rejected == 2
    assert res.final_parent_config_hash == res.baseline_config_hash


def test_candidate_trial_failure_cleanup_and_stop():
    rc = _rc(max_rounds=2)
    ex = _executor(fail_predicate=lambda t, s: t.top_k == 6)  # candidate fails
    res = _orch(rc, FakeAgentPlanner(), ex).run(splits=SPLITS)
    assert res.stop_reason == StopReason.FATAL_TRIAL_FAILURE
    assert res.status == AgentState.STOPPED
    assert res.final_parent_config_hash == res.baseline_config_hash
    assert res.residual_collections == []  # 失敗路徑也清乾淨


def test_baseline_failure_marks_failed():
    rc = _rc(max_rounds=2)
    ex = _executor(fail_predicate=lambda t, s: t.top_k == 5)  # baseline fails
    res = _orch(rc, FakeAgentPlanner(), ex).run(splits=SPLITS)
    assert res.status == AgentState.FAILED
    assert res.stop_reason == StopReason.BASELINE_FAILED
    assert res.residual_collections == []


def test_max_llm_calls_stop():
    rc = _rc(max_rounds=5, max_llm_calls=1, no_improvement_rounds=9, max_consecutive_rejections=9)
    # 第一輪就會用掉 llm_calls（proposal），下一輪前檢查即停。
    planner = FakeAgentPlanner(script=[make_proposal(top_k=6), make_proposal(top_k=7)])
    res = _orch(rc, planner, _executor()).run(splits=SPLITS)
    assert res.stop_reason in (StopReason.MAX_LLM_CALLS, StopReason.MAX_ROUNDS)


def test_llm_recommended_stop():
    rc = _rc(max_rounds=5, no_improvement_rounds=9, max_consecutive_rejections=9)
    planner = FakeAgentPlanner(script=[make_proposal(top_k=6, stop=True)])
    res = _orch(rc, planner, _executor()).run(splits=SPLITS)
    assert res.stop_reason == StopReason.LLM_RECOMMENDED_STOP
    assert res.status == AgentState.COMPLETED


def test_deterministic_end_to_end():
    rc = _rc(max_rounds=2, no_improvement_rounds=9, max_consecutive_rejections=9)
    a = _orch(rc, FakeAgentPlanner(), _executor()).run(splits=SPLITS)
    b = _orch(rc, FakeAgentPlanner(), _executor()).run(splits=SPLITS)
    # 相同輸入 → 相同 promotion/stop/metrics（決定性由 Orchestrator 負責）。
    assert a.to_dict() == b.to_dict()
