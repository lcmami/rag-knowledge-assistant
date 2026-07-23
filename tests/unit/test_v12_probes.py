"""Slice 4：Probe Comparison + 兩輪回饋 + failure 分類。"""
from __future__ import annotations

import json
import os

from evaluation.v12.agent.context import AgentContext
from evaluation.v12.agent.diagnosis import classify_failure
from evaluation.v12.agent.orchestrator import AgentOrchestrator, SplitPaths
from evaluation.v12.agent.planner import FakeAgentPlanner
from evaluation.v12.agent.probes import DiagnosticProbeRunner
from evaluation.v12.agent.proposal import FAILURE_TYPES
from evaluation.v12.agent.run_config import AgentRunConfig
from tests.unit.v12_agent_fakes import (
    FakeTrialExecutor,
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


def _executor():
    return FakeTrialExecutor(
        segments=(sample_segment(),), tune_failures=(sample_tune_failure(),)
    )


class _RecordingPlanner:
    """包住 FakeAgentPlanner，記錄每輪收到的 context。"""

    planner_type = "recording-fake"

    def __init__(self):
        self._inner = FakeAgentPlanner()
        self.contexts: list[AgentContext] = []

    def propose(self, context: AgentContext) -> str:
        self.contexts.append(context)
        return self._inner.propose(context)


# --------------------------------------------------------------------------
# failure 分類
# --------------------------------------------------------------------------
def test_classify_failure_mapping():
    assert classify_failure(
        case_status="miss", answerable=True, first_relevant_rank=None
    ) == "miss"
    assert classify_failure(
        case_status="miss", answerable=True, first_relevant_rank=3
    ) == "wrong_rank"
    assert classify_failure(
        case_status="miss", answerable=True, first_relevant_rank=None,
        conflicting_identifiers=["D200"],
    ) == "confusable_confusion"
    assert classify_failure(
        case_status="miss", answerable=True, first_relevant_rank=None, retrieved_any=False
    ) == "threshold_too_high"
    assert classify_failure(
        case_status="unanswerable_false_positive", answerable=False, first_relevant_rank=None
    ) == "unanswerable_false_positive"
    assert classify_failure(
        case_status="relevant_hit", answerable=True, first_relevant_rank=1
    ) is None
    # 所有非 None 結果都在受控詞彙內。
    assert classify_failure(
        case_status="miss", answerable=True, first_relevant_rank=2
    ) in FAILURE_TYPES


# --------------------------------------------------------------------------
# probe comparison
# --------------------------------------------------------------------------
def test_probe_comparison_runs_on_both_collections():
    rc = _rc(max_rounds=1)
    ex = _executor()
    runner = DiagnosticProbeRunner(ex)
    orch = AgentOrchestrator(rc, FakeAgentPlanner(), ex, probe_runner=runner, id_factory=lambda: "r")
    res = orch.run(splits=SPLITS)
    assert res.probe_count == 1
    comp = res.rounds[0].probe_comparison
    assert comp and comp[0]["candidate_better"] is True  # candidate top_k 較大 → 分數較高
    # 同一組 probe 同時打到 parent 與 candidate 的 collection。
    assert len(ex.retrieved) >= 2


def test_probes_do_not_affect_eligibility():
    rc = _rc(max_rounds=1)
    ex = _executor()
    orch = AgentOrchestrator(
        rc, FakeAgentPlanner(), ex, probe_runner=DiagnosticProbeRunner(ex), id_factory=lambda: "r"
    )
    res = orch.run(splits=SPLITS)
    assert res.recommendation_eligibility == "not_evaluated"


def test_rejected_candidate_still_saves_probe_comparison():
    # 讓 candidate 因 constraint 失敗被拒，但 probe comparison 仍保存。
    from tests.unit.v12_agent_fakes import default_metric_fn

    def ufp_fail(trial, split):
        m = default_metric_fn(trial, split)
        if trial.top_k == 6:
            m["unanswerable_false_positive_rate"] = 0.9
        return m

    rc = _rc(max_rounds=1)
    ex = FakeTrialExecutor(
        metric_fn=ufp_fail,
        segments=(sample_segment(),),
        tune_failures=(sample_tune_failure(),),
    )
    orch = AgentOrchestrator(
        rc, FakeAgentPlanner(), ex, probe_runner=DiagnosticProbeRunner(ex), id_factory=lambda: "r"
    )
    res = orch.run(splits=SPLITS)
    assert res.rounds[0].promotion_decision == "rejected"
    assert res.rounds[0].probe_comparison  # 被拒仍保存 probe comparison
    assert res.residual_collections == []  # candidate collection 已清理


# --------------------------------------------------------------------------
# 核心：兩輪回饋
# --------------------------------------------------------------------------
def test_two_round_probe_feedback_loop():
    rc = _rc(max_rounds=2, no_improvement_rounds=9, max_consecutive_rejections=9)
    ex = _executor()
    planner = _RecordingPlanner()
    orch = AgentOrchestrator(
        rc, planner, ex, probe_runner=DiagnosticProbeRunner(ex), id_factory=lambda: "r"
    )
    res = orch.run(splits=SPLITS)

    assert len(res.rounds) == 2
    # Round 1 產生 probe comparison。
    assert res.rounds[0].probe_comparison
    # Round 2 收到的 context 必須包含 Round 1 的 probe comparison（回饋證明）。
    assert len(planner.contexts) >= 2
    round2_ctx = planner.contexts[1]
    assert round2_ctx.previous_rounds[-1].probe_comparison
    assert (
        round2_ctx.previous_rounds[-1].probe_comparison[0]["question"]
        == res.rounds[0].probe_comparison[0]["question"]
    )
    # Round 2 依回饋提出「不同」proposal。
    assert res.rounds[0].proposal != res.rounds[1].proposal
    assert (
        res.rounds[0].candidate_config_hash != res.rounds[1].candidate_config_hash
    )
