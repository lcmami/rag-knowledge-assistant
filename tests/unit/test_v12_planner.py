"""Slice 2：FakeAgentPlanner 決定性單元測試。

決定性只涵蓋 Proposal / Probe / Config（相同 Context → 相同輸出）；Promotion / Stop /
Metrics 的決定性屬 Orchestrator 端到端測試（Slice 3）。
"""
from __future__ import annotations

import json
import os

from evaluation.v12.agent.context import (
    PreviousRoundSummary,
    SegmentExcerpt,
    TuneFailureView,
    build_agent_context,
)
from evaluation.v12.agent.parser import parse_proposal
from evaluation.v12.agent.planner import FakeAgentPlanner
from evaluation.v12.agent.run_config import AgentRunConfig, load_agent_run_config
from evaluation.v12.agent.validator import ProposalValidator

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")


def _rc() -> AgentRunConfig:
    return load_agent_run_config(SMOKE_CONFIG)


def _context(rc, previous_rounds=()):
    return build_agent_context(
        run_config=rc,
        current=rc.baseline,
        tune_failures=[
            TuneFailureView(
                case_id="c1",
                question="What is D900?",
                query_type="exact_code",
                case_status="miss",
                first_relevant_rank=None,
                expected_evidence_excerpt="alarm code D900",
            )
        ],
        segments=[
            SegmentExcerpt(
                segment_id="seg-1",
                section="Alarms",
                text="The alarm code D900 indicates an overheat fault.",
            )
        ],
        coarse_tune_metrics={"exact_code_top1_accuracy": 0.5},
        validation_gate="not_evaluated",
        previous_rounds=previous_rounds,
    )


def test_fake_default_produces_valid_proposal():
    rc = _rc()
    ctx = _context(rc)
    planner = FakeAgentPlanner()
    raw = planner.propose(ctx)
    proposal = parse_proposal(raw)
    tune_failure_types = {
        f["case_id"]: f["failure_type"]
        for f in ctx.to_prompt_payload().get("tune_failures", [])
    }
    res = ProposalValidator(rc).validate(
        proposal,
        parent=rc.baseline,
        tested_config_hashes=set(),
        segment_texts=ctx.segment_texts(),
        tune_failure_types=tune_failure_types,
    )
    assert res.ok, res.errors
    assert res.parameter_changes == ["retrieval.top_k: 5→6"]


def test_fake_is_deterministic():
    rc = _rc()
    ctx = _context(rc)
    planner = FakeAgentPlanner()
    assert planner.propose(ctx) == planner.propose(ctx)


def test_fake_reacts_to_previous_rejection():
    rc = _rc()
    base = FakeAgentPlanner().propose(_context(rc))
    prev = PreviousRoundSummary(
        round_number=0,
        proposal_summary={},
        promotion_decision="rejected",
        promotion_reason="regressed",
        probe_comparison=({"candidate_better": False},),
    )
    reacted = FakeAgentPlanner().propose(_context(rc, previous_rounds=[prev]))
    # 讀到上一輪被拒 → 改往另一方向（不同 proposal）。
    assert base != reacted
    assert parse_proposal(reacted).proposed_retrieval_config["top_k"] == 4


def test_fake_scripted_mode_indexes_by_round():
    rc = _rc()
    script = [
        {
            "diagnoses": [],
            "proposed_config": {
                "chunk_config": {"strategy": "current"},
                "retrieval_config": {"strategy": "dense", "top_k": 7},
            },
            "probe_questions": [],
            "expected_effects": [],
            "confidence": 0.4,
            "stop_recommendation": False,
        },
        {
            "diagnoses": [],
            "proposed_config": {
                "chunk_config": {"strategy": "current"},
                "retrieval_config": {"strategy": "dense", "top_k": 8},
            },
            "probe_questions": [],
            "expected_effects": [],
            "confidence": 0.4,
            "stop_recommendation": True,
        },
    ]
    planner = FakeAgentPlanner(script=script)
    r0 = parse_proposal(planner.propose(_context(rc)))
    assert r0.proposed_retrieval_config["top_k"] == 7
    prev = PreviousRoundSummary(
        round_number=0, proposal_summary={}, promotion_decision="promoted", promotion_reason="ok"
    )
    r1 = parse_proposal(planner.propose(_context(rc, previous_rounds=[prev])))
    assert r1.proposed_retrieval_config["top_k"] == 8
    assert r1.stop_recommendation is True
