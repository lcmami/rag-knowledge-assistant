"""Slice 6：LLMAgentPlanner + prompt builder 單元測試（以 fake async client，不需真實 LLM）。"""
from __future__ import annotations

import json
import os

from evaluation.v12.agent.context import (
    SegmentExcerpt,
    TuneFailureView,
    build_agent_context,
)
from evaluation.v12.agent.llm_planner import LLMAgentPlanner, build_agent_messages
from evaluation.v12.agent.parser import parse_proposal
from evaluation.v12.agent.run_config import load_agent_run_config
from evaluation.v12.agent.validator import ProposalValidator

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")


class _FakeAsyncLLM:
    def __init__(self, response: str, model: str = "fake-llm-1"):
        self._response = response
        self.model = model

    async def complete(self, messages):
        self._last_messages = messages
        return self._response


def _context(rc):
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
    )


def _canned_proposal() -> str:
    return json.dumps(
        {
            "diagnoses": [
                {"case_id": "c1", "failure_type": "miss", "reason": "not retrieved"}
            ],
            "proposed_config": {
                "chunk_config": {"strategy": "current"},
                "retrieval_config": {
                    "strategy": "dense",
                    "top_k": 6,
                    "score_threshold": 0.55,
                    "candidate_multiplier": 3,
                    "candidate_cap": 50,
                },
            },
            "probe_questions": [
                {
                    "question": "probe D900?",
                    "query_type": "exact_code",
                    "source_segment_ids": ["seg-1"],
                    "anchor_text_suggestions": ["D900"],
                }
            ],
            "expected_effects": [],
            "confidence": 0.6,
            "stop_recommendation": False,
        }
    )


def test_build_agent_messages_shape():
    rc = load_agent_run_config(SMOKE_CONFIG)
    msgs = build_agent_messages(_context(rc))
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "DATA" in msgs[0]["content"]  # 宣告 data delimiter
    user = json.loads(msgs[1]["content"])
    assert "context" in user and "output_schema" in user


def test_llm_planner_propose_parses_and_validates():
    rc = load_agent_run_config(SMOKE_CONFIG)
    ctx = _context(rc)
    planner = LLMAgentPlanner(_FakeAsyncLLM(_canned_proposal()))
    assert planner.planner_type == "llm"
    assert planner.model == "fake-llm-1"
    assert planner.provider == "openai_compatible"

    raw = planner.propose(ctx)
    proposal = parse_proposal(raw)
    res = ProposalValidator(rc).validate(
        proposal,
        parent=rc.baseline,
        tested_config_hashes=set(),
        segment_texts=ctx.segment_texts(),
    )
    assert res.ok, res.errors
