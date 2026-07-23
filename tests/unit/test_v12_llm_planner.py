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


def test_build_agent_messages_surfaces_hard_constraints():
    """Runtime-acceptance regression：受控 context 必須把 max_parameter_changes、
    max_probe_questions 與可用 segment_id 明確攤給 LLM，且 System Prompt 需說明變更上限與
    probe anchor 取材規則——否則真實 LLM 易產生超過參數上限 / anchor 臆造的非法 Proposal，
    使 Candidate Trial 與 Probe 永不執行。"""
    rc = load_agent_run_config(SMOKE_CONFIG)
    msgs = build_agent_messages(_context(rc))

    system = msgs[0]["content"]
    assert "max_parameter_changes" in system  # 明確告知參數變更上限
    assert "available_segment_ids" in system  # 明確告知 probe 只能綁既有 segment

    user = json.loads(msgs[1]["content"])
    hc = user["hard_constraints"]
    assert hc["max_parameter_changes"] == rc.budget.max_parameter_changes  # =2
    assert hc["max_probe_questions"] == rc.budget.max_probe_questions  # =3
    # 具體可用 segment_id 必須攤出，且與 Validator 檢查 anchor 的 segment_texts 一致。
    assert hc["available_segment_ids"] == ["seg-1"]

    # editable_current_config 只含 Validator 白名單欄位：canonical 版本欄位（照抄整個
    # current_config 會觸發白名單錯誤）不得出現，且與 Validator 允許欄位集合完全一致。
    space = rc.allowed_parameter_space
    edit = hc["editable_current_config"]
    assert set(edit["chunk_config"]) <= space.allowed_chunk_fields()
    assert set(edit["retrieval_config"]) <= space.allowed_retrieval_fields()
    assert "prefix_format_version" not in edit["chunk_config"]
    assert "ranking_version" not in edit["retrieval_config"]
    # 白名單內、current 實際有值的欄位必須被保留（供 LLM 逐欄位照抄）。
    cur_chunk = rc.baseline.chunk_config
    for f in space.allowed_chunk_fields():
        if f in cur_chunk:
            assert edit["chunk_config"][f] == cur_chunk[f]


def test_editable_current_config_copy_validates_with_single_change():
    """把 editable_current_config 照抄、只改 1 個 retrieval 數值欄位，必須通過 Validator——
    這是修好真實 LLM 路徑後，合法 Proposal 的最小可達路徑（Candidate Trial 得以執行）。"""
    rc = load_agent_run_config(SMOKE_CONFIG)
    ctx = _context(rc)
    msgs = build_agent_messages(ctx)
    hc = json.loads(msgs[1]["content"])["hard_constraints"]
    edit = hc["editable_current_config"]

    proposed_retrieval = dict(edit["retrieval_config"])
    proposed_retrieval["top_k"] = int(proposed_retrieval["top_k"]) + 1  # 單一變更
    raw = json.dumps(
        {
            "diagnoses": [{"case_id": "c1", "failure_type": "miss", "reason": "x"}],
            "proposed_config": {
                "chunk_config": dict(edit["chunk_config"]),
                "retrieval_config": proposed_retrieval,
            },
            "probe_questions": [
                {
                    "question": "probe?",
                    "query_type": "exact_code",
                    "source_segment_ids": [hc["available_segment_ids"][0]],
                    "anchor_text_suggestions": [],  # 空 anchor 合法
                }
            ],
            "expected_effects": [],
            "confidence": 0.5,
            "stop_recommendation": False,
        }
    )
    res = ProposalValidator(rc).validate(
        parse_proposal(raw),
        parent=rc.baseline,
        tested_config_hashes=set(),
        segment_texts=ctx.segment_texts(),
    )
    assert res.ok, res.errors
    assert res.parameter_changes == ["retrieval.top_k: 5→6"]


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
