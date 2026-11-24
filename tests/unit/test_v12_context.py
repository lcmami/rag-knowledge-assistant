"""Slice 2：受控原文 Context Builder 單元測試。"""
from __future__ import annotations

import json
import os

import pytest

from evaluation.v12.agent.context import (
    DATA_OPEN,
    DEFAULT_CAPS,
    PreviousRoundSummary,
    SegmentExcerpt,
    TuneFailureView,
    build_agent_context,
)
from evaluation.v12.agent.run_config import AgentRunConfig, load_agent_run_config

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")


def _run_config(**budget_overrides) -> AgentRunConfig:
    with open(SMOKE_CONFIG, encoding="utf-8") as f:
        raw = json.load(f)
    raw["budget"].update(budget_overrides)
    return AgentRunConfig.from_raw(raw)


def _segments():
    return [
        SegmentExcerpt(
            segment_id="seg-1",
            section="Alarms",
            text="The alarm code D900 indicates an overheat fault in the pump.",
            before="Chapter 3 preamble.",
            after="Next section discusses D901.",
            adjacent="Adjacent chunk about D901.",
        )
    ]


def _failures():
    return [
        TuneFailureView(
            case_id="c1",
            question="What is D900?",
            query_type="exact_code",
            case_status="miss",
            first_relevant_rank=None,
            failure_type="confusable_confusion",
            expected_evidence_excerpt="alarm code D900 indicates an overheat fault",
            actual_top_excerpts=("unrelated chunk about D200",),
            identifier_detail={"conflicting_identifiers": ["D200"]},
        )
    ]


def _build(run_config, **kw):
    kw.setdefault("validation_gate", "not_evaluated")
    return build_agent_context(
        run_config=run_config,
        current=run_config.baseline,
        tune_failures=_failures(),
        segments=_segments(),
        coarse_tune_metrics={"exact_code_top1_accuracy": 0.5},
        **kw,
    )


def test_context_only_tune_and_coarse_validation():
    ctx = _build(_run_config())
    p = ctx.to_prompt_payload()
    assert p["validation_gate"] == "not_evaluated"
    assert "tune_failures" in p and p["tune_failures"][0]["case_id"] == "c1"
    # 結構上不存在 validation per-case 欄位。
    assert "validation_failures" not in p
    assert "validation_case_results" not in p


def test_excerpt_caps_enforced():
    rc = _run_config()
    long_text = "X" * 2000
    ctx = build_agent_context(
        run_config=rc,
        current=rc.baseline,
        tune_failures=[],
        segments=[SegmentExcerpt(segment_id="seg-long", section=None, text=long_text)],
        coarse_tune_metrics={},
        validation_gate="not_evaluated",
    )
    seen = ctx.segment_texts()["seg-long"]
    assert len(seen) <= DEFAULT_CAPS["segment"] + 1  # + ellipsis


def test_document_text_wrapped_as_data():
    ctx = _build(_run_config())
    seg = ctx.to_prompt_payload()["segments"][0]
    assert DATA_OPEN in seg["text"]  # 原文以 data delimiter 包住
    assert "data_policy" in ctx.to_prompt_payload()


def test_context_hash_covers_content():
    rc = _run_config()
    a = _build(rc)
    b = _build(rc)
    assert a.context_hash == b.context_hash  # 相同輸入 → 相同 hash
    # 改動 segment 內容 → hash 改變。
    c = build_agent_context(
        run_config=rc,
        current=rc.baseline,
        tune_failures=_failures(),
        segments=[SegmentExcerpt(segment_id="seg-1", section="Alarms", text="different text")],
        coarse_tune_metrics={"exact_code_top1_accuracy": 0.5},
        validation_gate="not_evaluated",
    )
    assert c.context_hash != a.context_hash


def test_reject_precise_validation_metric():
    with pytest.raises(ValueError, match="coarse"):
        _build(_run_config(), validation_gate="mrr=0.83")


def test_global_budget_trims_and_notes():
    rc = _run_config(context_char_budget=200)  # 極小預算
    prev = PreviousRoundSummary(
        round_number=0,
        proposal_summary={"changes": ["retrieval.top_k: 5→6"]},
        promotion_decision="rejected",
        promotion_reason="regressed",
        probe_comparison=({"probe": "x", "candidate_better": False},),
    )
    ctx = _build(rc, previous_rounds=[prev])
    assert ctx.notes  # 有裁切紀錄
    # 至少保留 1 筆 tune failure（不得清空診斷來源）。
    assert len(ctx.to_prompt_payload()["tune_failures"]) >= 1


def test_segment_texts_match_llm_view():
    ctx = _build(_run_config())
    # Validator 用的 segment_texts 必須與 LLM 所見（裁切後）一致。
    assert "D900" in ctx.segment_texts()["seg-1"]
