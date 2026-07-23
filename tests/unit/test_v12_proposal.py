"""Slice 1：Proposal 嚴格解析單元測試。"""
from __future__ import annotations

import json

import pytest

from evaluation.v12.agent.parser import ProposalParseError, parse_proposal


def _valid_payload() -> dict:
    return {
        "diagnoses": [
            {
                "case_id": "c1",
                "failure_type": "miss",
                "reason": "expected chunk not retrieved",
                "supporting_observations": ["top1 was unrelated"],
            }
        ],
        "proposed_config": {
            "chunk_config": {"strategy": "current"},
            "retrieval_config": {"strategy": "dense", "top_k": 6},
        },
        "probe_questions": [
            {
                "question": "What is alarm D900?",
                "query_type": "exact_code",
                "source_segment_ids": ["seg-1"],
                "anchor_text_suggestions": ["D900"],
                "rationale": "probe identifier retrieval",
            }
        ],
        "expected_effects": [
            {"metric": "mrr", "direction": "increase", "rationale": "more candidates"}
        ],
        "confidence": 0.7,
        "stop_recommendation": False,
    }


def test_parse_valid():
    p = parse_proposal(json.dumps(_valid_payload()))
    assert p.diagnoses[0].failure_type == "miss"
    assert p.proposed_retrieval_config["top_k"] == 6
    assert p.probe_questions[0].query_type == "exact_code"
    assert p.confidence == 0.7
    assert p.stop_recommendation is False


def test_parse_strips_code_fence():
    raw = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    p = parse_proposal(raw)
    assert p.proposed_chunk_config["strategy"] == "current"


def test_reject_invalid_json():
    with pytest.raises(ProposalParseError, match="不是合法 JSON"):
        parse_proposal("{not json")


def test_reject_unknown_top_field():
    payload = _valid_payload()
    payload["surprise"] = 1
    with pytest.raises(ProposalParseError, match="含未知欄位"):
        parse_proposal(json.dumps(payload))


def test_reject_unknown_failure_type():
    payload = _valid_payload()
    payload["diagnoses"][0]["failure_type"] = "made_up"
    with pytest.raises(ProposalParseError, match="failure_type"):
        parse_proposal(json.dumps(payload))


def test_reject_unknown_query_type():
    payload = _valid_payload()
    payload["probe_questions"][0]["query_type"] = "banana"
    with pytest.raises(ProposalParseError, match="query_type"):
        parse_proposal(json.dumps(payload))


def test_reject_confidence_out_of_range():
    payload = _valid_payload()
    payload["confidence"] = 1.5
    with pytest.raises(ProposalParseError, match="confidence"):
        parse_proposal(json.dumps(payload))


def test_reject_non_bool_stop():
    payload = _valid_payload()
    payload["stop_recommendation"] = "yes"
    with pytest.raises(ProposalParseError, match="stop_recommendation"):
        parse_proposal(json.dumps(payload))


def test_reject_missing_proposed_config():
    payload = _valid_payload()
    del payload["proposed_config"]
    with pytest.raises(ProposalParseError, match="proposed_config"):
        parse_proposal(json.dumps(payload))


def test_reject_unknown_proposed_config_section():
    payload = _valid_payload()
    payload["proposed_config"]["embedding_model"] = "x"
    with pytest.raises(ProposalParseError, match="proposed_config 含未知欄位"):
        parse_proposal(json.dumps(payload))
