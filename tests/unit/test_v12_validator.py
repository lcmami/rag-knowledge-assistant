"""Slice 1：ProposalValidator 政策閘門單元測試。"""
from __future__ import annotations

import copy
import json
import os

import pytest

from evaluation.v12.agent.parser import parse_proposal
from evaluation.v12.agent.run_config import load_agent_run_config
from evaluation.v12.agent.validator import ProposalValidator

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")

SEGMENTS = {"seg-1": "The alarm code D900 indicates an overheat fault in the pump."}


@pytest.fixture
def run_config():
    return load_agent_run_config(SMOKE_CONFIG)


@pytest.fixture
def validator(run_config):
    return ProposalValidator(run_config)


def _payload(**overrides) -> dict:
    base = {
        "diagnoses": [],
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
                "question": "What is D900?",
                "query_type": "exact_code",
                "source_segment_ids": ["seg-1"],
                "anchor_text_suggestions": ["D900"],
            }
        ],
        "expected_effects": [],
        "confidence": 0.6,
        "stop_recommendation": False,
    }
    base.update(overrides)
    return base


def _validate(validator, run_config, payload, tested=None):
    proposal = parse_proposal(json.dumps(payload))
    return validator.validate(
        proposal,
        parent=run_config.baseline,
        tested_config_hashes=tested or set(),
        segment_texts=SEGMENTS,
    )


def test_valid_single_change(validator, run_config):
    res = _validate(validator, run_config, _payload())
    assert res.ok, res.errors
    assert res.candidate is not None
    assert res.config_hash
    assert res.parameter_changes == ["retrieval.top_k: 5→6"]


def test_reject_parent_identical(validator, run_config):
    payload = _payload()
    payload["proposed_config"]["retrieval_config"]["top_k"] = 5  # == baseline
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("完全相同" in e for e in res.errors)


def test_reject_out_of_bounds(validator, run_config):
    payload = _payload()
    payload["proposed_config"]["retrieval_config"]["top_k"] = 20  # bound [3,8]
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("超出 agent 界限" in e for e in res.errors)


def test_reject_whitelist_violation(validator, run_config):
    payload = _payload()
    payload["proposed_config"]["chunk_config"]["schema_version"] = "9.9"
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("白名單外欄位" in e for e in res.errors)


def test_reject_strategy_outside_allowed(validator, run_config):
    payload = _payload()
    # 'fixed' 是 v1.1 合法 strategy，但不在 smoke allowed chunk_strategies。
    payload["proposed_config"]["chunk_config"] = {"strategy": "fixed"}
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("不在允許集合" in e for e in res.errors)


def test_reject_too_many_changes(validator, run_config):
    payload = _payload()
    payload["proposed_config"]["chunk_config"] = {
        "strategy": "heading",
        "target_chars": 800,
        "prepend_section_title": True,
    }
    # strategy + target_chars + prepend_section_title = 3 changes > max 2
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("超過上限" in e for e in res.errors)


def test_reject_duplicate_config(validator, run_config):
    payload = _payload()
    first = _validate(validator, run_config, payload)
    assert first.ok
    res = _validate(validator, run_config, payload, tested={first.config_hash})
    assert not res.ok
    assert any("已測試 config 重複" in e for e in res.errors)


def test_reject_dense_with_identifier_field_via_v11(validator, run_config):
    payload = _payload()
    payload["proposed_config"]["retrieval_config"]["exact_match_boost"] = 0.3
    res = _validate(validator, run_config, payload)
    assert not res.ok
    # 白名單允許此欄位名，但 v1.1 RetrievalConfig 對 dense 拒絕 → 結構驗證失敗。
    assert any("結構驗證失敗" in e for e in res.errors)


def test_reject_probe_unknown_segment(validator, run_config):
    payload = _payload()
    payload["probe_questions"][0]["source_segment_ids"] = ["seg-missing"]
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("不存在的 segment" in e for e in res.errors)


def test_reject_probe_fabricated_anchor(validator, run_config):
    payload = _payload()
    payload["probe_questions"][0]["anchor_text_suggestions"] = ["Z999-not-in-text"]
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("臆造 anchor" in e for e in res.errors)


def test_reject_too_many_probes(validator, run_config):
    payload = _payload()
    probe = payload["probe_questions"][0]
    payload["probe_questions"] = [copy.deepcopy(probe) for _ in range(4)]  # max 3
    res = _validate(validator, run_config, payload)
    assert not res.ok
    assert any("probe 數" in e for e in res.errors)


def test_identifier_aware_valid_change(validator, run_config):
    payload = _payload()
    payload["proposed_config"]["retrieval_config"] = {
        "strategy": "identifier_aware",
        "top_k": 5,
        "score_threshold": 0.55,
        "candidate_multiplier": 3,
        "candidate_cap": 50,
    }
    # dense→identifier_aware 記為 1 個 strategy 變更（identifier 欄位為內含）。
    res = _validate(validator, run_config, payload)
    assert res.ok, res.errors
    assert res.parameter_changes == ["retrieval.strategy: dense→identifier_aware"]
