"""Slice 0：AgentRunConfig 合約單元測試（不呼叫 LLM、不碰 Qdrant）。"""
from __future__ import annotations

import copy
import json
import os

import pytest

from evaluation.v12.agent.run_config import (
    AgentBudget,
    AgentRunConfig,
    AgentRunConfigError,
    AllowedParameterSpace,
    load_agent_run_config,
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")


def _valid_raw() -> dict:
    with open(SMOKE_CONFIG, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_smoke_config_loads():
    cfg = load_agent_run_config(SMOKE_CONFIG)
    assert cfg.run_name == "v12-agent-smoke"
    assert cfg.dataset_path == "evaluation/datasets/v1.1/cases-v2.json"
    assert cfg.corpus_paths == ("sample_docs", "evaluation/fixtures")
    assert cfg.prompt_version == "v1"
    # baseline 重用 v1.1 TrialConfig，canonical 化後含展開欄位。
    assert cfg.baseline.chunk_config["strategy"] == "current"
    assert cfg.baseline.retrieval["strategy"] == "dense"
    assert cfg.baseline.top_k == 5


def test_budget_and_promotion_parsed():
    cfg = load_agent_run_config(SMOKE_CONFIG)
    assert cfg.budget.max_rounds == 2
    assert cfg.budget.max_consecutive_rejections == 2
    assert cfg.budget.context_char_budget == 12000
    assert len(cfg.promotion.hard_constraints) == 2
    # comparator ranking 最後一鍵必為 config_hash（deterministic tie-break）。
    assert cfg.promotion.comparator_ranking[-1].key == "config_hash"


def test_allowed_space_merges_defaults():
    cfg = load_agent_run_config(SMOKE_CONFIG)
    space = cfg.allowed_parameter_space
    # 明確指定的 bound 生效…
    assert space.bound_for("top_k") == (3, 8)
    # …未指定的沿用預設。
    assert space.bound_for("candidate_cap") == (1, 200)
    assert "strategy" in space.allowed_chunk_fields()
    assert "identifier_vector_floor" in space.allowed_retrieval_fields()
    assert space.is_int_field("top_k")
    assert not space.is_int_field("score_threshold")


def test_build_run_config_single_trial_keeps_collections():
    cfg = load_agent_run_config(SMOKE_CONFIG)
    rc = cfg.build_run_config("tmp/tune.cases.json", cfg.baseline, "v12-agent-smoke-tune")
    assert rc.dataset_path == "tmp/tune.cases.json"
    assert len(rc.trials) == 1
    assert rc.corpus_paths == cfg.corpus_paths
    # collection 生命週期由 Orchestrator 掌控 → 必須 keep。
    assert rc.keep_trial_collections is True


# --------------------------------------------------------------------------
# strict rejection
# --------------------------------------------------------------------------
def test_reject_unknown_top_level_field():
    raw = _valid_raw()
    raw["totally_unknown"] = 1
    with pytest.raises(AgentRunConfigError, match="未知欄位"):
        AgentRunConfig.from_raw(raw)


def test_reject_unknown_baseline_field():
    raw = _valid_raw()
    raw["baseline"]["mystery"] = True
    with pytest.raises(AgentRunConfigError, match="baseline 含未知欄位"):
        AgentRunConfig.from_raw(raw)


def test_reject_bad_budget_value():
    raw = _valid_raw()
    raw["budget"]["max_rounds"] = 0
    with pytest.raises(AgentRunConfigError, match=r"budget\.max_rounds"):
        AgentRunConfig.from_raw(raw)


def test_reject_unknown_bounds_field():
    raw = _valid_raw()
    raw["allowed_parameter_space"]["bounds"]["nonexistent"] = [1, 2]
    with pytest.raises(AgentRunConfigError, match="bounds 含未知欄位"):
        AgentRunConfig.from_raw(raw)


def test_reject_baseline_strategy_outside_allowed_space():
    raw = _valid_raw()
    raw["allowed_parameter_space"]["retrieval_strategies"] = ["identifier_aware"]
    with pytest.raises(AgentRunConfigError, match="baseline retrieval strategy"):
        AgentRunConfig.from_raw(raw)


def test_reject_comparator_without_config_hash_last():
    raw = _valid_raw()
    # 移除最後的 config_hash tie-break → 重用 v1.1 的 deterministic 要求應報錯。
    raw["promotion"]["comparator_ranking"] = raw["promotion"]["comparator_ranking"][:-1]
    with pytest.raises(AgentRunConfigError, match="promotion 設定無效"):
        AgentRunConfig.from_raw(raw)


def test_reject_invalid_baseline_chunk_config():
    raw = _valid_raw()
    # overlap >= target 由 v1.1 ChunkingConfig 擋下，經 TrialConfig 轉為 AgentRunConfigError。
    raw["baseline"]["chunk_config"] = {
        "strategy": "heading",
        "target_chars": 100,
        "overlap_chars": 100,
    }
    with pytest.raises(AgentRunConfigError, match="baseline 無效"):
        AgentRunConfig.from_raw(raw)


def test_budget_defaults_and_roundtrip():
    b = AgentBudget.from_raw({"max_rounds": 3})
    assert b.max_rounds == 3
    assert b.max_llm_calls == AgentBudget().max_llm_calls
    assert b.to_dict()["max_rounds"] == 3


def test_allowed_space_rejects_bad_bound_order():
    with pytest.raises(AgentRunConfigError, match="min.*不得大於"):
        AllowedParameterSpace.from_raw({"bounds": {"top_k": [9, 1]}})


def test_config_deepcopy_isolated():
    raw = _valid_raw()
    raw2 = copy.deepcopy(raw)
    AgentRunConfig.from_raw(raw)
    # from_raw 不得就地修改輸入。
    assert raw == raw2
