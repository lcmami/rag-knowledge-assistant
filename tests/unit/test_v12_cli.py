"""Slice 6：CLI（show / config 載入）單元測試（不需 Qdrant / LLM）。"""
from __future__ import annotations

import os

from evaluation.v12.agent import cli
from evaluation.v12.agent.orchestrator import AgentOrchestrator, SplitPaths
from evaluation.v12.agent.planner import FakeAgentPlanner
from evaluation.v12.agent.repository import AgentRepository
from tests.unit.v12_agent_fakes import (
    FakeTrialExecutor,
    sample_segment,
    sample_tune_failure,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")
SPLITS = SplitPaths(tune="t.json", validation="v.json", test="x.json")


def test_load_config_overrides_max_rounds():
    rc = cli._load_config(SMOKE_CONFIG, 1)
    assert rc.budget.max_rounds == 1
    rc2 = cli._load_config(SMOKE_CONFIG, None)
    assert rc2.budget.max_rounds == 2  # smoke 預設


def test_show_command_prints_run(tmp_path, capsys):
    rc = cli._load_config(SMOKE_CONFIG, 1)
    ex = FakeTrialExecutor(segments=(sample_segment(),), tune_failures=(sample_tune_failure(),))
    result = AgentOrchestrator(rc, FakeAgentPlanner(), ex, id_factory=lambda: "run-cli").run(
        splits=SPLITS
    )
    db = str(tmp_path / "agent.db")
    AgentRepository(db).write_run_summary(result)

    rc_code = cli.main(["show", "run-cli", "--agent-db", db])
    assert rc_code == 0
    out = capsys.readouterr().out
    assert "run-cli" in out
    assert "round 1" in out
    assert "not_evaluated" in out  # recommendation eligibility


def test_default_planner_is_fake():
    # 預設 planner=fake，不得因環境存在 API key 就走 llm。
    assert cli._build_planner("fake").planner_type == "fake"
