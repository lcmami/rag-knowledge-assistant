"""Slice 5A：獨立 Agent DB 最小 Audit 單元測試。"""
from __future__ import annotations

import json
import os

from evaluation.v12.agent.orchestrator import AgentOrchestrator, SplitPaths
from evaluation.v12.agent.planner import FakeAgentPlanner
from evaluation.v12.agent.repository import AgentRepository, redact_secrets
from evaluation.v12.agent.run_config import load_agent_run_config
from tests.unit.v12_agent_fakes import (
    FakeTrialExecutor,
    sample_segment,
    sample_tune_failure,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SMOKE_CONFIG = os.path.join(_REPO_ROOT, "evaluation", "configs", "v12-agent-smoke.json")
SPLITS = SplitPaths(tune="tune.json", validation="validation.json", test="test.json")


def _run_result():
    rc = load_agent_run_config(SMOKE_CONFIG)
    ex = FakeTrialExecutor(segments=(sample_segment(),), tune_failures=(sample_tune_failure(),))
    orch = AgentOrchestrator(rc, FakeAgentPlanner(), ex, id_factory=lambda: "run-1")
    return orch.run(splits=SPLITS)


def test_migration_idempotent(tmp_path):
    db = str(tmp_path / "agent.db")
    r1 = AgentRepository(db)
    assert r1.applied_migrations() == ["0001_agent_runs"]
    # 重新開啟不重複套用。
    r2 = AgentRepository(db)
    assert r2.applied_migrations() == ["0001_agent_runs"]


def test_write_and_read_roundtrip(tmp_path):
    repo = AgentRepository(str(tmp_path / "agent.db"))
    result = _run_result()
    repo.write_run_summary(result)

    run = repo.get_run("run-1")
    assert run is not None
    assert run["recommendation_eligibility"] == "not_evaluated"
    assert run["status"] in ("completed", "stopped")
    assert run["final_metric_summary"] is not None

    rounds = repo.list_rounds("run-1")
    assert len(rounds) == len(result.rounds)
    assert rounds[0]["promotion_decision"] in ("promoted", "rejected")
    assert rounds[0]["proposal"] is not None


def test_write_is_idempotent(tmp_path):
    repo = AgentRepository(str(tmp_path / "agent.db"))
    result = _run_result()
    repo.write_run_summary(result)
    repo.write_run_summary(result)  # 再寫一次
    assert len(repo.list_rounds("run-1")) == len(result.rounds)  # 不重複


def test_no_raw_response_stored_by_default(tmp_path):
    repo = AgentRepository(str(tmp_path / "agent.db"))
    result = _run_result()
    repo.write_run_summary(result)
    for r in repo.list_rounds("run-1"):
        assert r["raw_response_redacted"] is None  # 預設不存 raw

def test_debug_raw_response_is_redacted(tmp_path):
    repo = AgentRepository(str(tmp_path / "agent.db"))
    result = _run_result()
    secret_raw = "Authorization: Bearer abc123\napi_key=sk-SECRETKEY123\nleftover sk-anothersecret99"
    repo.write_run_summary(
        result,
        raw_responses={result.rounds[0].round_number: secret_raw},
        debug_store_raw=True,
    )
    stored = repo.list_rounds("run-1")[0]["raw_response_redacted"]
    assert stored is not None
    assert "abc123" not in stored
    assert "SECRETKEY" not in stored
    assert "anothersecret" not in stored
    assert "[REDACTED]" in stored


def test_redact_secrets_helper():
    out = redact_secrets("api_key=sk-ABCDEFGH12345 and Authorization: Bearer tok_xyz")
    assert "sk-ABCDEFGH12345" not in out
    assert "tok_xyz" not in out


def test_no_secret_in_persisted_json(tmp_path):
    repo = AgentRepository(str(tmp_path / "agent.db"))
    result = _run_result()
    repo.write_run_summary(result)
    # 掃描整個 DB 檔：不得出現 api key / bearer 字樣（正常路徑本就不存 raw/prompt）。
    with open(str(tmp_path / "agent.db"), "rb") as f:
        blob = f.read().lower()
    assert b"authorization:" not in blob
    assert b"bearer " not in blob
