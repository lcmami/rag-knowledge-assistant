"""Slice 6：真實 LLM Agent 一輪 Smoke（預設 Skip）。

驗收（§九）：真實 provider 回一份可解析 / 可驗證的 proposal，交給既有 Trial Runner，
產生 Parent/Candidate Probe Comparison 與正式 Metrics，做出 Promotion Decision 並停止。
**不**要求改善、**不** Apply、**不**發布 Dataset、**不**跑 test split、**不**碰 factory_knowledge。

預設 Skip：需要 ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` 與可連線的 Qdrant。
一般 CI 不帶 secret，故此測試被 ``llm_integration`` marker 排除。
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.llm_integration


def _qdrant_or_skip(url: str):
    import urllib.request

    try:
        urllib.request.urlopen(f"{url}/collections", timeout=3)
    except Exception:  # noqa: BLE001
        pytest.skip("Qdrant 未就緒，跳過真實 LLM smoke。")


@pytest.fixture
def llm_env():
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if not os.environ.get(key):
            pytest.skip(f"未設定 {key}，跳過真實 LLM smoke。")
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


def test_real_llm_one_round(llm_env, tmp_path):
    qdrant_url = llm_env
    _qdrant_or_skip(qdrant_url)

    from app.core.config import get_settings
    from app.services.llm_client import LLMClient
    from evaluation.v11.optimizer_repository import OptimizerRepository
    from evaluation.v11.real_services import RealTrialServiceProvider
    from evaluation.v11.search.split_dataset import (
        SPLIT_TUNE,
        SPLIT_VALIDATION,
        prepare_split_datasets,
    )
    from evaluation.v12.agent.cli import _abs
    from evaluation.v12.agent.executor import RealTrialExecutor
    from evaluation.v12.agent.llm_planner import LLMAgentPlanner
    from evaluation.v12.agent.orchestrator import AgentOrchestrator, SplitPaths
    from evaluation.v12.agent.probes import DiagnosticProbeRunner
    from evaluation.v12.agent.repository import AgentRepository
    from evaluation.v12.agent.run_config import load_agent_run_config

    config_path = os.path.join(_abs("evaluation/configs/v12-agent-smoke.json"))
    rc = load_agent_run_config(config_path)

    manifest = prepare_split_datasets(_abs(rc.dataset_path), str(tmp_path / "splits"))
    splits = SplitPaths(
        tune=manifest.splits[SPLIT_TUNE].dataset_path,
        validation=manifest.splits[SPLIT_VALIDATION].dataset_path,
    )

    settings = get_settings()
    assert settings.llm_configured, "LLM 未設定"
    planner = LLMAgentPlanner(LLMClient(settings))
    provider = RealTrialServiceProvider.create(qdrant_url, str(tmp_path / "provider"))
    repo = OptimizerRepository(str(tmp_path / "optimizer.db"))
    executor = RealTrialExecutor(rc, provider, repo, output_dir=str(tmp_path / "reports"))
    orch = AgentOrchestrator(rc, planner, executor, probe_runner=DiagnosticProbeRunner(executor))
    try:
        result = orch.run(splits=splits)  # smoke 設定 max_rounds=2；一輪即足以驗收
    finally:
        provider.close()

    # 真實 provider 至少完成 baseline，且未 Apply / 未發布 / 未評估 eligibility。
    assert result.status in ("completed", "stopped")
    assert result.recommendation_eligibility == "not_evaluated"
    assert result.residual_collections == []  # 全部隔離 collection 清乾淨
    # audit 寫入。
    agent_repo = AgentRepository(str(tmp_path / "agent.db"))
    agent_repo.write_run_summary(result)
    assert agent_repo.get_run(result.agent_run_id) is not None
    # 若真的產生候選，proposal 必已通過既有 validator（否則不會成為 executed candidate）。
    if result.candidates_proposed:
        assert result.rounds[0].proposal is not None
