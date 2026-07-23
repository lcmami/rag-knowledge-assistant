"""v1.2 Agent CLI（Slice 6 §十）。

    python -m evaluation.v12.agent.cli optimize --config <json> --max-rounds N [--planner fake|llm]
    python -m evaluation.v12.agent.cli show <agent_run_id> --agent-db <path>

``--planner`` 預設 ``fake``；**真實 LLM 必須明確指定 ``--planner llm``**，不會因環境存在
API Key 就偷偷呼叫真實 provider。整個流程不 Apply、不發布 Dataset、不碰 ``factory_knowledge``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Optional

from evaluation.v11 import hashing
from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.search.split_dataset import (
    SPLIT_TEST,
    SPLIT_TUNE,
    SPLIT_VALIDATION,
    prepare_split_datasets,
)
from evaluation.v12.agent.executor import RealTrialExecutor
from evaluation.v12.agent.orchestrator import AgentOrchestrator, SplitPaths
from evaluation.v12.agent.planner import FakeAgentPlanner
from evaluation.v12.agent.probes import DiagnosticProbeRunner
from evaluation.v12.agent.repository import AgentRepository
from evaluation.v12.agent.run_config import AgentRunConfig

ROOT = hashing.repo_root()


def _load_config(path: str, max_rounds: Optional[int]) -> AgentRunConfig:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if max_rounds is not None:
        raw.setdefault("budget", {})["max_rounds"] = max_rounds
    return AgentRunConfig.from_raw(raw)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _build_planner(kind: str):
    if kind == "fake":
        return FakeAgentPlanner()
    # 真實 LLM 必須明確指定；不得因環境存在 API Key 就自動呼叫。
    from app.core.config import get_settings
    from app.services.llm_client import LLMClient

    from evaluation.v12.agent.llm_planner import LLMAgentPlanner

    settings = get_settings()
    if not settings.llm_configured:
        raise SystemExit(
            "planner=llm 需要 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 環境變數；未設定。"
        )
    return LLMAgentPlanner(LLMClient(settings))


def _cmd_optimize(args) -> int:
    rc = _load_config(args.config, args.max_rounds)
    workdir = args.workdir or tempfile.mkdtemp(prefix="v12agent-")
    os.makedirs(workdir, exist_ok=True)
    agent_db = args.agent_db or os.path.join(workdir, "agent.db")
    optimizer_db = args.optimizer_db or os.path.join(workdir, "optimizer.db")

    manifest = prepare_split_datasets(_abs(rc.dataset_path), os.path.join(workdir, "splits"))
    tune = manifest.splits[SPLIT_TUNE].dataset_path
    validation = manifest.splits[SPLIT_VALIDATION].dataset_path
    if not tune or not validation:
        raise SystemExit("dataset 缺少 tune 或 validation split，無法執行 Agent。")
    test = manifest.splits[SPLIT_TEST].dataset_path if SPLIT_TEST in manifest.splits else None
    splits = SplitPaths(tune=tune, validation=validation, test=test)

    from evaluation.v11.real_services import RealTrialServiceProvider

    planner = _build_planner(args.planner)
    provider = RealTrialServiceProvider.create(
        args.qdrant_url, os.path.join(workdir, "provider")
    )
    repo = OptimizerRepository(optimizer_db)
    executor = RealTrialExecutor(rc, provider, repo, output_dir=os.path.join(workdir, "reports"))
    orch = AgentOrchestrator(
        rc, planner, executor, probe_runner=DiagnosticProbeRunner(executor)
    )
    try:
        result = orch.run(splits=splits)
    finally:
        provider.close()

    AgentRepository(agent_db).write_run_summary(result)
    _print_summary(result, agent_db)
    return 0


def _cmd_show(args) -> int:
    repo = AgentRepository(args.agent_db)
    run = repo.get_run(args.agent_run_id)
    if run is None:
        raise SystemExit(f"找不到 agent_run_id={args.agent_run_id!r}。")
    print(json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True))
    print("--- rounds ---")
    for r in repo.list_rounds(args.agent_run_id):
        print(
            f"round {r['round_number']}: {r['promotion_decision']} "
            f"({r['promotion_reason']}) candidate={r['candidate_config_hash']}"
        )
    return 0


def _print_summary(result, agent_db: str) -> None:
    lines = [
        "=== v1.2 Agent Run Summary ===",
        f"Run ID:                    {result.agent_run_id}",
        f"Planner Type:              {result.planner_type}",
        f"Baseline Config:           {result.baseline_config_hash[:16]}…",
        f"Final Active Parent Config:{result.final_parent_config_hash[:16]}…",
        f"Candidates Proposed:       {result.candidates_proposed}",
        f"Candidates Promoted:       {result.candidates_promoted}",
        f"Candidates Rejected:       {result.candidates_rejected}",
        f"Rounds:                    {len(result.rounds)}",
        f"LLM Calls:                 {result.llm_calls}",
        f"Probe Count:               {result.probe_count}",
        f"Metrics Before:            {json.dumps(result.baseline_metric_summary, ensure_ascii=False)}",
        f"Metrics After:             {json.dumps(result.final_metric_summary, ensure_ascii=False)}",
        f"Validation Gate (last):    {_last_gate(result)}",
        f"Stop Reason:               {result.stop_reason}",
        f"Status:                    {result.status}",
        f"Recommendation Eligibility:{result.recommendation_eligibility}",
        f"Residual Collections:      {result.residual_collections}",
        f"Agent DB:                  {agent_db}",
    ]
    print("\n".join(lines))


def _last_gate(result) -> str:
    for rec in reversed(result.rounds):
        cr = rec.constraint_result or {}
        if "feasible" in cr:
            return "pass" if cr["feasible"] else "fail"
    return "not_evaluated"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.v12.agent.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_opt = sub.add_parser("optimize", help="執行一次有界 Agent 最佳化")
    p_opt.add_argument("--config", required=True)
    p_opt.add_argument("--max-rounds", type=int, default=None)
    p_opt.add_argument("--planner", choices=["fake", "llm"], default="fake")
    p_opt.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    p_opt.add_argument("--workdir", default=None)
    p_opt.add_argument("--agent-db", default=None)
    p_opt.add_argument("--optimizer-db", default=None)

    p_show = sub.add_parser("show", help="顯示一次 Agent Run 的摘要與每輪決策")
    p_show.add_argument("agent_run_id")
    p_show.add_argument("--agent-db", required=True)

    args = parser.parse_args(argv)
    if args.command == "optimize":
        return _cmd_optimize(args)
    if args.command == "show":
        return _cmd_show(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
