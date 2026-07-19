"""Optimizer CLI（Slice 2 §十四）—— 與 import 共用同一核心 Runner。

    python -m evaluation.v11.cli run     --config <cfg.json> --db <optimizer.db> --output-dir <dir>
    python -m evaluation.v11.cli resume  --run-id <id> --db <optimizer.db> --output-dir <dir>
    python -m evaluation.v11.cli cleanup --run-id <id> --db <optimizer.db>

安全：不讀寫 .env、不呼叫 LLM、不觸碰正式 collection / metadata.db / uploads。
CLI 僅組裝依賴（config、DB path、Qdrant client、output dir）後委派 :mod:`trial_runner`。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

from evaluation.v11 import hashing, trial_runner
from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.run_models import RunConfig

DEFAULT_OUTPUT_DIR = os.path.join(hashing.repo_root(), "evaluation", "runs")


def _qdrant_url(explicit: str | None) -> str:
    return explicit or os.getenv("QDRANT_URL", "http://localhost:6333")


def _make_provider(qdrant_url: str, workdir: str):
    from evaluation.v11.real_services import RealTrialServiceProvider

    return RealTrialServiceProvider.create(qdrant_url, workdir)


def _load_run_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        return RunConfig.from_raw(json.load(f))


def cmd_run(args) -> int:
    run_config = _load_run_config(args.config)
    repo = OptimizerRepository(args.db)
    workdir = tempfile.mkdtemp(prefix="optimizer_run_")
    provider = _make_provider(_qdrant_url(args.qdrant_url), workdir)
    try:
        result = trial_runner.execute_run(
            run_config, repo, provider,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    finally:
        provider.close()
        shutil.rmtree(workdir, ignore_errors=True)
    _print_summary(result)
    return 0 if result.run.status == "completed" else 1


def cmd_resume(args) -> int:
    repo = OptimizerRepository(args.db)
    run = repo.get_run(args.run_id)
    if run is None:
        print(f"找不到 run_id={args.run_id!r}", file=sys.stderr)
        return 2
    if args.config:
        run_config = _load_run_config(args.config)
    else:
        if not run.config_json:
            print("該 run 無保存 config，且未提供 --config", file=sys.stderr)
            return 2
        run_config = RunConfig.from_raw(json.loads(run.config_json))
    workdir = tempfile.mkdtemp(prefix="optimizer_resume_")
    provider = _make_provider(_qdrant_url(args.qdrant_url), workdir)
    try:
        result = trial_runner.execute_run(
            run_config, repo, provider,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    finally:
        provider.close()
        shutil.rmtree(workdir, ignore_errors=True)
    _print_summary(result)
    return 0 if result.run.status == "completed" else 1


def cmd_cleanup(args) -> int:
    repo = OptimizerRepository(args.db)
    workdir = tempfile.mkdtemp(prefix="optimizer_cleanup_")
    provider = _make_provider(_qdrant_url(args.qdrant_url), workdir)
    try:
        results = trial_runner.cleanup_run(args.run_id, repo, provider)
    finally:
        provider.close()
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"Cleanup run_id={args.run_id}:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    return 0


def _print_summary(result) -> None:
    run = result.run
    print(f"Run {run.run_id} status={run.status}")
    print(f"  report.json: {result.report_json_path}")
    print(f"  report.md:   {result.report_markdown_path}")
    print(f"  executed trials: {len(result.executed_trial_ids)} "
          f"skipped: {len(result.skipped_trial_ids)}")
    if result.remaining_collections:
        print(f"  ⚠️ remaining collections: {result.remaining_collections}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation.v11.cli", description="v1.1 Retrieval Optimizer (Slice 2)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="執行一個 Run（明確列出的 Trial）")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--db", required=True)
    p_run.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--qdrant-url", default=None)
    p_run.set_defaults(func=cmd_run)

    p_res = sub.add_parser("resume", help="Resume 未完成 Trial")
    p_res.add_argument("--run-id", required=True)
    p_res.add_argument("--db", required=True)
    p_res.add_argument("--config", default=None)
    p_res.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_res.add_argument("--qdrant-url", default=None)
    p_res.set_defaults(func=cmd_resume)

    p_clean = sub.add_parser("cleanup", help="刪除一個 Run 的所有 Trial collection")
    p_clean.add_argument("--run-id", required=True)
    p_clean.add_argument("--db", required=True)
    p_clean.add_argument("--qdrant-url", default=None)
    p_clean.set_defaults(func=cmd_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
