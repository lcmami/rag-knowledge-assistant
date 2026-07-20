"""Optimizer Search CLI（Slice 5）。

    python -m evaluation.v11.search.cli search  --config <search.json> --db <optimizer.db> --output-dir <dir>
    python -m evaluation.v11.search.cli resume  --search-id <id> --config <search.json> --db <optimizer.db>
    python -m evaluation.v11.search.cli cleanup --search-id <id> --db <optimizer.db>

刻意與 Slice 2 的 :mod:`evaluation.v11.cli` 分開，避免更動既有 CLI 的行為。

安全：不讀寫 ``.env``、不呼叫 LLM、不觸碰正式 collection / metadata.db / uploads、
不套用任何 recommendation 到正式系統。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

from evaluation.v11 import hashing
from evaluation.v11.search import search_runner
from evaluation.v11.search.search_config import SearchConfig
from evaluation.v11.search.search_repository import SearchRepository

DEFAULT_OUTPUT_DIR = os.path.join(hashing.repo_root(), "evaluation", "searches")


def _qdrant_url(explicit: str | None) -> str:
    return explicit or os.getenv("QDRANT_URL", "http://localhost:6333")


def _make_provider(qdrant_url: str, workdir: str):
    from evaluation.v11.real_services import RealTrialServiceProvider

    return RealTrialServiceProvider.create(qdrant_url, workdir)


def _load(path: str) -> SearchConfig:
    with open(path, encoding="utf-8") as f:
        return SearchConfig.from_raw(json.load(f))


def _execute(args, search_id: str | None) -> int:
    search_config = _load(args.config)
    repo = SearchRepository(args.db)
    workdir = tempfile.mkdtemp(prefix="optimizer_search_")
    provider = _make_provider(_qdrant_url(args.qdrant_url), workdir)
    try:
        result = search_runner.execute_search(
            search_config,
            repo,
            provider,
            output_dir=args.output_dir,
            search_id=search_id,
        )
    finally:
        provider.close()
        shutil.rmtree(workdir, ignore_errors=True)
    _print_summary(result)
    return 0


def cmd_search(args) -> int:
    return _execute(args, None)


def cmd_resume(args) -> int:
    repo = SearchRepository(args.db)
    if repo.get_search(args.search_id) is None:
        print(f"找不到 search_id={args.search_id!r}", file=sys.stderr)
        return 2
    return _execute(args, args.search_id)


def cmd_cleanup(args) -> int:
    repo = SearchRepository(args.db)
    workdir = tempfile.mkdtemp(prefix="optimizer_search_cleanup_")
    provider = _make_provider(_qdrant_url(args.qdrant_url), workdir)
    try:
        results = search_runner.cleanup_search(args.search_id, repo, provider)
    finally:
        provider.close()
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"Cleanup search_id={args.search_id}:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    return 0


def _print_summary(result) -> None:
    s = result.search
    print(f"Search {s.search_id} status={s.status}")
    print(f"  mode={s.search_mode} seed={s.random_seed} budget={s.max_candidates}")
    g = result.generation.stats()
    print(
        f"  candidates: raw={g['raw_combination_count']} invalid={g['invalid_count']} "
        f"duplicate={g['duplicate_count']} executable={g['executable_candidate_count']}"
    )
    print(f"  recommended_config_hash: {s.recommended_config_hash}")
    print(f"  recommendation_locked:   {s.recommendation_locked}")
    print(f"  test_evaluated:          {s.test_evaluated}")
    print(f"  report.json: {result.report_json_path}")
    print(f"  report.md:   {result.report_markdown_path}")
    if result.remaining_collections:
        print(f"  ⚠️ remaining collections: {result.remaining_collections}")
    for note in result.notes:
        print(f"  note: {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation.v11.search.cli",
        description="v1.1 Retrieval Optimizer — Search / Constraints / Pareto / Recommendation (Slice 5)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="執行一次 optimizer search")
    p.add_argument("--config", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--qdrant-url", default=None)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("resume", help="Resume 未完成的 search")
    p.add_argument("--search-id", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--qdrant-url", default=None)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("cleanup", help="刪除該 search 全部 split run 的 trial collection")
    p.add_argument("--search-id", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--qdrant-url", default=None)
    p.set_defaults(func=cmd_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
