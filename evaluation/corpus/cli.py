"""External corpus CLI（docs/external-corpus-api-plan.md §10 Slice B）。

    python -m evaluation.corpus.cli fetch    --config <cfg.json> --output <raw.zip>
    python -m evaluation.corpus.cli snapshot --config <cfg.json> [--raw-zip <raw.zip>]
                                             [--output-root <dir>]

``fetch`` 是唯一會碰網路的子指令；``snapshot --raw-zip`` 可完全 offline 由既有
raw ZIP 重建 snapshot（Slice B 的驗證即走此路徑）。

安全：不讀寫 ``.env``、不呼叫 LLM、不觸碰 Qdrant 或任何正式 collection。此 API
不需認證，config 內不含任何 secret。

``stats`` 子指令屬 **Slice D**，此處刻意不實作。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

from evaluation.corpus import snapshot as snapshot_module
from evaluation.corpus.moj_law import MojLawError, fetch_raw
from evaluation.corpus.snapshot import (
    CorpusConfigError,
    SnapshotError,
    build_snapshot,
    load_config,
)


def cmd_fetch(args) -> int:
    """下載來源 ZIP 並原樣寫出。**唯一會碰網路的路徑。**"""
    config = load_config(args.config)
    raw = fetch_raw(config.source)
    parent = os.path.dirname(os.path.abspath(args.output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    snapshot_module._write_bytes(args.output, raw)
    print(f"endpoint:   {config.source.endpoint}")
    print(f"output:     {args.output}")
    print(f"raw_bytes:  {len(raw)}")
    print(f"raw_sha256: {hashlib.sha256(raw).hexdigest()}")
    return 0


def cmd_snapshot(args) -> int:
    config = load_config(args.config)
    raw_zip = None
    if args.raw_zip:
        with open(args.raw_zip, "rb") as handle:
            raw_zip = handle.read()

    result = build_snapshot(config, output_root=args.output_root, raw_zip=raw_zip)
    manifest = result.manifest
    print(f"snapshot_id:    {result.snapshot_id}")
    print(f"path:           {result.path}")
    print(f"corpus_hash:    {manifest.corpus_hash}")
    print(f"selection_hash: {manifest.selection_hash}")
    print(f"raw_sha256:     {manifest.raw_sha256}")
    print(
        f"documents:      {manifest.normalized_document_count} "
        f"(selected {manifest.selected_record_count} / source {manifest.source_record_count})"
    )
    print(f"articles:       {manifest.article_count}")
    if manifest.skipped_record_count:
        print(
            f"skipped:        {manifest.skipped_record_count} "
            f"{manifest.skipped_record_reason_counts}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation.corpus.cli",
        description="External legal corpus fetch / snapshot (Slice B)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="下載來源 ZIP 並寫出（唯一會碰網路的子指令）")
    p_fetch.add_argument("--config", required=True)
    p_fetch.add_argument("--output", required=True)
    p_fetch.set_defaults(func=cmd_fetch)

    p_snap = sub.add_parser("snapshot", help="產生 immutable snapshot（manifest + normalized）")
    p_snap.add_argument("--config", required=True)
    p_snap.add_argument("--raw-zip", default=None, help="改由既有 raw ZIP 建立，完全不碰網路")
    p_snap.add_argument("--output-root", default=None, help="覆寫 config 的 output_root")
    p_snap.set_defaults(func=cmd_snapshot)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CorpusConfigError as exc:
        print(f"config 錯誤：{exc}", file=sys.stderr)
        return 2
    except (SnapshotError, MojLawError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"I/O 錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
