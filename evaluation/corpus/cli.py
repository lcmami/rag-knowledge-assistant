"""External corpus CLI（docs/external-corpus-api-plan.md §10 Slice B / D）。

    python -m evaluation.corpus.cli fetch    --config <cfg.json> --output <raw.zip>
    python -m evaluation.corpus.cli snapshot --config <cfg.json> [--raw-zip <raw.zip>]
                                             [--output-root <dir>]
    python -m evaluation.corpus.cli stats    --snapshot <dir> [--json <f>] [--markdown <f>]

``fetch`` 是唯一會碰網路的子指令；``snapshot --raw-zip`` 可完全 offline 由既有
raw ZIP 重建 snapshot（Slice B 的驗證即走此路徑）。``stats``（Slice D）只讀既有
snapshot，不碰網路、不碰 Qdrant，但會載入 embedding tokenizer 以計算 token 分布。

安全：不讀寫 ``.env``、不呼叫 LLM、不觸碰 Qdrant 或任何正式 collection。此 API
不需認證，config 內不含任何 secret。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

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


def cmd_stats(args) -> int:
    """由既有 snapshot 產生 corpus statistics（Slice D）。

    token 分布需要真實 tokenizer，故此處建立 :class:`EmbeddingService`；模型必須
    已於本機快取（一律 offline，**絕不下載**）。
    """
    import json

    from evaluation.corpus import stats as stats_module
    from evaluation.v11.hashing import canonical_json

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from app.core.config import Settings
    from app.services.embedding_service import EmbeddingService

    settings = Settings(app_env="optimizer")
    embedding = EmbeddingService(
        model_name=settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        max_tokens=settings.embedding_max_tokens,
    )
    chunk_config = dict(stats_module.PILOT_CHUNK_CONFIG)
    chunk_config.update(
        target_chars=args.target_chars, overlap_chars=args.overlap_chars,
        max_tokens=args.max_tokens,
    )

    start = time.perf_counter()
    payload = stats_module.build_stats(
        args.snapshot, count_tokens=embedding.count_tokens, chunk_config=chunk_config
    )
    payload["runtime"] = {"stats_duration_ms": int((time.perf_counter() - start) * 1000)}

    if args.json:
        snapshot_module._write_text(args.json, canonical_json(payload) + "\n")
        print(f"json:     {args.json}")
    if args.markdown:
        snapshot_module._write_text(args.markdown, stats_module.render_markdown(payload))
        print(f"markdown: {args.markdown}")
    if not args.json and not args.markdown:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
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

    p_stats = sub.add_parser("stats", help="由既有 snapshot 產生 corpus statistics")
    p_stats.add_argument("--snapshot", required=True, help="snapshot 目錄（含 manifest.json）")
    p_stats.add_argument("--json", default=None, help="輸出 canonical JSON 至此路徑")
    p_stats.add_argument("--markdown", default=None, help="輸出 Markdown 報表至此路徑")
    p_stats.add_argument("--target-chars", type=int, default=450)
    p_stats.add_argument("--overlap-chars", type=int, default=80)
    p_stats.add_argument("--max-tokens", type=int, default=480)
    p_stats.set_defaults(func=cmd_stats)
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
