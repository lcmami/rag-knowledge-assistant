"""產生 Retrieval Optimizer UI 的 demo fixture（Slice 8 §十三）。

用法：
    python -m scripts.seed_optimizer_ui_demo --output-dir evaluation/demo
    python -m scripts.seed_optimizer_ui_demo --output-dir evaluation/demo --overwrite

產生三個 fixture DB（search / review / deployment）：
    <output-dir>/optimizer-demo.search.db
    <output-dir>/optimizer-demo.review.db
    <output-dir>/optimizer-demo.deployment.db

安全保證（§十三）：
- 只寫入明確傳入的 output-dir；既有檔案預設拒絕覆寫（--overwrite 才覆寫）。
- **不**修改真實 search / locked search、**不**建立 Qdrant collection / alias、**不**呼叫 LLM。
- Fixture 帶 demo_fixture=true 標記；忠實呈現 ineligible recommendation，絕不偽造 apply 成功。
- 產生的 DB 檔位於 gitignored 路徑，不入庫。

把這三個路徑設給 FastAPI（read-only 讀取）即可在 UI 展示：
    OPTIMIZER_SEARCH_DB_PATH=evaluation/demo/optimizer-demo.search.db
    OPTIMIZER_REVIEW_DB_PATH=evaluation/demo/optimizer-demo.review.db
    OPTIMIZER_DEPLOYMENT_DB_PATH=evaluation/demo/optimizer-demo.deployment.db
"""
from __future__ import annotations

import argparse
import sys

from evaluation.v11.reporting.demo_fixture import build_demo_fixtures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed Retrieval Optimizer UI demo fixtures.")
    parser.add_argument(
        "--output-dir", required=True,
        help="fixture DB 輸出目錄（必填；建議 evaluation/demo，已 gitignore）。",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆寫既有 fixture（預設拒絕覆寫）。",
    )
    args = parser.parse_args(argv)

    try:
        paths = build_demo_fixtures(args.output_dir, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"拒絕覆寫：{exc}", file=sys.stderr)
        return 2

    print("已產生 demo fixture（demo_fixture=true；非真實 production run）：")
    for kind, path in paths.items():
        print(f"  {kind:11s} {path}")
    print()
    print("設定下列環境變數後啟動 API 即可在 UI 展示（read-only）：")
    print(f"  OPTIMIZER_SEARCH_DB_PATH={paths['search']}")
    print(f"  OPTIMIZER_REVIEW_DB_PATH={paths['review']}")
    print(f"  OPTIMIZER_DEPLOYMENT_DB_PATH={paths['deployment']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
