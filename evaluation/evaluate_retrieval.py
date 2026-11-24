"""Retrieval Evaluation CLI（規格 §29 / Phase 4 §8、§9）。

v1.1 Slice 0：本檔改為 :mod:`evaluation.runner` 的薄包裝，核心邏輯（建服務 / 匯入 /
評估 / threshold 掃描 / 統計）已抽為可 import 模組；CLI 執行方式與 results.json 結構不變。

直接呼叫 Retrieval Service（不經 LLM），在獨立的 evaluation collection 上量測：
- Hit@1 / Hit@3（以 expected_document 判定），另記 section hit 與 page hit。
- 平均 / P50 / P95 retrieval latency。
- 可回答問題 Top-1 score 分布、無關問題最高 score 分布。
- 多個候選 threshold 下的接受 / 拒答 / 誤拒 / 誤接受統計，供 threshold 校準。

實際數字只能來自本次執行結果，不得預先捏造。

使用：
    python evaluation/evaluate_retrieval.py
              [--qdrant-url http://localhost:6333]
              [--collection factory_evaluation]
              [--top-k 10]
              [--keep-collection]      # 保留 evaluation collection（預設執行後刪除）

注意：預設使用獨立 collection「factory_evaluation」，不會觸碰正式 factory_knowledge。

輸出：``evaluation/runs/results.json``（runtime 產物，已被 .gitignore 排除）。
v1.0 當時的結果快照保存於 ``docs/reports/results.json``（historical evidence，本腳本不會覆寫）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 允許以 `python evaluation/evaluate_retrieval.py` 直接執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import runner  # noqa: E402
from evaluation.schemas import EvaluationConfig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
QUESTIONS_PATH = os.path.join(ROOT, "evaluation", "questions.json")
RESULTS_PATH = os.path.join(ROOT, "evaluation", "runs", "results.json")


def build_config(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        corpus_dirs=(SAMPLE_DIR,),
        cases_path=QUESTIONS_PATH,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        top_k=args.top_k,
        score_threshold=0.0,
        keep_collection=args.keep_collection,
        label="legacy-cli",
    )


def _print_summary(legacy: dict) -> None:
    print(f"[3/3] 評估完成，結果已寫入 {RESULTS_PATH}\n")
    print("=== 摘要 ===")
    print(
        f"題目總數={legacy['total_questions']}  "
        f"可回答={legacy['answerable_count']}  無關={legacy['unanswerable_count']}"
    )
    print(f"Hit@1={legacy['hit_at_1']}   Hit@3={legacy['hit_at_3']}")
    print(
        f"Section Hit@1={legacy['section_hit_at_1']} "
        f"(considered={legacy['section_hit_considered']})   "
        f"Page Hit@1={legacy['page_hit_at_1']} "
        f"(considered={legacy['page_hit_considered']})"
    )
    print(
        f"Latency  avg={legacy['average_latency_ms']}ms  "
        f"p50={legacy['p50_latency_ms']}ms  p95={legacy['p95_latency_ms']}ms"
    )
    print(f"可回答 Top-1 score 分布: {legacy['answerable_score_summary']}")
    print(f"無關   最高 score 分布: {legacy['unanswerable_score_summary']}")
    print("\n=== Threshold 掃描 ===")
    header = (
        f"{'thr':>5} {'ans_accept':>10} {'ans_falsereject':>15} "
        f"{'un_refuse':>10} {'un_falseaccept':>14} {'precision':>9}"
    )
    print(header)
    for rep in legacy["threshold_sweep"]:
        print(
            f"{rep['threshold']:>5.2f} {rep['answerable_accept_rate']:>10.2f} "
            f"{rep['answerable_false_reject_rate']:>15.2f} {rep['unanswerable_refuse_rate']:>10.2f} "
            f"{rep['unanswerable_false_accept_rate']:>14.2f} {rep['precision']:>9.2f}"
        )


def run_cli(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default="factory_evaluation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--keep-collection", action="store_true")
    args = parser.parse_args(argv)

    config = build_config(args)
    print(f"[1/3] 匯入 sample documents 至 collection「{config.collection}」…")
    print(f"[2/3] 執行檢索評估（top_k={config.top_k}, threshold=0.0）…")

    # 核心：CLI 與 import 共用同一 runner。
    report = runner.run_evaluation(config)
    legacy = report.to_legacy_dict()

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False, indent=2)

    _print_summary(legacy)
    if config.keep_collection:
        print(f"\n已保留 evaluation collection「{config.collection}」。")
    else:
        print(f"\n已刪除 evaluation collection「{config.collection}」。")
    return legacy


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
