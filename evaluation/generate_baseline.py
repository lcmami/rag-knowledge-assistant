"""產生 v1.1 Slice 0 Baseline 報告（規格 §九、§十）。

在隔離 collection（預設 factory_baseline_v1_1）上，對 sample_docs + evaluation/fixtures
的 corpus 執行 `evaluation/cases.json`，輸出：

    evaluation/baseline-v1.1.json
    evaluation/baseline-v1.1.md

並依實際 Baseline **推導**四項硬限制建議（不憑空填、附推導方式與可信度）。

使用：
    python evaluation/generate_baseline.py
              [--qdrant-url http://localhost:6333]
              [--collection factory_baseline_v1_1]
              [--top-k 5]
              [--report-threshold 0.55]

安全：不觸碰 factory_knowledge / 正式 SQLite / 正式文件；不呼叫任何 LLM。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import runner  # noqa: E402
from evaluation.schemas import EvaluationConfig, EvaluationReport  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
FIXTURES_DIR = os.path.join(ROOT, "evaluation", "fixtures")
CASES_PATH = os.path.join(ROOT, "evaluation", "cases.json")
JSON_OUT = os.path.join(ROOT, "evaluation", "baseline-v1.1.json")
MD_OUT = os.path.join(ROOT, "evaluation", "baseline-v1.1.md")

ALARM_DOC = "equipment_alarm_manual.md"
CONFUSABLE_DOC = "confusable_alarm_d900.md"

# 樣本數低於此值時，硬限制建議標為低可信度。
CONFIDENCE_MIN_SAMPLES = 15


def _confidence(n: int) -> str:
    if n == 0:
        return "none (no samples)"
    if n < CONFIDENCE_MIN_SAMPLES:
        return f"low (n={n} < {CONFIDENCE_MIN_SAMPLES})"
    return f"medium (n={n})"


def derive_hard_limits(report: EvaluationReport) -> dict:
    """依 Baseline 實際數字推導四項硬限制建議（§十）。

    每項都附：value、formula（推導方式）、baseline 依據、confidence。
    這些只是「建議值」，僅供寫入 Baseline 報告 / Slice 0 設定草案，
    不得直接修改正式 Retrieval Config，也不宣稱適用所有資料集。
    """
    m = report.metrics

    exact = m["exact_code_top1"]
    exact_n = exact["considered"]
    exact_floor = exact["accuracy"]  # 不得低於目前 baseline（no-regression floor）

    fp = m["unanswerable_false_positive"]
    fp_n = fp["considered"]
    fp_gran = round(1.0 / fp_n, 4) if fp_n else None
    # 上限 = baseline FP；若 baseline 為 0，取一個 case 的粒度作為可觀測上限。
    fp_ceiling = fp["rate"] if fp["rate"] > 0 else (fp_gran if fp_gran else 0.0)

    p95 = m["p95_latency_ms"]
    # 上限 = max(p95 × 1.5, p95 + 50ms)，給單機量測留合理餘裕，取整數毫秒。
    p95_ceiling = int(math.ceil(max(p95 * 1.5, p95 + 50)))

    sem = m["by_query_type"].get("semantic", {})
    sem_n = sem.get("hit_at_5_considered", 0)
    sem_recall5 = sem.get("hit_at_5", 0.0)
    # ε = 一個 case 翻轉的粒度（可觀測到的最小退化）。
    sem_epsilon = round(1.0 / sem_n, 4) if sem_n else None

    return {
        "exact_code_accuracy_floor": {
            "suggested_value": exact_floor,
            "formula": "baseline exact_code Top-1 accuracy（no-regression floor：候選不得低於目前值）",
            "baseline_basis": f"{exact['hits']}/{exact_n} exact_code 案例 Top-1 命中 = {exact_floor}",
            "confidence": _confidence(exact_n),
            "notes": "樣本少，建議 reviewer 保守採用；放寬一個 case 約等於 1/n 的幅度。",
        },
        "unanswerable_false_positive_ceiling": {
            "suggested_value": fp_ceiling,
            "formula": (
                f"baseline unanswerable FP rate @ threshold={fp['threshold']}"
                "；baseline 為 0 時採單一 case 粒度 1/n 作為可觀測上限"
            ),
            "baseline_basis": (
                f"{fp['false_positives']}/{fp_n} 無關題 top1_score ≥ {fp['threshold']}"
                f"（rate={fp['rate']}；粒度 1/n={fp_gran}）"
            ),
            "confidence": _confidence(fp_n),
            "notes": "此值與 report_threshold 綁定；更換 threshold 需重新推導。",
        },
        "p95_latency_ceiling_ms": {
            "suggested_value": p95_ceiling,
            "formula": "ceil(max(baseline_p95 × 1.5, baseline_p95 + 50ms))",
            "baseline_basis": f"baseline P95 = {p95} ms（單機、單次量測）",
            "confidence": "low (single-run, hardware-dependent)",
            "notes": "延遲高度依賴硬體與 Qdrant 部署，跨環境不可直接套用。",
        },
        "semantic_recall_at_5_epsilon": {
            "suggested_value": sem_epsilon,
            "formula": "ε = 1 / semantic_recall@5_considered（一個 case 翻轉的最小可觀測退化）",
            "baseline_basis": (
                f"semantic Recall@5 = {sem_recall5}（n={sem_n}）；"
                f"候選 Recall@5 應 ≥ baseline − ε"
            ),
            "confidence": _confidence(sem_n),
            "notes": "樣本少，ε 粒度粗；Slice 1 擴充 semantic 案例後應收斂。",
        },
    }


def _rank_summary(report: EvaluationReport, doc: str) -> list[dict]:
    """彙整某文件在各 E102 相關案例中的實際排名。"""
    rows = []
    for c in report.cases:
        if c.expected_document == ALARM_DOC or c.confusable_document == CONFUSABLE_DOC:
            rank = next((t["rank"] for t in c.top5 if t["document"] == doc), None)
            score = next((t["score"] for t in c.top5 if t["document"] == doc), None)
            rows.append({"case_id": c.id, "query_type": c.query_type, "rank": rank, "score": score})
    return rows


def render_markdown(report: EvaluationReport, hard_limits: dict, generated_at: str) -> str:
    env = report.environment
    m = report.metrics
    cfg = report.config
    git = env.get("git", {})

    lines: list[str] = []
    lines.append("# v1.1 Retrieval Optimizer — Slice 0 Baseline 報告")
    lines.append("")
    lines.append(
        "> 本報告為 v1.1 Slice 0 的 Baseline 固定化結果。Case 標註採 **Slice 0 Baseline 格式**"
        "（expected_document / expected_section / expected_page / should_answer），"
        "**Slice 1 才遷移至 Canonical Evidence Span**。"
    )
    lines.append("")

    # 執行資訊
    lines.append("## 執行資訊")
    lines.append("")
    lines.append(f"- 執行時間（UTC）：{generated_at}")
    lines.append(f"- Git commit：{git.get('commit')}")
    lines.append(f"- Git branch：{git.get('branch')}")
    lines.append(f"- Working tree dirty：{git.get('working_tree_dirty')}")
    lines.append(f"- Python：{env.get('python_version')}  /  Platform：{env.get('platform')}")
    lines.append("")

    # 設定
    lines.append("## Retrieval / Chunk 設定")
    lines.append("")
    lines.append(f"- Embedding model：`{env.get('embedding_model')}`（維度 {env.get('vector_size')}, {env.get('distance')}）")
    lines.append(f"- Chunk size / overlap：{env.get('chunk_size_chars')} / {env.get('chunk_overlap_chars')} chars")
    lines.append(f"- Embedding max tokens：{env.get('embedding_max_tokens')}")
    lines.append(f"- Top-K：{cfg.get('top_k')}")
    lines.append(f"- 檢索 score_threshold（收集原始分數）：{cfg.get('score_threshold')}")
    lines.append(f"- 報告 configured threshold（正式接受門檻）：{cfg.get('report_threshold')}")
    lines.append(f"- Candidate multiplier / cap：{cfg.get('candidate_multiplier')} / {cfg.get('candidate_cap')}")
    lines.append(f"- 固定 random seed：{cfg.get('seed')}")
    lines.append("")

    # Corpus
    corpus = report.corpus
    lines.append("## Corpus")
    lines.append("")
    lines.append(f"- 文件數：{corpus.get('document_count')}；總 chunk 數：{corpus.get('total_chunks')}")
    lines.append(f"- Corpus hash（sha256）：`{corpus.get('corpus_hash')}`")
    lines.append("")
    lines.append("| 文件 | chunks | sha256（前 12） |")
    lines.append("| --- | ---: | --- |")
    chunk_counts = corpus.get("ingested_chunk_counts", {})
    for d in corpus.get("documents", []):
        cc = chunk_counts.get(d["filename"], "-")
        lines.append(f"| {d['filename']} | {cc} | `{d['sha256'][:12]}` |")
    lines.append("")

    # Dataset
    ds = report.dataset
    lines.append("## Evaluation Dataset")
    lines.append("")
    lines.append(f"- Cases 檔：`{ds.get('cases_file')}`；案例數：{ds.get('case_count')}")
    lines.append(f"- Dataset hash（sha256）：`{ds.get('dataset_hash')}`")
    lines.append("")
    lines.append("query_type 分布：")
    lines.append("")
    lines.append("| query_type | 案例數 | Hit@1 | Hit@5 | MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for qt, g in m["by_query_type"].items():
        lines.append(f"| {qt} | {g['count']} | {g['hit_at_1']} | {g['hit_at_5']} | {g['mrr']} |")
    lines.append("")

    # Aggregate metrics
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| 指標 | 值 | 備註 |")
    lines.append("| --- | ---: | --- |")
    lines.append(f"| 案例總數 | {m['total_questions']} | 可回答 {m['answerable_count']} / 無關 {m['unanswerable_count']} |")
    lines.append(f"| Hit@1 | {m['hit_at_1']} | considered={m['hit_at_1_considered']} |")
    lines.append(f"| Hit@3 | {m['hit_at_3']} | considered={m['hit_at_3_considered']} |")
    lines.append(f"| Hit@5 / Recall@5 | {m['hit_at_5']} | considered={m['hit_at_5_considered']} |")
    lines.append(f"| MRR | {m['mrr']} | considered={m['mrr_considered']} |")
    lines.append(f"| Section Hit@1 | {m['section_hit_at_1']} | considered={m['section_hit_considered']} |")
    lines.append(f"| Page Hit@1 | {m['page_hit_at_1']} | considered={m['page_hit_considered']} |")
    ex = m["exact_code_top1"]
    lines.append(f"| Exact-code Top-1 Accuracy | {ex['accuracy']} | {ex['hits']}/{ex['considered']}；miss={ex['miss_ids']} |")
    cf = m["confusable_code_error"]
    lines.append(f"| Confusable-code Error Rate | {cf['rate']} | {cf['errors']}/{cf['considered']}；err={cf['error_ids']} |")
    ufp = m["unanswerable_false_positive"]
    lines.append(f"| Unanswerable False-positive Rate | {ufp['rate']} | @thr={ufp['threshold']}；{ufp['false_positives']}/{ufp['considered']}；fp={ufp['false_positive_ids']} |")
    lines.append(f"| Latency avg / P50 / P95 (ms) | {m['average_latency_ms']} / {m['p50_latency_ms']} / {m['p95_latency_ms']} |  |")
    lines.append("")

    # E102 / D900
    lines.append("## E102 / D900 實際排名（Confusable 壓力測試）")
    lines.append("")
    lines.append("同一組 E102 相關案例中，正確 E102 文件與虛構 D900 干擾文件的實際排名（非固定期望值）：")
    lines.append("")
    lines.append("| case | query_type | E102 rank (score) | D900 rank (score) |")
    lines.append("| --- | --- | --- | --- |")
    e102 = {r["case_id"]: r for r in _rank_summary(report, ALARM_DOC)}
    d900 = {r["case_id"]: r for r in _rank_summary(report, CONFUSABLE_DOC)}
    for cid in sorted(e102):
        er = e102[cid]
        dr = d900.get(cid, {})
        lines.append(
            f"| {cid} | {er['query_type']} | {er['rank']} ({er['score']}) | "
            f"{dr.get('rank')} ({dr.get('score')}) |"
        )
    lines.append("")

    # Unanswerable
    lines.append("## Unanswerable 實際結果")
    lines.append("")
    lines.append(f"- Configured threshold：{ufp['threshold']}")
    lines.append(f"- False-positive（top1_score ≥ threshold）：{ufp['false_positive_ids'] or '無'}")
    lines.append("")
    lines.append("| case | 問題 | top1 文件 | top1 score |")
    lines.append("| --- | --- | --- | ---: |")
    for c in report.cases:
        if not c.should_answer:
            lines.append(f"| {c.id} | {c.question} | {c.top1_document} | {c.top1_score} |")
    lines.append("")

    # Per-case Top-5
    lines.append("## 每題 Top-5")
    lines.append("")
    for c in report.cases:
        lines.append(f"### {c.id} [{c.query_type}] — {c.question}")
        lines.append("")
        lines.append(
            f"- expected：{c.expected_document} / {c.expected_section} / page={c.expected_page}"
            f"；should_answer={c.should_answer}；latency={c.retrieval_time_ms}ms"
        )
        lines.append("")
        lines.append("| rank | document | section | page | score |")
        lines.append("| ---: | --- | --- | ---: | ---: |")
        for t in c.top5:
            lines.append(
                f"| {t['rank']} | {t['document']} | {t['section']} | {t['page']} | {t['score']} |"
            )
        lines.append("")

    # Hard limits
    lines.append("## 硬限制門檻建議（由 Baseline 推導）")
    lines.append("")
    lines.append(
        "> 依 Slice 0 實際 Baseline 推導的**建議值**，僅供寫入 Baseline 報告 / Slice 0 設定草案，"
        "**不得直接修改正式 Retrieval Config**，也**不宣稱適用所有資料集**。"
    )
    lines.append("")
    lines.append("| 項目 | 建議值 | 推導方式 | Baseline 依據 | 可信度 |")
    lines.append("| --- | --- | --- | --- | --- |")
    label_map = {
        "exact_code_accuracy_floor": "Exact-code accuracy 下限",
        "unanswerable_false_positive_ceiling": "Unanswerable FP rate 上限",
        "p95_latency_ceiling_ms": "P95 latency 上限 (ms)",
        "semantic_recall_at_5_epsilon": "Semantic Recall@5 退化容忍 ε",
    }
    for key, label in label_map.items():
        h = hard_limits[key]
        lines.append(
            f"| {label} | {h['suggested_value']} | {h['formula']} | {h['baseline_basis']} | {h['confidence']} |"
        )
    lines.append("")

    # Limitations
    lines.append("## 已知限制與尚不可計算的未來 metrics")
    lines.append("")
    for n in report.notes:
        lines.append(f"- {n}")
    lines.append("- 硬限制建議樣本數偏低（尤其 exact_code / confusable / semantic），跨資料集不可直接沿用。")
    lines.append("- 未計算（留待 Slice 1／2）：Evidence Coverage、Graded NDCG、Evidence-level Precision@5、Canonical Evidence overlap。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default="factory_baseline_v1_1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report-threshold", type=float, default=0.55)
    parser.add_argument("--keep-collection", action="store_true")
    args = parser.parse_args()

    if args.collection == "factory_knowledge":
        raise SystemExit("拒絕使用正式 collection factory_knowledge 產生 baseline。")

    config = EvaluationConfig(
        corpus_dirs=(SAMPLE_DIR, FIXTURES_DIR),
        cases_path=CASES_PATH,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        top_k=args.top_k,
        score_threshold=0.0,
        report_threshold=args.report_threshold,
        keep_collection=args.keep_collection,
        label="baseline-v1.1",
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    print(f"[baseline] 執行評估 collection={config.collection} top_k={config.top_k} …")
    report = runner.run_evaluation(config)
    hard_limits = derive_hard_limits(report)

    payload = report.to_dict()
    payload["generated_at"] = generated_at
    payload["hard_limit_suggestions"] = hard_limits

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(render_markdown(report, hard_limits, generated_at))

    m = report.metrics
    print(f"[baseline] 完成。JSON → {JSON_OUT}\n            MD   → {MD_OUT}")
    print(
        f"[baseline] Hit@1={m['hit_at_1']} Hit@3={m['hit_at_3']} Hit@5={m['hit_at_5']} "
        f"MRR={m['mrr']} exact={m['exact_code_top1']['accuracy']} "
        f"confusable_err={m['confusable_code_error']['rate']} "
        f"unans_fp={m['unanswerable_false_positive']['rate']} "
        f"P95={m['p95_latency_ms']}ms"
    )


if __name__ == "__main__":
    main()
