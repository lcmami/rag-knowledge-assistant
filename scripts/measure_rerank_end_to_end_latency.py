"""Slice R2 —— **真正的** uncached end-to-end retrieval latency 實測（v1.2.1）。

R0/R0.5 產出的 ``provisional_end_to_end_p95_ms = 2222.62`` **不是量測值**：它是把
「孤立的 cross-encoder 前向」與「in-memory Qdrant 的 baseline 檢索」兩段**獨立**量測
相加、再補上雜訊邊際得到的估計。R0 產物本身已明載此事並要求 R2 重測。

本 script 就是那次重測。它對**完整 pipeline** 計時::

    query embedding
      -> vector search（真實 Qdrant，含網路 round-trip）
      -> boost/penalty（identifier_aware 時）
      -> admission（threshold / identifier floor）
      -> rerank window W = A[:rerank_top_n]，cross-encoder 依 raw logit 重排
      -> final ranking（W ++ T）[:top_k]

量測規則（與 R0 §0(4) 一致，且刻意更嚴格）：

- ``n >= 30``，warm-up **另外計時並單獨列出**，不混入樣本。
- **rerank cache 一律停用**。cached latency 只作資訊，不得作 feasibility 依據。
- 每次量測輪換 query，讓 token 長度分佈貼近真實。
- 產出 min / p50 / p95 / max / mean / stdev + **完整 raw samples**，
  並附前半/後半平均以偵測熱節流之類的單向漂移。
- 統計量一律由 raw samples 重算並自我校驗（min<=p50<=p95<=max）。

percentile 實作直接重用 :func:`evaluation.v11.retrieval_metrics._percentile` ——
量到的數字要拿來和 ``p95_retrieval_latency_ms`` 這個 metric 比較，兩者必須以完全相同
的方式計算，否則不可比較。

**本 script 不修改 app/、不動正式 ``factory_knowledge``、不修改 dataset。**
它只在唯一名稱的 ``optimizer_`` collection 上建索引、量測、然後刪掉自己建立的東西。

用法::

    python scripts/measure_rerank_end_to_end_latency.py --repeats 30 --warmup 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.v11.retrieval_metrics import _percentile  # noqa: E402

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CORPUS_DIRS = ("sample_docs", "evaluation/fixtures")
DATASET_PATH = "evaluation/datasets/v1.1/cases-v2.json"

#: 預設輸出為 runtime 產物（gitignored）。v1.2.1 R2 當時的實測快照保存於
#: docs/reports/rerank-end-to-end-latency-v1.2.1.{json,md}（historical evidence，不得被重跑覆寫）。
OUT_JSON = "evaluation/runs/rerank-end-to-end-latency-v1.2.1.json"
OUT_MD = "evaluation/runs/rerank-end-to-end-latency-v1.2.1.md"

#: §0(4)：低於此樣本數一律標為 low confidence。
CONFIDENCE_MIN_SAMPLES = 30

#: R0.5 選定並 FROZEN 的 window。**本 script 不得調整它**（R2 禁止調參）。
RERANK_TOP_N = 5
TOP_K = 5

#: v1.1 既有的硬性延遲約束，用於對照（不是本 script 決定的）。
EXISTING_V11_HARD_CONSTRAINT_MS = 86.0

#: **量測範圍的正式名稱。** 這個字串會進產物，用來防止數字被脫離情境引用。
#: 這是「本機原生 Qdrant + loopback」的 end-to-end，**不是** universal production latency。
MEASUREMENT_SCOPE = "local native-Qdrant loopback end-to-end measurement"

#: Qdrant 執行型態 —— 明寫 loopback，避免被讀成「含真實網路 hop」。
QDRANT_MODE = (
    "local native standalone Qdrant over loopback 127.0.0.1 "
    "(real server process, real HTTP/gRPC stack, but loopback RTT)"
)

#: **必須隨數字一起走的偏差聲明。** loopback 沒有實體網路 hop，因此比遠端部署便宜。
LOOPBACK_LIMITATION = (
    "loopback RTT is cheaper than a remote deployment and may underestimate remote "
    "production latency"
)

#: R0/R0.5 的 provisional 估計 —— 只作前後對照，**不是**本次量測值。
R0_PROVISIONAL = {
    "provisional_end_to_end_p95_ms": 2222.6165,
    "rerank_top_n": 5,
    "status": "PROVISIONAL estimate — sum of two independently measured components",
    "provenance": "docs/reports/rerank-latency-baseline-v1.2.1.json",
}

ARMS = {
    "dense_rerank": {
        "strategy": "dense", "top_k": TOP_K, "score_threshold": 0.55,
        "candidate_multiplier": 3, "candidate_cap": 50,
        "rerank_strategy": "cross_encoder", "rerank_top_n": RERANK_TOP_N,
    },
    "dense_baseline": {
        "strategy": "dense", "top_k": TOP_K, "score_threshold": 0.55,
        "candidate_multiplier": 3, "candidate_cap": 50,
    },
    "identifier_aware_rerank": {
        "strategy": "identifier_aware", "top_k": TOP_K, "score_threshold": 0.55,
        "candidate_multiplier": 3, "candidate_cap": 50,
        "identifier_candidate_cap": 20, "exact_match_boost": 0.2,
        "confusable_penalty": 0.15, "identifier_vector_floor": 0.4,
        "rerank_strategy": "cross_encoder", "rerank_top_n": RERANK_TOP_N,
    },
}


class MeasurementError(RuntimeError):
    """量測前提不成立 —— 必須中止，不得產出可疑數字。"""


# --------------------------------------------------------------------------
def _files() -> list[tuple[str, str]]:
    from evaluation.v11 import hashing

    return [
        (os.path.basename(rel), abs_path)
        for rel, abs_path in hashing.collect_corpus_files(list(CORPUS_DIRS), _ROOT)
    ]


def _questions() -> tuple[list[str], str, str]:
    import hashlib

    path = os.path.join(_ROOT, DATASET_PATH)
    with open(path, "rb") as fh:
        raw = fh.read()
    data = json.loads(raw.decode("utf-8"))
    cases = sorted(data.get("cases") or [], key=lambda c: c["id"])
    return (
        [c["question"] for c in cases],
        str(data.get("dataset_version")),
        hashlib.sha256(raw).hexdigest(),
    )


def _trial_config(name: str, retrieval: dict[str, Any]):
    from evaluation.v11.run_models import TrialConfig

    return TrialConfig.from_raw(
        {"trial_name": name, "chunk_config": {"strategy": "current"},
         "retrieval_config": retrieval},
        0,
    )


def _stats(samples: list[float], label: str) -> dict[str, Any]:
    """統計量 + **完整原始樣本**（見模組 docstring 的量測規則）。"""
    n = len(samples)
    mean = statistics.fmean(samples) if samples else None
    stdev = statistics.stdev(samples) if n >= 2 else None
    half = n // 2
    first_half = statistics.fmean(samples[:half]) if half else None
    second_half = statistics.fmean(samples[half:]) if half else None

    return {
        "label": label,
        "n": n,
        "min_ms": round(min(samples), 4) if samples else None,
        "p50_ms": round(_percentile(samples, 0.50), 4) if samples else None,
        "p95_ms": round(_percentile(samples, 0.95), 4) if samples else None,
        "max_ms": round(max(samples), 4) if samples else None,
        "mean_ms": round(mean, 4) if mean is not None else None,
        "stdev_ms": round(stdev, 4) if stdev is not None else None,
        "relative_stdev": round(stdev / mean, 4) if (stdev is not None and mean) else None,
        "first_half_mean_ms": round(first_half, 4) if first_half is not None else None,
        "second_half_mean_ms": round(second_half, 4) if second_half is not None else None,
        "drift_second_minus_first_ms": (
            round(second_half - first_half, 4)
            if (first_half is not None and second_half is not None) else None
        ),
        "raw_samples_ms": [round(v, 4) for v in samples],
        "confidence": (
            "low (below CONFIDENCE_MIN_SAMPLES)" if n < CONFIDENCE_MIN_SAMPLES
            else "moderate (single-machine, hardware-dependent)"
        ),
    }


def verify_stats_consistency(*blocks: dict[str, Any]) -> dict[str, Any]:
    """每個統計量都必須能由 raw samples 重算，且滿足 min<=p50<=p95<=max。

    沿用 R0 的自我稽核（同樣的錯誤曾真實發生過：混用兩次執行的數字）。
    """
    checks: list[dict[str, Any]] = []
    for s in blocks:
        raw = s.get("raw_samples_ms") or []
        if not raw:
            checks.append({"label": s.get("label"), "ok": False, "reason": "no raw samples"})
            continue
        recomputed = {
            "min_ms": min(raw), "p50_ms": _percentile(raw, 0.50),
            "p95_ms": _percentile(raw, 0.95), "max_ms": max(raw),
            "mean_ms": statistics.fmean(raw),
            "stdev_ms": statistics.stdev(raw) if len(raw) >= 2 else None,
        }
        mismatches = {
            k: {"recorded": s.get(k), "recomputed": round(v, 4)}
            for k, v in recomputed.items()
            if v is not None and abs(float(s.get(k)) - v) > 1e-3
        }
        ordered = (
            recomputed["min_ms"] <= recomputed["p50_ms"]
            <= recomputed["p95_ms"] <= recomputed["max_ms"]
        )
        checks.append({
            "label": s.get("label"), "n": len(raw),
            "recompute_mismatches": mismatches,
            "order_invariant_min_p50_p95_max": ordered,
            "ok": (not mismatches) and ordered,
        })
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise MeasurementError(
            "Stats consistency self-check 失敗："
            f"{json.dumps(failed, ensure_ascii=False)}"
        )
    return {
        "description": (
            "every reported statistic was recomputed from raw_samples_ms and the "
            "min<=p50<=p95<=max ordering was asserted"
        ),
        "checks": checks,
        "all_passed": True,
    }


# --------------------------------------------------------------------------
def measure_arm(
    provider,
    collection: str,
    arm_name: str,
    retrieval_cfg: dict[str, Any],
    questions: list[str],
    *,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    """對單一 arm 實測 end-to-end retrieval latency（rerank cache 已停用）。"""
    tc = _trial_config(arm_name, retrieval_cfg)
    index_outcome = provider.build_index(collection, tc, _files())

    rerank_on = bool(retrieval_cfg.get("rerank_strategy", "none") != "none")
    metas: list[dict[str, Any]] = []

    def one(qi: int) -> float:
        question = questions[qi % len(questions)]
        t0 = time.perf_counter()
        out = provider.retrieve(collection, tc, question, TOP_K)
        elapsed = (time.perf_counter() - t0) * 1000.0
        metas.append(out.meta or {})
        return elapsed

    # warm-up 單獨計時、單獨列出，**不混入樣本**。
    warm = [one(i) for i in range(warmup)]
    warm_metas, metas = list(metas), []
    samples = [one(warmup + i) for i in range(repeats)]

    block = _stats(samples, f"{arm_name}_end_to_end_ms")
    block["warm_up_discarded_ms"] = [round(v, 4) for v in warm]
    block["arm"] = arm_name
    block["retrieval_config"] = retrieval_cfg
    block["indexed_point_count"] = index_outcome.point_count
    block["qdrant_mode"] = QDRANT_MODE
    block["measured_pipeline"] = (
        "query embedding -> vector search -> boost/penalty -> admission -> "
        "rerank window -> final ranking"
    )

    if rerank_on:
        # 快取狀態要能被獨立稽核 —— 全 miss 才算 uncached。
        hits = sum(int(m.get("rerank_cache_hits", 0)) for m in metas)
        misses = sum(int(m.get("rerank_cache_misses", 0)) for m in metas)
        windows = [int(m.get("rerank_window_count", 0)) for m in metas]
        rerank_ms = [float(m.get("rerank_latency_ms", 0.0)) for m in metas]
        block["rerank"] = {
            "cache": "DISABLED (uncached measurement)",
            "cache_hits": hits,
            "cache_misses": misses,
            "uncached_verified": hits == 0,
            "rerank_top_n": RERANK_TOP_N,
            "window_size_min": min(windows) if windows else None,
            "window_size_max": max(windows) if windows else None,
            "window_size_mean": round(statistics.fmean(windows), 4) if windows else None,
            "rerank_component_ms": _stats(rerank_ms, f"{arm_name}_rerank_component_ms"),
            "warm_up_window_sizes": [
                int(m.get("rerank_window_count", 0)) for m in warm_metas
            ],
        }
        if hits:
            raise MeasurementError(
                f"arm={arm_name} 出現 {hits} 次 rerank cache hit —— 這不是 uncached 量測。"
                " feasibility 數字不得由暖快取產生。"
            )
    return block


def _git_state() -> dict[str, Any]:
    def run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(args, cwd=_ROOT, capture_output=True, text=True, timeout=15)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def compare_with_slo(arms: dict[str, Any]) -> dict[str, Any]:
    """把實測 p95 與既有 SLO / R0 provisional 估計並列。**只陳述，不決策。**"""
    rerank_arm = arms.get("dense_rerank") or {}
    baseline_arm = arms.get("dense_baseline") or {}
    measured = rerank_arm.get("p95_ms")
    baseline = baseline_arm.get("p95_ms")
    prov = R0_PROVISIONAL["provisional_end_to_end_p95_ms"]

    return {
        "metric_name": "p95_retrieval_latency_ms",
        "status": "MEASURED (real end-to-end, uncached, native standalone Qdrant)",
        "measurement_scope": MEASUREMENT_SCOPE,
        "scope_limitation": LOOPBACK_LIMITATION,
        "must_not_be_described_as": "universal production latency",
        "measured_p95_retrieval_latency_ms": measured,
        "baseline_without_rerank_p95_ms": baseline,
        "rerank_overhead_p95_ms": (
            round(measured - baseline, 4)
            if (measured is not None and baseline is not None) else None
        ),
        "existing_v11_hard_constraint_ms": EXISTING_V11_HARD_CONSTRAINT_MS,
        "ratio_vs_existing_constraint": (
            round(measured / EXISTING_V11_HARD_CONSTRAINT_MS, 2) if measured else None
        ),
        "meets_existing_constraint": (
            bool(measured is not None and measured <= EXISTING_V11_HARD_CONSTRAINT_MS)
        ),
        "r0_provisional_estimate_ms": prov,
        "measured_vs_provisional_ratio": (
            round(measured / prov, 4) if measured else None
        ),
        "provisional_estimate_superseded": True,
        "note": (
            "This is the measured value R3's deterministic gate must use, and it must always "
            f"be cited with its scope: {MEASUREMENT_SCOPE}. The R0/R0.5 provisional estimate "
            f"({prov} ms) was never a measurement and is now superseded. The comparison against "
            "the existing v1.1 constraint is a statement of fact about this implementation on "
            "this machine over loopback; it is NOT a decision about whether to ship, change "
            "model/backend, or restrict rerank to offline analysis, and it is NOT a claim about "
            "universal production latency."
        ),
        "not_decided_here": (
            "No optimisation was attempted or evaluated: no ONNX/OpenVINO, no quantisation, "
            "no model swap, no smaller rerank_top_n, no GPU. R2 is integration + measurement only."
        ),
    }


# --------------------------------------------------------------------------
def render_markdown(doc: dict[str, Any]) -> str:
    lines: list[str] = []
    ap = lines.append
    ap("# Rerank End-to-End Latency — v1.2.1 Slice R2 (MEASURED)")
    ap("")
    ap(f"- Slice: `{doc['slice']}`")
    ap(f"- Git: `{doc['git']['commit']}` on `{doc['git']['branch']}` "
       f"(dirty={doc['git']['dirty']})")
    ap(f"- Qdrant: `{doc['qdrant_mode']}`")
    ap(f"- Dataset: `{doc['dataset']['dataset_version']}` / "
       f"`{doc['dataset']['dataset_hash'][:12]}…`")
    fp = doc.get("rerank_fingerprint") or {}
    ap(f"- Rerank Fingerprint: `{(doc.get('rerank_fingerprint_hash') or '')[:12]}…` "
       f"(model=`{fp.get('model_name')}` revision=`{fp.get('model_revision')}` "
       f"activation=`{fp.get('scored_activation')}`)")
    ap(f"- rerank_top_n: **{RERANK_TOP_N}** (FROZEN by R0.5) · batch_size: "
       f"**{doc.get('rerank_batch_size')}**")
    ap("")
    ap("## Why this file exists")
    ap("")
    ap(f"> {doc['purpose']}")
    ap("")

    ap("## Measured end-to-end latency (uncached)")
    ap("")
    ap(f"**Scope: {doc['measurement_scope']}.**")
    ap("")
    ap(f"> ⚠️ {LOOPBACK_LIMITATION}. This is NOT universal production latency.")
    ap("")
    ap("| arm | n | min | p50 | **p95** | max | mean | stdev | rel.stdev | drift |")
    ap("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, a in doc["arms"].items():
        ap(f"| `{name}` | {a['n']} | {a['min_ms']} | {a['p50_ms']} | **{a['p95_ms']}** | "
           f"{a['max_ms']} | {a['mean_ms']} | {a['stdev_ms']} | {a['relative_stdev']} | "
           f"{a['drift_second_minus_first_ms']} |")
    ap("")
    ap("All values in milliseconds. Warm-up runs were timed separately and are listed "
       "per arm in the JSON (`warm_up_discarded_ms`); they are excluded from every "
       "statistic above.")
    ap("")

    ap("## Rerank component (subset of the end-to-end time, NOT additive)")
    ap("")
    ap("| arm | cache | hits | misses | window | rerank p50 | rerank p95 |")
    ap("| --- | --- | --- | --- | --- | --- | --- |")
    for name, a in doc["arms"].items():
        rr = a.get("rerank")
        if not rr:
            continue
        comp = rr["rerank_component_ms"]
        ap(f"| `{name}` | {rr['cache']} | {rr['cache_hits']} | {rr['cache_misses']} | "
           f"{rr['window_size_mean']} | {comp['p50_ms']} | {comp['p95_ms']} |")
    ap("")

    cmp_ = doc["slo_comparison"]
    ap("## Comparison with the current project SLO")
    ap("")
    ap(f"- **Measured `p95_retrieval_latency_ms` = {cmp_['measured_p95_retrieval_latency_ms']} ms**"
       f" ({cmp_['status']})")
    ap(f"- Scope: **{cmp_['measurement_scope']}** · must not be described as "
       f"*{cmp_['must_not_be_described_as']}*")
    ap(f"- Baseline without rerank: {cmp_['baseline_without_rerank_p95_ms']} ms · "
       f"rerank overhead: {cmp_['rerank_overhead_p95_ms']} ms")
    ap(f"- Existing v1.1 hard constraint: {cmp_['existing_v11_hard_constraint_ms']} ms · "
       f"ratio: **{cmp_['ratio_vs_existing_constraint']}×** · "
       f"meets constraint: **{cmp_['meets_existing_constraint']}**")
    ap(f"- R0/R0.5 provisional estimate was {cmp_['r0_provisional_estimate_ms']} ms "
       f"(ratio measured/provisional = {cmp_['measured_vs_provisional_ratio']}) — "
       f"**superseded by this measurement**")
    ap("")
    ap(f"> {cmp_['note']}")
    ap("")
    ap(f"> {cmp_['not_decided_here']}")
    ap("")

    ap("## Self-check")
    ap("")
    ap(f"- {doc['stats_consistency_self_check']['description']}")
    ap(f"- all_passed: **{doc['stats_consistency_self_check']['all_passed']}**")
    ap("")
    ap("## Scope / limitations")
    for lim in doc["limitations"]:
        ap(f"- {lim}")
    ap("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    ap_.add_argument("--repeats", type=int, default=30)
    ap_.add_argument("--warmup", type=int, default=5)
    ap_.add_argument("--out-json", default=OUT_JSON)
    ap_.add_argument("--out-md", default=OUT_MD)
    args = ap_.parse_args()

    if args.repeats < CONFIDENCE_MIN_SAMPLES:
        raise SystemExit(
            f"--repeats 必須 >= {CONFIDENCE_MIN_SAMPLES}（§0(4)）；得到 {args.repeats}。"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch

    torch.set_grad_enabled(False)

    from evaluation.v11 import hashing
    from evaluation.v11.real_services import RealTrialServiceProvider

    questions, dataset_version, dataset_hash = _questions()
    token = uuid.uuid4().hex[:8]
    workdir = os.path.join(_ROOT, ".r2-latency-workdir", token)

    print(f"[1/3] 連線真實 Qdrant {args.qdrant_url} ...", flush=True)
    provider = RealTrialServiceProvider.create(
        args.qdrant_url,
        workdir,
        cache_db_path=os.path.join(workdir, "r2-latency.embedding-cache.db"),
        # **關鍵**：rerank cache 停用 —— 這是 uncached 量測。
        rerank_cache_enabled=False,
    )

    created: list[str] = []
    arms: dict[str, Any] = {}
    try:
        for i, (arm_name, cfg) in enumerate(ARMS.items()):
            collection = f"optimizer_r2lat_{token}_{i}"
            created.append(collection)
            print(f"[2/3] 量測 arm={arm_name}（warmup={args.warmup} "
                  f"repeats={args.repeats}）...", flush=True)
            arms[arm_name] = measure_arm(
                provider, collection, arm_name, cfg, questions,
                repeats=args.repeats, warmup=args.warmup,
            )
            print(f"      p50={arms[arm_name]['p50_ms']} ms "
                  f"p95={arms[arm_name]['p95_ms']} ms", flush=True)
            provider.delete_collection(collection)
            created.remove(collection)

        self_check = verify_stats_consistency(*arms.values())
        rerank_stats = provider.rerank_stats() or {}

        doc = {
            "schema_version": "1.0",
            "slice": "v1.2.1-R2",
            "purpose": (
                "REAL measured uncached end-to-end retrieval latency with the rerank stage "
                "wired into the pipeline against a real Qdrant. This supersedes the R0/R0.5 "
                "provisional estimate, which was a sum of two independently measured "
                "components and was never a measurement of the integrated pipeline."
            ),
            "git": _git_state(),
            "qdrant_mode": "real server (network round-trip included)",
            "dataset": {
                "path": DATASET_PATH,
                "dataset_version": dataset_version,
                "dataset_hash": dataset_hash,
                "question_count": len(questions),
            },
            "corpus": {"dirs": list(CORPUS_DIRS)},
            "measurement_protocol": {
                "repeats": args.repeats,
                "warmup": args.warmup,
                "warmup_timed_separately": True,
                "rerank_cache": "DISABLED for every arm (uncached)",
                "queries": "rotated across the full dataset question list",
                "percentile_implementation": (
                    "evaluation.v11.retrieval_metrics._percentile — the SAME implementation "
                    "used for the p95_retrieval_latency_ms metric, so the two are comparable"
                ),
            },
            "rerank_top_n": RERANK_TOP_N,
            "top_k": TOP_K,
            "rerank_batch_size": rerank_stats.get("rerank_batch_size"),
            "rerank_fingerprint": rerank_stats.get("rerank_fingerprint"),
            "rerank_fingerprint_hash": rerank_stats.get("rerank_fingerprint_hash"),
            "arms": arms,
            "slo_comparison": compare_with_slo(arms),
            "prior_r0_provisional": R0_PROVISIONAL,
            "stats_consistency_self_check": self_check,
            "measurement_scope": MEASUREMENT_SCOPE,
            "limitations": [
                LOOPBACK_LIMITATION + " — the Qdrant server is a real separate process with "
                "the full HTTP/gRPC stack, but there is no physical network hop.",
                "This is NOT universal production latency and must never be described as such.",
                "Single machine, CPU only. Hardware-dependent; not a cross-machine claim.",
                "Measured against the tracked demo corpus and the v1.1 dataset questions, "
                "not against production traffic.",
                "Cached rerank latency is NOT measured here and must never be used as a "
                "feasibility number.",
                "No optimisation path was evaluated (ONNX/OpenVINO, quantisation, GPU, "
                "model swap, smaller window). R2 is integration + measurement only.",
                "rerank_top_n stays FROZEN at 5 as selected by the R0.5 coverage audit; it "
                "was NOT re-tuned against these latency numbers.",
            ],
        }

        for out_rel in (args.out_json, args.out_md):
            os.makedirs(os.path.dirname(os.path.join(_ROOT, out_rel)), exist_ok=True)
        with open(os.path.join(_ROOT, args.out_json), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        with open(os.path.join(_ROOT, args.out_md), "w", encoding="utf-8") as fh:
            fh.write(render_markdown(doc))

        print(f"[3/3] 已寫出 {args.out_json} / {args.out_md}", flush=True)
        cmp_ = doc["slo_comparison"]
        print(f"      MEASURED p95_retrieval_latency_ms = "
              f"{cmp_['measured_p95_retrieval_latency_ms']} ms "
              f"({cmp_['ratio_vs_existing_constraint']}x vs "
              f"{EXISTING_V11_HARD_CONSTRAINT_MS} ms constraint)", flush=True)
        return 0
    finally:
        # 只清理自己建立的 collection。
        for name in created:
            try:
                provider.delete_collection(name)
            except Exception:  # noqa: BLE001
                pass
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
