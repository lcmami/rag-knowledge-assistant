"""Slice R0 — Cross-Encoder rerank 前提檢查 + provisional 延遲預算推導（v1.2.1）。

**本 script 不修改 retrieval runtime pipeline**,不寫入任何 production 資源,不呼叫 LLM,
不連線正式 Qdrant。它只做四件事,前兩件是 gate,未通過即中止:

1. **Activation semantics gate（§0.9(1)）** —— ``CrossEncoder.predict()`` 在 ``num_labels==1``
   時**預設套用 Sigmoid**。本專案排序主鍵是 raw logit,故必須顯式傳入 identity activation,
   並於執行期驗證取得的確實是 logit（``sigmoid(identity) ≈ default``）。
2. **Batch invariance gate（§0(2)）** —— 以 >= 32 組固定真實 (query, chunk) pair,
   分別以 batch = 1 / 8 / 32 計分,**逐 pair 對齊**算 ``max_abs_delta``。
   ``> 1e-6`` 即中止,不進 benchmark、不進 R1。
3. **Latency benchmark（§0(4)）** —— N >= 30,丟棄前 warm-up;量 uncached reranker 推論
   與 baseline retrieval。**不量正式 cache-hit**（正式 rerank cache 於 R1 才存在）。
4. **Provisional budget 推導** —— 明確記錄推導方法與 noise margin。

**產出是 provisional,不是 feasibility 數字。** 這裡量到的是「孤立的 cross-encoder 前向」加上
「現況 baseline 檢索」,**不是** rerank 啟用後真正的 end-to-end ``p95_retrieval_latency_ms``。
R2 pipeline 整合後必須重新實測 end-to-end;R3 的 deterministic gate 一律以該實測值為準。

用法::

    python scripts/derive_rerank_latency_budget.py --repeats 30 --warmup 3 --pairs 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

# 專案根目錄
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.v11.hashing import canonical_json  # noqa: E402
from evaluation.v11.rerank_fingerprint import (  # noqa: E402
    ACTIVATION_IDENTITY,
    RerankFingerprint,
)
# 刻意重用 metrics 內部的 percentile 實作:推導出的預算日後要用來約束
# p95_retrieval_latency_ms,兩者必須以**完全相同**的方式計算,否則不可比較。
from evaluation.v11.retrieval_metrics import _percentile  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-reranker-base"

#: **Repository Owner decision（R0 remediation）**：reranker 一律以 batch=1 計分。
#:
#: R0 首輪實測顯示跨 batch 的分數偏差達 4.292e-06（> 原訂 1e-6 門檻）。成因已定位為
#: **padding 改變 reduction order** 造成的 float32 累加變異（約 3.5× ULP）——
#: 「完全相同輸入」對照組 delta 為 0,「同 batch size 重跑」對照組 delta 亦為 0。
#: batch=1 定義上無 padding,故 padding-induced drift 恆為 0;且 CPU 實測 batch=1
#: （218.96 ms/pair）比原訂 batch=16（259.51 ms/pair）**更快**。
#:
#: 因此 batch size 是 **runtime implementation constant**：
#: 不進 RetrievalConfig、不進 config hash、不進 cache key、不進 optimizer search space。
RERANK_BATCH_SIZE = 1

#: 跨 batch 比較**已不再是阻塞 gate**（batch 不再是可變參數),保留為 audit evidence,
#: 說明「為何固定 batch=1」。此 tolerance 僅用於在報告中標示觀察值的相對量級。
BATCH_INVARIANCE_REFERENCE_TOLERANCE = 1e-6
BATCH_SIZES = (1, 8, 32)

#: 固定輸入的 determinism 檢查 —— 這才是**阻塞** gate。batch 固定後,同一輸入
#: 連續兩次計分必須完全相同（bit-identical),否則整個 rerank 設計不成立。
DETERMINISM_TOLERANCE = 0.0

#: Chunking 參數:明確採用 Settings **class defaults**,不讀 .env（比照 Slice 3 §一 的
#: 可重現性規則 —— baseline 不得因本機 .env 而改變）。
CHUNK_SIZE_CHARS = 600
CHUNK_OVERLAP_CHARS = 100
EMBEDDING_MAX_TOKENS = 480
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
RETRIEVAL_SCORE_THRESHOLD = 0.55
RETRIEVAL_TOP_K = 5

#: 受版控的示範語料（§Slice 1 決定:optimizer corpus 綁定 tracked demo corpus,
#: 不是 docker volume 內的正式文件）。
#: 一律以 POSIX 相對路徑書寫 —— 這些字串會進入版控的 baseline 產物,
#: 用 os.sep 會讓 Windows 與 Linux 產生不同內容。
CORPUS_DIRS = ("sample_docs", "evaluation/fixtures")
DATASET_PATH = "evaluation/datasets/v1.1/cases-v2.json"

OUT_JSON = "evaluation/rerank-latency-baseline-v1.2.1.json"
OUT_MD = "evaluation/rerank-latency-baseline-v1.2.1.md"

#: 樣本數低於此值一律標為 low confidence（沿用 generate_baseline.py 的慣例）。
CONFIDENCE_MIN_SAMPLES = 30

#: R0.5 candidate coverage audit 檢查的 window 大小。
COVERAGE_WINDOWS = (5, 10, 20)

#: R0 初次量測（window=20）的結果,僅供 R0.5 的前後對照。
#: 來源:同一台機器、同一 fingerprint(e0dedc02…)、同一 batch=1、同樣 30 次 uncached
#: repeats,記錄於 R0.5 之前的 baseline 產物。保留為常數是為了讓對照表可稽核;
#: 它**不是**本次量測值,不得與本次 benchmark 混用。
PRIOR_R0_WINDOW20 = {
    "rerank_top_n": 20,
    "reranker_p95_ms": 8919.6873,
    "baseline_retrieval_p95_ms": 125.9583,
    "provisional_end_to_end_p95_ms": 11491.3037,
    "provenance": "R0 initial benchmark, same machine / fingerprint / batch=1 / n=30",
}

#: 「多秒級」門檻 —— provisional end-to-end p95 >= 此值即歸類為 B（production-infeasible
#: for the current CPU reference implementation）。
MULTI_SECOND_THRESHOLD_MS = 1000.0

#: **正式** window selection 只能用這些 split（test 必須維持 holdout 語義）。
SELECTION_SPLITS = ("tune", "validation")

#: coverage 探測時的 top_k。必須 >= max(COVERAGE_WINDOWS),否則 coverage@10 / @20 無從量測。
#: 這**不是**修改 ranking parameter —— threshold / boost / penalty / floor / multiplier / cap
#: 全部維持文件化預設值,只把「輸出幾筆」放大到足以觀察 rank 20 為止。
COVERAGE_PROBE_TOP_K = max(COVERAGE_WINDOWS)


class GateFailure(RuntimeError):
    """Gate 未通過 —— 必須中止,不得繼續。"""


# --------------------------------------------------------------------------
# 語料 / pairs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusChunk:
    filename: str
    chunk_index: int
    section: Optional[str]
    page: Optional[int]
    text: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus_chunks(count_tokens: Callable[[str], int]) -> tuple[list[CorpusChunk], str]:
    """以**正式** loader + TextSplitter 切出 chunk,回傳 (chunks, corpus_hash)。

    檔案依相對路徑排序,chunk 依 (filename, chunk_index) 排序 → 完全 deterministic。
    """
    from app.services.document_loader import load_document
    from app.services.identifier_extractor import IdentifierExtractor
    from app.services.pdf_loader import load_pdf
    from app.services.text_splitter import TextSplitter

    paths: list[str] = []
    for d in CORPUS_DIRS:
        abs_d = os.path.join(_ROOT, d)
        if not os.path.isdir(abs_d):
            continue
        for name in os.listdir(abs_d):
            full = os.path.join(abs_d, name)
            if os.path.isfile(full):
                paths.append(full)
    paths.sort(key=lambda p: os.path.relpath(p, _ROOT).replace(os.sep, "/"))

    splitter = TextSplitter(
        chunk_size_chars=CHUNK_SIZE_CHARS,
        chunk_overlap_chars=CHUNK_OVERLAP_CHARS,
        max_tokens=EMBEDDING_MAX_TOKENS,
        count_tokens=count_tokens,
        identifier_extractor=IdentifierExtractor(),
    )

    chunks: list[CorpusChunk] = []
    hasher = hashlib.sha256()
    for path in paths:
        rel = os.path.relpath(path, _ROOT).replace(os.sep, "/")
        with open(path, "rb") as fh:
            raw = fh.read()
        file_hash = _sha256_bytes(raw)
        hasher.update(rel.encode("utf-8"))
        hasher.update(file_hash.encode("utf-8"))

        name = os.path.basename(path)
        try:
            if name.lower().endswith(".pdf"):
                elements = load_pdf(raw)
            else:
                elements = load_document(name, raw)
        except Exception as exc:  # noqa: BLE001 - 語料有問題要顯性中止
            raise GateFailure(f"語料載入失敗 {rel}: {type(exc).__name__}: {exc}") from exc

        for ch in splitter.split(
            elements, document_id=rel, filename=name, file_hash=file_hash
        ):
            chunks.append(
                CorpusChunk(
                    filename=name,
                    chunk_index=ch.chunk_index,
                    section=ch.section,
                    page=ch.page,
                    text=ch.text,
                )
            )

    chunks.sort(key=lambda c: (c.filename, c.chunk_index))
    return chunks, hasher.hexdigest()


def load_questions() -> tuple[list[str], str, str]:
    """回傳 (questions, dataset_version, dataset_hash);questions 依 case id 排序。"""
    path = os.path.join(_ROOT, DATASET_PATH)
    with open(path, "rb") as fh:
        raw = fh.read()
    data = json.loads(raw.decode("utf-8"))
    cases = sorted(data.get("cases") or [], key=lambda c: c["id"])
    return (
        [c["question"] for c in cases],
        str(data.get("dataset_version")),
        _sha256_bytes(raw),
    )


def build_fixed_pairs(
    questions: list[str], chunks: list[CorpusChunk], target: int
) -> list[tuple[str, str]]:
    """>= ``target`` 組固定真實 (query, chunk) pair（§0(2)：不得只用單一 pair）。

    以 deterministic round-robin 走訪 question × chunk,讓文字長度分佈自然分散
    （cross-encoder 的前向成本取決於 token 長度,與相關與否無關）。
    """
    if not questions or not chunks:
        raise GateFailure("語料或問題集為空,無法建立 pairs。")
    pairs: list[tuple[str, str]] = []
    qn, cn = len(questions), len(chunks)
    total = max(target, 0)
    for i in range(total):
        pairs.append((questions[i % qn], chunks[(i * 7 + i // qn) % cn].text))
    return pairs


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def gate_activation_semantics(model: Any, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """§0.9(1)：確認我們拿到的是 raw logit,而非已套 sigmoid 的值。"""
    import torch

    default_activation = type(getattr(model, "activation_fn", None)).__name__
    num_labels = int(getattr(model, "num_labels", 0) or 0)

    probe = pairs[:8]
    raw = model.predict(
        probe, activation_fn=torch.nn.Identity(), batch_size=8, show_progress_bar=False
    )
    default = model.predict(probe, batch_size=8, show_progress_bar=False)

    raw_list = [float(x) for x in raw]
    default_list = [float(x) for x in default]
    sigmoid_of_raw = [float(torch.sigmoid(torch.tensor(v))) for v in raw_list]
    max_delta = max(abs(a - b) for a, b in zip(sigmoid_of_raw, default_list))

    # raw logit 必須能離開 [0,1]（sigmoid 輸出永遠在 [0,1] 內）。若 raw 全落在 [0,1]
    # 且與 default 幾乎相同,代表我們其實拿到的是已激活值 → 設計會靜默失效。
    raw_outside_unit_interval = any(v < 0.0 or v > 1.0 for v in raw_list)
    identical_to_default = max(
        abs(a - b) for a, b in zip(raw_list, default_list)
    ) < 1e-9

    verified = (max_delta <= 1e-5) and raw_outside_unit_interval and not identical_to_default

    result = {
        "num_labels": num_labels,
        "model_default_activation": default_activation,
        "activation_used_for_scoring": ACTIVATION_IDENTITY,
        "probe_pair_count": len(probe),
        "raw_score_min": min(raw_list),
        "raw_score_max": max(raw_list),
        "raw_outside_unit_interval": raw_outside_unit_interval,
        "default_score_min": min(default_list),
        "default_score_max": max(default_list),
        "max_abs_delta_sigmoid_raw_vs_default": max_delta,
        "raw_identical_to_default": identical_to_default,
        "verified_raw_logit": verified,
    }
    if not verified:
        raise GateFailure(
            "Activation semantics gate 未通過:無法確認取得的是 raw logit。"
            f" 詳情={json.dumps(result, ensure_ascii=False)}"
        )
    return result


def gate_fixed_input_determinism(
    model: Any, pairs: list[tuple[str, str]], *, batch_size: int = RERANK_BATCH_SIZE
) -> dict[str, Any]:
    """**阻塞 gate**：固定 batch=1 下,同一輸入連跑兩次必須 bit-identical。

    batch size 固定後,跨 batch 比較已無意義（batch 不再是可變參數）;真正需要保證的是
    「同一輸入 → 同一分數」。這是 rerank 可重現、可快取、可比較的最低前提。
    """
    import torch

    def score(ps: list[tuple[str, str]]) -> list[float]:
        return [
            float(x)
            for x in model.predict(
                ps,
                activation_fn=torch.nn.Identity(),
                batch_size=batch_size,
                show_progress_bar=False,
            )
        ]

    run1 = score(pairs)
    run2 = score(pairs)
    run3 = score(list(reversed(pairs)))[::-1]  # 順序不得影響單筆分數

    d_repeat = max(abs(a - b) for a, b in zip(run1, run2))
    d_order = max(abs(a - b) for a, b in zip(run1, run3))
    worst = max(d_repeat, d_order)

    result = {
        "batch_size": batch_size,
        "pair_count": len(pairs),
        "tolerance": DETERMINISM_TOLERANCE,
        "max_abs_delta_repeat_run": d_repeat,
        "max_abs_delta_reversed_input_order": d_order,
        "max_abs_delta": worst,
        "passed": worst <= DETERMINISM_TOLERANCE,
    }
    if not result["passed"]:
        raise GateFailure(
            f"Fixed-input determinism gate 未通過（batch={batch_size}）:"
            f"max_abs_delta={worst:.3e} > {DETERMINISM_TOLERANCE:.1e}。"
            " 同一輸入必須產生同一分數,否則 rerank 不可重現、不可快取。"
        )
    return result


def audit_batch_invariance(
    model: Any, pairs: list[tuple[str, str]], token_lengths: list[int]
) -> dict[str, Any]:
    """**非阻塞 audit evidence**：記錄「為何固定 batch=1」的實測依據。

    batch size 已由 Repository Owner 裁示固定為 1（見 :data:`RERANK_BATCH_SIZE`）,
    因此跨 batch 的偏差**不再是阻塞條件**。此函式保留原始 batch=1/8/32 量測,
    連同三項成因對照一起寫進 baseline 產物,使該決策可被獨立稽核:

    - ``identical_input_control``：整批餵入完全相同的 pair（無長度差異 → 無 padding 差異）。
      delta 為 0 即證明偏差來自 **padding 造成的 reduction order 改變**。
    - ``repeat_determinism``：同一 batch size 連跑兩次。
    - ``float32_ulp_context``：把觀察到的 delta 換算成該量級下的 float32 ULP 倍數。
    """
    import torch

    def score(ps: list[tuple[str, str]], bs: int) -> list[float]:
        return [
            float(x)
            for x in model.predict(
                ps,
                activation_fn=torch.nn.Identity(),
                batch_size=bs,
                show_progress_bar=False,
            )
        ]

    scores_by_batch: dict[int, list[float]] = {}
    for bs in BATCH_SIZES:
        out = score(pairs, bs)
        if len(out) != len(pairs):
            raise GateFailure(
                f"batch={bs} 回傳 {len(out)} 筆,與 pairs {len(pairs)} 不符。"
            )
        scores_by_batch[bs] = out

    base = BATCH_SIZES[0]
    comparisons = []
    overall = 0.0
    per_pair: list[dict[str, Any]] = []
    for bs in BATCH_SIZES[1:]:
        deltas = [abs(a - b) for a, b in zip(scores_by_batch[base], scores_by_batch[bs])]
        mx = max(deltas)
        overall = max(overall, mx)
        comparisons.append(
            {
                "batch_a": base,
                "batch_b": bs,
                "max_abs_delta": mx,
                "mean_abs_delta": statistics.fmean(deltas),
                "nonzero_pair_count": sum(1 for d in deltas if d > 0.0),
                "argmax_pair_index": deltas.index(mx),
            }
        )
    for i in range(len(pairs)):
        per_pair.append(
            {
                "index": i,
                "token_length": token_lengths[i],
                "score_batch_1": scores_by_batch[base][i],
                "abs_delta_vs_batch_8": abs(
                    scores_by_batch[base][i] - scores_by_batch[8][i]
                ),
                "abs_delta_vs_batch_32": abs(
                    scores_by_batch[base][i] - scores_by_batch[32][i]
                ),
            }
        )

    # --- 對照 1：完全相同輸入（無 padding 差異）---
    uniform = [pairs[0]] * len(pairs)
    u1, u8 = score(uniform, 1), score(uniform, 8)
    identical_control = {
        "description": "all pairs identical -> no length variation -> no padding variation",
        "max_abs_delta_b1_vs_b8": max(abs(a - b) for a, b in zip(u1, u8)),
        "within_batch_spread_b8": max(u8) - min(u8),
    }

    # --- 對照 2：固定 batch size 重跑 ---
    repeat = {}
    for bs in BATCH_SIZES:
        r1, r2 = score(pairs, bs), score(pairs, bs)
        repeat[str(bs)] = max(abs(a - b) for a, b in zip(r1, r2))

    # --- 對照 3：float32 ULP 量級 ---
    max_abs_score = max(abs(v) for v in scores_by_batch[base])
    ulp = max_abs_score * 1.1920929e-7
    ulp_context = {
        "max_abs_score": max_abs_score,
        "float32_ulp_at_that_magnitude": ulp,
        "observed_delta_in_ulps": (overall / ulp) if ulp else None,
    }

    return {
        "role": "audit evidence (NON-BLOCKING) — justifies the fixed batch=1 decision",
        "pair_count": len(pairs),
        "batch_sizes": list(BATCH_SIZES),
        "reference_tolerance": BATCH_INVARIANCE_REFERENCE_TOLERANCE,
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
        "token_length_mean": statistics.fmean(token_lengths),
        "comparisons": comparisons,
        "per_pair": per_pair,
        "identical_input_control": identical_control,
        "repeat_determinism_max_abs_delta": repeat,
        "float32_ulp_context": ulp_context,
        "max_abs_delta": overall,
        "exceeds_reference_tolerance": overall > BATCH_INVARIANCE_REFERENCE_TOLERANCE,
        "root_cause": (
            "padding-induced reduction-order variance in float32. Confirmed by the "
            "identical-input control (delta 0) and repeat determinism (delta 0)."
        ),
        "remediation": (
            f"Repository Owner decision: fix RERANK_BATCH_SIZE = {RERANK_BATCH_SIZE}. "
            "batch=1 has no padding by construction, so padding-induced drift is 0, and "
            "it is also faster on CPU than the originally planned batch=16. Batch size is "
            "therefore a runtime constant: not in RetrievalConfig, config hash, cache key, "
            "or optimizer search space."
        ),
    }


def probe_batch_cost(
    model: Any, pairs: list[tuple[str, str]], *, reps: int = 3
) -> dict[str, Any]:
    """各 batch size 的相對成本 —— 供 remediation 決策（強制 batch=1 的代價是多少）。

    這是 **gate 診斷**,不是 §0(4) 的 feasibility benchmark。
    """
    import torch

    out: dict[str, Any] = {}
    for bs in (1, 8, 16, 32):
        model.predict(
            pairs[:8],
            activation_fn=torch.nn.Identity(),
            batch_size=bs,
            show_progress_bar=False,
        )  # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            model.predict(
                pairs,
                activation_fn=torch.nn.Identity(),
                batch_size=bs,
                show_progress_bar=False,
            )
        total = (time.perf_counter() - t0) / reps * 1000.0
        out[str(bs)] = {
            "total_ms_for_all_pairs": round(total, 2),
            "ms_per_pair": round(total / len(pairs), 3),
        }
    return out


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
def verify_stats_consistency(*stats: dict[str, Any]) -> dict[str, Any]:
    """自我稽核:每個統計量都必須能由 ``raw_samples_ms`` 重算,且滿足序關係。

    加這道檢查的原因是一次**真實的回報錯誤**:曾經在報告中把某次執行的表格與另一次執行的
    p95 混用,產生 `p95 > max` 這種數學上不可能的組合。產物本身當時是對的,
    但這類錯誤不該靠人眼發現 —— 這裡讓它變成機器可攔截的失敗。
    """
    checks: list[dict[str, Any]] = []
    for s in stats:
        raw = s.get("raw_samples_ms") or []
        if not raw:
            checks.append({"label": s.get("label"), "ok": False, "reason": "no raw samples"})
            continue
        recomputed = {
            "min_ms": min(raw),
            "p50_ms": _percentile(raw, 0.50),
            "p95_ms": _percentile(raw, 0.95),
            "max_ms": max(raw),
            "mean_ms": statistics.fmean(raw),
            "stdev_ms": statistics.stdev(raw) if len(raw) >= 2 else None,
        }
        mismatches = {
            k: {"recorded": s.get(k), "recomputed": round(v, 4)}
            for k, v in recomputed.items()
            if v is not None and abs(float(s.get(k)) - v) > 1e-3
        }
        ordered = (
            recomputed["min_ms"]
            <= recomputed["p50_ms"]
            <= recomputed["p95_ms"]
            <= recomputed["max_ms"]
        )
        checks.append(
            {
                "label": s.get("label"),
                "n": len(raw),
                "recompute_mismatches": mismatches,
                "order_invariant_min_p50_p95_max": ordered,
                "ok": (not mismatches) and ordered,
            }
        )
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise GateFailure(
            "Stats consistency self-check 失敗（統計量與 raw samples 不符,或違反 "
            f"min<=p50<=p95<=max）:{json.dumps(failed, ensure_ascii=False)}"
        )
    return {
        "description": (
            "every reported statistic was recomputed from raw_samples_ms and the "
            "min<=p50<=p95<=max ordering was asserted"
        ),
        "checks": checks,
        "all_passed": True,
    }


def audit_candidate_coverage(embedding: Any, workdir: str) -> dict[str, Any]:
    """R0.5 Step 1 —— candidate coverage audit（**不修改 retrieval pipeline**）。

    完全重用既有元件:`RealTrialServiceProvider`(語料載入 / canonical / chunk / index /
    retrieve)與 `EvidenceMatcher`(relevance 判定)。**不另寫一套 relevance 邏輯** ——
    relevance 一律以專案既有定義為準(`relevance_level >= 1`,見 `retrieval_metrics.py:196-200`)。

    對每個 **answerable** case 記錄 `first_relevant_rank` 與 coverage@{5,10,20};
    另記 `admitted_count`(= union 去重後扣掉兩種 admission 拒絕),用以檢查
    `rerank_top_n=20` 是否根本超過實際 admitted 集合大小。

    dense 與 identifier_aware **兩個 arm 都審計** —— search space 兩者皆在,
    選出的 window 必須對兩者都足夠。
    """
    from qdrant_client import QdrantClient

    from app.core.config import Settings
    from evaluation.v11.case_schema import load_dataset
    from evaluation.v11.evidence_matcher import EvidenceMatcher
    from evaluation.v11.real_services import RealTrialServiceProvider
    from evaluation.v11.retrieval_metrics import _span_from_payload
    from evaluation.v11.run_models import TrialConfig

    dataset = load_dataset(os.path.join(_ROOT, DATASET_PATH))
    cases = sorted(dataset.cases, key=lambda c: c.id)
    answerable = [c for c in cases if c.answerable]

    files: list[tuple[str, str]] = []
    for d in CORPUS_DIRS:
        abs_d = os.path.join(_ROOT, d)
        if not os.path.isdir(abs_d):
            continue
        for name in sorted(os.listdir(abs_d)):
            full = os.path.join(abs_d, name)
            if os.path.isfile(full):
                files.append((name, full))
    files.sort(key=lambda t: t[0])

    client = QdrantClient(":memory:")
    provider = RealTrialServiceProvider(
        client,
        embedding,
        workdir,
        Settings(app_env="optimizer", qdrant_url=":memory:"),
        cache_db_path=os.path.join(workdir, "r05-audit.embedding-cache.db"),
        cache_enabled=True,
    )
    matcher = EvidenceMatcher()

    # 兩個 arm 的 retrieval config —— 全部沿用 RetrievalConfig 文件化預設值,
    # 只把 top_k 放大到 COVERAGE_PROBE_TOP_K 以便觀察到 rank 20。
    arms = {
        "dense": {
            "strategy": "dense",
            "top_k": COVERAGE_PROBE_TOP_K,
            "score_threshold": 0.55,
            "candidate_multiplier": 3,
            "candidate_cap": 50,
        },
        "identifier_aware": {
            "strategy": "identifier_aware",
            "top_k": COVERAGE_PROBE_TOP_K,
            "score_threshold": 0.55,
            "candidate_multiplier": 3,
            "candidate_cap": 50,
            "identifier_candidate_cap": 20,
            "exact_match_boost": 0.2,
            "confusable_penalty": 0.15,
            "identifier_vector_floor": 0.4,
        },
    }

    per_arm: dict[str, Any] = {}
    try:
        for arm_name, retrieval_cfg in arms.items():
            collection = f"optimizer_r05_audit_{arm_name}"
            tc = TrialConfig(
                trial_name=f"r05-audit-{arm_name}",
                chunk_config={"strategy": "current"},
                retrieval=retrieval_cfg,
            )
            index_outcome = provider.build_index(collection, tc, files)

            rows: list[dict[str, Any]] = []
            for case in answerable:
                out = provider.retrieve(
                    collection, tc, case.question, COVERAGE_PROBE_TOP_K
                )
                first_rank: Optional[int] = None
                for res in out.results:
                    spans = [_span_from_payload(d) for d in (res.provenance_spans or [])]
                    mr = matcher.match(case.expected_evidence, spans, res.text)
                    if mr.relevance_level >= 1:
                        first_rank = res.rank
                        break

                meta = out.meta or {}
                admitted = (
                    meta.get("deduplicated_count", 0)
                    - meta.get("threshold_rejected_count", 0)
                    - meta.get("identifier_floor_rejected_count", 0)
                )
                rows.append(
                    {
                        "case_id": case.id,
                        "query_type": case.query_type,
                        "split": case.split,
                        "first_relevant_rank": first_rank,
                        "returned_count": len(out.results),
                        "admitted_count_estimate": admitted,
                        "union_candidate_count": meta.get("union_candidate_count"),
                        "deduplicated_count": meta.get("deduplicated_count"),
                        **{
                            f"coverage_at_{n}": (
                                first_rank is not None and first_rank <= n
                            )
                            for n in COVERAGE_WINDOWS
                        },
                    }
                )

            def _summarise(subset: list[dict[str, Any]]) -> dict[str, Any]:
                cov = {
                    f"coverage_at_{n}": sum(1 for x in subset if x[f"coverage_at_{n}"])
                    for n in COVERAGE_WINDOWS
                }
                rk = [x["first_relevant_rank"] for x in subset if x["first_relevant_rank"]]
                ac = [x["admitted_count_estimate"] for x in subset]
                return {
                    "answerable_case_count": len(subset),
                    "case_ids": [x["case_id"] for x in subset],
                    "coverage_counts": cov,
                    "coverage_rates": {
                        k: round(v / len(subset), 4) if subset else None
                        for k, v in cov.items()
                    },
                    "cases_with_no_relevant_in_top20": [
                        x["case_id"] for x in subset if x["first_relevant_rank"] is None
                    ],
                    "first_relevant_rank_distribution": {
                        "min": min(rk) if rk else None,
                        "max": max(rk) if rk else None,
                        "mean": round(statistics.fmean(rk), 4) if rk else None,
                    },
                    "admitted_count_estimate": {
                        "min": min(ac) if ac else None,
                        "max": max(ac) if ac else None,
                        "mean": round(statistics.fmean(ac), 4) if ac else None,
                    },
                }

            selection_rows = [r for r in rows if r["split"] in SELECTION_SPLITS]
            test_rows = [r for r in rows if r["split"] == "test"]

            per_arm[arm_name] = {
                "retrieval_config": retrieval_cfg,
                "indexed_chunk_count": getattr(index_outcome, "chunk_count", None),
                # **正式選擇依據** —— 只含 tune + validation。
                "selection_scope": _summarise(selection_rows),
                # 僅供資訊揭露,**不得**作為 rerank_top_n 的選擇依據。
                "test_split_observed": _summarise(test_rows),
                "all_splits_informational": _summarise(rows),
                "per_case": rows,
            }
            provider.delete_collection(collection)
    finally:
        provider.close()

    return {
        "probe_top_k": COVERAGE_PROBE_TOP_K,
        "windows_examined": list(COVERAGE_WINDOWS),
        "selection_splits": list(SELECTION_SPLITS),
        "split_leakage_disclosure": {
            "formal_selection_evidence": (
                "tune + validation ONLY. rerank_top_n was selected from selection_scope; "
                "test_split_observed is reported for transparency and is NOT selection evidence."
            ),
            "test_split_was_inspected": True,
            "consequence": (
                "The existing test split is NO LONGER a pristine blind holdout for "
                "rerank_top_n selection: its per-case first_relevant_rank values were "
                "observed during the R0.5 audit."
            ),
            "binding_constraint": (
                "Test feedback must NOT be used to tune rerank_top_n, thresholds, the model, "
                "or any other hyperparameter from this point on."
            ),
            "not_mitigated_by": (
                "The dataset and its splits were NOT modified to hide this. The disclosure "
                "stands on the record instead."
            ),
        },
        "relevance_definition": (
            "EvidenceMatcher relevance_level >= 1 against the case's expected_evidence "
            "(the project's existing definition; not reimplemented here)"
        ),
        "unmodified": (
            "dataset, relevance labels, score_threshold, identifier_vector_floor, "
            "exact_match_boost, confusable_penalty, candidate_multiplier and candidate_cap "
            "are all at their documented defaults. Only top_k was raised to "
            f"{COVERAGE_PROBE_TOP_K} so that rank<=20 is observable at all."
        ),
        "arms": per_arm,
    }


def select_rerank_window(audit: dict[str, Any]) -> dict[str, Any]:
    """R0.5 Step 2 —— 在 {5,10,20} 中選出 coverage 不低於 top_n=20 的最小 N。

    規則（Repository Owner 指定,逐字實作）:
      - 若 5 已與 20 完全相同 → 5
      - 若 5 有退化但 10 與 20 相同 → 10
      - 若 10 仍退化 → 保留 20

    **對所有 audited arm 同時成立才算數** —— 任一 arm 在該 window 退化即不採用。
    """
    arms = audit["arms"]
    reference = max(COVERAGE_WINDOWS)

    per_window: dict[int, dict[str, Any]] = {}
    for n in COVERAGE_WINDOWS:
        degraded: list[str] = []
        for arm_name, a in arms.items():
            # **只讀 selection_scope（tune + validation）** —— test split 不得參與選擇。
            cc = a["selection_scope"]["coverage_counts"]
            cov_n = cc[f"coverage_at_{n}"]
            cov_ref = cc[f"coverage_at_{reference}"]
            if cov_n < cov_ref:
                degraded.append(f"{arm_name}: {cov_n} < {cov_ref} at N={n}")
        per_window[n] = {"degraded_arms": degraded, "adequate": not degraded}

    selected = reference
    for n in sorted(COVERAGE_WINDOWS):
        if per_window[n]["adequate"]:
            selected = n
            break

    if selected == reference:
        classification = "C"
        rationale = (
            f"Coverage requires the full window: every smaller N degrades relevant-candidate "
            f"coverage versus N={reference}, so rerank_top_n stays {reference}. "
            "Window was NOT shrunk for speed."
        )
    else:
        classification = "reduced"
        rationale = (
            f"N={selected} is the smallest window whose relevant-candidate coverage equals "
            f"N={reference} on every audited arm, so the larger window was pure cost."
        )

    return {
        "selected_rerank_top_n": selected,
        "reference_window": reference,
        "evidence_scope": list(SELECTION_SPLITS),
        "evidence_scope_note": (
            "Selection used tune + validation ONLY. The test split was inspected during the "
            "audit and is disclosed, but contributed NO evidence to this decision."
        ),
        "rule": (
            "smallest N in {5,10,20} whose candidate coverage is not lower than N=20, "
            "required to hold for every audited arm, evaluated on tune+validation only"
        ),
        "per_window_adequacy": {str(k): v for k, v in per_window.items()},
        "classification_hint": classification,
        "rationale": rationale,
    }


def _stats(samples: list[float], label: str) -> dict[str, Any]:
    """統計量 + **完整原始樣本**。

    原始樣本必須保留:本量測的相對變異很高,只給 p50/p95 無法讓讀者判斷數字是否可信,
    也無法區分「隨機雜訊」與「熱節流之類的單向漂移」。另附前半 / 後半平均以偵測漂移。
    """
    n = len(samples)
    mean = statistics.fmean(samples) if samples else None
    stdev = statistics.stdev(samples) if n >= 2 else None

    half = n // 2
    first_half = statistics.fmean(samples[:half]) if half else None
    second_half = statistics.fmean(samples[half:]) if half else None
    drift = (
        round(second_half - first_half, 4)
        if (first_half is not None and second_half is not None)
        else None
    )

    return {
        "label": label,
        "n": n,
        "min_ms": round(min(samples), 4) if samples else None,
        "max_ms": round(max(samples), 4) if samples else None,
        "mean_ms": round(mean, 4) if mean is not None else None,
        "p50_ms": round(_percentile(samples, 0.50), 4) if samples else None,
        "p95_ms": round(_percentile(samples, 0.95), 4) if samples else None,
        "stdev_ms": round(stdev, 4) if stdev is not None else None,
        "relative_stdev": (
            round(stdev / mean, 4) if (stdev is not None and mean) else None
        ),
        "first_half_mean_ms": round(first_half, 4) if first_half is not None else None,
        "second_half_mean_ms": round(second_half, 4) if second_half is not None else None,
        "drift_second_minus_first_ms": drift,
        "raw_samples_ms": [round(v, 4) for v in samples],
        "confidence": (
            "low (below CONFIDENCE_MIN_SAMPLES)"
            if n < CONFIDENCE_MIN_SAMPLES
            else "moderate (single-machine, hardware-dependent)"
        ),
    }


def bench_reranker(
    model: Any,
    questions: list[str],
    chunks: list[CorpusChunk],
    *,
    rerank_top_n: int,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    """量測「對一個 query 的 rerank window 計分」的 uncached 成本。

    這是 rerank stage 在單次檢索中實際要付的代價（window 大小 = ``rerank_top_n``）。
    每次量測輪換不同 query,讓 token 長度分佈貼近真實。
    """
    import torch

    window = chunks[:rerank_top_n]
    if len(window) < rerank_top_n:
        # 語料 chunk 數不足時如實縮小 window,並在報告標註。
        rerank_top_n = len(window)

    def one(qi: int) -> float:
        pairs = [(questions[qi % len(questions)], c.text) for c in window]
        t0 = time.perf_counter()
        model.predict(
            pairs,
            activation_fn=torch.nn.Identity(),
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
        return (time.perf_counter() - t0) * 1000.0

    warm = [one(i) for i in range(warmup)]
    samples = [one(warmup + i) for i in range(repeats)]

    out = _stats(samples, "uncached_reranker_window_ms")
    out["warm_up_discarded_ms"] = [round(v, 4) for v in warm]
    out["rerank_top_n"] = rerank_top_n
    out["batch_size"] = RERANK_BATCH_SIZE
    out["cache"] = "none (R0 has no rerank cache; formal cache-hit latency deferred to R1/R2)"
    return out


def bench_baseline_retrieval(
    embedding: Any,
    chunks: list[CorpusChunk],
    questions: list[str],
    *,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    """現況 baseline 檢索延遲,使用 **in-memory Qdrant**（不啟動、不連線正式 Qdrant）。

    誠實限制:in-memory 模式**沒有網路 round-trip**,故會**低估**真實部署的檢索成本。
    這是 provisional 預算的已知偏差來源,已於報告標註。
    """
    from qdrant_client import QdrantClient

    from app.services.retrieval_service import RetrievalService
    from app.services.vector_store import VectorStore

    client = QdrantClient(":memory:")
    collection = "r0_baseline_probe"
    store = VectorStore(client, collection)
    store.ensure_collection(embedding.dimension)

    # 直接以 upsert 寫入,不動任何 production 資源。
    from app.models.chunk import Chunk

    vectors = embedding.embed_documents([c.text for c in chunks])
    # Qdrant point id 必須是 UUID 或 unsigned int;以 uuid5 由 chunk identity 導出,
    # 保持 deterministic（同一語料每次產生相同 id,量測可重現）。
    ns = uuid.UUID("00000000-0000-5000-8000-000000000001")
    payload_chunks = [
        Chunk(
            id=str(uuid.uuid5(ns, f"{c.filename}:{c.chunk_index}")),
            document_id=c.filename,
            filename=c.filename,
            file_hash="r0-probe",
            page=c.page,
            section=c.section,
            chunk_index=c.chunk_index,
            text=c.text,
            created_at="1970-01-01T00:00:00+00:00",
        )
        for c in chunks
    ]
    store.upsert_chunks(payload_chunks, vectors)

    service = RetrievalService(
        embedding_service=embedding,
        vector_store=store,
        score_threshold=RETRIEVAL_SCORE_THRESHOLD,
    )

    def one(qi: int) -> float:
        t0 = time.perf_counter()
        service.retrieve(questions[qi % len(questions)], RETRIEVAL_TOP_K)
        return (time.perf_counter() - t0) * 1000.0

    warm = [one(i) for i in range(warmup)]
    samples = [one(warmup + i) for i in range(repeats)]

    out = _stats(samples, "baseline_retrieval_ms")
    out["warm_up_discarded_ms"] = [round(v, 4) for v in warm]
    out["top_k"] = RETRIEVAL_TOP_K
    out["score_threshold"] = RETRIEVAL_SCORE_THRESHOLD
    out["indexed_point_count"] = len(payload_chunks)
    out["qdrant_mode"] = "in-memory (no network round-trip)"
    out["known_bias"] = (
        "in-memory Qdrant has no network round-trip, so this UNDERSTATES real deployed "
        "retrieval latency. Treated as a lower bound in the provisional budget."
    )

    try:
        client.delete_collection(collection)
    except Exception:  # noqa: BLE001 - in-memory client,清理失敗無副作用
        pass
    return out


# --------------------------------------------------------------------------
def derive_provisional_budget(
    rerank: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """由兩段獨立量測組合出 provisional end-to-end 預算,並記錄推導方法與 noise margin。"""
    r_p95 = float(rerank["p95_ms"])
    b_p95 = float(baseline["p95_ms"])
    r_sd = float(rerank["stdev_ms"] or 0.0)
    b_sd = float(baseline["stdev_ms"] or 0.0)

    # 兩段各自獨立量測 → 以平方和合成雜訊邊際,再取 2 sigma。
    noise_margin = 2.0 * ((r_sd ** 2 + b_sd ** 2) ** 0.5)
    combined = r_p95 + b_p95
    provisional = combined + noise_margin

    return {
        "status": "PROVISIONAL — not a feasibility number",
        "method": (
            "provisional_end_to_end_p95 = p95(uncached reranker window) "
            "+ p95(baseline retrieval) + 2*sqrt(sd_rerank^2 + sd_baseline^2). "
            "The two terms are measured INDEPENDENTLY and simply summed; this is an "
            "estimate, NOT a measured end-to-end value."
        ),
        "rerank_p95_ms": round(r_p95, 4),
        "baseline_retrieval_p95_ms": round(b_p95, 4),
        "combined_p95_ms": round(combined, 4),
        "noise_margin_ms": round(noise_margin, 4),
        "provisional_end_to_end_p95_ms": round(provisional, 4),
        "existing_v11_hard_constraint_ms": 86.0,
        "ratio_vs_existing_constraint": round(provisional / 86.0, 2),
        "confidence": "low — derived from two independently measured components, not an end-to-end run",
        "feasibility_implication": (
            f"The reranker term dominates by ~{r_p95 / b_p95:.0f}x over retrieval. At "
            f"{provisional / 1000:.1f}s per query, the CURRENT CPU REFERENCE IMPLEMENTATION "
            f"FAILS THE PROJECT'S CURRENT PRODUCTION LATENCY BUDGET / SLO, and it also makes "
            f"a full optimizer grid search expensive (cost scales with cases x candidates x "
            f"window size). This statement is scoped to this implementation and this SLO; it "
            f"is NOT a claim that Cross-Encoder rerank is universally production-infeasible. "
            f"Unevaluated levers: smaller or quantised reranker, ONNX/OpenVINO backend, GPU, "
            f"caching. None were attempted."
        ),
        "measurement_quality_warning": (
            "The reranker term has high relative variance on this machine. Inspect "
            "raw_samples_ms and drift_second_minus_first_ms before treating any single "
            "percentile as stable."
        ),
        "must_not_be_used_as": "p95_retrieval_latency_ms in any deterministic gate",
        "required_next_step": (
            "R2 must re-measure REAL uncached end-to-end p95_retrieval_latency_ms with the "
            "rerank stage wired into the trial runner against a real Qdrant. R3's "
            "deterministic gate uses that measured value, never this provisional estimate."
        ),
    }


def classify_feasibility(
    budget: dict[str, Any], selection: dict[str, Any], rerank_bench: dict[str, Any]
) -> dict[str, Any]:
    """R0.5 Step 4 —— 只做分類,**不自行進 R1**。

    A / B / C 並非互斥:window 可能同時「顯著改善」(A) 且「仍為多秒級」(B)。
    因此回傳一組 flag,並以最保守者作為 headline。
    """
    now = float(budget["provisional_end_to_end_p95_ms"])
    prior = float(PRIOR_R0_WINDOW20["provisional_end_to_end_p95_ms"])
    speedup = prior / now if now else None
    reduction_pct = (1.0 - now / prior) * 100.0 if prior else None

    coverage_forced_full_window = (
        int(selection["selected_rerank_top_n"]) == max(COVERAGE_WINDOWS)
    )
    still_multi_second = now >= MULTI_SECOND_THRESHOLD_MS
    improved = speedup is not None and speedup > 1.2

    classes: list[str] = []
    if improved:
        classes.append("A")
    if still_multi_second:
        classes.append("B")
    if coverage_forced_full_window:
        classes.append("C")

    headline = "B" if still_multi_second else ("A" if improved else "A")
    if coverage_forced_full_window and still_multi_second:
        headline = "B+C"

    return {
        "classes": classes,
        "headline": headline,
        "improvement": {
            "prior_window": PRIOR_R0_WINDOW20["rerank_top_n"],
            "prior_provisional_p95_ms": prior,
            "selected_window": selection["selected_rerank_top_n"],
            "new_provisional_p95_ms": round(now, 4),
            "speedup_x": round(speedup, 2) if speedup else None,
            "reduction_pct": round(reduction_pct, 1) if reduction_pct else None,
            "prior_provenance": PRIOR_R0_WINDOW20["provenance"],
        },
        "A_significant_improvement": improved,
        "B_still_multi_second": still_multi_second,
        "C_coverage_requires_full_window": coverage_forced_full_window,
        "statement": (
            (
                f"A: shrinking the window from {PRIOR_R0_WINDOW20['rerank_top_n']} to "
                f"{selection['selected_rerank_top_n']} cut provisional end-to-end p95 by "
                f"{reduction_pct:.1f}% ({speedup:.2f}x) with ZERO loss of relevant-candidate "
                f"coverage on tune+validation. "
                if improved
                else ""
            )
            + (
                f"B: at {now / 1000:.2f}s per query, the CURRENT CPU REFERENCE "
                f"IMPLEMENTATION FAILS THE PROJECT'S CURRENT PRODUCTION LATENCY BUDGET / SLO. "
                if still_multi_second
                else ""
            )
            + (
                "C: candidate coverage requires the full window; it was NOT shrunk for speed. "
                if coverage_forced_full_window
                else ""
            )
        ).strip(),
        # 四個獨立狀態軸 —— 不得互相推論,尤其不得由 latency 推論到方法本身。
        "status_axes": {
            "quality_feasibility": "unknown",
            "offline_evaluation_feasibility": "yes",
            "production_latency_feasibility_under_current_slo": "no",
            "future_optimization_feasibility": "not_evaluated",
        },
        "scope_of_negative_finding": (
            "The negative result applies ONLY to the current CPU reference implementation "
            "measured against the project's current production latency budget / SLO. It must "
            "NOT be generalised to 'Cross-Encoder rerank is universally production-infeasible'. "
            "No optimisation path (smaller/quantised model, ONNX/OpenVINO, GPU, batching "
            "strategy, caching) has been evaluated."
        ),
        "not_decided_here": (
            "Whether to proceed to R1, change model/backend, or restrict rerank to offline "
            "analysis is a Repository Owner decision. This script only classifies."
        ),
    }


def _git_state() -> dict[str, Any]:
    def run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                args, cwd=_ROOT, capture_output=True, text=True, timeout=15
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help=(
            "覆寫 R0.5 coverage audit 選出的 window。預設 None = 由 audit 決定。"
            "只在需要重現特定 window 的量測時使用;覆寫會如實記入產物。"
        ),
    )
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-md", default=OUT_MD)
    args = ap.parse_args()

    if args.repeats < CONFIDENCE_MIN_SAMPLES:
        print(
            f"[warn] repeats={args.repeats} < {CONFIDENCE_MIN_SAMPLES}；"
            "所有數字將標為 low confidence（§0(4) 要求 N>=30）。",
            file=sys.stderr,
        )
    if args.pairs < 32:
        raise SystemExit(
            f"--pairs 必須 >= 32（§0(2)：不得只用少量 pair）；得到 {args.pairs}。"
        )

    import torch

    torch.set_grad_enabled(False)

    from sentence_transformers import CrossEncoder

    from app.services.embedding_service import EmbeddingService

    print(f"[1/6] 載入 embedding model {EMBEDDING_MODEL} ...", flush=True)
    embedding = EmbeddingService(
        model_name=EMBEDDING_MODEL,
        query_instruction=EMBEDDING_QUERY_INSTRUCTION,
        max_tokens=EMBEDDING_MAX_TOKENS,
    )

    print("[2/6] 切分受版控示範語料 ...", flush=True)
    chunks, corpus_hash = load_corpus_chunks(embedding.count_tokens)
    questions, dataset_version, dataset_hash = load_questions()
    pairs = build_fixed_pairs(questions, chunks, args.pairs)
    print(
        f"      chunks={len(chunks)} questions={len(questions)} pairs={len(pairs)}",
        flush=True,
    )

    print(f"[3/6] 載入 cross-encoder {args.model} ...", flush=True)
    model = CrossEncoder(args.model)
    model.model.eval()

    fingerprint, manifest = RerankFingerprint.from_cross_encoder(
        model, model_name=args.model
    )
    print(
        f"      fingerprint={fingerprint.fingerprint_hash()[:16]} "
        f"shards={fingerprint.weights_shard_count}",
        flush=True,
    )

    print("[4/6] GATE 1 — activation semantics ...", flush=True)
    activation = gate_activation_semantics(model, pairs)
    print(
        f"      OK  num_labels={activation['num_labels']} "
        f"default={activation['model_default_activation']} "
        f"raw_range=[{activation['raw_score_min']:.4f}, {activation['raw_score_max']:.4f}]",
        flush=True,
    )

    print(
        f"[5/7] GATE 2 — fixed-input determinism (batch={RERANK_BATCH_SIZE}) ...",
        flush=True,
    )
    determinism = gate_fixed_input_determinism(model, pairs)
    print(
        f"      OK  repeat={determinism['max_abs_delta_repeat_run']:.3e} "
        f"reversed_order={determinism['max_abs_delta_reversed_input_order']:.3e}",
        flush=True,
    )

    print(
        f"[6/7] AUDIT — batch invariance evidence (non-blocking, {len(pairs)} pairs) ...",
        flush=True,
    )
    token_lengths = [len(model.tokenizer(q, c)["input_ids"]) for q, c in pairs]
    invariance = audit_batch_invariance(model, pairs, token_lengths)
    invariance["batch_cost_probe"] = probe_batch_cost(model, pairs)
    print(
        f"      recorded  cross-batch max_abs_delta={invariance['max_abs_delta']:.3e} "
        f"(evidence only; batch fixed at {RERANK_BATCH_SIZE})",
        flush=True,
    )

    print("[7/9] R0.5 Step 1 — candidate coverage audit ...", flush=True)
    workdir = os.path.join(_ROOT, "evaluation", "_r05_audit_workdir")
    coverage_audit = audit_candidate_coverage(embedding, workdir)
    for arm_name, a in coverage_audit["arms"].items():
        s = a["selection_scope"]
        t = a["test_split_observed"]
        print(
            f"      {arm_name:17s} [selection tune+val] @5/10/20 = "
            + "/".join(
                str(s["coverage_counts"][f"coverage_at_{n}"]) for n in COVERAGE_WINDOWS
            )
            + f" of {s['answerable_case_count']}"
            f"   [test, disclosure only] "
            + "/".join(
                str(t["coverage_counts"][f"coverage_at_{n}"]) for n in COVERAGE_WINDOWS
            )
            + f" of {t['answerable_case_count']}",
            flush=True,
        )

    print("[8/9] R0.5 Step 2 — window selection ...", flush=True)
    selection = select_rerank_window(coverage_audit)
    selected_top_n = int(selection["selected_rerank_top_n"])
    if args.rerank_top_n is not None and args.rerank_top_n != selected_top_n:
        selection["cli_override"] = {
            "audit_selected": selected_top_n,
            "overridden_to": args.rerank_top_n,
            "note": "benchmark ran with an operator-supplied window, NOT the audited one",
        }
        selected_top_n = int(args.rerank_top_n)
    print(
        f"      selected rerank_top_n = {selected_top_n} "
        f"({selection['classification_hint']})",
        flush=True,
    )

    print(
        f"[9/9] Benchmark (warmup={args.warmup}, repeats={args.repeats}, "
        f"batch={RERANK_BATCH_SIZE}, window={selected_top_n}) ...",
        flush=True,
    )
    rerank_bench = bench_reranker(
        model,
        questions,
        chunks,
        rerank_top_n=selected_top_n,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    baseline_bench = bench_baseline_retrieval(
        embedding, chunks, questions, repeats=args.repeats, warmup=args.warmup
    )
    stats_check = verify_stats_consistency(rerank_bench, baseline_bench)
    print("      stats consistency self-check: OK", flush=True)
    budget = derive_provisional_budget(rerank_bench, baseline_bench)
    feasibility = classify_feasibility(budget, selection, rerank_bench)
    print(
        f"      feasibility class = {feasibility['headline']} "
        f"({feasibility['improvement']['speedup_x']}x vs window "
        f"{PRIOR_R0_WINDOW20['rerank_top_n']})",
        flush=True,
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "slice": "v1.2.1-R0",
        "purpose": (
            "Cross-encoder rerank prerequisite gates + PROVISIONAL latency budget. "
            "Not a feasibility measurement."
        ),
        "git": _git_state(),
        "corpus": {
            "dirs": list(CORPUS_DIRS),
            "corpus_hash": corpus_hash,
            "chunk_count": len(chunks),
            "chunking": {
                "chunk_size_chars": CHUNK_SIZE_CHARS,
                "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
                "max_tokens": EMBEDDING_MAX_TOKENS,
                "source": "Settings class defaults (NOT .env)",
            },
        },
        "dataset": {
            "path": DATASET_PATH,
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "question_count": len(questions),
        },
        "rerank_fingerprint": fingerprint.to_canonical_dict(),
        "rerank_fingerprint_hash": fingerprint.fingerprint_hash(),
        "weights_manifest": manifest,
        "rerank_batch_size": RERANK_BATCH_SIZE,
        "rerank_batch_size_ownership": (
            "runtime implementation constant — NOT in RetrievalConfig, config hash, "
            "cache key, or optimizer search space"
        ),
        "gate_status": {
            "activation_semantics": "PASSED" if activation["verified_raw_logit"] else "FAILED",
            "fixed_input_determinism": "PASSED" if determinism["passed"] else "FAILED",
            "batch_invariance": "AUDIT EVIDENCE (non-blocking)",
            "overall": "PASSED",
            "benchmark_executed": True,
        },
        "activation_semantics": activation,
        "fixed_input_determinism": determinism,
        "batch_invariance_audit": invariance,
        "candidate_coverage_audit": coverage_audit,
        "window_selection": selection,
        "benchmarks": {
            "uncached_reranker": rerank_bench,
            "baseline_retrieval": baseline_bench,
            "cache_hit": None,
        },
        "cache_hit_note": (
            "NOT MEASURED in R0. The formal rerank SQLite cache does not exist until R1 "
            "(rerank_cache.py). R0 must not claim a cache-hit latency. Feasibility uses "
            "uncached numbers only."
        ),
        "provisional_budget": budget,
        "prior_r0_window20_measurement": PRIOR_R0_WINDOW20,
        "feasibility_classification": feasibility,
        "stats_consistency_self_check": stats_check,
    }

    out_json = os.path.join(_ROOT, args.out_json)
    with open(out_json, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(report))
        fh.write("\n")

    out_md = os.path.join(_ROOT, args.out_md)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    print(f"\nWrote {args.out_json}\nWrote {args.out_md}")
    return 0


def render_markdown(r: dict[str, Any]) -> str:
    fp = r["rerank_fingerprint"]
    act = r["activation_semantics"]
    det = r["fixed_input_determinism"]
    inv = r["batch_invariance_audit"]
    rr = r["benchmarks"]["uncached_reranker"]
    bl = r["benchmarks"]["baseline_retrieval"]
    bg = r["provisional_budget"]
    gs = r["gate_status"]
    ca = r["candidate_coverage_audit"]
    sel = r["window_selection"]
    fc = r["feasibility_classification"]
    imp = fc["improvement"]
    sld = ca["split_leakage_disclosure"]
    scc = r["stats_consistency_self_check"]

    def row(s: dict[str, Any]) -> str:
        return (
            f"| {s['label']} | {s['n']} | {s['min_ms']} | {s['p50_ms']} | "
            f"{s['p95_ms']} | {s['max_ms']} | {s['mean_ms']} | {s['stdev_ms']} | "
            f"{s['relative_stdev']} | {s['drift_second_minus_first_ms']} |"
        )

    lines = [
        "# v1.2.1 Slice R0 — Rerank Prerequisite Gates + Provisional Latency Budget",
        "",
        f"> **Gate status — activation semantics: {gs['activation_semantics']} · "
        f"fixed-input determinism: {gs['fixed_input_determinism']} · "
        f"overall: {gs['overall']}**",
        "",
        f"> **`RERANK_BATCH_SIZE = {r['rerank_batch_size']}`**（Repository Owner 裁示）——"
        f"{r['rerank_batch_size_ownership']}",
        "",
    ]
    lines += [
        "> **本文件的延遲數字（若有）是 PROVISIONAL,不是 feasibility 依據。**",
        "> R0 量的是「孤立的 cross-encoder 前向」與「現況 baseline 檢索」兩段**獨立**量測,",
        "> **不是** rerank 啟用後真正的 end-to-end `p95_retrieval_latency_ms`。",
        "> R2 pipeline 整合後必須重新實測 end-to-end;R3 的 deterministic gate 以該實測值為準。",
        "",
        "## Provenance",
        "",
        f"- git commit: `{r['git']['commit']}` (branch `{r['git']['branch']}`, "
        f"dirty={r['git']['dirty']})",
        f"- corpus: {', '.join(r['corpus']['dirs'])} — `corpus_hash={r['corpus']['corpus_hash'][:16]}…`, "
        f"{r['corpus']['chunk_count']} chunks",
        f"- chunking: {r['corpus']['chunking']['chunk_size_chars']}/"
        f"{r['corpus']['chunking']['chunk_overlap_chars']}/"
        f"{r['corpus']['chunking']['max_tokens']} ({r['corpus']['chunking']['source']})",
        f"- dataset: `{r['dataset']['dataset_version']}` — "
        f"`dataset_hash={r['dataset']['dataset_hash'][:16]}…`, "
        f"{r['dataset']['question_count']} questions",
        "",
        "## Rerank Fingerprint",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| `model_name` | `{fp['model_name']}` |",
        f"| `model_revision` | `{fp['model_revision']}` |",
        f"| `num_labels` | {fp['num_labels']} |",
        f"| `max_length` | {fp['max_length']} |",
        f"| `scored_activation` | `{fp['scored_activation']}` |",
        f"| `weights_manifest_sha256` | `{fp['weights_manifest_sha256']}` |",
        f"| `weights_shard_count` | {fp['weights_shard_count']} |",
        f"| **fingerprint_hash** | `{r['rerank_fingerprint_hash']}` |",
        "",
        "### Weights manifest (full, auditable)",
        "",
        "| # | relative path | size (bytes) | sha256 |",
        "| --- | --- | --- | --- |",
    ]
    for i, m in enumerate(r["weights_manifest"], start=1):
        lines.append(
            f"| {i} | `{m['relative_posix_path']}` | {m['size_bytes']} | `{m['sha256']}` |"
        )

    lines += [
        "",
        "## GATE 1 — Activation semantics (§0.9(1))",
        "",
        f"- 模型 `num_labels` = **{act['num_labels']}**,"
        f"`CrossEncoder` 預設 activation = **`{act['model_default_activation']}`**",
        f"- 本專案計分一律顯式傳入 **`{act['activation_used_for_scoring']}`**,取得 raw logit",
        f"- raw score 範圍 `[{act['raw_score_min']:.4f}, {act['raw_score_max']:.4f}]` — "
        f"落在 [0,1] 之外 = **{act['raw_outside_unit_interval']}**（sigmoid 輸出不可能如此）",
        f"- `sigmoid(raw)` vs 預設輸出 max_abs_delta = **{act['max_abs_delta_sigmoid_raw_vs_default']:.3e}** "
        "（確認兩者確為 sigmoid 關係）",
        f"- **verified_raw_logit = {act['verified_raw_logit']}**",
        "",
        "> 若不顯式傳 identity,`predict()` 會套 Sigmoid,排序主鍵設計會**靜默失效**。",
        "",
        f"## GATE 2 — Fixed-input determinism — **{gs['fixed_input_determinism']}**",
        "",
        f"batch 固定為 {det['batch_size']} 後,跨 batch 比較已無意義;真正需保證的是"
        "「同一輸入 → 同一分數」。",
        "",
        "| 檢查 | max_abs_delta | tolerance |",
        "| --- | --- | --- |",
        f"| 同一輸入連跑兩次 | `{det['max_abs_delta_repeat_run']:.3e}` | "
        f"`{det['tolerance']:.1e}` |",
        f"| 反轉輸入順序後逐筆比對 | `{det['max_abs_delta_reversed_input_order']:.3e}` | "
        f"`{det['tolerance']:.1e}` |",
        "",
        f"- **passed = {det['passed']}**（{det['pair_count']} pairs）",
        "",
        "## AUDIT — Batch invariance evidence（**非阻塞**）",
        "",
        f"> {inv['role']}",
        "",
        f"- 固定真實 pair 數:**{inv['pair_count']}**;"
        f"token 長度 {inv['token_length_min']}–{inv['token_length_max']}"
        f"（mean {inv['token_length_mean']:.1f}）",
        f"- batch sizes:{inv['batch_sizes']};"
        f"參考量級 = `{inv['reference_tolerance']:.1e}`"
        f"（超出 = {inv['exceeds_reference_tolerance']}）",
        "",
        "| A | B | max_abs_delta | mean_abs_delta | 非零 pair 數 | argmax idx |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in inv["comparisons"]:
        lines.append(
            f"| {c['batch_a']} | {c['batch_b']} | {c['max_abs_delta']:.3e} | "
            f"{c['mean_abs_delta']:.3e} | {c['nonzero_pair_count']}/{inv['pair_count']} | "
            f"{c['argmax_pair_index']} |"
        )

    ic = inv["identical_input_control"]
    uc = inv["float32_ulp_context"]
    lines += [
        "",
        f"- **overall cross-batch max_abs_delta = {inv['max_abs_delta']:.3e}**",
        "",
        "### 成因對照（決定 remediation 的依據,非事後解釋）",
        "",
        "| 對照 | 結果 | 推論 |",
        "| --- | --- | --- |",
        f"| 完全相同輸入（無長度差異→無 padding 差異）b1 vs b8 | "
        f"`{ic['max_abs_delta_b1_vs_b8']:.3e}` | 偏差**來自 padding**,非模型不確定性 |",
        f"| 同 batch size 連跑兩次 | "
        + " / ".join(
            f"b{k}=`{v:.3e}`" for k, v in inv["repeat_determinism_max_abs_delta"].items()
        )
        + " | 固定 batch 組成下**完全 deterministic** |",
        f"| float32 ULP @ max\\|score\\|={uc['max_abs_score']:.4f} | "
        f"`{uc['float32_ulp_at_that_magnitude']:.3e}` | 觀察到的偏差約 "
        f"**{uc['observed_delta_in_ulps']:.1f}× ULP** —— float32 累加順序的固有變異 |",
        "",
        "### 各 batch size 成本（remediation 決策用,非 §0(4) benchmark）",
        "",
        "| batch | 全部 pairs 總計 (ms) | ms/pair |",
        "| --- | --- | --- |",
    ]
    for bs, v in (inv.get("batch_cost_probe") or {}).items():
        marker = " **← 採用**" if int(bs) == r["rerank_batch_size"] else ""
        lines.append(
            f"| {bs}{marker} | {v['total_ms_for_all_pairs']} | {v['ms_per_pair']} |"
        )

    lines += [
        "",
        f"### 成因與 remediation",
        "",
        f"- **root cause**：{inv['root_cause']}",
        f"- **remediation**：{inv['remediation']}",
        "",
        "## R0.5 — Candidate coverage audit (Step 1)",
        "",
        f"- relevance 定義:{ca['relevance_definition']}",
        f"- 未修改項目:{ca['unmodified']}",
        f"- probe `top_k` = {ca['probe_top_k']};檢查 window = {ca['windows_examined']}",
        "",
        "### ⚠ Split leakage disclosure",
        "",
        f"- **正式選擇依據**:{sld['formal_selection_evidence']}",
        f"- **test split 已被檢視**:{sld['test_split_was_inspected']}",
        f"- **後果**:{sld['consequence']}",
        f"- **約束**:{sld['binding_constraint']}",
        f"- {sld['not_mitigated_by']}",
        "",
        f"### Selection scope — {'+'.join(ca['selection_splits'])}（**正式依據**）",
        "",
        "| arm | answerable | cov@5 | cov@10 | cov@20 | FRR min/max/mean | \\|A\\| min/max/mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm_name, a in ca["arms"].items():
        s = a["selection_scope"]
        cc, fr, ac = (
            s["coverage_counts"],
            s["first_relevant_rank_distribution"],
            s["admitted_count_estimate"],
        )
        lines.append(
            f"| `{arm_name}` | {s['answerable_case_count']} | "
            + " | ".join(str(cc[f"coverage_at_{n}"]) for n in COVERAGE_WINDOWS)
            + f" | {fr['min']}/{fr['max']}/{fr['mean']} "
            f"| {ac['min']}/{ac['max']}/{ac['mean']} |"
        )

    lines += [
        "",
        "### Test split — observed（**揭露用,不是選擇依據**）",
        "",
        "| arm | answerable | cov@5 | cov@10 | cov@20 | FRR min/max/mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm_name, a in ca["arms"].items():
        t = a["test_split_observed"]
        cc, fr = t["coverage_counts"], t["first_relevant_rank_distribution"]
        lines.append(
            f"| `{arm_name}` | {t['answerable_case_count']} | "
            + " | ".join(str(cc[f"coverage_at_{n}"]) for n in COVERAGE_WINDOWS)
            + f" | {fr['min']}/{fr['max']}/{fr['mean']} |"
        )

    lines += ["", "### Per-case first_relevant_rank（全 split,揭露用）", ""]
    for arm_name, a in ca["arms"].items():
        lines += [
            f"**`{arm_name}`**",
            "",
            "| case_id | query_type | split | FRR | \\|A\\| | @5 | @10 | @20 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        # NOTE: 迴圈變數不得叫 `r` —— 會遮蔽本函式的 report 參數 `r`,
        # 導致其後所有 r[...] 讀到的是 per-case row。
        for case_row in a["per_case"]:
            marks = " | ".join(
                "✅" if case_row[f"coverage_at_{n}"] else "❌" for n in COVERAGE_WINDOWS
            )
            lines.append(
                f"| `{case_row['case_id']}` | {case_row['query_type']} | "
                f"{case_row['split']} | {case_row['first_relevant_rank']} | "
                f"{case_row['admitted_count_estimate']} | {marks} |"
            )
        misses = a["all_splits_informational"]["cases_with_no_relevant_in_top20"]
        if misses:
            lines.append("")
            lines.append(
                "> 前 20 名內無 relevant candidate:"
                + ", ".join(f"`{c}`" for c in misses)
            )
        lines.append("")

    lines += [
        "## R0.5 — Window selection (Step 2)",
        "",
        f"- 規則:{sel['rule']}",
        f"- **證據範圍**:{'+'.join(sel['evidence_scope'])} —— {sel['evidence_scope_note']}",
        "",
        "| N | adequate | 退化的 arm |",
        "| --- | --- | --- |",
    ]
    for n, v in sel["per_window_adequacy"].items():
        lines.append(
            f"| {n} | {'✅' if v['adequate'] else '❌'} | "
            f"{'—' if not v['degraded_arms'] else '; '.join(v['degraded_arms'])} |"
        )
    lines += [
        "",
        f"- **selected `rerank_top_n` = {sel['selected_rerank_top_n']}**"
        f"（reference N={sel['reference_window']}）",
        f"- 理由:{sel['rationale']}",
        "",
        "## Latency measurements (Step 3 — selected window)",
        "",
        "| measurement | n | min | p50 | p95 | max | mean | stdev | rel.sd | drift |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        row(rr),
        row(bl),
        "",
        "> `rel.sd` = stdev/mean;`drift` = 後半平均 − 前半平均（正值代表越跑越慢,"
        "可能是熱節流而非隨機雜訊）。完整原始樣本見 JSON 的 `raw_samples_ms`。",
        "",
        f"- reranker window = `rerank_top_n={rr['rerank_top_n']}`,batch_size={rr['batch_size']},"
        f" cache = {rr['cache']}",
        f"- warm-up 丟棄:reranker {rr['warm_up_discarded_ms']} / baseline {bl['warm_up_discarded_ms']}",
        f"- baseline retrieval:{bl['qdrant_mode']},indexed {bl['indexed_point_count']} points,"
        f" top_k={bl['top_k']}, threshold={bl['score_threshold']}",
        f"- **已知偏差**:{bl['known_bias']}",
        f"- **cache-hit 未量測**:{r['cache_hit_note']}",
        "",
        "## Provisional budget",
        "",
        f"- **status: {bg['status']}**",
        f"- 推導方法:{bg['method']}",
        "",
        "| Term | ms |",
        "| --- | --- |",
        f"| p95 uncached reranker window | {bg['rerank_p95_ms']} |",
        f"| p95 baseline retrieval | {bg['baseline_retrieval_p95_ms']} |",
        f"| combined | {bg['combined_p95_ms']} |",
        f"| noise margin (2σ, 平方和合成) | {bg['noise_margin_ms']} |",
        f"| **provisional end-to-end p95** | **{bg['provisional_end_to_end_p95_ms']}** |",
        "",
        f"- 對照既有 v1.1 hard constraint **{bg['existing_v11_hard_constraint_ms']} ms** → "
        f"約 **{bg['ratio_vs_existing_constraint']}×**",
        f"- confidence:{bg['confidence']}",
        "",
        "### ⚠ Feasibility implication（需 Repository Owner 裁示,本 script 不決定）",
        "",
        f"{bg['feasibility_implication']}",
        "",
        f"**量測品質警告**:{bg['measurement_quality_warning']}",
        f"- **不得用作**:{bg['must_not_be_used_as']}",
        f"- **必要後續**:{bg['required_next_step']}",
        "",
        "## R0.5 — Previous vs New (Step 3 對照)",
        "",
        f"| | window {PRIOR_R0_WINDOW20['rerank_top_n']}（R0 初次） | "
        f"window {imp['selected_window']}（R0.5） | 變化 |",
        "| --- | --- | --- | --- |",
        f"| p95 uncached reranker | {PRIOR_R0_WINDOW20['reranker_p95_ms']} ms | "
        f"{rr['p95_ms']} ms | — |",
        f"| p95 baseline retrieval | {PRIOR_R0_WINDOW20['baseline_retrieval_p95_ms']} ms | "
        f"{bl['p95_ms']} ms | — |",
        f"| **provisional end-to-end p95** | **{imp['prior_provisional_p95_ms']} ms** | "
        f"**{imp['new_provisional_p95_ms']} ms** | "
        f"**−{imp['reduction_pct']}%（{imp['speedup_x']}×）** |",
        "",
        f"> 對照基準來源:{imp['prior_provenance']}。"
        "兩次量測的 fingerprint、batch size、repeats 相同,唯一變數是 window 大小。",
        "",
        f"## R0.5 — Feasibility classification (Step 4) — **{fc['headline']}**",
        "",
        f"- 命中類別:{', '.join(fc['classes'])}",
        f"- **A** 顯著改善:{fc['A_significant_improvement']}",
        f"- **B** 仍為多秒級:{fc['B_still_multi_second']}"
        f"（門檻 {MULTI_SECOND_THRESHOLD_MS:.0f} ms）",
        f"- **C** coverage 要求保留完整 window:{fc['C_coverage_requires_full_window']}",
        "",
        f"{fc['statement']}",
        "",
        "### 四個獨立狀態軸（不得互相推論）",
        "",
        "| 軸 | 狀態 |",
        "| --- | --- |",
        f"| quality feasibility | **{fc['status_axes']['quality_feasibility']}** |",
        f"| offline evaluation feasibility | "
        f"**{fc['status_axes']['offline_evaluation_feasibility']}** |",
        f"| production latency feasibility under current SLO | "
        f"**{fc['status_axes']['production_latency_feasibility_under_current_slo']}** |",
        f"| future optimization feasibility | "
        f"**{fc['status_axes']['future_optimization_feasibility']}** |",
        "",
        f"> **負面結論的適用範圍**:{fc['scope_of_negative_finding']}",
        "",
        f"> {fc['not_decided_here']}",
        "",
        "## Stats consistency self-check",
        "",
        f"- {scc['description']}",
        "",
        "| measurement | n | recompute mismatches | min<=p50<=p95<=max | ok |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in scc["checks"]:
        lines.append(
            f"| {c['label']} | {c['n']} | "
            f"{'—' if not c['recompute_mismatches'] else c['recompute_mismatches']} | "
            f"{'✅' if c['order_invariant_min_p50_p95_max'] else '❌'} | "
            f"{'✅' if c['ok'] else '❌'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as exc:
        print(f"\n[GATE FAILURE] {exc}", file=sys.stderr)
        raise SystemExit(2)
