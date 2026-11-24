# v1.2.1 Slice R0 — Rerank Prerequisite Gates + Provisional Latency Budget

> **Gate status — activation semantics: PASSED · fixed-input determinism: PASSED · overall: PASSED**

> **`RERANK_BATCH_SIZE = 1`**（Repository Owner 裁示）——runtime implementation constant — NOT in RetrievalConfig, config hash, cache key, or optimizer search space

> **本文件的延遲數字（若有）是 PROVISIONAL,不是 feasibility 依據。**
> R0 量的是「孤立的 cross-encoder 前向」與「現況 baseline 檢索」兩段**獨立**量測,
> **不是** rerank 啟用後真正的 end-to-end `p95_retrieval_latency_ms`。
> R2 pipeline 整合後必須重新實測 end-to-end;R3 的 deterministic gate 以該實測值為準。

## Provenance

- git commit: `ad12330919522ee938e8df525ff301e672a5a0fd` (branch `main`, dirty=True)
- corpus: sample_docs, evaluation/fixtures — `corpus_hash=8d67e239ae296dd0…`, 24 chunks
- chunking: 600/100/480 (Settings class defaults (NOT .env))
- dataset: `v1.1-coverage-0002` — `dataset_hash=beca6001ab419194…`, 16 questions

## Rerank Fingerprint

| Field | Value |
| --- | --- |
| `model_name` | `BAAI/bge-reranker-base` |
| `model_revision` | `2cfc18c9415c912f9d8155881c133215df768a70` |
| `num_labels` | 1 |
| `max_length` | 512 |
| `scored_activation` | `identity` |
| `weights_manifest_sha256` | `65c6915c562321b524d0256ca0b24ccfdc9cc28c0af339b691a4d916e122281b` |
| `weights_shard_count` | 1 |
| **fingerprint_hash** | `e0dedc0255c2b8a9367c331769664ec0116a85f2c4a8cacb7816a284a3356732` |

### Weights manifest (full, auditable)

| # | relative path | size (bytes) | sha256 |
| --- | --- | --- | --- |
| 1 | `model.safetensors` | 1112206140 | `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd` |

## GATE 1 — Activation semantics (§0.9(1))

- 模型 `num_labels` = **1**,`CrossEncoder` 預設 activation = **`Sigmoid`**
- 本專案計分一律顯式傳入 **`identity`**,取得 raw logit
- raw score 範圍 `[-10.1967, 3.5685]` — 落在 [0,1] 之外 = **True**（sigmoid 輸出不可能如此）
- `sigmoid(raw)` vs 預設輸出 max_abs_delta = **0.000e+00** （確認兩者確為 sigmoid 關係）
- **verified_raw_logit = True**

> 若不顯式傳 identity,`predict()` 會套 Sigmoid,排序主鍵設計會**靜默失效**。

## GATE 2 — Fixed-input determinism — **PASSED**

batch 固定為 1 後,跨 batch 比較已無意義;真正需保證的是「同一輸入 → 同一分數」。

| 檢查 | max_abs_delta | tolerance |
| --- | --- | --- |
| 同一輸入連跑兩次 | `0.000e+00` | `0.0e+00` |
| 反轉輸入順序後逐筆比對 | `0.000e+00` | `0.0e+00` |

- **passed = True**（32 pairs）

## AUDIT — Batch invariance evidence（**非阻塞**）

> audit evidence (NON-BLOCKING) — justifies the fixed batch=1 decision

- 固定真實 pair 數:**32**;token 長度 33–312（mean 111.5）
- batch sizes:[1, 8, 32];參考量級 = `1.0e-06`（超出 = True）

| A | B | max_abs_delta | mean_abs_delta | 非零 pair 數 | argmax idx |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 4.292e-06 | 6.594e-07 | 14/32 | 1 |
| 1 | 32 | 3.874e-06 | 7.469e-07 | 13/32 | 6 |

- **overall cross-batch max_abs_delta = 4.292e-06**

### 成因對照（決定 remediation 的依據,非事後解釋）

| 對照 | 結果 | 推論 |
| --- | --- | --- |
| 完全相同輸入（無長度差異→無 padding 差異）b1 vs b8 | `0.000e+00` | 偏差**來自 padding**,非模型不確定性 |
| 同 batch size 連跑兩次 | b1=`0.000e+00` / b8=`0.000e+00` / b32=`0.000e+00` | 固定 batch 組成下**完全 deterministic** |
| float32 ULP @ max\|score\|=10.1969 | `1.216e-06` | 觀察到的偏差約 **3.5× ULP** —— float32 累加順序的固有變異 |

### 各 batch size 成本（remediation 決策用,非 §0(4) benchmark）

| batch | 全部 pairs 總計 (ms) | ms/pair |
| --- | --- | --- |
| 1 **← 採用** | 5882.43 | 183.826 |
| 8 | 7281.4 | 227.544 |
| 16 | 7088.89 | 221.528 |
| 32 | 11458.06 | 358.064 |

### 成因與 remediation

- **root cause**：padding-induced reduction-order variance in float32. Confirmed by the identical-input control (delta 0) and repeat determinism (delta 0).
- **remediation**：Repository Owner decision: fix RERANK_BATCH_SIZE = 1. batch=1 has no padding by construction, so padding-induced drift is 0, and it is also faster on CPU than the originally planned batch=16. Batch size is therefore a runtime constant: not in RetrievalConfig, config hash, cache key, or optimizer search space.

## R0.5 — Candidate coverage audit (Step 1)

- relevance 定義:EvidenceMatcher relevance_level >= 1 against the case's expected_evidence (the project's existing definition; not reimplemented here)
- 未修改項目:dataset, relevance labels, score_threshold, identifier_vector_floor, exact_match_boost, confusable_penalty, candidate_multiplier and candidate_cap are all at their documented defaults. Only top_k was raised to 20 so that rank<=20 is observable at all.
- probe `top_k` = 20;檢查 window = [5, 10, 20]

### ⚠ Split leakage disclosure

- **正式選擇依據**:tune + validation ONLY. rerank_top_n was selected from selection_scope; test_split_observed is reported for transparency and is NOT selection evidence.
- **test split 已被檢視**:True
- **後果**:The existing test split is NO LONGER a pristine blind holdout for rerank_top_n selection: its per-case first_relevant_rank values were observed during the R0.5 audit.
- **約束**:Test feedback must NOT be used to tune rerank_top_n, thresholds, the model, or any other hyperparameter from this point on.
- The dataset and its splits were NOT modified to hide this. The disclosure stands on the record instead.

### Selection scope — tune+validation（**正式依據**）

| arm | answerable | cov@5 | cov@10 | cov@20 | FRR min/max/mean | \|A\| min/max/mean |
| --- | --- | --- | --- | --- | --- | --- |
| `dense` | 7 | 7 | 7 | 7 | 1/1/1.0 | 2/16/6.2857 |
| `identifier_aware` | 7 | 7 | 7 | 7 | 1/1/1.0 | 2/16/6.2857 |

### Test split — observed（**揭露用,不是選擇依據**）

| arm | answerable | cov@5 | cov@10 | cov@20 | FRR min/max/mean |
| --- | --- | --- | --- | --- | --- |
| `dense` | 5 | 5 | 5 | 5 | 1/2/1.4 |
| `identifier_aware` | 5 | 5 | 5 | 5 | 1/2/1.4 |

### Per-case first_relevant_rank（全 split,揭露用）

**`dense`**

| case_id | query_type | split | FRR | \|A\| | @5 | @10 | @20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eval-api-token-001` | semantic | validation | 1 | 2 | ✅ | ✅ | ✅ |
| `eval-e101-001` | exact_code | validation | 1 | 16 | ✅ | ✅ | ✅ |
| `eval-e102-001` | exact_code | test | 2 | 13 | ✅ | ✅ | ✅ |
| `eval-e102-002` | exact_code | test | 1 | 3 | ✅ | ✅ | ✅ |
| `eval-e102-comparison` | confusable | test | 1 | 10 | ✅ | ✅ | ✅ |
| `eval-e102-confusable` | confusable | test | 2 | 8 | ✅ | ✅ | ✅ |
| `eval-e102-semantic` | semantic | test | 1 | 10 | ✅ | ✅ | ✅ |
| `eval-e203-001` | exact_code | tune | 1 | 3 | ✅ | ✅ | ✅ |
| `eval-maint-comm-001` | cross_section | tune | 1 | 14 | ✅ | ✅ | ✅ |
| `eval-maint-cycle-001` | semantic | tune | 1 | 4 | ✅ | ✅ | ✅ |
| `eval-maint-filter-001` | semantic | tune | 1 | 2 | ✅ | ✅ | ✅ |
| `eval-mes-status-001` | procedure | validation | 1 | 3 | ✅ | ✅ | ✅ |

**`identifier_aware`**

| case_id | query_type | split | FRR | \|A\| | @5 | @10 | @20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eval-api-token-001` | semantic | validation | 1 | 2 | ✅ | ✅ | ✅ |
| `eval-e101-001` | exact_code | validation | 1 | 16 | ✅ | ✅ | ✅ |
| `eval-e102-001` | exact_code | test | 2 | 13 | ✅ | ✅ | ✅ |
| `eval-e102-002` | exact_code | test | 1 | 5 | ✅ | ✅ | ✅ |
| `eval-e102-comparison` | confusable | test | 1 | 10 | ✅ | ✅ | ✅ |
| `eval-e102-confusable` | confusable | test | 2 | 8 | ✅ | ✅ | ✅ |
| `eval-e102-semantic` | semantic | test | 1 | 10 | ✅ | ✅ | ✅ |
| `eval-e203-001` | exact_code | tune | 1 | 3 | ✅ | ✅ | ✅ |
| `eval-maint-comm-001` | cross_section | tune | 1 | 14 | ✅ | ✅ | ✅ |
| `eval-maint-cycle-001` | semantic | tune | 1 | 4 | ✅ | ✅ | ✅ |
| `eval-maint-filter-001` | semantic | tune | 1 | 2 | ✅ | ✅ | ✅ |
| `eval-mes-status-001` | procedure | validation | 1 | 3 | ✅ | ✅ | ✅ |

## R0.5 — Window selection (Step 2)

- 規則:smallest N in {5,10,20} whose candidate coverage is not lower than N=20, required to hold for every audited arm, evaluated on tune+validation only
- **證據範圍**:tune+validation —— Selection used tune + validation ONLY. The test split was inspected during the audit and is disclosed, but contributed NO evidence to this decision.

| N | adequate | 退化的 arm |
| --- | --- | --- |
| 5 | ✅ | — |
| 10 | ✅ | — |
| 20 | ✅ | — |

- **selected `rerank_top_n` = 5**（reference N=20）
- 理由:N=5 is the smallest window whose relevant-candidate coverage equals N=20 on every audited arm, so the larger window was pure cost.

## Latency measurements (Step 3 — selected window)

| measurement | n | min | p50 | p95 | max | mean | stdev | rel.sd | drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uncached_reranker_window_ms | 30 | 803.9899 | 980.7408 | 1586.6272 | 1647.6098 | 1045.3374 | 225.9898 | 0.2162 | 258.0442 |
| baseline_retrieval_ms | 30 | 29.9815 | 43.6918 | 175.009 | 216.3802 | 59.3698 | 45.3247 | 0.7634 | -0.1066 |

> `rel.sd` = stdev/mean;`drift` = 後半平均 − 前半平均（正值代表越跑越慢,可能是熱節流而非隨機雜訊）。完整原始樣本見 JSON 的 `raw_samples_ms`。

- reranker window = `rerank_top_n=5`,batch_size=1, cache = none (R0 has no rerank cache; formal cache-hit latency deferred to R1/R2)
- warm-up 丟棄:reranker [954.1673, 862.8498, 918.0046] / baseline [39.4854, 42.2809, 37.3276]
- baseline retrieval:in-memory (no network round-trip),indexed 24 points, top_k=5, threshold=0.55
- **已知偏差**:in-memory Qdrant has no network round-trip, so this UNDERSTATES real deployed retrieval latency. Treated as a lower bound in the provisional budget.
- **cache-hit 未量測**:NOT MEASURED in R0. The formal rerank SQLite cache does not exist until R1 (rerank_cache.py). R0 must not claim a cache-hit latency. Feasibility uses uncached numbers only.

## Provisional budget

- **status: PROVISIONAL — not a feasibility number**
- 推導方法:provisional_end_to_end_p95 = p95(uncached reranker window) + p95(baseline retrieval) + 2*sqrt(sd_rerank^2 + sd_baseline^2). The two terms are measured INDEPENDENTLY and simply summed; this is an estimate, NOT a measured end-to-end value.

| Term | ms |
| --- | --- |
| p95 uncached reranker window | 1586.6272 |
| p95 baseline retrieval | 175.009 |
| combined | 1761.6362 |
| noise margin (2σ, 平方和合成) | 460.9803 |
| **provisional end-to-end p95** | **2222.6165** |

- 對照既有 v1.1 hard constraint **86.0 ms** → 約 **25.84×**
- confidence:low — derived from two independently measured components, not an end-to-end run

### ⚠ Feasibility implication（需 Repository Owner 裁示,本 script 不決定）

The reranker term dominates by ~9x over retrieval. At 2.2s per query, the CURRENT CPU REFERENCE IMPLEMENTATION FAILS THE PROJECT'S CURRENT PRODUCTION LATENCY BUDGET / SLO, and it also makes a full optimizer grid search expensive (cost scales with cases x candidates x window size). This statement is scoped to this implementation and this SLO; it is NOT a claim that Cross-Encoder rerank is universally production-infeasible. Unevaluated levers: smaller or quantised reranker, ONNX/OpenVINO backend, GPU, caching. None were attempted.

**量測品質警告**:The reranker term has high relative variance on this machine. Inspect raw_samples_ms and drift_second_minus_first_ms before treating any single percentile as stable.
- **不得用作**:p95_retrieval_latency_ms in any deterministic gate
- **必要後續**:R2 must re-measure REAL uncached end-to-end p95_retrieval_latency_ms with the rerank stage wired into the trial runner against a real Qdrant. R3's deterministic gate uses that measured value, never this provisional estimate.

## R0.5 — Previous vs New (Step 3 對照)

| | window 20（R0 初次） | window 5（R0.5） | 變化 |
| --- | --- | --- | --- |
| p95 uncached reranker | 8919.6873 ms | 1586.6272 ms | — |
| p95 baseline retrieval | 125.9583 ms | 175.009 ms | — |
| **provisional end-to-end p95** | **11491.3037 ms** | **2222.6165 ms** | **−80.7%（5.17×）** |

> 對照基準來源:R0 initial benchmark, same machine / fingerprint / batch=1 / n=30。兩次量測的 fingerprint、batch size、repeats 相同,唯一變數是 window 大小。

## R0.5 — Feasibility classification (Step 4) — **B**

- 命中類別:A, B
- **A** 顯著改善:True
- **B** 仍為多秒級:True（門檻 1000 ms）
- **C** coverage 要求保留完整 window:False

A: shrinking the window from 20 to 5 cut provisional end-to-end p95 by 80.7% (5.17x) with ZERO loss of relevant-candidate coverage on tune+validation. B: at 2.22s per query, the CURRENT CPU REFERENCE IMPLEMENTATION FAILS THE PROJECT'S CURRENT PRODUCTION LATENCY BUDGET / SLO.

### 四個獨立狀態軸（不得互相推論）

| 軸 | 狀態 |
| --- | --- |
| quality feasibility | **unknown** |
| offline evaluation feasibility | **yes** |
| production latency feasibility under current SLO | **no** |
| future optimization feasibility | **not_evaluated** |

> **負面結論的適用範圍**:The negative result applies ONLY to the current CPU reference implementation measured against the project's current production latency budget / SLO. It must NOT be generalised to 'Cross-Encoder rerank is universally production-infeasible'. No optimisation path (smaller/quantised model, ONNX/OpenVINO, GPU, batching strategy, caching) has been evaluated.

> Whether to proceed to R1, change model/backend, or restrict rerank to offline analysis is a Repository Owner decision. This script only classifies.

## Stats consistency self-check

- every reported statistic was recomputed from raw_samples_ms and the min<=p50<=p95<=max ordering was asserted

| measurement | n | recompute mismatches | min<=p50<=p95<=max | ok |
| --- | --- | --- | --- | --- |
| uncached_reranker_window_ms | 30 | — | ✅ | ✅ |
| baseline_retrieval_ms | 30 | — | ✅ | ✅ |

