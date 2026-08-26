# Cross-Encoder Rerank — Quality Evaluation (v1.2.1 Slice R3)

Canonical, committed record of the R3 rerank comparison. The raw search output lives under
`evaluation/searches/`, which is **gitignored** (`.gitignore:27`); this file is the permanent
evidence and is reproducible from the committed config (see [Reproduction](#reproduction)).

## Provenance

| field | value |
| --- | --- |
| slice | `v1.2.1-R3` |
| code commit | `e3a103b575272803534332489cf5d48631b639e2` (working tree dirty: R3 search-space changes uncommitted at run time) |
| search id | `v121-rerank-comparison` |
| search config hash | `c861f671e052ff21` |
| dataset version | `v1.1-coverage-0002` |
| dataset hash | `d827ad34d32addbe835cb2c099bbe8581b115206f4acdd7eca3ded1cb324aa3a` |
| corpus hash | `3bec437a97e1b02d9b2859d6abdd155afb8a720972ea6cb34fc5109a37888a82` |
| random seed | 42 |
| rerank fingerprint | `e0dedc0255c2b8a9367c331769664ec0116a85f2c4a8cacb7816a284a3356732` |
| rerank model | `BAAI/bge-reranker-base` @ `2cfc18c9415c912f9d8155881c133215df768a70` |
| split sizes | tune **n=4** · validation **n=5** · test **n=7** |

Dataset, relevance labels and splits were **not modified** by R3.

## The four frozen arms

Every ranking parameter is at its documented default. The only dimensions varied are
`strategy` and `rerank_strategy`; `rerank_top_n` is pinned to the single value `5`.

| arm | strategy | rerank | `ranking_version` | config hash |
| --- | --- | --- | --- | --- |
| `dense` | dense | none | `dense_v1` | `87947b1d981a4e3f` |
| `dense+CE` | dense | cross_encoder | `dense_v1+cross_encoder_v1` | `7721e3c9d53bebc0` |
| `identifier_aware` | identifier_aware | none | `identifier_aware_v1` | `8667a690a774fd0e` |
| `identifier_aware+CE` | identifier_aware | cross_encoder | `identifier_aware_v1+cross_encoder_v1` | `17f83c9c0861e227` |

Frozen and unchanged: `top_k=5`, `score_threshold=0.55`, `candidate_multiplier=3`,
`candidate_cap=50`, `exact_match_boost=0.2`, `confusable_penalty=0.15`,
`identifier_vector_floor=0.4`, `identifier_candidate_cap=20`, `rerank_top_n=5`
(FROZEN by the R0.5 coverage audit), `RERANK_BATCH_SIZE=1`, identity activation.

Candidate generation: 4 raw combinations → 4 executable, **0 invalid, 0 duplicate**.

## Formal metrics — tune + validation ONLY

This is the **only** evidence admissible for selection and comparison reasoning.

| metric (tune / validation) | dense | identifier_aware | dense+CE | identifier_aware+CE |
| --- | --- | --- | --- | --- |
| `mrr` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| `recall_at_1` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| `recall_at_5` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| `relevant_context_hit_at_1` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| `strong_evidence_hit_at_1` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| `exact_code_top1_accuracy` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| `confusable_code_error_rate` | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| `unanswerable_false_positive_rate` | n/a / 0.0 | n/a / 0.0 | n/a / 0.0 | n/a / 0.0 |
| `graded_ndcg_at_5` | 0.9225 / 0.9009 | **0.9316** / 0.9193 | 0.9194 / 0.9193 | 0.9194 / 0.9193 |
| `duplicate_ratio` | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

Every per-case `first_relevant_rank` on tune and validation is **1** for all four arms.
Unanswerable cases correctly return empty for all four arms.

**These splits are saturated.** Eight of the ten metrics sit at a ceiling for every arm, so
they cannot discriminate between rerank on and off. The only moving metric is
`graded_ndcg_at_5`, which shifts by at most 0.012 and moves in *both* directions:
rerank is marginally lower on tune (0.9194 vs 0.9225 / 0.9316) and marginally higher for
dense on validation (0.9009 → 0.9193).

## Descriptive metrics — test split

> **These numbers are descriptive final evidence only.** They were not used to select,
> tune, or reject anything. See [Test-split leakage disclosure](#test-split-leakage-disclosure).

| metric | dense | dense+CE | identifier_aware | identifier_aware+CE |
| --- | --- | --- | --- | --- |
| `mrr` | 0.80 | 0.90 | 0.80 | **1.00** |
| `relevant_context_hit_at_1` | 0.60 | 0.80 | 0.60 | **1.00** |
| `recall_at_1` | 0.60 | 0.80 | 0.60 | **1.00** |
| `recall_at_5` | 1.00 | 1.00 | 1.00 | 1.00 |
| `exact_code_top1_accuracy` | 0.50 | 0.50 | 0.50 | **1.00** |
| `confusable_code_error_rate` | 0.50 | 0.25 | 0.50 | **0.00** |
| `unanswerable_false_positive_rate` | 0.00 | 0.00 | **0.50** | **0.50** |
| `strong_evidence_hit_at_5` | 0.80 | 0.80 | 1.00 | 1.00 |
| `evidence_coverage_at_5` | 0.7857 | 0.7857 | 0.9857 | 0.9857 |

Per-case `first_relevant_rank`:

| case | dense | dense+CE | identifier_aware | identifier_aware+CE |
| --- | --- | --- | --- | --- |
| `eval-e102-001` | 2 | **1** | 2 | **1** |
| `eval-e102-002` | 1 | **2** ⚠ | 1 | 1 |
| `eval-e102-comparison` | 1 | 1 | 1 | 1 |
| `eval-e102-confusable` | 2 | **1** | 2 | **1** |
| `eval-e102-semantic` | 1 | 1 | 1 | 1 |
| `eval-unanswerable-capital` | – | – | – | – |
| `eval-unanswerable-price` | – | – | – | – |

⚠ `dense+CE` **regressed** `eval-e102-002` from rank 1 to 2. Rerank is not uniformly
beneficial. `identifier_aware+CE` regressed no case.

### Unanswerable false positives are inherited, not caused by rerank

`identifier_aware` scores `unanswerable_false_positive_rate = 0.50` (1 of 2 unanswerable
test cases) **both with and without rerank**. This behaviour is inherited from
identifier_aware candidate admission and is **structurally unchangeable by reranking** —
see H3 below. Because the test split was already inspected during R0.5, this is **not**
recorded as a new confirmatory test failure; it is a descriptive observation. It does mean
that latency is **not** the only production concern attached to `identifier_aware+CE`.

## Pre-registered hypotheses

### H1 — `eval-e102-001` first_relevant_rank 2 → 1: **CONFIRMED (descriptive)**

```
dense             2  ->  dense+CE             1
identifier_aware  2  ->  identifier_aware+CE  1
```

### H2 — `eval-e102-confusable` first_relevant_rank 2 → 1: **CONFIRMED (descriptive)**

```
dense             2  ->  dense+CE             1
identifier_aware  2  ->  identifier_aware+CE  1

confusable_code_error_rate:  dense            0.50 -> dense+CE            0.25
                             identifier_aware 0.50 -> identifier_aware+CE 0.00
```

Both are confirmed on **descriptive, already-inspected** test evidence. They are therefore
supporting evidence, not confirmatory proof.

### H3 — rerank has no admission power: **HOLDS**

The structural safety invariant: `results non-empty ⇔ admitted set non-empty`.

- **28 / 28** (arm, case) pairs consistent, **0 violations**.
- `unanswerable_false_positive_rate` identical with rerank on and off:
  dense `0.00 → 0.00`, identifier_aware `0.50 → 0.50`.

H3 is exactly why the identifier_aware false positive above survives reranking: the rerank
stage sits **after** admission and has no authority to add or remove candidates. It can only
reorder the admitted window. Any future attempt to fix that FP by thresholding rerank scores
would break H3 and is prohibited.

## Latency evidence

### Engineering conclusion — R2 dedicated benchmark (authoritative)

`evaluation/rerank-end-to-end-latency-v1.2.1.json` · n=30 per arm · warm-up timed separately
and discarded · **rerank cache disabled and verified at 0 hits** · scope
**local native-Qdrant loopback end-to-end measurement**.

| arm | n | p50 (ms) | **p95 (ms)** | stdev (ms) |
| --- | --- | --- | --- | --- |
| `dense` (baseline) | 30 | 21.15 | **32.83** | 5.06 |
| `dense+CE` | 30 | 491.31 | **866.47** | 338.25 |
| `identifier_aware+CE` | 30 | 557.75 | **917.36** | 316.95 |
| `identifier_aware` (no rerank) | — | — | **not independently benchmarked in R2** | — |

> **Scope limitation:** loopback RTT is cheaper than a remote deployment and may
> underestimate remote production latency. This is not universal production latency.

Against the current 86 ms SLO: `dense` passes (0.38×); `dense+CE` is **10.08×** over;
`identifier_aware+CE` is **10.67×** over.

### Search-internal per-trial latency — LOW CONFIDENCE, gate input only

The deterministic gate consumed `p95_retrieval_latency_ms` computed per trial from the
split itself. Those measurements carry:

- **tune n=4, validation n=5** — at this size p95 is essentially the maximum sample
- `latency_measurement_repetitions = 1`
- `policy_confidence = low`
- **near-threshold decisions from these numbers are not reliable**

Concretely, the plain `dense` arm recorded `p95 = 131.2 ms` on validation and was therefore
marked infeasible by the gate. **That must not be read as an engineering conclusion that
plain dense is production-infeasible** — the R2 dedicated benchmark puts dense at
**32.83 ms (n=30)**, comfortably inside the SLO. The validation figure is small-sample noise.

The rerank verdict does not depend on this noise: 520–578 ms per-trial and 866–917 ms
dedicated are both an order of magnitude clear of 86 ms.

## Deterministic gate

Constraint matrix (evaluated on validation, n=5):

| arm | feasible | pass / fail | failing constraint |
| --- | --- | --- | --- |
| `identifier_aware` | **True** | 4 / 0 | — |
| `dense` | False | 3 / 1 | `p95_latency_ceiling_ms` (131.2 > 86) — low-confidence, see above |
| `dense+CE` | False | 3 / 1 | `p95_latency_ceiling_ms` (578.2 > 86) |
| `identifier_aware+CE` | False | 3 / 1 | `p95_latency_ceiling_ms` (520.0 > 86) |

Both rerank arms pass every **quality and safety** constraint on the formal splits:
`exact_code_top1_floor` 1.0 ≥ 0.75, `unanswerable_false_positive_ceiling` 0.0 ≤ 0.25,
`semantic_no_regression` 1.0 ≥ 0.8.

Recommendation: `identifier_aware` (no rerank), `apply_eligible = False`,
confidence `low`, ineligible for `small_validation_sample`, `test_constraint_failed`,
`demo_corpus_only`. Overall search status: `test_constraint_failed` — the recommendation
itself failed `exact_code_top1_floor` and `unanswerable_false_positive_ceiling` on test.

## Quality conclusion

**Formal (tune + validation): NO MEASURABLE QUALITY IMPROVEMENT / NO DISCRIMINATION**,
because the splits are saturated.

**Descriptive (inspected test): PROMISING IMPROVEMENT for `identifier_aware+CE`** —
H1 and H2 confirmed, MRR / hit@1 / exact_code_top1 all reach 1.00, confusable errors reach
zero, no case regressed. Not confirmatory.

> **Cross-Encoder reranking shows a promising ranking signal, but quality improvement is
> not formally established on the current dataset.**

This is not a claim that Cross-Encoder reranking "proved better", and it is not a claim
that it failed. The formal splits are too easy and too small (n=4, n=5) to measure the
effect, and the split that does show the effect is no longer a blind holdout.

## Production eligibility conclusion

> On the formal tune/validation gate, rerank candidates are rejected by latency.
> Descriptive test evidence additionally shows that identifier-aware+CE inherits the
> identifier-aware unanswerable false-positive behavior, which reranking is structurally
> unable to fix.

Latency is the constraint that actually fires on the formal gate. It is **not** the only
production concern on the record: the inherited unanswerable false-positive behaviour of
`identifier_aware+CE` is a separate, structurally distinct issue that reranking cannot
address (H3), and it would need to be resolved at the admission layer, not by the reranker.

Status axes:

| axis | status |
| --- | --- |
| quality feasibility | **indicated, not formally established** |
| offline evaluation feasibility | **yes** |
| production latency feasibility under current SLO | **no** |
| future optimization feasibility | **not_evaluated** |

## Test-split leakage disclosure

The test split's per-case `first_relevant_rank` values were observed during the **R0.5**
candidate-coverage audit. The test split is therefore **no longer a pristine blind
holdout** for anything in the v1.2.1 rerank line of work.

- Formal selection and comparison reasoning used **tune + validation only**.
- Test results are reported as **descriptive final evidence** and were not fed back into
  any parameter, model, threshold, window, or split decision.
- The dataset and its splits were **not** modified to hide this. The disclosure stands on
  the record instead.
- A future confirmatory result requires a **new, untouched, harder holdout** — see
  `docs/v1.2.1-rerank-design.md` § Future work.

## No optimization attempted

R3 evaluated quality and applied the deterministic gate. **No performance optimisation was
attempted or evaluated**: no ONNX, no OpenVINO, no quantisation, no model swap, no GPU, no
change to `rerank_top_n`, no BM25 / sparse / RRF. `future_optimization_feasibility` remains
`not_evaluated`.

## Reproduction

Requires a real Qdrant on `127.0.0.1:6333` and a locally cached
`BAAI/bge-reranker-base`. The rerank cache must be **disabled** so the gate consumes
uncached latency.

```bash
# Quality comparison (four arms, tune -> validation -> test)
python -m evaluation.v11.search.cli search \
  --config evaluation/configs/v121-rerank-comparison.json \
  --db evaluation/searches/v121-rerank.search.db \
  --output-dir evaluation/searches \
  --qdrant-url http://127.0.0.1:6333

# Dedicated end-to-end latency benchmark (n=30, cache disabled)
python scripts/measure_rerank_end_to_end_latency.py --repeats 30 --warmup 5
```

> **Caveat:** the search CLI builds its provider with the rerank cache **enabled** by
> default. The results above were produced with an injected provider using
> `rerank_cache_enabled=False`, so that the `p95_retrieval_latency_ms` feeding the
> deterministic gate is uncached. Reproducing via the plain CLI will yield cache-warmed
> latency, which must not be used as feasibility evidence.
