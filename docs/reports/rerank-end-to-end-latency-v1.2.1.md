# Rerank End-to-End Latency — v1.2.1 Slice R2 (MEASURED)

- Slice: `v1.2.1-R2`
- Git: `4be5d7e4a652e22db1a35877cd8cd575e59d5bbb` on `feat/v1.2.1-reranker` (dirty=True)
- Qdrant: `real server (network round-trip included)`
- Dataset: `v1.1-coverage-0002` / `beca6001ab41…`
- Rerank Fingerprint: `e0dedc0255c2…` (model=`BAAI/bge-reranker-base` revision=`2cfc18c9415c912f9d8155881c133215df768a70` activation=`identity`)
- rerank_top_n: **5** (FROZEN by R0.5) · batch_size: **1**

## Why this file exists

> REAL measured uncached end-to-end retrieval latency with the rerank stage wired into the pipeline against a real Qdrant. This supersedes the R0/R0.5 provisional estimate, which was a sum of two independently measured components and was never a measurement of the integrated pipeline.

## Measured end-to-end latency (uncached)

**Scope: local native-Qdrant loopback end-to-end measurement.**

> ⚠️ loopback RTT is cheaper than a remote deployment and may underestimate remote production latency. This is NOT universal production latency.

| arm | n | min | p50 | **p95** | max | mean | stdev | rel.stdev | drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense_rerank` | 30 | 19.2383 | 491.3136 | **866.4739** | 887.5464 | 467.9694 | 338.2494 | 0.7228 | 18.6396 |
| `dense_baseline` | 30 | 19.4358 | 21.1519 | **32.8265** | 41.3654 | 22.6636 | 5.0622 | 0.2234 | -1.6181 |
| `identifier_aware_rerank` | 30 | 23.4608 | 557.7539 | **917.3581** | 949.7259 | 500.648 | 316.9549 | 0.6331 | 6.006 |

All values in milliseconds. Warm-up runs were timed separately and are listed per arm in the JSON (`warm_up_discarded_ms`); they are excluded from every statistic above.

## Rerank component (subset of the end-to-end time, NOT additive)

| arm | cache | hits | misses | window | rerank p50 | rerank p95 |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_rerank` | DISABLED (uncached measurement) | 0 | 86 | 2.8667 | 467.4501 | 838.437 |
| `identifier_aware_rerank` | DISABLED (uncached measurement) | 0 | 92 | 3.0667 | 534.0238 | 884.1885 |

## Comparison with the current project SLO

- **Measured `p95_retrieval_latency_ms` = 866.4739 ms** (MEASURED (real end-to-end, uncached, native standalone Qdrant))
- Scope: **local native-Qdrant loopback end-to-end measurement** · must not be described as *universal production latency*
- Baseline without rerank: 32.8265 ms · rerank overhead: 833.6474 ms
- Existing v1.1 hard constraint: 86.0 ms · ratio: **10.08×** · meets constraint: **False**
- R0/R0.5 provisional estimate was 2222.6165 ms (ratio measured/provisional = 0.3898) — **superseded by this measurement**

> This is the measured value R3's deterministic gate must use, and it must always be cited with its scope: local native-Qdrant loopback end-to-end measurement. The R0/R0.5 provisional estimate (2222.6165 ms) was never a measurement and is now superseded. The comparison against the existing v1.1 constraint is a statement of fact about this implementation on this machine over loopback; it is NOT a decision about whether to ship, change model/backend, or restrict rerank to offline analysis, and it is NOT a claim about universal production latency.

> No optimisation was attempted or evaluated: no ONNX/OpenVINO, no quantisation, no model swap, no smaller rerank_top_n, no GPU. R2 is integration + measurement only.

## Self-check

- every reported statistic was recomputed from raw_samples_ms and the min<=p50<=p95<=max ordering was asserted
- all_passed: **True**

## Scope / limitations
- loopback RTT is cheaper than a remote deployment and may underestimate remote production latency — the Qdrant server is a real separate process with the full HTTP/gRPC stack, but there is no physical network hop.
- This is NOT universal production latency and must never be described as such.
- Single machine, CPU only. Hardware-dependent; not a cross-machine claim.
- Measured against the tracked demo corpus and the v1.1 dataset questions, not against production traffic.
- Cached rerank latency is NOT measured here and must never be used as a feasibility number.
- No optimisation path was evaluated (ONNX/OpenVINO, quantisation, GPU, model swap, smaller window). R2 is integration + measurement only.
- rerank_top_n stays FROZEN at 5 as selected by the R0.5 coverage audit; it was NOT re-tuned against these latency numbers.
