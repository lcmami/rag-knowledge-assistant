# Reports — Historical Evidence

> 本目錄只存放**受版控的歷史證據**：被 README / design / runbook 引用、且目前 `main` 已無法重現
> 相同數值的結果快照，以及人工整理的結論報告。
>
> **每次執行產生的 runtime 產物不放這裡**：一律輸出至 gitignored 的 `evaluation/runs/`
> （以及 `evaluation/searches/`、`reviews/`、`deployments/`、`cache/` 等），不進版控。
> 評估腳本**不會**覆寫本目錄的任何檔案。

## 為什麼這些結果無法在目前 `main` 重現

v1.3 起 Embedding 由本地 `BAAI/bge-small-zh-v1.5` 改為外部 OpenAI-compatible API。
以下 v1.0–v1.2.1 的數值皆以本地模型量測，延遲數字另含當時的單機硬體條件；
重新執行對應腳本會得到不同結果。要對照原始環境，請切回各報告記載的 commit / tag。

## Generated snapshots（腳本產生，作為歷史證據保存）

| 檔案 | 產生腳本（現預設輸出至 `evaluation/runs/`） | 版本 / 來源 | 主要引用處 |
| --- | --- | --- | --- |
| [results.json](results.json) | `evaluation/evaluate_retrieval.py` | v1.0 檢索評估（20 題） | README、[v1.0-runtime-verification](../v1.0-runtime-verification.md) |
| [baseline-v1.1.md](baseline-v1.1.md) / [.json](baseline-v1.1.json) | `evaluation/generate_baseline.py` | v1.1 Slice 0 baseline（commit `98884e9`）；`slice5` config 的 hard constraint 門檻由此推導 | README、[testing](../testing.md) |
| [rerank-latency-baseline-v1.2.1.md](rerank-latency-baseline-v1.2.1.md) / [.json](rerank-latency-baseline-v1.2.1.json) | `scripts/derive_rerank_latency_budget.py` | v1.2.1 Slice R0：prerequisite gates + provisional budget | [rerank design](../v1.2.1-rerank-design.md)、[runbook](../v1.2.1-rerank-runbook.md) |
| [rerank-end-to-end-latency-v1.2.1.md](rerank-end-to-end-latency-v1.2.1.md) / [.json](rerank-end-to-end-latency-v1.2.1.json) | `scripts/measure_rerank_end_to_end_latency.py` | v1.2.1 Slice R2：實測 end-to-end latency（commit `4be5d7e`） | [rerank design](../v1.2.1-rerank-design.md)、[runbook](../v1.2.1-rerank-runbook.md) |

快照內容維持產生當時的原樣（包含其中記錄的舊路徑字串），僅修正相對連結。

## Curated reports（人工整理的結論報告）

| 檔案 | 內容 |
| --- | --- |
| [rerank-quality-evaluation-v1.2.1.md](rerank-quality-evaluation-v1.2.1.md) | v1.2.1 Slice R3：Cross-Encoder rerank 品質比較與 deterministic gate 結論 |
| [corpus-moj-law-pilot-report.md](corpus-moj-law-pilot-report.md) | 外部法規 corpus Slice D：pcode-first pilot（NOT YET SUITABLE） |
| [corpus-moj-law-challenge-report.md](corpus-moj-law-challenge-report.md) | 外部法規 corpus Slice D2：confusability-enriched challenge corpus（SUITABLE） |

## 不屬於本目錄的東西

- **評估輸入**（datasets、corpus、fixtures、configs）：維持在 `evaluation/`、`sample_docs/` 原位。
  其路徑參與 `corpus_hash` / `search_config_hash`，用於 resume、deployment eligibility 與
  review publish 的一致性驗證，**不得搬移**。
- **Runtime 產物**：`evaluation/runs/` 等 gitignored 目錄。
