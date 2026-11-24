# MOJ 法規 Corpus Pilot 結果（Slice D）

> 語言：繁體中文。程式名稱、class / function / config field、command 保留英文。
>
> **範圍：** 記錄 Slice D 的實測結果——一份 bounded（100 部）的真實外部法規 corpus，
> 從 live API 取得、正規化、以既有 pipeline 索引，並產出 corpus statistics 與少量
> retrieval sanity 觀察。
>
> **這不是品質評測。** 本文不含 metric threshold、不含 hard label，也**未**建立任何
> evaluation dataset。所有 retrieval 觀察皆為人工可理解的 sanity probe，用途只有一個：
> 判斷這份 corpus 是否值得支撐下一階段的 hard holdout。
>
> Production 行為完全未變更；`app/**` 與 `evaluation/v11/**` 零修改。

相關文件：
- [docs/external-corpus-api-plan.md](../external-corpus-api-plan.md) — 完整計畫（Slice A–D）
- [docs/testing.md](../testing.md) — 測試分層與 marker
- [docs/v1.2.1-rerank-design.md](../v1.2.1-rerank-design.md) — Cross-Encoder rerank 設計

---

## 1. Snapshot Identity

| 項目 | 值 |
| --- | --- |
| `snapshot_id` | `moj-chlaw-20260821-2e815ced-9629909b` |
| `source_endpoint` | `https://law.moj.gov.tw/api/Ch/Law/JSON` |
| `source_update_date` | `2026/8/21 上午 12:00:00` |
| `raw_sha256` | `9629909b87688f0a26eb4dd12da79c54c61ec9e9db79532e60e8e0d436620425` |
| `raw_bytes` | 6,115,185（約 5.83 MB） |
| `selection_hash` | `2e815ced283adb3d04556db44b4300078bc3fb09e39c87fc2450da7c1476ad14` |
| `corpus_hash` | `0310b7ec962b3446c254f784cbb8dd587b4ef9956ce465375de2f797fa9b1519` |
| selection | `max_laws=100`、`category_prefix=null`、`include_repealed=true`、`order_by=pcode` |

重現指令（`--raw-zip` 讓第二次以後完全不必再碰網路）：

```
python -m evaluation.corpus.cli fetch    --config evaluation/configs/corpus-moj-law-pilot.json --output <raw.zip>
python -m evaluation.corpus.cli snapshot --config evaluation/configs/corpus-moj-law-pilot.json --raw-zip <raw.zip>
python -m evaluation.corpus.cli stats    --snapshot evaluation/corpora/moj-law/<snapshot_id> --markdown <out.md>
```

入庫體積：`manifest.json` 942 B + `normalized/` 1,336,332 B = **約 1.28 MB**，
未觸及計畫 §6.7 的 10 MB size trigger。`raw/`（30.7 MB）依 `.gitignore` 規則不入庫。

## 2. 來源契約驗證

Slice D 是本計畫第一次真正呼叫 MOJ live endpoint。實測結果與計畫 §5.1–§5.3
規劃期記錄**完全相符**，無契約漂移：

| 項目 | 規劃期記錄 | Slice D 實測 |
| --- | --- | --- |
| ZIP 成員 | `ChLaw.json` / `schema.csv` / `manifest.csv` | 相同 |
| 編碼 | UTF-8 with BOM | 相同 |
| 頂層結構 | `{UpdateDate, Laws}` | 相同 |
| 法規總數 | 1,346 | 1,346 |
| type-`A` 條文 | 47,257 | 47,257 |
| type-`C` 編章節 | 4,062 | 4,062 |
| 已廢止法規 | 322 | 322 |

契約漂移偵測已固化為 `tests/integration/test_corpus_external_api.py`
（marker `external_api`，**預設不執行**，須顯式 `-m external_api`）。

## 3. Corpus Statistics

### 3.1 規模

| 項目 | 值 |
| --- | --- |
| 來源法規數 | 1,346 |
| 挑選法規數 | 100 |
| normalized 文件數 | 100 |
| type-`A` 條文數 | 3,437 |
| 已廢止 / 現行 | 23 / 77 |
| skipped record / article | 0 / 0 |
| normalized 字元數 | 488,783 |

### 3.2 Chunk（`target_chars=450` / `overlap_chars=80` / `max_tokens=480`）

| 項目 | 值 |
| --- | --- |
| canonical segment | 7,079 |
| chunk 總數 | 3,643 |
| 條文 chunk | 3,543 |
| 法規標頭 chunk | 100 |
| 標頭占比 | **2.74%** |
| 標頭含前言 / 僅 metadata | 4 / 96 |
| hard token split | **68 / 3,437（1.98%）** |
| 切分後超過 token 上限 | **0** |
| strategy fallback | **0** |

| 分布 | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chunk 字元數 | 11 | 101 | 273 | 361 | 438 | 450 | 129.96 |
| chunk token 數 | 7 | 92 | 263 | 353 | 427 | 447 | 121.56 |
| 每部法規 chunk 數 | 2 | 18 | 96 | 127 | 178 | 396 | 36.43 |
| 每條條文 chunk 數 | 1 | 1 | 1 | 1 | 2 | 5 | 1.03 |

「hard token split」在此的定義是**條文全文本身即超過 `max_tokens`**——這類條文
無論如何都會被切在 token 邊界而非語意邊界。1.98% 與計畫 L2 以 400 條抽樣估計的
約 1.5% 一致。**450/80/480 在此語料上正常工作**（0 chunk 超標、0 fallback）；
本文**不**主張 450 優於 600，那需要另做比較實驗。

### 3.3 Identity / metadata

| 項目 | 值 |
| --- | --- |
| unique pcode / 標題 | 100 / 100 |
| 重複 pcode / 標題 | 0 / 0 |
| metadata 六欄全備 | 100 / 100 |
| 有前言 / 無前言 | 4 / 96 |
| 已廢止且有廢止註記 | 23 / 23 |

### 3.4 混淆性——**本 pilot 最重要的發現**

同一組確定性規則，套用於 pilot 的 100 部與完整的 1,346 部：

| 指標 | pilot（前 100 部，依 pcode） | 完整 corpus（1,346 部） |
| --- | --- | --- |
| 完全同名群組 | 0 | 0 |
| 標題詞幹碰撞 | **0 群 / 0 部** | **26 群 / 52 部** |
| 共用 6 字前綴 | 4 群 / 8 部 | 95 群 / 288 部 |
| 現行／已廢止交錯詞幹 | **0 群** | **19 群** |
| 已廢止法規 | 23 | 322 |

**成因：** `max_laws=100` + `order_by=pcode` 是決定性截斷，取到的是 pcode
`A0000001`–`A0030243`，亦即憲法／總統府／行政院／立法院／司法院／考試院／監察院
的組織法規這一個狹窄區段，而非全 corpus 的代表性樣本。完整 corpus 中如
`內政部警政署組織條例` vs `內政部警政署組織法` 這類詞幹碰撞（26 組，其中 19 組
一現行一廢止）**完全沒有落在這 100 部裡**。

依 Owner 指示，**未**依 retrieval 結果重選 corpus，亦**未**人工挑選法規；此處只
如實記錄。

### 3.5 條號歧義（pilot 的實際強項）

| 項目 | 值 |
| --- | --- |
| 相異條號標籤 | 522 |
| `第 1 條`～`第 6 條` 各自出現於 | 94 / 100 部法規 |
| 出現於 ≥10 部法規的條號標籤 | 77 |
| `第 N-N 條` 型條號 | 253 |

同一條號在 94 部法規中重複出現，**條號歧義極強**。

### 3.6 Runtime（觀察值，非 SLO）

| 階段 | 時間 |
| --- | --- |
| live fetch（5.83 MB） | 約 3 s |
| parse + normalize + snapshot 寫入 | 1,557 ms |
| corpus statistics | 6,422 ms |
| chunk + embed + Qdrant index（3,643 chunk） | 160,190 ms |

## 4. Header Chunk 觀察（承接 Slice C）

Slice C 在 4 部法規的 fixture 上發現：法規名稱 heading、metadata 清單與前言的
`section` 同為法規名稱，`TextSplitter` 依 `(page, section)` 分組後合併為單一
**法規標頭 chunk**。Slice D 在 100 部規模上量化：

| 項目 | 值 |
| --- | --- |
| 標頭 chunk 數 | 100（每部法規恰好 1 個） |
| 占全部 chunk | **2.74%**（Slice C 的 4 部 fixture 為 10.3%） |
| 含前言 | 4 |
| 僅 metadata | 96 |

**污染程度量測**：決定性抽樣 30 條真實條文，以其前 40 字為 query，取 top-5：

| 項目 | 值 |
| --- | --- |
| 查詢數 | 30 |
| top-5 結果總數 | 150 |
| 其中為標頭 chunk | **1（0.67%）** |
| 標頭排在第 1 名的查詢數 | 1 |

標頭 chunk 在一般條文查詢的 top-5 佔比（0.67%）**低於**其在索引中的基準比例
（2.74%）。**未觀察到 header pollution**，不構成 blocker。標頭 chunk 仍會被
metadata 式查詢命中（見 §5 probe 7），這是預期且無害的行為。

依指示，`normalize.py` 與 `TextSplitter` **未做任何修改**。

## 5. Retrieval Sanity Probes

dense、`score_threshold=0.55`、`top_k=5`。**非評測**，未調整任何參數，未依結果改動 corpus。
`nature` 欄位：`ARTICLE` = 條文 chunk，`HEADER` = 法規標頭 chunk。

| # | 類型 | 查詢 | Top-1 | 觀察 |
| --- | --- | --- | --- | --- |
| 1 | 法規名稱 + 條文內容 | 法院組織法 關於法官員額與庭長設置的規定 | `A0030243` 法官法 第 24 條（0.718） | ⚠️ **取到 法官法 而非 法院組織法**；top-4 全為法官法 |
| 2 | 純 semantic paraphrase | 各級法院的書記官負責哪些職務 | `A0010053` 法院組織法 第 52 條（0.7695） | ✅ 正確 |
| 3 | 法規名稱 + 第 N 條 | 立法院組織法 第 3 條 | `A0010043` 立法院**各委員會**組織法 第 21 條（0.6667） | ❌ **法規錯、條號錯** |
| 4 | `第 N-N 條` | 法院組織法 第 14-1 條 | `A0010043` 立法院各委員會組織法 第 1 條（0.6778） | ❌ **完全失準** |
| 5 | 相似法規名稱 | 行政法院組織法 院長的任命與資格 | `A0000001` 中華民國憲法 第 55 條（0.6861） | ⚠️ 目標法規落在第 3 名 |
| 6 | 已廢止 vs 現行 | 國策顧問委員會的組織與職權 | `A0010013` 國策顧問委員會組織條例 第 2 條（0.7273） | ✅ 正確取到**已廢止**法規，top-4 皆同一部 |
| 7 | metadata-style | 法規類別 廢止法規 總統府 法規位階 法律 | `A0020039` HEADER（0.6611） | 📋 標頭 chunk 佔 top-5 的 4 席 |
| 8 | foreword-style | 中華民國國民大會受全體國民之付託 制定本憲法 | `A0000001` 中華民國憲法 第 25 條（0.7531） | 📋 取到憲法**條文**，前言標頭未進 top-5 |
| 9 | cross-law confusable | 院長副院長由總統任命 任期與員額之規定 | `A0000001` 憲法 第 55 條（0.7762） | ⚠️ top-5 橫跨**憲法／立法院／司法院／中央研究院**四部法規 |

**綜合觀察：**

- **條號定位完全失效**（probe 3、4）。查詢中的「第 N 條」/「第 N-N 條」對 dense
  retrieval 幾乎沒有作用，結果由條文語意主導。這與 §6 的 identifier limitation 同源。
- **法規名稱幾乎不具區辨力**（probe 1、5）。查詢中的法規名稱無法把結果限制在該部
  法規內；組織法家族的 boilerplate 高度相似，跨法規語意混淆真實存在（probe 9）。
- **已廢止法規可正常檢索**（probe 6），未因 `include_repealed=true` 造成異常。

### Rerank 接線觀察

dense → Cross-Encoder（`BAAI/bge-reranker-base`），`ranking_version =
dense_v1+cross_encoder_v1`。以 probe 5 為例，rerank 把 `行政法院組織法 第 19 條`
由第 3 名提到第 1 名，raw logit 由 0.6456 遞減至 -3.8922，排序主鍵確為 raw logit。

**這只是接線 smoke**：確認 rerank stage 能接受法律語料的 chunk 並產出合法結果。
未比較品質、未量測 latency、未建立任何 quality metric。

## 6. Identifier-aware Limitation（L1）在真實規模的確認

| 項目 | 值 |
| --- | --- |
| 索引 chunk 數 | 3,643 |
| `identifiers` payload 非空的 chunk | **0（0.00%）** |
| 9 個 probe query 的 `extract_normalized()` | **全部為 `[]`** |
| `identifier_candidate_count` | 0 |
| 候選來源 | 僅 `dense` |
| `exact_match_boost` / `confusable_penalty` | 0.0 / 0.0 |

計畫 L1 在 100 部、3,643 chunk 的真實規模上**完全成立**：`IdentifierExtractor`
的 regex 要求 ASCII 字母 + 2–6 位數字，CJK「第 N 條」永不匹配，pcode 為 1 字母 +
7 位數字亦超出位數上限。`IdentifierAwareRetrievalStrategy` 因此 100% 退化為 dense。

依 Owner Decision Q1，**未修改 `IdentifierExtractor`**。此數字即為未來是否開啟
Slice E 的證據。

## 7. Pilot 結論

### A. Corpus 是否顯著大於 demo corpus？

**是。** demo corpus 為 2 份 production 文件（10 points）／5 份 `sample_docs`；
本 pilot 為 100 部法規、3,437 條文、**3,643 chunk**，規模高兩個數量級。

### B. 是否存在足夠的困難度？

**部分。逐項分開回答：**

| 困難度維度 | 是否具備 | 證據 |
| --- | --- | --- |
| 條號歧義 | ✅ **強** | `第 1 條` 出現於 94/100 部；77 個條號標籤出現於 ≥10 部；253 個 `第 N-N 條` |
| 跨法規語意歧義 | ✅ **強** | probe 9 top-5 橫跨 4 部法規；probe 1 取到錯誤法規 |
| 標題混淆（title confuser） | ❌ **幾乎沒有** | 0 詞幹碰撞（完整 corpus 有 26 組）；4 群共用前綴（完整 corpus 有 95 群） |
| 現行／已廢止混淆 | ❌ **沒有** | 0 組交錯詞幹（完整 corpus 有 19 組）；23 部廢止法規彼此無同幹對照組 |

### C. Header chunk 是否造成明顯 retrieval pollution？

**否。** 索引占比 2.74%，一般條文查詢 top-5 佔比 0.67%，低於基準。

### D. 450/80/480 的 hard-token-split 情況？

**68 / 3,437 = 1.98%**，與計畫 L2 估計的約 1.5% 一致。0 chunk 超過上限、0 fallback。
設定可用。

### E. Identifier-aware 是否如預期基本 inert？

**是，且比預期更徹底**：0.00% 的 chunk 帶有 identifier，退化率 100%。

### F. 是否足以支撐下一階段的 hard evaluation holdout？

## **NOT YET SUITABLE**

**理由（單一且明確）：** 這份 corpus 缺少計畫最想壓測的那個維度——**標題混淆與
現行／已廢止混淆**。完整 1,346 部中存在 26 組詞幹碰撞（其中 19 組一現行一廢止），
但因 `max_laws=100` + `order_by=pcode` 的決定性截斷，**這 100 部裡一組都沒有**
（§3.4）。以此 slice 建立 holdout，會系統性地漏測 confusable-title 這一類問題，
而那正是 v1.1 identifier-aware 與 v1.2.1 reranker 想解決的核心難題。

**這不是缺陷，是 selection scope 的結果。** pipeline 本身已被 Slice C／D 證明可用：
ingest、index、retrieval、rerank 全部正常，統計可重現，`app/**` 與
`evaluation/v11/**` 零修改。

**需要 Owner 決定的是 selection policy**，例如（僅列出，本文不實作、不預設偏好）：

1. 提高 `max_laws`，讓截斷涵蓋更多 pcode 區段；
2. 改為 category-stratified 或 confuser-aware 的決定性挑選（需新增 `order_by` /
   filter 語義，屬既有 config 契約的擴充）；
3. 直接對完整 1,346 部建 snapshot——但 normalized 體積約 18 MB，**會觸發 §6.7 的
   10 MB size trigger**，需先取得入庫決定。

在 selection policy 決定之前，**不建立 hard evaluation holdout**。

## 8. 已知限制

- **L1 — identifier-aware 在此 corpus 退化為 dense。** 見 §6，0.00% chunk 帶
  identifier。Slice A–D 依 Owner Decision Q1 不修改 `IdentifierExtractor`。
- **L2 — 預設 chunk 大小在中文語料上超出 embedding token 上限。** pilot 以 config
  值（450/80）因應，實測 hard token split 1.98%（§3.2）。
- **L3（本 slice 新增）— pilot 的 100 部並非全 corpus 的代表性樣本。** pcode 升冪
  截斷落在憲政／院級組織法規區段，混淆性顯著低於完整 corpus（§3.4）。任何基於本
  snapshot 的結論**不得**外推至完整 1,346 部 corpus。
- 本文所有 retrieval 觀察皆為 sanity probe，樣本極小、無 hard label、無 metric
  threshold，**不得**當作品質評測結果引用。
