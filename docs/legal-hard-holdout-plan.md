# 法律 Hard Holdout Evaluation Plan

> 語言：繁體中文。程式名稱、class / function / config field、metric name、command 保留英文。
>
> **本文為 PLAN ONLY。** 撰寫本文的這一輪**未**建立任何 question、label、dataset、
> split 檔，**未**執行任何 retrieval / reranker，**未**修改 `app/**`、
> `evaluation/v11/**` 或 `IdentifierExtractor`。
>
> 相關文件：
> - [external-corpus-api-plan.md](external-corpus-api-plan.md) — corpus Slice A–D 計畫
> - [reports/corpus-moj-law-pilot-report.md](reports/corpus-moj-law-pilot-report.md) — Slice D（pcode-first pilot，NOT YET SUITABLE）
> - [reports/corpus-moj-law-challenge-report.md](reports/corpus-moj-law-challenge-report.md) — Slice D2（challenge corpus，SUITABLE）
> - [v1.1-retrieval-optimizer-requirements.md](v1.1-retrieval-optimizer-requirements.md) — 既有 evaluation schema / metric 需求來源
> - [v1.2.1-rerank-design.md](v1.2.1-rerank-design.md) — Cross-Encoder rerank 設計

---

## 0. 研究目的與固定前提

### 0.1 目的

在一份**已 freeze** 的 100-law confusability-enriched challenge corpus 上，建立一份
新的 retrieval evaluation dataset，用來比較五個 arm：

| Arm | 說明 | 現況 |
| --- | --- | --- |
| A1 `dense` | 現行 dense baseline | 已存在 |
| A2 `identifier_aware`（現行） | 預期在法律語料上 **inert**（0.00% chunk 帶 identifier） | 已存在 |
| A3 `dense + cross_encoder` | v1.2.1 reranker | 已存在 |
| A4 `legal_identifier_aware` | **Legal Identifier Support，尚未實作** | 不存在 |
| A5 `legal_identifier_aware + cross_encoder` | 同上 + reranker | 不存在 |

**A4 / A5 尚未實作，因此 benchmark 必須在實作之前先 freeze。** 這是本計畫的核心
時序約束，也是所有 anti-leakage 規則的來源。

### 0.2 固定 corpus

| 項目 | 值 |
| --- | --- |
| `snapshot_id` | `moj-chlaw-20260821-60e8123b-9629909b` |
| `corpus_hash` | `b42622b01c5e68b38f3e77df31632c17366b62a141fa75da9f6e79cf50c9dbc3` |
| `selection_hash` | `60e8123bccf5c8743bef8d281f06906c9946019085f7150e3792241d6b9f7bbc` |
| config | `evaluation/configs/corpus-moj-law-challenge.json` |
| 規模 | 100 部法規 / 2,440 條文 / 2,637 chunk（450/80/480） |
| 混淆性 | 26 群 title-stem 碰撞（52 部）；19 群現行／已廢止交錯 |

### 0.3 結果描述的強制用語

本 corpus 中 title-stem confuser 佔 **52%**，完整 MOJ source 僅 **3.9%**。

因此，**所有**基於本 dataset 的 evaluation report 必須將結論描述為：

> **challenge-set performance**

**不得**描述、暗示或推論為 real-world frequency-weighted performance。
此規則適用於 report.md、report.json 的摘要文字、以及任何對外敘述。
H4 的 dataset freeze 檢查與 H5 的報告產出皆須驗證此用語存在。

---

## 1. Repository Findings — 既有可重用資產

**結論：既有 v1.1 schema 與 metric 基礎建設足以承載本 benchmark 的絕大部分需求，
不需要另造平行 evaluation framework。** 以下逐項盤點，缺口集中於 §1.5。

### 1.1 Case / Dataset schema — `evaluation/v11/case_schema.py`

`SCHEMA_VERSION = "1.1"`。

| 欄位 | 型別 | 本 benchmark 的用途 |
| --- | --- | --- |
| `id` | str（不得重複） | legal case id |
| `question` | str | query |
| `query_type` | enum | 見 §1.5 Gap A |
| `answerable` | bool | 直接支援 unanswerable case |
| `group_id` | str | **split 的最小不可分割單位**（同 group 不得跨 split，schema 強制） |
| `split` | `tune` / `validation` / `test` | 直接支援 |
| `expected_evidence` | `EvidenceSpan[]` | gold evidence |
| `dataset_version` | str | 版本識別 |
| `status` | str | 自由字串（`migrated` / `pending` / `accepted`…） |

已內建的驗證（`parse_dataset` / `parse_case`）：

1. `answerable=true` ⇒ 至少一個 evidence span；
2. `answerable=true` ⇒ **每個** span 至少一個 `anchor_text`；
3. `answerable=false` 可無 evidence；
4. `split` 僅三值；
5. **同 `group_id` 不得跨 split**（`_check_group_split_leakage`）；
6. case id 不重複；
7. evidence 的 `document_sha256` / `canonical_text_version` 不得為空。

**實測發現：`parse_dataset` / `parse_case` 不拒絕未知 key。** 既有
`evaluation/datasets/v1.1/cases-v2.json` 即帶有 top-level `notes` 並正常通過。
這代表 corpus provenance 與額外 legal label **可以在零程式碼修改下**攜帶於
dataset 檔中——但它們**不會被驗證**。取捨見 §1.5 Gap B / Owner Decision 5。

### 1.2 Stable evidence identity — 直接可用，無需新機制

`EvidenceSpan` 綁的是**原始文件的 evidence span**，**不是** chunk id、
**不是** Qdrant point id、**不是** runtime `document_id`：

| `EvidenceSpan` 欄位 | 本 corpus 的對應 | 穩定性 |
| --- | --- | --- |
| `document_sha256` | normalized `<pcode>.md` 檔案 bytes 的 SHA-256 | ✅ snapshot 為 immutable artifact，且 `corpus_hash` 可重新驗算 |
| `filename` | `<pcode>.md`（例如 `D0000159.md`） | ✅ pcode 為來源穩定識別 |
| `section` | `第 N 條` 或 `章節路徑｜第 N 條`（`normalize.py` 寫出的 `##` heading） | ✅ 由來源 `ArticleNo` + 編章節決定 |
| `canonical_text_version` | `"v1"`（`app/services/canonical_normalizer.py`） | ✅ 常數，變更時報告會標示不可比較 |
| `loader_version` | `"v1"`（同上） | ✅ |
| `offset_scope` / `char_start` / `char_end` | document scope，指向 normalized markdown 中該條文正文的字元範圍 | ✅ 由 snapshot 檔案內容決定 |
| `anchor_text` | 條文中的關鍵詞片段 | ✅ 內容導出 |

**因此 ground truth 在重新 build snapshot / 重新 index 後保持一致**，只要
snapshot 檔案 bytes 不變（immutable + `corpus_hash` 驗證）。

法規 identity（pcode）、條號、章節脈絡三者皆已可表示，**不需要新增 identity 機制**。

`EvidenceMatcher`（`evaluation/v11/evidence_matcher.py`）的判級：

- **Level 2**：offset 覆蓋率 ≥ `offset_coverage_threshold`（預設 0.5）且 version 相容；
  或 offset 不可用時 **anchor 全命中 + section 相符**。
- **Level 1**：section 相符、或 anchor 部分命中。
- **Level 0**：錯誤文件 / 無比對。

→ 對本 corpus，「取到正確法規的正確條文」自然落在 Level 2，「取到正確法規的錯誤條文」
落在 Level 0（section 不符且 anchor 不中），「取到錯誤法規」落在 Level 0
（`wrong_document`）。**判級語意與法律 benchmark 的需求天然吻合。**

### 1.3 既有 metrics — `evaluation/v11/retrieval_metrics.py`

版本化參數集中於 `evaluation/v11/metric_config.py`（`METRIC_CONFIG_VERSION = "v1"`，
`top_k=5`、NDCG gain `{2:3, 1:1, 0:0}`、`ndcg_variant="trial_index_gold_v1"`）。

| Owner 要求的 metric | 既有實作 | 可否直接沿用 |
| --- | --- | --- |
| MRR | `mrr`（aggregate + per `query_type`） | ✅ 直接沿用 |
| hit / recall @1 | `recall_at_1` / `relevant_context_hit_at_1` / `strong_evidence_hit_at_1` | ✅ 直接沿用 |
| hit / recall @k | 同上 @3 / @5 | ✅ 直接沿用 |
| nDCG | `graded_ndcg_at_5`（`trial_index_gold_v1`） | ✅ **前提：gold evidence 帶 document-scope offset**（見 §2.4） |
| exact target-law top1 | ❌ 無 | ⚠️ 見 Gap C |
| exact target-article top1 | 部分：`exact_code_top1_accuracy` | ⚠️ 見 Gap C / Gap D |
| confusable-version error | 有同名 metric 但**在此語料上恆為 0** | ❌ 見 Gap D |
| unanswerable false-positive | `unanswerable_false_positive_count` / `_rate` | ✅ 直接沿用 |

其他可直接沿用、且對本研究有價值的既有 metric：

- `relevant_section_precision_at_5` — top-5 中相關結果比例（會反映 confuser 稀釋）。
- `evidence_coverage_at_5` — Level-2 結果對 gold offset span 的覆蓋（需 offset）。
- `duplicate_ratio` — near-duplicate 比例。**對本 corpus 特別有意義**：confuser
  pair 的兩部法規條文高度相似，但屬**不同 document**，故不會被判為 duplicate
  （判定需 `document_id` 相同）——這正好讓 duplicate_ratio 不會被 confuser 汙染。
- `mean/p50/p95_retrieval_latency_ms`、`mean/p50/p95_rerank_latency_ms`、
  rerank cache telemetry — 供 §6 的 deployability 討論。
- per-`query_type` 的 `relevant_context_hit_at_{1,5}` / `mrr` / `case_count`。

**誠實性設計已內建**：不可計算的 metric 以 `computable=false` +
`not_computable_reason` 標示，**不得**偽裝為 0；每個 metric 附
`numerator` / `denominator` / `sample_count`。本 benchmark 沿用此約定。

### 1.4 Split handling / provenance / artifact — 直接可用

| 資產 | 位置 | 說明 |
| --- | --- | --- |
| **物理 split 隔離** | `evaluation/v11/search/split_dataset.py::prepare_split_datasets` | 每個 split 寫出**獨立 dataset 檔**（`<split>.cases.json`），Trial Runner 只拿到單一檔案路徑——**不是**靠呼叫端記得過濾，而是 runner 物理上拿不到其他 split 的題目 |
| group leakage 檢查 | 同上 `check_group_leakage` + `case_schema._check_group_split_leakage` | 兩道防線 |
| split manifest | `SplitManifest`：`dataset_version`、`full_dataset_hash`、每 split 的 `case_ids` / `group_ids` / `query_type_counts` / `answerable_count` / `dataset_hash` | 可直接作為 freeze 憑證 |
| dataset hash | `evaluation/v11/hashing.py::dataset_hash` = `sha256(canonical_json(dataset))` | ✅ |
| corpus hash | `hashing.corpus_hash` | ✅ 已用於 snapshot |
| run provenance | `EvaluationRun`：`dataset_version`、`dataset_hash`、`corpus_hash`、`code_commit`、`working_tree_dirty`、`canonical_text_version`、`loader_version`、**`identifier_normalization_version`**、`random_seed`、`config_json` | ✅ 已足以識別 A2 與 A4 的差異 |
| run config | `RunConfig`：`run_name` / `dataset_path` / `corpus_paths` / `trials[]` / `random_seed` | ✅ 五個 arm = 五個 `TrialConfig` |
| 報告產物 | `evaluation/runs/<run_id>/report.json` + `report.md`（`.gitignore` 已涵蓋） | ✅ |
| constraint 可計算性事前預測 | `split_dataset.py::_AGGREGATE_METRIC_REQUIRES` | ⚠️ 目前只列 `exact_code` / `confusable` / `unanswerable`，新 metric 需同步登錄 |

**`identifier_normalization_version`（目前 `"v1"`，`app/services/identifier_extractor.py`）
已被記錄於每個 run**。Legal Identifier Support 若 bump 此版本，A2 與 A4 的
provenance 自然可區分——**這是既有機制，不需新增。**

### 1.5 Gaps — 現有 schema / metric 無法直接表示的部分

> 以下**皆不在本輪實作**。每項標註是否構成 Owner decision。

#### Gap A — `ALLOWED_QUERY_TYPES` 是固定集合，且與法律 category 不同構

```
{exact_code, semantic, procedure, cross_section, confusable, unanswerable}
```

§2.1 的 9 個法律 category 無法一對一對映：`第 N 條` 與 `第 N-N 條`、
similar-title confuser 與 active/repealed confuser，在既有 enum 中會塌縮成同一值，
使 per-`query_type` metric 失去區辨力——而**區辨這幾類正是本研究的目的**。

兩個選項（Owner Decision 5）：

| 選項 | 作法 | 代價 |
| --- | --- | --- |
| **A-1（零程式碼修改）** | `query_type` 沿用既有 enum；細分類放在**未驗證**的 case 額外欄位 `legal_category`，per-category 統計於**報告端**由 `report.json` 的 per-case payload 彙總 | per-`query_type` metric 仍可用但粒度粗；`legal_category` 無 schema 驗證，靠 H4 的 dataset validator 自行檢查 |
| **A-2（最小 schema 擴充）** | 於 `case_schema.py` 新增 `legal_category` 欄位與其 enum 驗證，並擴充 `ALLOWED_QUERY_TYPES` | 需改 `evaluation/v11/**`（本輪禁止）；需 bump `SCHEMA_VERSION`，既有 v1.1 dataset 需確認相容 |

**Plan 推薦 A-1**：不為法律 benchmark 破壞既有 schema，且 per-category 分析在報告端
完全可行（`top_results_payload` 已含 `filename` / `section` / `relevance_level`）。

#### Gap B — dataset 無 corpus provenance 欄位

`corpus_hash` 目前記在 **run** 上，不在 **dataset** 上。一份綁定單一 snapshot 的
frozen holdout 應自我宣告它屬於哪份 corpus。

因 `parse_dataset` 不拒絕未知 key（§1.1），可**零程式碼修改**於 dataset 頂層加入：

```json
{
  "schema_version": "1.1",
  "dataset_version": "legal-holdout-0001",
  "corpus": {
    "snapshot_id": "moj-chlaw-20260821-60e8123b-9629909b",
    "corpus_hash": "b42622b0…c9dbc3",
    "selection_hash": "60e8123b…7bbc",
    "config_path": "evaluation/configs/corpus-moj-law-challenge.json"
  },
  "created_at": "…",
  "notes": "challenge-set …",
  "cases": [ … ]
}
```

這些欄位會進入 `dataset_hash`（canonical JSON 全序列化），故**受 hash 保護**，
只是不受 parser 驗證。H4 的 dataset validator 負責驗證。

#### Gap C — 無 `exact target-law top1` / `exact target-article top1` metric

既有最接近者是 `exact_code_top1_accuracy`，但它：

1. 只計 `query_type == "exact_code"` 的 case；
2. 以 `IdentifierExtractor` 從 question 抽 identifier，再與 `result.identifiers` 取交集。

**可在報告端零程式碼修改導出**：`report.json` 的每個 case 已含 top-k 的
`filename` / `section` / `relevance_level`，故

- `exact_target_law_top1` = top-1 的 `filename` == gold evidence 的 `filename`
- `exact_target_article_top1` = 上式且 top-1 的 `section` == gold evidence 的 `section`

皆可由 report 後處理算出。**Plan 推薦報告端導出**，不動 `evaluation/v11/**`。

#### Gap D — `confusable_code_error_*` 在本語料上恆為 0（**最重要的缺口**）

`_confusable_errors()` 的判定是：找出帶有「不屬於 query identifier」之 identifier
的 Level-0 結果。但本 corpus 的 chunk **0.00% 帶 identifier**（Slice D/D2 皆實測），
因此 `conflicting` 恆為空清單 → **error count 恆為 0**。

**後果：現有 metric 無法量測「取到已廢止版本而非現行版本」這一類錯誤——而這正是
D2 觀察到最嚴重的問題（5/5 confuser pair 兩版本同進 top-5，3/5 的 top-1 是已廢止版）。**

需要的最小資訊：每個 answerable confuser case 標註其 **confuser 對照法規**
（`confusable_counterpart`，值為對方的 `<pcode>.md`）。有了它，即可在報告端導出：

- `confusable_version_error_rate` = top-1 的 `filename` == `confusable_counterpart`
  的 case 比例；
- `confusable_pair_both_in_topk` = top-k 同時含 target 與 counterpart 的比例；
- `repealed_over_active_top1` = target 為現行、top-1 為已廢止對照版的比例。

`confusable_counterpart` 是 **corpus-intrinsic** 的（由 frozen 的 title-stem
collision group 決定），不依賴任何模型輸出，故不違反 §3 的 selection 規則。

**Plan 推薦：以未驗證額外欄位攜帶 `confusable_counterpart`，metric 於報告端導出。**
若 Owner 要求 metric 進入 `trial_metrics`（可被 search constraint 使用），則需
擴充 `retrieval_metrics.py` + `metric_names.py` + `_AGGREGATE_METRIC_REQUIRES`
——屬 Owner Decision 6，本輪不實作。

#### Gap E — `exact_code_top1_accuracy` 的語意會在 A4 上線時改變（比較性風險）

在 A1–A3（`IdentifierExtractor` inert）下，`query_ids` 為空 →
`ok = (top1.relevance_level >= 1)`，該 metric 實際等同「top-1 相關率」。

Legal Identifier Support 上線後 `query_ids` 非空 → 改為要求
`query_ids & top1.identifiers` 非空，**判定條件變嚴**。

**這使 A2 與 A4 的 `exact_code_top1_accuracy` 不可直接相減。** 處置：

1. 每個 run 已記錄 `identifier_normalization_version`，報告必須並列顯示；
2. **主要 target 定位結論改用 Gap C 的 `exact_target_law_top1` /
   `exact_target_article_top1`**——它們只看 `filename` / `section`，與
   identifier 實作完全無關，故**跨 arm 恆可比較**；
3. `exact_code_top1_accuracy` 降為輔助指標，並在報告中明寫版本差異警語。

---

## 2. Proposed Dataset

### 2.1 Categories

| # | `legal_category` | 說明 | `query_type` 對映（A-1） | `answerable` |
| --- | --- | --- | --- | --- |
| 1 | `law_and_article_semantic` | 明確法規名稱 + 條文語意 | `semantic` | true |
| 2 | `paraphrase` | 純語意改寫，不提法規名稱 | `semantic` | true |
| 3 | `law_and_article_number` | 法規名稱 + `第 N 條` | `exact_code` | true |
| 4 | `law_and_subarticle_number` | 法規名稱 + `第 N-N 條` | `exact_code` | true |
| 5 | `similar_title_confuser` | 目標法規與同詞幹對照版並存 | `confusable` | true |
| 6 | `active_repealed_confuser` | 目標為現行（或已廢止）版，對照版為另一狀態 | `confusable` | true |
| 7 | `cross_law_semantic_confuser` | boilerplate 條文跨多部法規語意重疊 | `cross_section` | true |
| 8 | `unanswerable_out_of_corpus` | 語料中確實不存在的法律問題 | `unanswerable` | false |
| 9 | `unanswerable_confusable` | 表面像 corpus 內某法規，實際無對應條文（例如指名不存在的條號） | `unanswerable` | false |

Category 5 與 6 的差別：5 的兩部皆為現行（7 群 active/active），6 的兩部一現行一
已廢止（19 群 mixed）。兩者的失敗模式不同，必須分開統計。

### 2.2 metadata / foreword 是否列為正式 category — **建議不列入**

> Owner Decision 4。以下是判斷依據，非既定結論。

| 候選 category | 母體 | 評估 | 建議 |
| --- | --- | --- | --- |
| `foreword` | **僅 1 部法規有前言**（`A0000001` 中華民國憲法） | 母體為 1 ⇒ 只能形成 1 個 `group_id` ⇒ 只能落在單一 split ⇒ 無法同時出現在 tune 與 test，統計上無意義 | ❌ 不列入 |
| `metadata` | 100 個標頭 chunk | 檢索標頭 chunk 並非目前系統設計要解決的任務（無 metadata filter）；且 header 行為已由 `law_header_chunk_share` 與 D/D2 的 header pollution 量測覆蓋（索引占比 3.79%、條文查詢 top-5 占比 0.00%） | ❌ 不列入 |

兩者**保留為 sanity probe**（如 Slice D/D2 所做），不進正式 dataset。
**不因為 D/D2 做過 probe 就自動升格為正式 category。**

若 Owner 決定列入 `metadata`，代價是：需額外定義「標頭 chunk 是正確答案」的
gold evidence（section == 法規名稱），且該 category 的結論只能描述系統的
metadata 記憶行為，不能作為 retrieval 品質結論。

### 2.3 Size options

成本基準（用於估算，非承諾）：

- 人工 ground truth 建立 + 覆核：每個 answerable case 約 **4–8 分鐘**
  （閱讀條文 → 定義 evidence → 撰寫 query → 覆核 anchor 命中）；unanswerable case
  約 3–5 分鐘（需確認語料中確實無對應條文）。
- 執行成本：challenge corpus 索引一次約 **230 s**（2,637 chunk，D2 實測）。
  5 個 arm ⇒ 5 次索引 ≈ 19 min，加上 per-case retrieval；CE arm 顯著較慢
  （見 `docs/reports/rerank-end-to-end-latency-v1.2.1.md`）。**此為估算，H5 需實測。**

| 規模 | cases | answerable / unanswerable | 每 category 平均 | 建置工時估算 | test split（35%） | 適合的結論粒度 |
| --- | --- | --- | --- | --- | --- | --- |
| **small** | 100 | 75 / 25 | ~11 | ~8 h | 35 | 只能下 aggregate 結論；per-category 不可用 |
| **medium（推薦）** | 180 | 135 / 45 | ~20 | ~15 h | 63 | aggregate 結論穩健；per-category 為方向性參考 |
| **large** | 300 | 225 / 75 | ~33 | ~25 h | 105 | aggregate + per-category 皆可下結論 |

**推薦：medium（180 cases）。**

理由：

1. 母體本身有上限——26 個 confuser group、19 個 mixed group。category 5/6 的
   case 數受 group 數與 §2.5 的 per-group 上限限制，把 dataset 放大到 large
   並不會等比放大最關鍵的 confuser 訊號，只會放大 category 1/2/7 的比重，
   反而稀釋研究焦點。
2. medium 讓 test split 的 aggregate metric（63 cases，其中 ~47 answerable）
   足以支撐 arm 間比較；per-category 每類 ~7 例，只作方向性觀察。
3. 建置工時約 15 h 屬單人可維護範圍；large 的 25 h 會讓 dataset 覆核品質下降，
   而覆核品質正是 hard holdout 的價值來源。

**若 Owner 需要 per-category 的 test 結論（而非僅 aggregate），必須選 large。**
這是 small / medium / large 之間唯一實質的取捨。

### 2.4 Gold evidence 的形式 — 建議帶 document-scope offset

`graded_ndcg_at_5` 與 `evidence_coverage_at_5` 皆要求 gold evidence 具備
`char_start` / `char_end`（`offset_scope="document"`）；缺少時會誠實回報
`insufficient_graded_evidence` 而**不可計算**。

本 corpus 的 normalized markdown 結構固定（`## {section}` + 空行 + 正文 + 空行），
故條文正文的字元範圍**可由程式決定性導出**，人工成本接近 0。

**因此 plan 決定：所有 answerable case 的 gold evidence 一律帶 document-scope
offset + anchor_text。** 這讓 NDCG 與 coverage 兩個既有 metric 直接可用，
且 anchor 仍作為 fallback 保留（schema 要求）。

### 2.5 Confuser-group coverage policy

> Owner Decision 3。以下為建議值。

母體：26 個 title-stem collision group（其中 19 個 active/repealed mixed、
7 個 active/active）。

**規則：**

1. 每個 collision group 貢獻 **1–3 個 answerable case**（硬上限 3）。
   上限的目的是避免單一 group 支配 category 5/6；**不要求**每個 group 產出相同數量。
2. `group_id` 對 confuser case 一律為 `stem:<title_stem>`（例如 `stem:國家圖書館`）。
   ⇒ 同一 confuser pair 的所有 case **必然同 split**（schema 強制），
   pair 的兩部法規不可能一部在 tune、一部在 test。
3. **test split 的硬性下限**：至少 **8** 個 collision group，其中至少 **6** 個為
   active/repealed mixed group。此下限由 §2.6 的分層規則保證，並由 H4 的
   validator 斷言——**這是防止重演 Slice D「0 confuser」問題的具體檢查。**
4. 26 個 group 不必全部使用；未被使用的 group 仍屬 corpus，只是不產生 case。

### 2.6 Proposed split policy — deterministic、group-level、分層

**單純 case-level random split 明確不採用**，理由：confuser pair 的兩部法規條文
高度相似，若同一 pair 的 case 分散於 tune 與 test，開發 A4 時等同直接看到 test
的困難樣態 ⇒ 高度 leakage。

**規則（deterministic，無 RNG）：**

1. **split 的單位是 `group_id`，不是 case。**
2. `group_id` 命名約定：
   - confuser case（category 5/6）：`stem:<title_stem>`
   - 其他 answerable case：`law:<pcode>:<topic-slug>`（同一法規的同一主題族）
   - paraphrase family：**與其所改寫的來源 case 共用同一 `group_id`**
     ⇒ paraphrase family 不可能跨 split
   - unanswerable case：`unans:<theme-slug>`；若其誘餌指向某 confuser group，
     則**必須沿用該 group 的 `group_id`**（否則等於把該 pair 的訊息洩漏到另一 split）
3. **分層**（stratum）：
   - S1：active/repealed mixed confuser groups（19）
   - S2：active/active confuser groups（7）
   - S3：非 confuser 的 `law:` families
   - S4：`unans:` families（未綁定 confuser group 者）
4. **每個 stratum 內**，以
   `sha256(f"{dataset_version}|{group_id}")` 的十六進位字串升冪排序，
   前 40% → `tune`，次 25% → `validation`，其餘 35% → `test`（比例可調，
   見 Owner Decision 2）。
5. 因為分層在前、排序在後，**每個 split 必然同時含有 S1/S2/S3/S4 的成員**，
   confuser 覆蓋不靠運氣。
6. `dataset_version` 進入 hash 輸入 ⇒ 版本一改，split 重新洗牌；同一
   `dataset_version` 則**永遠得到相同 split**，不需要記錄 seed。

分層下限（19 個 S1 group × 35% ≈ 6.65 → 至少 6）恰好滿足 §2.5 規則 3 的
「test 至少 6 個 mixed group」。

**本輪不實際執行 split。**

### 2.7 Dataset provenance（freeze 契約）

Dataset 檔頂層必須包含（§1.5 Gap B 的形式）：

| 欄位 | 來源 |
| --- | --- |
| `schema_version` | `"1.1"` |
| `dataset_version` | `legal-holdout-NNNN` |
| `corpus.snapshot_id` / `corpus.corpus_hash` / `corpus.selection_hash` / `corpus.config_path` | §0.2 |
| `created_at` | ISO-8601 UTC |
| `notes` | 必須含 §0.3 的 challenge-set 用語 |
| 每 case：`legal_category`、`confusable_counterpart`（適用時）、`split`、`group_id`、`answerable`、`expected_evidence` | §2.1 / §2.5 |
| dataset hash | `hashing.dataset_hash(raw)`（涵蓋上述全部欄位） |
| split assignment | `SplitManifest`（含 `full_dataset_hash` 與每 split 的 `dataset_hash`） |

**Freeze 之後不得原地修改。** 任何 question / label / split 變更一律
bump `dataset_version` 並產生新的 `dataset_hash`；舊版本檔案保留、不刪除、不覆寫
（比照 corpus snapshot 的 immutability 慣例）。

---

## 3. Anti-Leakage Rules

### 3.1 Evidence-first question construction（防 difficulty mining）

**每個 answerable case 的建立順序是強制的：**

```
選定 source law（pcode）
  → 選定 article / section
  → 定義 relevant evidence（offset + anchor）
  → 才撰寫 query
```

**明文禁止**：

- 先寫一批 query → 跑 retrieval → 保留「模型答錯的題」；
- 依 dense / reranker / 未來 identifier 的結果決定某 case 是否進 dataset；
- 依「看起來模型會答錯」的直覺挑選法規或條文。

Case 的**選擇**只能依據：

- corpus / source intrinsic properties（pcode、title、title_stem、廢止狀態、
  LawCategory、條號、章節結構）；
- frozen 的 confuser group；
- evidence 內容本身。

建立 question 時**可以**閱讀 corpus evidence——這是必要的，且不構成 leakage，
因為 evidence 是 ground truth 的定義來源，不是模型輸出。

H3 的產出必須包含每個 case 的 `construction_order` 稽核記錄
（先有 evidence 才有 query），供 H4 檢核。

### 3.2 Test split 保護

**Dataset freeze 後，開發 Legal Identifier Support 期間只能使用 `tune` 與
`validation`。**

明文禁止（直到 §3.3 的解封條件全部滿足）：

- 查看 test 的 retrieval output；
- 查看 test 的任何 metric；
- 查看 test 的個別失敗案例；
- 以 test 結果調整任何 config、normalization、candidate selection 或程式碼。

**機制性保護（非僅靠自律）：**

1. `prepare_split_datasets` 已提供**物理隔離**——每個 split 是獨立檔案，
   Trial Runner 只收到單一 split 的路徑。
2. H5 的 run config **只列出** `tune.cases.json` 與 `validation.cases.json`；
   `test.cases.json` 不出現在任何 run config 中。
3. Test split 檔案存放於獨立目錄並標記 `SEALED`；H1–H5 的所有 command
   不得引用該路徑。
4. 每個 run 的 `EvaluationRun.dataset_hash` 已記錄實際使用的 split dataset hash
   ⇒ 事後可稽核「H5 期間是否曾對 test 執行過任何 run」。

### 3.3 Test 解封條件（必須**全部**滿足）

Final test evaluation 只能在下列項目**全部 freeze** 之後執行**一次**：

1. Legal Identifier Support 的實作（程式碼）；
2. 全部 config（chunk config、retrieval config、identifier 參數）；
3. normalization version（`canonical_text_version` / `loader_version` /
   `identifier_normalization_version`）；
4. retrieval behavior（candidate selection、admission、ranking 規則）。

解封後，**在同一份 pristine test 上同時執行全部五個 arm**：

```
A1 dense
A2 identifier_aware（現行，預期 inert）
A3 dense + cross_encoder
A4 legal_identifier_aware
A5 legal_identifier_aware + cross_encoder
```

**舊 baseline（A1–A3）不需要現在先跑 test**——這正是「同時在 pristine test 上跑
全部 arm」的目的：所有 arm 面對的是同一份從未被觀察過的 test split，
比較基礎完全對等。

### 3.4 Group separation

- 同一 `group_id` 不得跨 split（schema + `check_group_leakage` 雙重強制）。
- 同一 title-stem collision group 的**兩部法規**必然同 `group_id` ⇒ 必然同 split。
- 同一 question paraphrase family 共用 `group_id` ⇒ 必然同 split。
- unanswerable 誘餌若指向某 confuser group，沿用該 group 的 `group_id`。

**例外必須明文論證。** 目前本 plan 不提出任何例外。

---

## 4. Execution Phases

> 通則：**每個 phase 結束即 STOP，等待 Owner review**；不得自行推進下一 phase。
> 所有 phase 皆禁止修改 `app/**`、`evaluation/v11/**`、`evaluation/runner.py`、
> production API，以及 `IdentifierExtractor`。所有 phase 皆禁止開始 Slice E。

### H1 — Schema / category / split design

| 項目 | 內容 |
| --- | --- |
| **Objective** | 把本 plan 的 §2 定案為可執行的具體規格：category enum、`group_id` 命名規則、split 演算法、dataset 檔格式（含 provenance 欄位）、validator 規格 |
| **Allowed files** | `docs/legal-hard-holdout-plan.md`（更新）、`docs/legal-holdout-schema.md`（新增，如需要） |
| **Forbidden files** | `app/**`、`evaluation/v11/**`、`evaluation/datasets/**`、`evaluation/configs/**`、任何 dataset 檔 |
| **Tasks** | 1. 依 Owner Decisions 定案 size / 比例 / coverage / category；2. 寫出 `legal_category` enum 與 `query_type` 對映表；3. 寫出 split 演算法的 pseudo-code 與可驗證性質；4. 寫出 dataset validator 的檢查清單 |
| **Tests** | 無（純文件） |
| **Artifacts** | 定案後的 plan / schema 文件 |
| **Acceptance** | 每個 Owner Decision 皆有明確結論；split 演算法可由第三人重現；validator 檢查清單涵蓋 §2.5 規則 3 與 §2.7 全部欄位 |
| **STOP** | 文件完成即 STOP。**不得**開始建立任何 case |

### H2 — Evidence inventory

| 項目 | 內容 |
| --- | --- |
| **Objective** | 由 frozen snapshot 決定性導出**候選 evidence 清單**——每筆為「某 pcode 的某 section 的字元範圍 + 可用 anchor 候選」。**這是資料盤點，不是題目** |
| **Allowed files** | `evaluation/corpus/**`（唯讀使用）、新增 inventory 產出檔（建議 `evaluation/holdout/evidence-inventory.json`）、其 unit test |
| **Forbidden files** | `app/**`、`evaluation/v11/**`、`evaluation/datasets/**`、`IdentifierExtractor`、任何 retrieval / reranker 程式碼 |
| **Tasks** | 1. 讀 snapshot `normalized/*.md`，導出每個 `## section` 的 `char_start` / `char_end`（document scope）；2. 計算每個檔案的 `document_sha256`；3. 標註 confuser group 歸屬與 `confusable_counterpart`；4. 標註條號型態（`第 N 條` / `第 N-N 條`）、章節脈絡、廢止狀態；5. 產出決定性 inventory（同 snapshot 必得同輸出） |
| **Tests** | inventory 決定性（重跑 byte-identical）；`document_sha256` 與 `corpus_hash` 一致；offset 導出正確（切片回原文相符）；confuser group 數 == 26、mixed == 19 |
| **Artifacts** | `evidence-inventory.json` + 其 hash |
| **Acceptance** | 100 部法規、2,440 條文全數涵蓋；0 筆 offset 導出失敗；inventory 可重現 |
| **STOP** | inventory 完成即 STOP。**不得**執行任何 retrieval，**不得**撰寫任何 question |

### H3 — Question + ground-truth construction

| 項目 | 內容 |
| --- | --- |
| **Objective** | 依 §3.1 evidence-first 規則建立全部 case 與 gold label |
| **Allowed files** | `evaluation/datasets/legal-holdout/**`（新增）、建構輔助腳本、其 unit test |
| **Forbidden files** | `app/**`、`evaluation/v11/**`、`evaluation/datasets/v1.1/**`、`IdentifierExtractor`、任何 retrieval / reranker 執行 |
| **Tasks** | 1. 自 H2 inventory 依 corpus-intrinsic 規則抽出 case 候選；2. 對每個候選：先鎖定 evidence（pcode + section + offset + anchor）；3. **再**撰寫 query；4. 標註 `legal_category` / `query_type` / `group_id` / `confusable_counterpart`；5. 建立 unanswerable case 並**逐一確認語料中確實無對應條文**；6. 記錄 `construction_order` 稽核欄位 |
| **Tests** | anchor 確實出現在 evidence 指向的條文文字中；offset 切片與 anchor 一致；unanswerable case 的誘餌法規存在但目標條文不存在；`group_id` 命名符合 §2.5 規則 2 |
| **Artifacts** | 未 freeze 的 draft dataset（`status="pending"`） |
| **Acceptance** | 達成 Owner 核可的 case 數；每個 answerable case 皆有 ≥1 offset-grounded evidence + ≥1 anchor；**零** retrieval 執行紀錄 |
| **STOP** | draft 完成即 STOP。**不得** freeze，**不得**跑 baseline |

### H4 — Dataset validation / freeze

| 項目 | 內容 |
| --- | --- |
| **Objective** | 驗證 draft、指派 split、freeze 為不可變 dataset 版本 |
| **Allowed files** | `evaluation/datasets/legal-holdout/**`、validator 與其 unit test |
| **Forbidden files** | 同 H3；此外**禁止**任何形式的 test split 內容檢視（除 count / hash 等聚合統計外） |
| **Tasks** | 1. 執行 `case_schema.parse_dataset`（既有驗證）；2. 執行本 benchmark 專屬 validator（§2.7 欄位、`legal_category` enum、`confusable_counterpart` 指向存在的法規且屬同一 group）；3. 依 §2.6 分層規則指派 split；4. 執行 `check_group_leakage`；5. 斷言 §2.5 規則 3（test ≥8 groups、≥6 mixed）；6. 以 `prepare_split_datasets` 產出三個 split 檔；7. 計算 `dataset_hash` 與 `SplitManifest`；8. 將 test split 標記為 SEALED |
| **Tests** | group leakage == 0；同一 confuser pair 同 split；paraphrase family 同 split；test confuser 下限達成；split 指派決定性（重跑相同）；freeze 後 dataset 檔 byte-identical |
| **Artifacts** | frozen dataset + `tune/validation/test.cases.json` + `SplitManifest`（含 `full_dataset_hash`）+ freeze 記錄（dataset_version / created_at / corpus provenance） |
| **Acceptance** | 全部驗證通過；`dataset_hash` 記錄於文件；test split 未被任何 run 引用 |
| **STOP** | freeze 完成即 STOP |

### H5 — Baseline on tune + validation ONLY

| 項目 | 內容 |
| --- | --- |
| **Objective** | 在 **tune + validation** 上建立 A1 / A2 / A3 的 baseline，作為 Legal Identifier Support 的開發起點與可行性依據 |
| **Allowed files** | `evaluation/configs/**`（新增 run config）、報告輸出（`evaluation/runs/**`，已 gitignored）、結果文件 |
| **Forbidden files** | `app/**`、`evaluation/v11/**`、`IdentifierExtractor`、**`test.cases.json`（絕對禁止引用）**、frozen dataset（不得修改） |
| **Tasks** | 1. 建立 run config，`dataset_path` 只指向 tune / validation；2. 執行 A1 / A2 / A3；3. 產出既有 metric + §1.5 Gap C/D 的報告端導出指標；4. 記錄 A2 的 inert 證據（`identifier_candidate_count`、`candidate_sources`）；5. 記錄 CE 的 quality 與 latency，**且兩者分開陳述**（§6） |
| **Tests** | run config 不含 test path（自動化斷言）；`EvaluationRun.dataset_hash` 等於 tune / validation 的 split hash；正式 Qdrant collection before/after 一致；只建立與刪除自己的 `optimizer_` collection |
| **Artifacts** | `evaluation/runs/<run_id>/report.{json,md}` + 一份摘要文件；摘要必須含 §0.3 的 challenge-set 用語 |
| **Acceptance** | A1/A2/A3 皆完成；A2 的 inert 行為被明確量化；**零** test split 存取（可由 dataset_hash 稽核） |
| **STOP** | **STOP。H5 完成後，才由 Owner 決定是否開始 Slice E — Legal Identifier Support。** 本 plan 不預設會開始 |

### H5 之後（不屬本 plan 範圍，僅標示邊界）

Legal Identifier Support 開發期間持續適用 §3.2；解封條件見 §3.3。
Final test evaluation 為**單次**執行、五個 arm 同時進行，且必須另行取得 Owner 核准。

---

## 5. Metrics — 最終清單

### 5.1 直接沿用（零程式碼修改）

| Metric | Scope | 說明 |
| --- | --- | --- |
| `mrr` | aggregate + query_type | 主要排序品質指標 |
| `recall_at_{1,3,5}` | aggregate | == `relevant_context_hit_at_k` |
| `relevant_context_hit_at_{1,3,5}` | aggregate + query_type | Level ≥1 |
| `strong_evidence_hit_at_{1,3,5}` | aggregate | Level 2 |
| `graded_ndcg_at_5` | aggregate | `trial_index_gold_v1`；因 §2.4 帶 offset 故可計算 |
| `evidence_coverage_at_5` | aggregate | 需 offset |
| `relevant_section_precision_at_5` | aggregate | confuser 稀釋的直接反映 |
| `unanswerable_false_positive_count` / `_rate` | aggregate | **直接支援 Owner 要求的 unanswerable 評估** |
| `duplicate_ratio` | aggregate | confuser pair 屬不同 document，不會誤判為 duplicate |
| `mean/p50/p95_retrieval_latency_ms` | aggregate | |
| `mean/p50/p95_rerank_latency_ms`、rerank cache telemetry | aggregate | 僅 CE arm 產生 |
| `case_count` | query_type | 覆蓋度稽核 |

### 5.2 報告端導出（零程式碼修改，由 `report.json` 後處理）

| Metric | 定義 | 為何需要 |
| --- | --- | --- |
| `exact_target_law_top1` | top-1 的 `filename` == gold 的 `filename` | Gap C；**跨 arm 恆可比較**（不依賴 identifier 實作） |
| `exact_target_article_top1` | 上式且 `section` 相符 | Gap C |
| `confusable_version_error_rate` | top-1 == `confusable_counterpart` 的比例 | **Gap D，本研究最關鍵的失敗模式** |
| `confusable_pair_both_in_topk` | top-k 同時含 target 與 counterpart | D2 實測 5/5，需追蹤是否改善 |
| `repealed_over_active_top1` | target 為現行、top-1 為已廢止對照版 | D2 實測 3/5 |
| per-`legal_category` 彙總 | 依 §2.1 的 9 類 | Gap A（A-1 方案） |

### 5.3 明確不新增

- 不為法律語料新造 vanity metric。
- `exact_code_top1_accuracy` **保留但降為輔助**，並在報告中標註 Gap E 的
  版本比較風險。
- 任何需要**新 label** 的 metric，其 label 成本已在 §1.5 逐項列出；
  除 `confusable_counterpart`（corpus-intrinsic、成本極低）外，本 plan 不引入新 label。

---

## 6. Cross-Encoder：quality 與 deployability 必須分離

已知事實：CE 的 quality signal 有前景，但 **production latency 不符目前 SLO**
（見 `docs/v1.2.1-rerank-design.md` 與 `docs/reports/rerank-end-to-end-latency-v1.2.1.md`）。

本 benchmark **保留 A3 / A5 兩個 CE arm**，因為本次目的首先是研究 ranking quality。

**報告結構強制要求：**

1. **Ranking quality 章節**：只陳述 §5 的品質指標，可自由比較五個 arm。
2. **Production deployability 章節**：獨立章節，只陳述 latency 與資源成本，
   並明寫「CE 目前不符 production SLO」。
3. **禁止**在 quality 章節中出現任何「可以上線 / 建議部署」的結論；
   **禁止**以 cached rerank latency 作為 feasibility 依據
   （既有 `_rerank_metrics` 的 docstring 已載明此約束，本 plan 重申）。
4. 兩章節的結論不得合併為單一「CE 較好」的敘述。

---

## 7. Open Owner Decisions

> 僅列真正需要 Owner 拍板、且會改變後續工作內容的事項。
> 純 implementation detail 不列入。

### D1 — Dataset target size

| 選項 | cases | 工時估算 | 可下的結論粒度 |
| --- | --- | --- | --- |
| small | 100 | ~8 h | 僅 aggregate |
| **medium（plan 推薦）** | **180** | **~15 h** | aggregate 穩健 + per-category 方向性 |
| large | 300 | ~25 h | aggregate + per-category 皆可下結論 |

**唯一實質取捨**：是否需要 per-category 的 test 結論。需要 ⇒ large。

### D2 — tune / validation / test 配置比例

Plan 推薦 **40 / 25 / 35**（依 `group_id` 分層計算，非 case 數）。

理由：test 需足夠大以支撐單次 final evaluation 的 aggregate 結論；
tune 需足夠大以支撐 Legal Identifier Support 的開發迭代。
若 Owner 更重視開發迭代空間，可改 50/20/30；更重視 final 結論穩健性，可改 35/25/40。

### D3 — Confuser-group coverage policy

Plan 推薦：

- 每個 collision group 貢獻 **1–3** 個 case（硬上限 3，不要求相同數量）；
- **test split 至少 8 個 collision group，其中至少 6 個為 active/repealed mixed**。

此下限是防止重演 Slice D「0 confuser」問題的具體檢查點。Owner 可調整下限數字。

### D4 — `metadata` / `foreword` 是否列為正式 category

Plan 推薦 **兩者皆不列入**（理由見 §2.2：foreword 母體僅 1 部法規，
統計上無法 split；metadata 非目前系統設計要解決的任務，且行為已被 header
pollution 量測覆蓋）。兩者保留為 sanity probe。

### D5 — 額外 legal label 的攜帶方式

`legal_category` 與 `confusable_counterpart` 兩個欄位是 §5.2 關鍵 metric 的前提。

| 選項 | 作法 | 影響 |
| --- | --- | --- |
| **D5-a（plan 推薦）** | 以**未驗證**的額外欄位攜帶（`parse_dataset` 不拒絕未知 key，且欄位進入 `dataset_hash`），由 H4 的專屬 validator 驗證 | 零 `evaluation/v11/**` 修改；代價是 schema 本身不保證這些欄位存在 |
| D5-b | 擴充 `case_schema.py`（新增欄位 + enum），bump `SCHEMA_VERSION` | 需修改 `evaluation/v11/**`；需確認既有 v1.1 dataset 相容性；schema 層級保證 |

### D6 — §5.2 的導出指標是否需要進入 `trial_metrics`

| 選項 | 作法 | 影響 |
| --- | --- | --- |
| **D6-a（plan 推薦）** | 只在報告端導出 | 零 `evaluation/v11/**` 修改；代價是這些指標**不能**被 optimizer search 的 hard constraint 使用 |
| D6-b | 擴充 `retrieval_metrics.py` + `search/metric_names.py` + `_AGGREGATE_METRIC_REQUIRES` | 可作為 search constraint；需修改 `evaluation/v11/**` 並 bump `METRIC_CONFIG_VERSION`（既有報告的可比較性標示會改變） |

若本研究只做 arm 比較、不做 optimizer search，D6-a 已足夠。

---

## 8. 本輪明確未做的事

- 未建立任何 question、label、evidence、dataset、split 檔。
- 未執行任何 retrieval、reranker、index build、Qdrant 操作。
- 未修改 `IdentifierExtractor`、`app/**`、`evaluation/v11/**`、
  `evaluation/datasets/**`、`evaluation/runner.py`。
- 未開始 Slice E。
- 未跑任何 benchmark。
