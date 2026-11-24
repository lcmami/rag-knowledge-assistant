# MOJ 法規 Challenge Corpus 結果（Slice D2）

> 語言：繁體中文。程式名稱、class / function / config field、command 保留英文。
>
> **範圍：** 記錄 Slice D2 的實測結果——第二份 bounded（100 部）真實外部法規 corpus，
> 保留 Slice D 的可控規模與 reproducibility，但**刻意納入完整來源中已知存在的
> title confuser 與 active/repealed confuser**。
>
> **這不是品質評測。** 本文不含 metric threshold、不含 hard label，也**未**建立任何
> evaluation dataset。所有 retrieval 觀察皆為人工可理解的 sanity probe。
>
> Production 行為完全未變更；`app/**` 與 `evaluation/v11/**` 零修改。

## ⚠️ 本 corpus 的性質限制（必讀）

**這是一份 challenge corpus，不是完整來源的 representative sample。**

其 52/100 部（52%）是刻意納入的 title-stem confuser，而完整 1,346 部來源中
confuser 僅佔 52 部（3.9%）。因此本 corpus 的混淆密度**遠高於**真實分佈：

| 維度 | 完整來源（1,346 部） | D2 challenge（100 部） |
| --- | --- | --- |
| title-stem 碰撞法規佔比 | 52 / 1,346 = **3.9%** | 52 / 100 = **52%** |
| 已廢止法規佔比 | 322 / 1,346 = 23.9% | 27 / 100 = 27% |

**後果：**

- 本 corpus 上的 retrieval 行為**不得**外推為完整 corpus 的預期表現——它是刻意
  加壓過的上界情境。
- 反過來說，若某個 retrieval 策略在此 corpus 上仍能區辨 confuser，該證據是有意義的；
  若無法區辨，也不代表在真實分佈下同樣嚴重。
- 與 Slice D 的 pcode-first pilot 是**互補**而非取代關係：D 是「真實分佈的一個
  狹窄切片」，D2 是「刻意加壓的 confusability 切片」。兩份 snapshot 皆保留。

相關文件：
- [docs/external-corpus-api-plan.md](../docs/external-corpus-api-plan.md) — 完整計畫
- [evaluation/corpus-moj-law-pilot-report.md](corpus-moj-law-pilot-report.md) — Slice D（pcode-first pilot，NOT YET SUITABLE）

---

## 1. Selection Policy

### 1.1 Mandatory set — 全部 title-stem collision group

以 `evaluation/corpus/stats.py` 已驗證的 `title_stem` 規則（剝除單一法制形式尾綴，
詞幹至少 2 字）對**完整 1,346 部來源**建 collision group：

| 項目 | 完整來源實測 | 全數納入 D2 |
| --- | --- | --- |
| title-stem collision group | 26 群 | ✅ 26 / 26 |
| 上述群組的法規 | 52 部 | ✅ 52 / 52 |
| 其中 active/repealed mixed group | 19 群 | ✅ 19 / 19 |

`mandatory_count = 52 <= max_laws = 100`，故不觸發 STOP 條件。26 群全部是
「同一機關的『組織條例』 vs 『組織法』」——這正是計畫最想壓測的那一類混淆。

### 1.2 Fill — 剩餘 48 slot 的 deterministic broad-coverage 取樣

剩餘法規**不做任何人工挑選**，規則為：

1. 非 mandatory 的 1,294 筆依 `pcode` 升冪排成 list（長度 N = 1,294）；
2. 取 index `(i * N) // k`，i = 0..k-1，k = 48 —— 即在**完整排序範圍內**
   evenly-spaced 取樣（刻意**不**取前 48 筆）；
3. 最終 100 個 pcode 依 `pcode` canonical 排序 freeze。

此規則完全 source-intrinsic：只用 `title` / `title_stem` / repealed status /
`pcode`，**未**使用任何 retrieval outcome、reranker outcome 或未來評測題目。

取樣結果橫跨 A/B/C/D/E/F/G/H/I/J/K/L/M/N/O/R/S 共 17 個 pcode 字母域，
相較 Slice D 只涵蓋 `A0000001`–`A0030243` 單一區段，coverage 顯著加寬。

### 1.3 Freeze

最終清單 freeze 於新的受追蹤 config：

```
evaluation/configs/corpus-moj-law-challenge.json
```

使用 `include_pcodes: [...]`（100 個 pcode）。**未來來源更新時，此 config 仍會挑出
同一批法規**，不會因 collision group 變動而悄悄產生不同的 challenge corpus 卻共用
同一份 config identity。

`evaluation/configs/corpus-moj-law-pilot.json` **未做任何修改**。

## 2. Backward Compatibility

新增的 `include_pcodes` 是 **optional selection capability**：只有在實際使用時才會
進入 selection payload 與 hash。

| 項目 | 值 | 狀態 |
| --- | --- | --- |
| 舊 pilot `selection_hash` | `2e815ced283adb3d04556db44b4300078bc3fb09e39c87fc2450da7c1476ad14` | ✅ 未變 |
| 舊 pilot `selection` payload | `{max_laws, category_prefix, include_repealed, order_by}` | ✅ 未新增欄位 |
| 舊 snapshot `moj-chlaw-20260821-2e815ced-9629909b` | `corpus_hash` = `0310b7ec…b1519` | ✅ 未覆寫、重新驗算通過 |

回歸測試（`tests/unit/test_corpus_snapshot.py`）：

- `test_pilot_selection_hash_is_backward_compatible`
- `test_optional_selection_field_stays_out_of_payload_when_unused`
- `test_include_pcodes_null_is_equivalent_to_absent`
- `test_tracked_pilot_snapshot_manifest_still_matches_the_pilot_config`
  —— 直接讀**已 commit 的** manifest，斷言其 `selection` / `selection_hash` /
  `snapshot_id` 仍能由今天的程式碼重算出來。

## 3. Challenge Snapshot Identity

| 項目 | 值 |
| --- | --- |
| `snapshot_id` | `moj-chlaw-20260821-60e8123b-9629909b` |
| `source_update_date` | `2026/8/21 上午 12:00:00`（與 Slice D **同一份來源版本**） |
| `raw_sha256` | `9629909b87688f0a26eb4dd12da79c54c61ec9e9db79532e60e8e0d436620425`（與 D 相同） |
| `selection_hash` | `60e8123bccf5c8743bef8d281f06906c9946019085f7150e3792241d6b9f7bbc` |
| `corpus_hash` | `b42622b01c5e68b38f3e77df31632c17366b62a141fa75da9f6e79cf50c9dbc3` |

入庫體積：`manifest.json` 2,060 B + `normalized/` 936,405 B = **約 0.89 MB**，
未觸及計畫 §6.7 的 10 MB size trigger。`raw/`（32.2 MB，與 D 共用同一份來源 bytes）
依 `.gitignore` 規則不入庫。

重現指令（完全 offline）：

```
python -m evaluation.corpus.cli snapshot --config evaluation/configs/corpus-moj-law-challenge.json --raw-zip <raw.zip>
python -m evaluation.corpus.cli stats    --snapshot evaluation/corpora/moj-law/moj-chlaw-20260821-60e8123b-9629909b --markdown <out.md>
```

## 4. Corpus Statistics

`target_chars=450` / `overlap_chars=80` / `max_tokens=480`（**未修改**，與 Slice D 相同）。

### 4.1 規模

| 項目 | Slice D（pcode-first） | Slice D2（challenge） |
| --- | --- | --- |
| 挑選法規數 | 100 | 100 |
| type-A 條文數 | 3,437 | 2,440 |
| chunk 總數 | 3,643 | **2,637** |
| 條文 chunk | 3,543 | 2,537 |
| 法規標頭 chunk | 100 | 100 |
| 已廢止 / 現行 | 23 / 77 | **27 / 73** |
| canonical segment | — | 5,081 |
| skipped record / article | 0 / 0 | 0 / 0 |

條文數較少是因為 confuser 多為短篇組織法規（多數 5–20 條）。

### 4.2 Chunk 與 token 分布

| 分布 | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chunk 字元數 | 11 | 87 | 282 | 386 | 443 | 450 | 127.08 |
| chunk token 數 | 7 | 79 | 274 | 380 | 434 | 450 | 118.84 |
| 每部法規 chunk 數 | 6 | 13 | 70 | 90 | 161 | 176 | 26.37 |
| 每條條文 chunk 數 | 1 | 1 | 1 | 1 | 2 | 3 | 1.04 |

| 項目 | 值 |
| --- | --- |
| hard token split | **61 / 2,440（2.5%）** |
| 切分後超過 token 上限 | 0 |
| strategy fallback | 0 |

hard split 2.5%（Slice D 為 1.98%），仍屬同一量級；450/80/480 設定可用，**未修改**。

### 4.3 Identity / metadata

| 項目 | 值 |
| --- | --- |
| unique pcode / 標題 | 100 / 100 |
| 重複 pcode / 標題 | 0 / 0 |
| metadata 六欄全備 | 100 / 100 |
| 有前言 / 無前言 | 1 / 99 |
| 已廢止且有廢止註記 | 27 / 27 |

### 4.4 混淆性——**D2 的核心目標**

| 指標 | Slice D | **Slice D2** | 完整來源 |
| --- | --- | --- | --- |
| 完全同名群組 | 0 | 0 | 0 |
| **title-stem 碰撞** | **0 群 / 0 部** | **26 群 / 52 部** | 26 群 / 52 部 |
| **現行／已廢止交錯詞幹** | **0 群** | **19 群** | 19 群 |
| 共用 6 字前綴 | 4 群 / 8 部 | **26 群 / 53 部** | 95 群 / 288 部 |
| 已廢止法規 | 23 | 27 | 322 |

**Confusability acceptance 全數達成**：完整來源目前的 26 組 collision group 與
19 組 active/repealed mixed group **全部**存在於本 corpus。

19 組 mixed confuser（`組織條例` 已廢止 vs `組織法` 現行）：

內政部建築研究所、內政部役政署、內政部警政署、國家圖書館、國立中正紀念堂管理處、
國立國父紀念館、國立教育廣播電臺、國立歷史博物館、國立自然科學博物館、
國立臺灣科學教育館、國防部主計局、國防部參謀本部、國防部軍備局、國防部軍醫局、
外交部領事事務局、法務部法醫研究所、法務部行政執行署、財政部國庫署、財政部賦稅署。

另 7 組為 active/active confuser（交通部民用航空局、交通部運輸研究所、內政部消防署、
經濟部智慧財產局、經濟部標準檢驗局、經濟部水利署、經濟部產業園區管理局）。

### 4.5 條號歧義

| 項目 | Slice D | Slice D2 |
| --- | --- | --- |
| 相異條號標籤 | 522 | 240 |
| `第 1 條`～`第 5 條` 各自出現於 | 94 / 100 部 | **100 / 100 部** |
| `第 6 條` 出現於 | — | 95 / 100 部 |
| 出現於 ≥10 部法規的條號標籤 | 77 | 64 |
| `第 N-N 條` 型標籤 | 253 | 65（分佈於 19 部法規） |

條號歧義**比 Slice D 更強**：`第 1 條`～`第 5 條` 在 100 部中**無一例外**全部出現。
相異標籤總數較少只是因為組織法規篇幅短（多數 5–20 條），不影響歧義強度。

## 5. Header Chunk 觀察

| 項目 | Slice D | Slice D2 |
| --- | --- | --- |
| 標頭 chunk 數 | 100（每部 1 個） | 100（每部 1 個） |
| 索引中占比 | 2.74% | **3.79%** |
| 含前言 / 僅 metadata | 4 / 96 | 1 / 99 |

**污染程度量測**（與 Slice D 同法：決定性抽樣 30 條真實條文，取其前 40 字為 query，
top-5）：

| 項目 | Slice D | Slice D2 |
| --- | --- | --- |
| 查詢數 | 30 | 30 |
| top-5 結果總數 | 150 | 150 |
| 其中為標頭 chunk | 1（0.67%） | **0（0.00%）** |
| 標頭排在第 1 名的查詢數 | 1 | **0** |

索引占比從 2.74% 上升到 3.79%（因為法規短、條文少，標頭的相對比重自然上升），
但一般條文查詢 top-5 中的標頭占比反而**降到 0**。**未觀察到 header pollution**，
不構成 blocker。

標頭 chunk 仍會被 metadata 式查詢（probe 7）與短法規的整體性查詢（probe 4、6）命中，
這是預期且無害的行為。`normalize.py` 與 `TextSplitter` **未做任何修改**。

## 6. Retrieval Sanity Probes

dense、`score_threshold=0.55`、`top_k=5`。**非評測**，未調整任何參數，
**未依結果改動 corpus**。`nature`：`ARTICLE` = 條文 chunk，`HEADER` = 法規標頭 chunk。

| # | 類型 | 查詢 | Top-1 | 觀察 |
| --- | --- | --- | --- | --- |
| 1 | 法規名稱 + 條文內容 | 內政部警政署組織法 署長的官等職務列等與副署長的設置 | `D0000242` 內政部**消防署**組織法 第 3 條（0.7044） | ⚠️ **取到完全不同機關**；目標 `D0000159` 落在第 4 名，top-5 橫跨 5 部法規 |
| 2 | 純 semantic paraphrase | 國家圖書館掌理的圖書資料蒐集整理與國際交換業務 | `H0000011` 國家圖書館組織**條例**（**已廢止**）第 1 條（0.8084） | ⚠️ **已廢止版本排在現行版本之前**；現行 `H0000118` 為第 2 名 |
| 3 | 法規名稱 + 第 N 條 | 國防部參謀本部組織法 第 3 條 | `F0000074` 國防部參謀本部組織法 **第 1 條**（0.7022） | ❌ **法規對、條號錯**；top-5 有 3 席是已廢止的組織條例 |
| 4 | `第 N-N 條` | 九二一震災重建暫行條例 第 17-1 條 | `A0030110` **HEADER**（0.8039） | ❌ 法規對，但 **`第 17-1 條` 未出現在 top-5**（取到 第 74-1 / 1 / 2 / 5 條） |
| 5 | 相似法規名稱 | 經濟部智慧財產局組織條例 局長的職務列等與副局長員額 | `J0000096` 經濟部智慧財產局組織**法** 第 1 條（0.7310） | ⚠️ **查詢指名「條例」卻取到「法」**；top-5 另含標準檢驗局、民用航空局 |
| 6 | 現行 vs 已廢止 | 法務部行政執行署的組織編制與所屬行政執行處 | `I0000034` 現行組織法 HEADER（0.7488） | ⚠️ 第 2 名即為**已廢止**的組織條例（0.7301），兩版本交錯出現 |
| 7 | metadata-style | 法規類別 廢止法規 國防部 法規位階 法律 | `F0000067` HEADER（0.6942） | 📋 top-5 **全部**為 HEADER 且**全部為已廢止**——metadata 查詢確實可用 |
| 8 | foreword-style | 中華民國國民大會受全體國民之付託 制定本憲法 | `A0000001` 憲法 第 25 條（0.7531） | 📋 取到憲法**條文**，前言標頭在第 4 名 |
| 9 | cross-law semantic | 本法所需人員之員額另以編制表定之 並得依聘用人員相關規定進用 | `A0010037` 行政院公共工程委員會組織條例 第 14 條（0.7691） | ⚠️ top-5 橫跨**5 部不同法規**，分數 0.7572–0.7691 幾乎並列 |

### 6.1 Confuser-pair 專項觀察（D2 新增）

對 5 組 title-stem confuser，以**不含法制形式字樣**的中性查詢（`{機關名}的設置與職掌`）
觀察 top-5：

| 查詢 | 配對 | 兩者皆進 top-5 | Top-1 | Top-1 狀態 |
| --- | --- | --- | --- | --- |
| 內政部警政署的設置與職掌 | `D0000015` / `D0000159` | ✅ | `D0000015` 組織條例 | **已廢止** |
| 法務部行政執行署的設置與職掌 | `I0000016` / `I0000034` | ✅ | `I0000016` 組織條例 | **已廢止** |
| 經濟部智慧財產局的設置與職掌 | `J0000061` / `J0000096` | ✅ | `J0000096` 組織法 | 現行 |
| 國防部參謀本部的設置與職掌 | `F0000002` / `F0000074` | ✅ | `F0000074` 組織法 | 現行 |
| 國家圖書館的設置與職掌 | `H0000011` / `H0000118` | ✅ | `H0000011` 組織條例 | **已廢止** |

**5 / 5 組的兩個版本同時出現在 top-5；3 / 5 組的 Top-1 是已廢止版本。**
這正是 Slice D 完全無法觀察到的現象。

### 6.2 綜合觀察

- **similar-title 現在真的產生 confusable top-k**（probe 1、5、§6.1）。查詢中的
  法規名稱**無法**把結果限制在該部法規，甚至無法限制在正確機關（probe 1）。
- **active/repealed 現在真的產生 confusable top-k**（probe 2、6、§6.1）。dense
  retrieval 對「現行 vs 已廢止」**完全沒有區辨能力**——兩版本文字高度相似，
  廢止狀態只存在於標頭 chunk 的 metadata 行，條文 chunk 本身不帶此訊號。
- **條號定位仍完全失效**（probe 3、4），與 Slice D 一致，與 §7 的 identifier
  limitation 同源。
- **cross-law semantic ambiguity 存在**（probe 9）：5 部法規的 boilerplate 條文
  分數幾乎並列。

### 6.3 Rerank 接線觀察

dense → Cross-Encoder（`BAAI/bge-reranker-base`），`ranking_version =
dense_v1+cross_encoder_v1`，raw logit 遞減排序正常（1.969 → -2.271）。
以 probe 5 為例，rerank 把 `J0000061 第 9 條`（原第 3 名）與
`J0000051 第 9 條` 並列提到前段，原 Top-1 `J0000096 第 1 條` 降至第 4 名。

**這只是接線 smoke**：確認 rerank stage 能吃下這份語料並產出合法結果。
未比較品質、未量測 latency、未建立任何 quality metric。

## 7. Identifier-aware

| 項目 | Slice D | Slice D2 |
| --- | --- | --- |
| 索引 chunk 數 | 3,643 | 2,637 |
| `identifiers` payload 非空的 chunk | 0（0.00%） | **0（0.00%）** |
| 9 個 probe query 的 `extract_normalized()` | 全部 `[]` | **全部 `[]`** |
| `identifier_candidate_count` | 0 | 0 |
| 候選來源 | 僅 `dense` | 僅 `dense` |
| `ranking_version` | `identifier_aware_v1` | `identifier_aware_v1` |

計畫 L1 在 challenge corpus 上**同樣完全成立**：`IdentifierAwareRetrievalStrategy`
100% 退化為 dense。依指示**未修改 `IdentifierExtractor`**。此數字是未來 Slice E 的
evidence，不是 D2 要修的功能。

## 8. D2 Suitability 結論

| 判斷依據 | 狀態 | 證據 |
| --- | --- | --- |
| 1. corpus 規模足夠 | ✅ | 100 部、2,440 條文、2,637 chunk（demo corpus 為 10 points） |
| 2. title confuser 已真正存在 | ✅ | 26 群 / 52 部詞幹碰撞（D 為 0）；probe 1、5 與 §6.1 5/5 產生 confusable top-k |
| 3. active/repealed confuser 已真正存在 | ✅ | 19 群交錯詞幹（D 為 0）；§6.1 中 3/5 組 Top-1 為已廢止版本 |
| 4. article-number ambiguity 足夠 | ✅ | `第 1 條`～`第 5 條` 各出現於 **100/100 部**；64 個標籤出現於 ≥10 部；65 個 `第 N-N 條` |
| 5. cross-law semantic ambiguity 存在 | ✅ | probe 9 top-5 橫跨 5 部法規，分數 0.7572–0.7691 幾近並列 |
| 6. header pollution 沒有形成 blocker | ✅ | 索引占比 3.79%，條文查詢 top-5 占比 **0.00%**，0 次排第 1 |
| 7. pipeline 仍可穩定 ingest/index/retrieve | ✅ | 100 檔 → 2,637 chunk → 2,637 point，0 fallback、0 超限 chunk；dense / identifier_aware / rerank 三條路徑皆正常 |

# **SUITABLE**

七項判斷依據全部成立。Slice D 唯一的 blocker（標題混淆與現行／廢止混淆完全缺席）
在本 corpus 中已被明確消除，且經 retrieval 觀察確認**這些混淆是真實可觸發的**，
而不只是統計上存在。

**下一步由 Owner 決定**：本文**未**建立任何評測題目、hard label 或 holdout dataset。

## 9. 已知限制

- **本 corpus 不是 representative sample**（見文首）。confuser 密度 52% vs 真實
  分佈 3.9%，任何 retrieval 觀察**不得**外推為完整 corpus 的預期表現。
- **26 組 confuser 全為同一種型態**：「機關組織條例 → 機關組織法」的法制形式更替。
  完整來源中的其他混淆型態（例如 95 群共用 6 字前綴中的其餘 69 群）並未被
  mandatory 規則涵蓋，只有 48 個 fill slot 隨機觸及其中 26 群。
- **L1 — identifier-aware 退化為 dense**（§7，0.00%）。未修改 `IdentifierExtractor`。
- **L2 — hard token split 61 / 2,440 = 2.5%**，450/80/480 未修改。
- 本文所有 retrieval 觀察皆為 sanity probe，樣本極小、無 hard label、無 metric
  threshold，**不得**當作品質評測結果引用。

## 10. 安全與隔離

- Qdrant：自固定位置 `D:\qdrant-v121-test\bin\qdrant.exe` 啟動，loopback only
  （127.0.0.1:6333 / 6334），isolated storage。
- 只建立自己的 collection `optimizer_d2_666d8992`，測試後只刪除該 collection。
- 正式 `factory_knowledge`：before **不存在** / after **不存在**，狀態一致；
  結束時 Qdrant collection 清單為空。
- 未呼叫 LLM、未讀寫 `.env`、未碰 production API、未修改 `app/**` 與
  `evaluation/v11/**`。
- 本 slice 僅一次網路存取來自既有 `external_api` marker 測試；D2 snapshot 由
  **既有 raw ZIP** 離線重建，未重複下載。
