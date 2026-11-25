# v1.1 Retrieval Optimizer — Slice 0 Baseline 報告

> 本報告為 v1.1 Slice 0 的 Baseline 固定化結果。Case 標註採 **Slice 0 Baseline 格式**（expected_document / expected_section / expected_page / should_answer），**Slice 1 才遷移至 Canonical Evidence Span**。

## 執行資訊

- 執行時間（UTC）：2026-07-19T07:59:09.361621+00:00
- Git commit：98884e90d5490656d40533df84b1696289d4d5b0
- Git branch：feature/v1.1-retrieval-optimizer
- Working tree dirty：False
- Python：3.10.11  /  Platform：Windows-10-10.0.26200-SP0

## Retrieval / Chunk 設定

- Embedding model：`BAAI/bge-small-zh-v1.5`（維度 512, Cosine）
- Chunk size / overlap：600 / 100 chars
- Embedding max tokens：480
- Top-K：5
- 檢索 score_threshold（收集原始分數）：0.0
- 報告 configured threshold（正式接受門檻）：0.55
- Candidate multiplier / cap：3 / 50
- 固定 random seed：20240501

## Corpus

- 文件數：6；總 chunk 數：24
- Corpus hash（sha256）：`9f575f0c912b161cb048a421cc07da619bb3957ab5f6a7334a4290d2b09d8ece`

| 文件 | chunks | sha256（前 12） |
| --- | ---: | --- |
| api_integration_spec.md | 5 | `680315276611` |
| confusable_alarm_d900.md | 4 | `c55288e033fd` |
| equipment_alarm_manual.md | 5 | `042f36719ba2` |
| equipment_maintenance_manual.pdf | 5 | `490ff764f373` |
| manufacturing_data_dictionary.txt | 1 | `9e12848195e2` |
| mes_operation_sop.md | 4 | `8432bccf03a1` |

## Evaluation Dataset

- Cases 檔：`cases.json`；案例數：15
- Dataset hash（sha256）：`181a00a20712b2f1d485b52484fcc008e836580961822eb43fdd870e6cff1334`

query_type 分布：

| query_type | 案例數 | Hit@1 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| confusable | 2 | 0.0 | 1.0 | 0.5 |
| exact_code | 4 | 0.75 | 1.0 | 0.875 |
| semantic | 5 | 1.0 | 1.0 | 1.0 |
| unanswerable | 4 | 0.0 | 0.0 | 0.0 |

## Aggregate Metrics

| 指標 | 值 | 備註 |
| --- | ---: | --- |
| 案例總數 | 15 | 可回答 11 / 無關 4 |
| Hit@1 | 0.7273 | considered=11 |
| Hit@3 | 1.0 | considered=11 |
| Hit@5 / Recall@5 | 1.0 | considered=11 |
| MRR | 0.8636 | considered=11 |
| Section Hit@1 | 0.6364 | considered=11 |
| Page Hit@1 | 1.0 | considered=2 |
| Exact-code Top-1 Accuracy | 0.75 | 3/4；miss=['C01'] |
| Confusable-code Error Rate | 0.4 | 2/5；err=['C01', 'C10'] |
| Unanswerable False-positive Rate | 0.0 | @thr=0.55；0/4；fp=[] |
| Latency avg / P50 / P95 (ms) | 26.53 / 29.0 / 35.8 |  |

## E102 / D900 實際排名（Confusable 壓力測試）

同一組 E102 相關案例中，正確 E102 文件與虛構 D900 干擾文件的實際排名（非固定期望值）：

| case | query_type | E102 rank (score) | D900 rank (score) |
| --- | --- | --- | --- |
| C01 | exact_code | 2 (0.6842) | 1 (0.707) |
| C02 | exact_code | 1 (0.6299) | 2 (0.6164) |
| C03 | exact_code | 1 (0.828) | 5 (0.6761) |
| C04 | exact_code | 1 (0.803) | 2 (0.5912) |
| C05 | semantic | 1 (0.7814) | 2 (0.6934) |
| C10 | confusable | 2 (0.6299) | 1 (0.7251) |
| C11 | confusable | 2 (0.7019) | 4 (0.6227) |

## Unanswerable 實際結果

- Configured threshold：0.55
- False-positive（top1_score ≥ threshold）：無

| case | 問題 | top1 文件 | top1 score |
| --- | --- | --- | ---: |
| C12 | CX-500 的市場售價是多少？ | equipment_maintenance_manual.pdf | 0.4986 |
| C13 | 台灣的首都是哪一個城市？ | confusable_alarm_d900.md | 0.3054 |
| C14 | 產線作業員的月薪是多少？ | api_integration_spec.md | 0.4523 |
| C15 | 公司股票今天的收盤價是多少？ | api_integration_spec.md | 0.3595 |

## 每題 Top-5

### C01 [exact_code] — 設備出現 E102 警報時應如何處理？

- expected：equipment_alarm_manual.md / E102 感測器通訊異常 / page=None；should_answer=True；latency=21ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.707 |
| 2 | equipment_alarm_manual.md | E102 感測器通訊異常 | None | 0.6842 |
| 3 | equipment_alarm_manual.md | 警報代碼對照表 | None | 0.6689 |
| 4 | equipment_alarm_manual.md | E101 電源模組異常 | None | 0.6615 |
| 5 | confusable_alarm_d900.md | D900 安全門異常 | None | 0.6583 |

### C02 [exact_code] — E102 警報代表什麼？

- expected：equipment_alarm_manual.md / E102 感測器通訊異常 / page=None；should_answer=True；latency=18ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_alarm_manual.md | 警報代碼對照表 | None | 0.6299 |
| 2 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.6164 |
| 3 | confusable_alarm_d900.md | 警報代碼對照表 | None | 0.6023 |
| 4 | equipment_alarm_manual.md | E102 感測器通訊異常 | None | 0.5451 |
| 5 | equipment_alarm_manual.md | E203 溫度過高警報 | None | 0.5371 |

### C03 [exact_code] — 設備顯示 E101 電源模組異常時，操作員應先確認什麼？

- expected：equipment_alarm_manual.md / E101 電源模組異常 / page=None；should_answer=True；latency=40ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_alarm_manual.md | E101 電源模組異常 | None | 0.828 |
| 2 | equipment_maintenance_manual.pdf | 通訊模組檢查 | 3 | 0.7385 |
| 3 | equipment_alarm_manual.md | E102 感測器通訊異常 | None | 0.6959 |
| 4 | equipment_alarm_manual.md | 警報代碼對照表 | None | 0.6778 |
| 5 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.6761 |

### C04 [exact_code] — E203 溫度過高警報的安全閾值是攝氏幾度？

- expected：equipment_alarm_manual.md / E203 溫度過高警報 / page=None；should_answer=True；latency=29ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_alarm_manual.md | E203 溫度過高警報 | None | 0.803 |
| 2 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.5912 |
| 3 | equipment_alarm_manual.md | 警報代碼對照表 | None | 0.5543 |
| 4 | confusable_alarm_d900.md | 安全門警報處理手冊（D900） | None | 0.5492 |
| 5 | equipment_alarm_manual.md | E102 感測器通訊異常 | None | 0.5477 |

### C05 [semantic] — 感測器無法與控制器正常通訊時應如何排除？

- expected：equipment_alarm_manual.md / E102 感測器通訊異常 / page=None；should_answer=True；latency=16ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_alarm_manual.md | E102 感測器通訊異常 | None | 0.7814 |
| 2 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.6934 |
| 3 | equipment_maintenance_manual.pdf | 通訊模組檢查 | 3 | 0.6513 |
| 4 | equipment_alarm_manual.md | 警報代碼對照表 | None | 0.6352 |
| 5 | confusable_alarm_d900.md | D900 安全門異常 | None | 0.6147 |

### C06 [semantic] — CX-500 設備的基本保養每幾小時執行一次？

- expected：equipment_maintenance_manual.pdf / 定期保養週期 / page=2；should_answer=True；latency=30ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_maintenance_manual.pdf | 定期保養週期 | 2 | 0.7283 |
| 2 | equipment_maintenance_manual.pdf | 異常保養 | 3 | 0.7012 |
| 3 | equipment_maintenance_manual.pdf | 設備基本資訊 | 1 | 0.6709 |
| 4 | manufacturing_data_dictionary.txt | None | None | 0.5644 |
| 5 | equipment_maintenance_manual.pdf | 通訊模組檢查 | 3 | 0.514 |

### C07 [semantic] — 空氣過濾網每幾小時需要清潔一次？

- expected：equipment_maintenance_manual.pdf / 定期保養週期 / page=2；should_answer=True；latency=29ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_maintenance_manual.pdf | 定期保養週期 | 2 | 0.6096 |
| 2 | equipment_maintenance_manual.pdf | 異常保養 | 3 | 0.5462 |
| 3 | equipment_maintenance_manual.pdf | 通訊模組檢查 | 3 | 0.5182 |
| 4 | equipment_alarm_manual.md | E203 溫度過高警報 | None | 0.4995 |
| 5 | manufacturing_data_dictionary.txt | None | None | 0.4925 |

### C08 [semantic] — MES 工單如何從 WAITING 切換為 RUNNING？

- expected：mes_operation_sop.md / 工單狀態流程 / page=None；should_answer=True；latency=33ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | mes_operation_sop.md | 工單狀態流程 | None | 0.7101 |
| 2 | mes_operation_sop.md | 工單啟動 | None | 0.5778 |
| 3 | api_integration_spec.md | 上報工單進度請求欄位 | None | 0.5546 |
| 4 | mes_operation_sop.md | MES 系統操作標準作業程序 | None | 0.509 |
| 5 | manufacturing_data_dictionary.txt | None | None | 0.4962 |

### C09 [semantic] — API Token 的有效時間是多久？

- expected：api_integration_spec.md / 認證與 Token / page=None；should_answer=True；latency=15ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | api_integration_spec.md | 認證與 Token | None | 0.6965 |
| 2 | api_integration_spec.md | Token 規格 | None | 0.5877 |
| 3 | api_integration_spec.md | 生產資料介接 API 規格 | None | 0.4778 |
| 4 | mes_operation_sop.md | 工單狀態流程 | None | 0.4674 |
| 5 | equipment_maintenance_manual.pdf | 異常保養 | 3 | 0.4415 |

### C10 [confusable] — E102 和其他設備警報的處理方式有何不同？

- expected：equipment_alarm_manual.md / E102 感測器通訊異常 / page=None；should_answer=True；latency=28ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.7251 |
| 2 | equipment_alarm_manual.md | E102 感測器通訊異常 | None | 0.6299 |
| 3 | confusable_alarm_d900.md | 安全門警報處理手冊（D900） | None | 0.6237 |
| 4 | equipment_alarm_manual.md | 警報代碼對照表 | None | 0.6166 |
| 5 | equipment_alarm_manual.md | E203 溫度過高警報 | None | 0.6087 |

### C11 [confusable] — 通訊模組指示燈閃紅燈時屬於哪一種警報，應如何處理？

- expected：equipment_alarm_manual.md / E102 感測器通訊異常 / page=None；should_answer=True；latency=17ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_maintenance_manual.pdf | 通訊模組檢查 | 3 | 0.771 |
| 2 | equipment_alarm_manual.md | E102 感測器通訊異常 | None | 0.7019 |
| 3 | equipment_alarm_manual.md | 警報代碼對照表 | None | 0.6301 |
| 4 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.6227 |
| 5 | equipment_alarm_manual.md | E101 電源模組異常 | None | 0.6225 |

### C12 [unanswerable] — CX-500 的市場售價是多少？

- expected：None / None / page=None；should_answer=False；latency=29ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | equipment_maintenance_manual.pdf | 設備基本資訊 | 1 | 0.4986 |
| 2 | equipment_maintenance_manual.pdf | 定期保養週期 | 2 | 0.4301 |
| 3 | manufacturing_data_dictionary.txt | None | None | 0.3626 |
| 4 | confusable_alarm_d900.md | 與 E102 的差異 | None | 0.3614 |
| 5 | equipment_alarm_manual.md | E203 溫度過高警報 | None | 0.3593 |

### C13 [unanswerable] — 台灣的首都是哪一個城市？

- expected：None / None / page=None；should_answer=False；latency=31ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | confusable_alarm_d900.md | 警報代碼對照表 | None | 0.3054 |
| 2 | api_integration_spec.md | 上報工單進度請求欄位 | None | 0.2898 |
| 3 | manufacturing_data_dictionary.txt | None | None | 0.2883 |
| 4 | mes_operation_sop.md | 狀態轉換條件 | None | 0.2766 |
| 5 | mes_operation_sop.md | 工單啟動 | None | 0.2697 |

### C14 [unanswerable] — 產線作業員的月薪是多少？

- expected：None / None / page=None；should_answer=False；latency=34ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | api_integration_spec.md | 上報工單進度請求欄位 | None | 0.4523 |
| 2 | manufacturing_data_dictionary.txt | None | None | 0.4503 |
| 3 | mes_operation_sop.md | 工單啟動 | None | 0.4444 |
| 4 | equipment_maintenance_manual.pdf | 異常保養 | 3 | 0.4173 |
| 5 | mes_operation_sop.md | 工單狀態流程 | None | 0.4171 |

### C15 [unanswerable] — 公司股票今天的收盤價是多少？

- expected：None / None / page=None；should_answer=False；latency=28ms

| rank | document | section | page | score |
| ---: | --- | --- | ---: | ---: |
| 1 | api_integration_spec.md | 上報工單進度請求欄位 | None | 0.3595 |
| 2 | manufacturing_data_dictionary.txt | None | None | 0.3373 |
| 3 | equipment_maintenance_manual.pdf | 定期保養週期 | 2 | 0.331 |
| 4 | mes_operation_sop.md | 工單啟動 | None | 0.3302 |
| 5 | equipment_maintenance_manual.pdf | 通訊模組檢查 | 3 | 0.3299 |

## 硬限制門檻建議（由 Baseline 推導）

> 依 Slice 0 實際 Baseline 推導的**建議值**，僅供寫入 Baseline 報告 / Slice 0 設定草案，**不得直接修改正式 Retrieval Config**，也**不宣稱適用所有資料集**。

| 項目 | 建議值 | 推導方式 | Baseline 依據 | 可信度 |
| --- | --- | --- | --- | --- |
| Exact-code accuracy 下限 | 0.75 | baseline exact_code Top-1 accuracy（no-regression floor：候選不得低於目前值） | 3/4 exact_code 案例 Top-1 命中 = 0.75 | low (n=4 < 15) |
| Unanswerable FP rate 上限 | 0.25 | baseline unanswerable FP rate @ threshold=0.55；baseline 為 0 時採單一 case 粒度 1/n 作為可觀測上限 | 0/4 無關題 top1_score ≥ 0.55（rate=0.0；粒度 1/n=0.25） | low (n=4 < 15) |
| P95 latency 上限 (ms) | 86 | ceil(max(baseline_p95 × 1.5, baseline_p95 + 50ms)) | baseline P95 = 35.8 ms（單機、單次量測） | low (single-run, hardware-dependent) |
| Semantic Recall@5 退化容忍 ε | 0.2 | ε = 1 / semantic_recall@5_considered（一個 case 翻轉的最小可觀測退化） | semantic Recall@5 = 1.0（n=5）；候選 Recall@5 應 ≥ baseline − ε | low (n=5 < 15) |

## 已知限制與尚不可計算的未來 metrics

- Case 標註為 Slice 0 Baseline 格式（expected_document / expected_section / expected_page / should_answer）；Slice 1 才遷移至 Canonical Evidence Span。
- Evidence Coverage、Graded NDCG、Evidence-level Precision@5、Canonical Evidence overlap 於 Slice 0 尚無可靠標註，未計算（留給 Slice 1／2）。
- 分數與排名為本次真實 BGE 執行結果，僅供 Baseline 記錄，不作為固定期望值斷言。
- 硬限制建議樣本數偏低（尤其 exact_code / confusable / semantic），跨資料集不可直接沿用。
- 未計算（留待 Slice 1／2）：Evidence Coverage、Graded NDCG、Evidence-level Precision@5、Canonical Evidence overlap。
