# factory-rag-assistant

## 智慧製造文件 RAG 知識助理

### 專案需求與技術規格書

---

# 版本狀態（Version Status）

> 本節為後續版本的實作狀態摘要；以下第 0 節起為原始 v1.0 規格書，內容保留不變。

- **v1.0 — RAG MVP**：已交付（tag `v1.0-rag-mvp`）。
- **v1.1 — Human-in-the-loop Retrieval Optimizer**：已交付（tag `v1.1-retrieval-optimizer`）。
- **v1.2 — LLM-guided Retrieval Optimization Agent**：**implemented and runtime-verified**，
  已 merge 至 `main`（merge commit `aec93b7`）、tag `v1.2-llm-agent-mvp`，GitHub Release 已發布。

## v1.2 狀態

**Completed**

- bounded LLM agent loop（有界的 rounds／LLM calls）
- controlled context（受控原文 context；validation 對 LLM 只給 coarse gate）
- strict proposal validation（whitelist-only、界限、probe 綁定與 anchor 存在性）
- candidate trial（隔離 `optimizer_` collection 的真實試驗）
- parent/candidate probe comparison
- deterministic promotion（重用 v1.1 constraint + recommendation ordering；LLM 不參與勝負）
- minimal audit（匿名化，不存 secret／完整 prompt／文件全文）
- real-provider verification（一次 OpenAI-compatible Provider 單輪端到端驗收）

**Deferred**

- held-out recommendation lock（held-out test 與 recommendation eligibility 仍為 `not_evaluated`）
- full persistence / resume
- reporting UI
- dataset expansion loop
- adaptive multi-proposal search

詳見 [docs/v1.2-runtime-acceptance.md](docs/v1.2-runtime-acceptance.md) 與 README 的
「v1.2 — LLM-guided Retrieval Optimization Agent」章節。

---

# 0. 給 Claude CLI 的任務指示

請先完整閱讀本規格書，分析需求、依賴關係、技術風險與開發順序，然後產生一份可實際執行的開發計畫。

本階段請遵守以下規則：

1. 只產生執行計畫，不要直接建立或修改程式碼。
2. 不要擴充本規格未要求的功能。
3. 不要導入 LangChain、LlamaIndex、Agent Framework 或大型前端框架。
4. 優先考慮兩天內完成可展示的 MVP。
5. 執行計畫必須拆分成可驗收的小步驟。
6. 每個步驟需包含：

   * 目標
   * 預計修改或建立的檔案
   * 核心實作內容
   * 相依步驟
   * 驗收方式
   * 可能風險
7. 請標記：

   * P0：MVP 必須完成
   * P1：時間允許才完成
   * Future Work：本次不實作
8. 若規格存在技術衝突，請在執行計畫前先列出，但不要自行改變專案目標。
9. 不要假設已有任何葛拉漢面試專案的程式碼可供使用。
10. 專案必須能獨立建立、啟動、測試與展示。

---

# 1. 專案基本資訊

## 1.1 專案名稱

```text
factory-rag-assistant
```

## 1.2 中文名稱

智慧製造文件 RAG 知識助理

## 1.3 專案性質

本專案為獨立作品集專案，用於展示企業文件 RAG 系統的完整實作能力。

本專案不得：

* 修改既有葛拉漢面試專案。
* 引用既有葛拉漢面試專案程式碼。
* 依賴既有葛拉漢面試專案服務。
* 使用真實公司機密或未授權文件。
* 將其他專案的資料庫、API 或環境視為必要條件。

所有範例資料均使用自行建立的虛構智慧製造文件。

---

# 2. 專案目標

建立一套可以獨立執行的企業文件 RAG 系統。

使用者上傳 PDF、Markdown 或 TXT 文件後，系統必須能夠：

1. 驗證文件格式與大小。
2. 計算文件 SHA-256。
3. 辨識完全相同的重複文件。
4. 解析文件內容。
5. 保留文件頁碼、標題與段落資訊。
6. 根據段落與句子切分 Chunk。
7. 使用本地模型產生 Embedding。
8. 將向量及 metadata 儲存至 Qdrant。
9. 保存文件層級的管理資料。
10. 根據使用者問題進行 Top-K 相似度檢索。
11. 根據設定門檻過濾低相關內容。
12. 將合格 Context 傳送給 OpenAI-compatible LLM。
13. 產生只依據文件內容的回答。
14. 在回答中標註具體引用來源。
15. 回傳檔名、頁碼、章節、Chunk 編號、分數與原始內容。
16. 文件資訊不足時直接拒答。
17. 提供基本測試與檢索評估。
18. 使用 Docker Compose 啟動 FastAPI 與 Qdrant。

---

# 3. 專案定位

本專案不是單純的 ChatGPT API 聊天介面。

專案應展示以下能力：

* Python 後端開發
* FastAPI API 設計
* 文件上傳與驗證
* PDF、Markdown、TXT 文件解析
* 結構化資料清洗
* Chunking 策略
* Token 長度控制
* 本地 Embedding
* Qdrant 向量資料庫
* Metadata 與文件生命週期管理
* 相似度檢索
* Threshold 拒答
* RAG Prompt 設計
* 引用來源追蹤
* LLM 輸出引用驗證
* Docker Compose 部署
* 錯誤處理與回滾
* 單元測試與整合測試
* Hit@K 與延遲評估
* 可重現的 README

---

# 4. 專案情境

系統模擬智慧製造企業內部知識庫。

知識庫可能包含：

1. 設備警報排除手冊
2. MES 系統操作 SOP
3. 品質檢測流程
4. API 串接規格
5. 製程資料欄位說明
6. 設備保養規範
7. 常見問題文件

使用者可能詢問：

* 設備出現 E102 警報時，應該先檢查什麼？
* MES 工單如何從 WAITING 切換為 RUNNING？
* AOI 檢測結果為 NG 時，需要記錄哪些欄位？
* API Token 的有效時間是多少？
* 設備定期保養的週期為多久？
* 文件中不存在的員工薪資、供應商價格或其他無關問題。

---

# 5. MVP 範圍

## 5.1 P0 必須完成

* FastAPI 後端
* Qdrant 向量資料庫
* SQLite 文件 Registry
* PDF、Markdown、TXT 上傳
* 文件格式與大小驗證
* SHA-256 重複檢查
* 文件解析
* 結構化 Chunking
* Token hard limit
* 本地 Embedding
* Qdrant Collection 自動建立
* 文件 indexing
* 文件列表
* 文件刪除
* Top-K Retrieval
* Retrieval-only API
* RAG 問答 API
* 後端直接拒答
* 來源引用
* 簡易靜態前端
* 基礎測試
* Retrieval 評估
* Docker Compose
* README
* 五份以上虛構範例文件
* 至少一份真正的 PDF 範例

## 5.2 P1 時間允許才完成

* 更完整的前端視覺樣式
* 批次 Embedding 進度顯示
* 更詳細的 readiness 診斷
* 回答中未使用來源與全部檢索結果分開顯示
* 額外測試邊界案例
* 自動產生評估報告 Markdown
* 額外製造領域範例文件

## 5.3 本次不實作

* OCR
* 掃描型 PDF 辨識
* 使用者登入
* 權限管理
* 多租戶
* Kubernetes
* 大型前端框架
* React、Vue、Angular
* 混合搜尋
* BM25
* Reranker
* RAGAS
* LLM Judge
* LoRA
* Fine-tuning
* 多 Agent
* 長期記憶
* 自動網頁搜尋
* 任意 SQL
* 任意 Shell
* 正式雲端部署
* 背景工作佇列
* Celery
* Redis
* 多向量資料庫支援

---

# 6. 技術棧

## 6.1 後端

* Python 3.11 或 Python 3.12
* FastAPI
* Uvicorn
* Pydantic
* Pydantic Settings
* python-multipart
* PyMuPDF
* sentence-transformers
* transformers tokenizer
* Qdrant Client
* SQLite
* OpenAI Python SDK
* httpx
* pytest

## 6.2 Embedding

預設模型：

```text
BAAI/bge-small-zh-v1.5
```

模型名稱必須由環境變數設定：

```env
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

Embedding Service 必須：

* 在應用程式啟動時載入一次。
* 不得每次 Request 重新載入模型。
* 使用 normalized embedding。
* 從模型動態取得向量維度。
* 對 Query 使用可設定的 query instruction。
* 對文件 Chunk 不加入 query instruction。
* 在 embedding 前檢查 token 數。
* 不可依賴默默 truncation。

預設 Query instruction：

```text
为这个句子生成表示以用于检索相关文章：
```

環境變數：

```env
EMBEDDING_QUERY_INSTRUCTION=为这个句子生成表示以用于检索相关文章：
EMBEDDING_MAX_TOKENS=480
```

## 6.3 LLM

使用 OpenAI-compatible API。

必要環境變數：

```env
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
```

程式碼不得寫死特定供應商。

系統應能連接：

* OpenAI API
* 本地 vLLM
* 其他支援 OpenAI Chat Completions 格式的服務

LLM 並不包含在本次 Docker Compose 服務內。

README 必須明確說明需要外部 LLM endpoint。

## 6.4 部署

Docker Compose 至少啟動：

* FastAPI
* Qdrant

Embedding 模型與 FastAPI 放在同一 Container。

必須保存：

* Qdrant 資料
* Hugging Face 模型快取
* SQLite metadata
* 上傳文件

---

# 7. 建議專案目錄

> 本目錄為 v1.0 原始草案，**已過時**（未包含 `evaluation/v11`、`evaluation/v12`、`docs/` 等後續新增）。
> 目前實際結構見 README 的「專案結構」章節。

```text
factory-rag-assistant/
├── app/
│   ├── main.py
│   ├── dependencies.py
│   │
│   ├── api/
│   │   ├── health.py
│   │   ├── documents.py
│   │   ├── retrieval.py
│   │   └── query.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── constants.py
│   │
│   ├── schemas/
│   │   ├── common.py
│   │   ├── document.py
│   │   ├── retrieval.py
│   │   └── query.py
│   │
│   ├── models/
│   │   ├── document_element.py
│   │   └── chunk.py
│   │
│   ├── repositories/
│   │   └── document_repository.py
│   │
│   ├── services/
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   ├── indexing_service.py
│   │   ├── retrieval_service.py
│   │   ├── llm_client.py
│   │   └── rag_service.py
│   │
│   ├── prompts/
│   │   └── rag_prompt.py
│   │
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
│
├── sample_docs/
│   ├── equipment_alarm_manual.md
│   ├── mes_operation_sop.md
│   ├── quality_inspection_guideline.md
│   ├── api_integration_spec.md
│   ├── manufacturing_data_dictionary.md
│   └── equipment_maintenance_manual.pdf
│
├── evaluation/
│   ├── questions.json
│   └── evaluate_retrieval.py
│
├── tests/
│   ├── fakes.py
│   ├── fixtures/
│   ├── unit/
│   │   ├── test_text_splitter.py
│   │   ├── test_document_loader.py
│   │   ├── test_indexing_service.py
│   │   ├── test_retrieval_service.py
│   │   └── test_query_api.py
│   └── integration/
│       ├── test_qdrant_indexing.py
│       └── test_retrieval.py
│
├── storage/
│   ├── uploads/
│   └── metadata.db
│
├── scripts/
│   └── generate_sample_pdf.py
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

實際實作時可微調目錄，但不得將大量商業邏輯直接放入 API Router。

---

# 8. 系統架構

```text
Browser
   │
   ▼
FastAPI Static Frontend
   │
   ├── Document API
   ├── Retrieval API
   ├── Query API
   └── Health API
          │
          ▼
   Application Services
   ├── Indexing Service
   ├── Retrieval Service
   └── RAG Service
          │
          ├── Document Loader
          ├── Text Splitter
          ├── Embedding Service
          ├── Qdrant Vector Store
          ├── SQLite Document Repository
          └── OpenAI-compatible LLM
```

---

# 9. 核心資料模型

## 9.1 DocumentElement

Loader 不得只回傳整份文件字串。

應先解析為結構化 Element。

```python
class DocumentElement:
    text: str
    page: int | None
    section: str | None
    element_type: str
```

`element_type` 可包含：

* heading
* paragraph
* table
* code
* list

## 9.2 Chunk

```python
class Chunk:
    id: str
    document_id: str
    filename: str
    file_hash: str
    page: int | None
    section: str | None
    chunk_index: int
    text: str
    created_at: str
```

## 9.3 Qdrant Payload

每個 Point Payload 至少包含：

```json
{
  "document_id": "uuid",
  "filename": "mes_operation_sop.md",
  "file_hash": "sha256",
  "page": 3,
  "section": "工單啟動流程",
  "chunk_index": 12,
  "text": "原始文字內容",
  "created_at": "ISO datetime"
}
```

Markdown 或 TXT 沒有頁碼時：

```json
{
  "page": null
}
```

PDF 頁碼一律使用 1-based。

---

# 10. SQLite 文件 Registry

Qdrant 負責 Chunk 向量與 Payload。

SQLite 負責文件層級資訊與文件生命週期。

## 10.1 documents 資料表

```text
documents
- document_id TEXT PRIMARY KEY
- filename TEXT NOT NULL
- file_hash TEXT NOT NULL UNIQUE
- content_type TEXT NOT NULL
- file_size INTEGER NOT NULL
- chunk_count INTEGER NOT NULL DEFAULT 0
- status TEXT NOT NULL
- created_at TEXT NOT NULL
- error_message TEXT NULL
```

## 10.2 文件狀態

```text
indexing
indexed
failed
deleting
```

## 10.3 重複文件定義

SHA-256 必須根據原始檔案 bytes 計算。

相同 bytes 視為重複文件。

相同文字但不同格式，例如 PDF 與 Markdown，不必視為重複。

重複文件上傳應回傳：

```http
409 Conflict
```

並提供既有 `document_id`。

---

# 11. 文件解析規則

## 11.1 支援格式

* `.pdf`
* `.md`
* `.markdown`
* `.txt`

## 11.2 PDF

使用 PyMuPDF。

規則：

* 只支援可擷取文字的 PDF。
* 每頁分別解析。
* Chunk 原則上不得跨頁。
* 頁碼從 1 開始。
* 無法擷取文字的掃描型 PDF 回傳明確錯誤。
* 本次不實作 OCR。

## 11.3 Markdown

規則：

* 辨識標題層級。
* 追蹤目前所在 section。
* 依標題與空行分段。
* 表格在未超過硬限制時盡量保持完整。
* 程式碼區塊在未超過硬限制時盡量保持完整。

## 11.4 TXT

規則：

* 依空行分段。
* 沒有 section 時可設為 null。
* 忽略完全空白段落。

## 11.5 空白文件

解析後沒有有效文字內容時，拒絕匯入。

回傳：

```http
400 Bad Request
```

---

# 12. Chunking 規格

## 12.1 初始參數

```env
CHUNK_SIZE_CHARS=600
CHUNK_OVERLAP_CHARS=100
EMBEDDING_MAX_TOKENS=480
```

## 12.2 切分優先順序

1. 文件頁面
2. Markdown 標題
3. 空行
4. 段落
5. 句號或其他句尾符號
6. 字元長度
7. Token hard limit

## 12.3 規則

* 不要優先從句子中間截斷。
* `CHUNK_SIZE_CHARS` 是軟目標。
* `EMBEDDING_MAX_TOKENS` 是硬限制。
* 每個 Chunk 在 embedding 前必須經 tokenizer 驗證。
* 不允許依靠模型靜默截斷。
* Chunk 不得為空白。
* Chunk 必須保留來源 metadata。
* `chunk_index` 在單一文件內從 0 開始且保持穩定。
* overlap 不得產生完全重複或無意義 Chunk。
* 單一超長句仍需按照安全長度拆分。

---

# 13. Qdrant 規格

## 13.1 Collection

```text
factory_knowledge
```

Collection 名稱由環境變數設定：

```env
QDRANT_COLLECTION=factory_knowledge
```

## 13.2 距離函數

```text
Cosine
```

## 13.3 向量維度

不得完全寫死。

啟動時使用：

```python
model.get_sentence_embedding_dimension()
```

取得實際向量維度。

## 13.4 啟動檢查

應用程式啟動時：

1. 檢查 Collection 是否存在。
2. 不存在則自動建立。
3. 已存在則檢查：

   * 距離函數
   * 向量維度
4. 不相容時：

   * 不得默默重建或刪除資料。
   * `/ready` 回傳 503。
   * 回傳明確錯誤訊息。
5. README 必須說明，更換 Embedding 模型後需要重新 indexing。

## 13.5 Payload Index

至少為以下欄位建立 keyword index：

```text
document_id
file_hash
```

用途：

* 文件刪除
* 文件 Chunk 查詢
* 回滾
* 文件過濾

---

# 14. 文件匯入流程

```text
Upload File
→ 驗證副檔名
→ 驗證檔案大小
→ 讀取原始 bytes
→ 計算 SHA-256
→ SQLite 查詢是否重複
→ 建立 documents 記錄，status=indexing
→ 保存原始檔案
→ Loader 解析 DocumentElement
→ 驗證非空白文件
→ Splitter 建立 Chunk
→ 驗證 token hard limit
→ 批次產生 Embedding
→ 寫入 Qdrant
→ 更新 SQLite chunk_count
→ 更新 status=indexed
→ 回傳文件資訊
```

## 14.1 匯入失敗回滾

若解析、Chunking、Embedding 或 Qdrant Upsert 失敗：

1. 刪除已寫入的該 `document_id` Qdrant Points。
2. 更新 SQLite：

   * status=`failed`
   * error_message=錯誤摘要
3. 視情況保留或刪除原始檔案。
4. API 回傳明確錯誤。
5. 不得讓文件顯示為成功 indexed。

---

# 15. 文件刪除流程

```text
DELETE Request
→ 確認 document_id 存在
→ status=deleting
→ 根據 document_id 刪除 Qdrant Points
→ 刪除原始上傳檔案
→ 刪除 SQLite 文件記錄
→ 回傳 204
```

若 Qdrant 刪除失敗：

* 不得先刪除 SQLite 記錄。
* 保留可重試狀態。
* 回傳 503 或適當錯誤。

---

# 16. Retrieval 流程

```text
User Question
→ 驗證 question
→ 加入 Query Instruction
→ 產生 Query Embedding
→ Qdrant Top-K Search
→ Score Threshold 過濾
→ 回傳合格結果
```

## 16.1 Top-K

環境變數：

```env
RETRIEVAL_DEFAULT_TOP_K=5
RETRIEVAL_MAX_TOP_K=10
```

API 輸入限制：

```text
1 <= top_k <= 10
```

## 16.2 Score Threshold

環境變數：

```env
RETRIEVAL_SCORE_THRESHOLD=0.0
```

初始開發階段可設為 0.0，以收集分數分布。

最終門檻必須根據 evaluation 測試集校準，不可任意假設 0.7 或 0.8。

## 16.3 無合格結果

Threshold 過濾後沒有 Chunk 時：

* 後端直接回傳固定 fallback。
* 不呼叫 LLM。
* `sources=[]`
* `generation_time_ms=0`

固定 fallback：

```text
目前提供的文件中沒有足夠資訊可以回答此問題。
```

拒答是應用程式行為，不可完全交由 Prompt 或 LLM 決定。

---

# 17. RAG 問答流程

```text
User Question
→ Retrieval Service
→ 合格 Chunk
→ 分配 Source ID
→ 組合 Context
→ 呼叫 LLM
→ 驗證答案中的 Citation
→ 後端組裝 Sources
→ 回傳 Answer 與 Sources
```

## 17.1 Source ID

檢索結果由後端分配：

```text
S1
S2
S3
```

傳給 LLM 的 Context 範例：

```text
[S1]
filename: equipment_alarm_manual.md
page: null
section: E102 感測器通訊異常
chunk_index: 4
text:
E102 表示感測器通訊異常……

[S2]
filename: equipment_maintenance_manual.pdf
page: 3
section: 通訊模組檢查
chunk_index: 8
text:
……
```

## 17.2 Metadata 可信來源

以下欄位必須由後端檢索結果建立，不得信任 LLM 生成：

* filename
* page
* section
* chunk_index
* score
* text
* document_id

LLM 只負責：

* 產生自然語言答案。
* 在答案主要結論後標註 `[S1]`、`[S2]`。

## 17.3 Citation 驗證

系統應解析答案中的：

```regex
\[S(\d+)\]
```

只允許引用實際存在的 Source ID。

若 LLM 引用了不存在的來源：

* 移除無效來源或重新處理。
* 不得回傳捏造的 metadata。
* 記錄 warning log。

## 17.4 Sources 定義

`response.sources` 應優先包含答案實際引用的來源。

若實作 `retrieved_sources`，可額外回傳所有檢索結果；但不是 P0 必須項目。

---

# 18. RAG Prompt 規格

System Prompt 必須包含：

1. 只能根據提供的 Context 回答。
2. 不得將模型自身知識當成文件內容。
3. 每個主要結論都要附來源，例如 `[S1]`。
4. 不得捏造頁碼、文件名稱、章節或來源。
5. 資訊不足時回答：
   `目前提供的文件中沒有足夠資訊可以回答此問題。`
6. 若提供的 Context 中出現矛盾，必須指出矛盾。
7. 不得自行選擇其中一份文件為正確答案。
8. 可整合多個來源，但必須標註各自來源。
9. 回答應使用繁體中文，除非問題明確要求其他語言。

Prompt 結構：

```text
System Instruction
+
Retrieved Context
+
User Question
```

系統只能保證辨識「已檢索 Context 中的矛盾」，不能保證發現知識庫內所有未被檢索的矛盾。

---

# 19. API 規格

API prefix：

```text
/api
```

---

## 19.1 Health Check

```http
GET /health
```

Response：

```json
{
  "status": "ok"
}
```

用途：

* 確認 FastAPI process 正常。
* 不進行外部依賴完整檢查。

Status：

```http
200 OK
```

---

## 19.2 Readiness Check

```http
GET /ready
```

檢查：

* FastAPI 已完成初始化
* Qdrant 可連線
* Collection 相容
* Embedding 模型已載入
* SQLite 可讀寫
* LLM 必要設定存在

成功：

```json
{
  "status": "ready",
  "checks": {
    "qdrant": "ok",
    "collection": "ok",
    "embedding": "loaded",
    "metadata_db": "ok",
    "llm_config": "ok"
  }
}
```

失敗：

```http
503 Service Unavailable
```

```json
{
  "status": "not_ready",
  "checks": {
    "qdrant": "error",
    "collection": "unknown",
    "embedding": "loaded",
    "metadata_db": "ok",
    "llm_config": "ok"
  }
}
```

---

## 19.3 上傳文件

```http
POST /api/documents
Content-Type: multipart/form-data
```

欄位：

```text
file
```

允許：

* PDF
* Markdown
* TXT

成功：

```http
201 Created
```

```json
{
  "document_id": "uuid",
  "filename": "mes_operation_sop.md",
  "chunk_count": 15,
  "status": "indexed",
  "created_at": "ISO datetime"
}
```

錯誤：

* 不支援格式：`415 Unsupported Media Type`
* 檔案過大：`413 Payload Too Large`
* 重複文件：`409 Conflict`
* 空白文件：`400 Bad Request`
* 文件解析失敗：`400 Bad Request`
* Qdrant 或服務不可用：`503 Service Unavailable`

---

## 19.4 文件列表

```http
GET /api/documents
```

Response：

```json
{
  "documents": [
    {
      "document_id": "uuid",
      "filename": "mes_operation_sop.md",
      "content_type": "text/markdown",
      "file_size": 8420,
      "chunk_count": 15,
      "status": "indexed",
      "created_at": "ISO datetime"
    }
  ],
  "total": 1
}
```

---

## 19.5 刪除文件

```http
DELETE /api/documents/{document_id}
```

成功：

```http
204 No Content
```

文件不存在：

```http
404 Not Found
```

---

## 19.6 Retrieval-only API

```http
POST /api/retrieve
```

Request：

```json
{
  "question": "E102 警報代表什麼？",
  "top_k": 5
}
```

Response：

```json
{
  "results": [
    {
      "source_id": "S1",
      "document_id": "uuid",
      "filename": "equipment_alarm_manual.md",
      "page": null,
      "section": "E102 感測器通訊異常",
      "chunk_index": 4,
      "score": 0.87,
      "text": "E102 表示感測器通訊異常……"
    }
  ],
  "retrieval_time_ms": 52
}
```

用途：

* 單獨驗證檢索。
* 不呼叫 LLM。
* 提供 evaluation 使用。
* 在 LLM 不可用時仍可檢查系統前半段。

---

## 19.7 RAG 問答

```http
POST /api/query
```

Request：

```json
{
  "question": "設備發生 E102 警報時應如何處理？",
  "top_k": 5
}
```

成功：

```json
{
  "answer": "依據設備警報手冊，應先檢查感測器連線狀態與通訊線材。[S1]",
  "sources": [
    {
      "source_id": "S1",
      "document_id": "uuid",
      "filename": "equipment_alarm_manual.md",
      "page": null,
      "section": "E102 感測器通訊異常",
      "chunk_index": 4,
      "score": 0.87,
      "text": "E102 表示感測器通訊異常……"
    }
  ],
  "retrieval_time_ms": 52,
  "generation_time_ms": 1210
}
```

拒答：

```json
{
  "answer": "目前提供的文件中沒有足夠資訊可以回答此問題。",
  "sources": [],
  "retrieval_time_ms": 41,
  "generation_time_ms": 0
}
```

Request 驗證：

* question 不可為空白。
* question 最大 2000 字元。
* top_k 範圍 1 至 10。

缺少 question：

```http
422 Unprocessable Entity
```

---

# 20. 檔案限制

環境變數：

```env
MAX_UPLOAD_SIZE_MB=10
```

至少驗證：

* 副檔名
* MIME type
* 檔案大小
* 檔案是否為空
* 解析後是否有文字

不得只依賴前端驗證。

---

# 21. FastAPI 生命週期與依賴管理

使用 FastAPI lifespan 初始化共用資源。

啟動時初始化：

* Settings
* SQLite connection 或 repository
* Qdrant Client
* Embedding Model
* LLM Client
* Collection 檢查

服務不得在每次 Request 建立 Embedding 模型。

應使用 dependency injection，讓測試可以替換：

* Embedding Service
* Vector Store
* LLM Client
* Document Repository

---

# 22. 錯誤處理

需定義清楚的 application exceptions，例如：

* UnsupportedFileTypeError
* FileTooLargeError
* EmptyDocumentError
* DuplicateDocumentError
* DocumentParseError
* EmbeddingError
* VectorStoreError
* CollectionMismatchError
* LLMConfigurationError
* LLMServiceError
* DocumentNotFoundError

API 應將例外轉為一致格式：

```json
{
  "error": {
    "code": "DUPLICATE_DOCUMENT",
    "message": "此文件已經上傳。",
    "details": {
      "document_id": "existing-uuid"
    }
  }
}
```

不得直接將完整 Python traceback 回傳給前端。

---

# 23. Logging

至少記錄：

* Request method 與 path
* 文件上傳開始與完成
* document_id
* filename
* file_hash 前數碼
* Chunk 數量
* Embedding 耗時
* Qdrant Upsert 耗時
* Retrieval 耗時
* LLM generation 耗時
* 回滾事件
* 無效 Citation
* 外部服務錯誤

不得記錄：

* LLM API Key
* 完整敏感環境變數
* 未經遮蔽的秘密資訊

---

# 24. 簡易前端

使用：

* FastAPI StaticFiles
* HTML
* CSS
* Vanilla JavaScript

不使用大型前端框架。

## 24.1 畫面區域

### 文件上傳區

顯示：

* 檔案選擇
* 上傳按鈕
* 支援格式
* 最大檔案大小
* 上傳狀態
* 錯誤訊息

### 文件列表區

顯示：

* 文件名稱
* 狀態
* Chunk 數
* 建立時間
* 刪除按鈕

### 問答區

顯示：

* 問題輸入
* Top-K
* 查詢按鈕
* Loading 狀態
* 回答
* 引用來源
* 相似度分數
* 頁碼或章節
* 原始 Chunk
* Retrieval 耗時
* Generation 耗時

## 24.2 前端限制

前端只負責展示。

不得將關鍵驗證、Threshold、拒答或引用組裝邏輯放在 JavaScript。

---

# 25. Docker 規格

## 25.1 服務

```text
api
qdrant
```

## 25.2 Volumes

至少保存：

```text
qdrant_data
hf_cache
app_storage
```

用途：

* `qdrant_data`：Qdrant 持久化
* `hf_cache`：Embedding 模型快取
* `app_storage`：SQLite 與上傳文件

## 25.3 啟動

基本指令：

```bash
cp .env.example .env
docker compose up --build
```

README 必須說明：

* 第一次啟動會下載 Embedding 模型。
* 第一次啟動需要網路。
* 模型下載後會使用 volume cache。
* LLM endpoint 需另外提供。
* 若使用本機 vLLM，Container 可能需透過 `host.docker.internal` 連線。

---

# 26. 環境變數

`.env.example` 至少包含：

```env
APP_NAME=factory-rag-assistant
APP_ENV=development
LOG_LEVEL=INFO

HOST=0.0.0.0
PORT=8000

MAX_UPLOAD_SIZE_MB=10
UPLOAD_DIR=/app/storage/uploads
SQLITE_PATH=/app/storage/metadata.db

CHUNK_SIZE_CHARS=600
CHUNK_OVERLAP_CHARS=100
EMBEDDING_MAX_TOKENS=480

EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_QUERY_INSTRUCTION=为这个句子生成表示以用于检索相关文章：

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=factory_knowledge

RETRIEVAL_DEFAULT_TOP_K=5
RETRIEVAL_MAX_TOP_K=10
RETRIEVAL_SCORE_THRESHOLD=0.0

LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=60
```

不得提交真實 `.env` 或 API Key。

---

# 27. 範例文件

`sample_docs` 至少包含五份自行撰寫的虛構文件，以及一份真正的 PDF。

每份文件頂端標註：

```text
本文件為作品集示範所建立的虛構資料，不代表任何真實公司或設備。
```

內容需包含：

* 可精確問答的規則
* 不同設備代碼
* 流程步驟
* 狀態轉換
* 表格
* API 欄位
* 資料字典
* 容易混淆的相似資訊
* 不同但相近的保養週期
* 文件之間可能有條件差異
* 明確不存在的問題

建議文件：

1. `equipment_alarm_manual.md`
2. `mes_operation_sop.md`
3. `quality_inspection_guideline.md`
4. `api_integration_spec.md`
5. `manufacturing_data_dictionary.md`
6. `equipment_maintenance_manual.pdf`

PDF 可由專案內腳本根據虛構 Markdown 或文字內容產生，確保 Repository 可重現。

---

# 28. 測試策略

測試分為：

* Unit Test
* Integration Test

不得讓所有 API 測試都依賴真實 Embedding 模型、真實 Qdrant 與真實 LLM。

## 28.1 Fake Components

Unit Test 使用：

* FakeEmbeddingService
* FakeVectorStore
* FakeLLMClient
* Temporary SQLite
* Temporary upload directory

## 28.2 Unit Test

至少測試：

1. Markdown Loader 可解析。
2. TXT Loader 可解析。
3. 文字型 PDF Loader 可解析。
4. 掃描型或無文字 PDF 被拒絕。
5. 不支援副檔名被拒絕。
6. 空白文件被拒絕。
7. SHA-256 重複文件可辨識。
8. Chunk 保留 filename。
9. Chunk 保留 page。
10. Chunk 保留 section。
11. Chunk 保留 chunk_index。
12. Chunk 不超過 token hard limit。
13. Chunk 不產生空內容。
14. indexing 成功時更新 status。
15. indexing 失敗時執行回滾。
16. Retrieval threshold 能過濾低分結果。
17. 無結果時不呼叫 Fake LLM。
18. 回答包含 sources。
19. 無效 Citation 不會產生捏造來源。
20. `/health` 回傳 200。
21. 缺少 question 回傳 422。
22. 不支援格式回傳 415。
23. 重複文件回傳 409。

## 28.3 Integration Test

使用真實：

* sentence-transformers 模型
* Qdrant
* 固定測試文件

至少測試：

1. 文件可實際寫入 Qdrant。
2. Qdrant Point 數量符合 Chunk 數。
3. 根據 document_id 可刪除所有 Points。
4. 固定問題的正確文件出現在 Top-3。
5. Collection 維度不相容時 readiness 失敗。

Integration Test 不必呼叫付費 LLM。

## 28.4 pytest 標記

```python
@pytest.mark.integration
```

執行方式：

```bash
pytest -m "not integration"
pytest -m integration
```

---

# 29. Retrieval 評估

建立：

```text
evaluation/questions.json
```

## 29.1 格式

```json
[
  {
    "question": "E102 警報代表什麼？",
    "expected_document": "equipment_alarm_manual.md",
    "expected_section": "E102 感測器通訊異常",
    "should_answer": true
  },
  {
    "question": "MES 工單啟動前需要檢查哪些欄位？",
    "expected_document": "mes_operation_sop.md",
    "expected_section": "工單啟動",
    "should_answer": true
  },
  {
    "question": "公司今年員工旅遊的目的地是哪裡？",
    "expected_document": null,
    "expected_section": null,
    "should_answer": false
  }
]
```

## 29.2 指標

至少計算：

* Hit@1
* Hit@3
* 平均 Retrieval latency
* P50 Retrieval latency
* 無關問題拒答率
* 正確問題誤拒率
* 各問題 Top-1 score
* 無關問題最高 score

## 29.3 Threshold 校準

評估腳本需要輸出分數分布，協助決定：

```env
RETRIEVAL_SCORE_THRESHOLD
```

不得預先捏造評估結果。

README 中的數字只能來自實際評估輸出。

## 29.4 評估範圍

評估直接呼叫 Retrieval Service。

不透過 LLM。

目的為隔離：

* 檢索品質
* 生成品質

本次不實作：

* RAGAS
* LLM Judge
* Answer correctness 自動評分

---

# 30. README 規格

README 必須包含：

1. 專案目的
2. 專案情境
3. 系統架構圖
4. 技術棧
5. RAG 資料流程
6. 文件匯入流程
7. 查詢與拒答流程
8. 專案目錄
9. 快速啟動方式
10. 環境變數
11. LLM endpoint 設定方式
12. API 範例
13. 範例問題
14. 測試方式
15. Retrieval 評估方式
16. 實際評估結果
17. Demo 截圖
18. 已知限制
19. Future Work
20. 虛構資料聲明
21. 更換 Embedding 模型後需重新 indexing 的說明
22. 第一次啟動需下載模型的說明

---

# 31. 兩天開發順序

開發原則：

先完成最窄的 end-to-end vertical slice，再補文件管理、PDF、前端與測試。

---

## 第一天上午：核心 Retrieval Vertical Slice

### 工作內容

1. 建立 Git Repository。
2. 建立 Python 專案與依賴。
3. 建立 FastAPI 基本架構。
4. 建立 config。
5. 建立 Dockerfile。
6. 建立 docker-compose。
7. 啟動 Qdrant。
8. 建立 Markdown 與 TXT Loader baseline。
9. 建立 Text Splitter baseline。
10. 載入 Embedding 模型。
11. 建立 Qdrant Collection。
12. 將固定 Markdown 文件寫入 Qdrant。
13. 完成 Retrieval Service。
14. 使用固定問題找到正確 Chunk。

### 第一階段驗收

必須能完成：

```text
固定 Markdown 文件
→ Chunk
→ Embedding
→ Qdrant
→ Query Embedding
→ Top-K Result
```

上午結束前，必須能看到正確 Chunk 出現在檢索結果中。

---

## 第一天下午：完整文件生命週期

### 工作內容

1. 建立 SQLite documents Registry。
2. 建立 Document Repository。
3. 完成 SHA-256。
4. 完成重複文件判斷。
5. 完成上傳 API。
6. 完成 indexing status。
7. 完成 rollback。
8. 完成文件列表 API。
9. 完成文件刪除 API。
10. 完成 PDF Loader。
11. 補一份真正 PDF 範例。
12. 完成 `/api/retrieve`。

### 第二階段驗收

必須能：

* 上傳三種文件。
* 重複文件回傳 409。
* 文件可顯示於列表。
* Qdrant 可看到向量。
* 文件刪除後 Points 一併消失。
* indexing 失敗不會顯示為成功。

---

## 第一天晚上：RAG 問答

### 工作內容

1. 建立 OpenAI-compatible LLM Client。
2. 建立 RAG Prompt。
3. 建立 RAG Service。
4. 分配 Source ID。
5. 完成後端直接拒答。
6. 完成 Citation 解析與驗證。
7. 完成 `/api/query`。
8. 使用 Swagger 手動測試。
9. 測試無關問題不呼叫 LLM。
10. 測試來源 metadata 由後端提供。

### 當天結束驗收

必須完整跑通：

```text
上傳文件
→ 問問題
→ 檢索文件
→ LLM 回答
→ 回答附來源
```

並且：

```text
無相關內容
→ 後端直接拒答
→ 不呼叫 LLM
```

---

## 第二天上午：前端與主要測試

### 工作內容

1. 建立靜態前端。
2. 完成文件上傳畫面。
3. 完成文件列表。
4. 完成刪除功能。
5. 完成問答區。
6. 顯示來源與分數。
7. 顯示原始 Chunk。
8. 顯示查詢耗時。
9. 加入 Loading 與錯誤訊息。
10. 建立 Fake Services。
11. 完成主要 Unit Test。
12. 完成主要 API Test。

### 驗收

* 瀏覽器可完整操作。
* Unit Test 不依賴真實 LLM。
* 測試可重複執行。

---

## 第二天下午：評估與交付

### 工作內容

1. 建立 evaluation questions。
2. 完成 Hit@1。
3. 完成 Hit@3。
4. 完成拒答率。
5. 輸出 Retrieval latency。
6. 根據分數分布設定 Threshold。
7. 執行 Integration Test。
8. 撰寫 README。
9. 畫架構圖。
10. 建立 Demo 截圖。
11. 清理 logging 與環境變數。
12. 確認 `.env` 未提交。
13. 從乾淨環境測試 Docker Compose。
14. 建立 Git tag。

Git tag：

```text
v1.0-rag-mvp
```

---

# 32. MVP 完成條件

只有同時符合以下條件，才能視為 MVP 完成：

* 可上傳 PDF、Markdown、TXT。
* PDF 為可擷取文字的文件。
* 文件會計算 SHA-256。
* 重複文件會被辨識。
* 文件狀態會保存於 SQLite。
* 文件會切分為保留 metadata 的 Chunk。
* Chunk 不超過 Embedding token hard limit。
* 使用本地模型產生 Embedding。
* Qdrant Collection 會自動建立。
* Collection 維度與模型相容。
* Chunk 與 Payload 寫入 Qdrant。
* 可根據問題進行 Top-K Retrieval。
* 提供 Retrieval-only API。
* Threshold 可過濾低相關結果。
* 無合格內容時由後端直接拒答。
* 拒答時不呼叫 LLM。
* LLM 只根據 Retrieved Context 回答。
* 回答包含有效 Source ID。
* Sources metadata 由後端建立。
* 不會接受 LLM 捏造的來源。
* 可列出已匯入文件。
* 可刪除文件與其全部向量。
* Docker Compose 可啟動。
* Embedding cache 有持久化。
* 至少有基本 Unit Test。
* 至少有 Retrieval Integration Test。
* 有 Hit@1 與 Hit@3 評估。
* 有無關問題拒答率。
* README 可讓其他人重新啟動。
* 有五份以上虛構文件。
* 有至少一份真正 PDF。
* 不依賴既有葛拉漢專案。

---

# 33. 第二階段 Agent 擴充

> **廢棄路線**：本節描述的工具型 Agent（`search_documents` / `check_service_health` /
> `list_documents`，tag `v1.1-tool-agent`）**從未實作**。實際的 v1.1 改為
> Human-in-the-loop Retrieval Optimizer（見 `docs/v1.1-*`），v1.2 為 LLM-guided Retrieval
> Optimization Agent（見 `docs/v1.2-runtime-acceptance.md`）。本節保留供設計演進紀錄。

只有在 RAG MVP 全部完成後才開始。

Git tag：

```text
v1.1-tool-agent
```

## 33.1 Agent 工具

### search_documents

用途：

* 使用既有 Retrieval Service 搜尋文件。
* 不得繞過既有 Threshold 與來源處理。

### check_service_health

用途：

* 查詢 FastAPI、Qdrant、Embedding 與 LLM 設定狀態。

### list_documents

用途：

* 列出目前已匯入文件。

## 33.2 Routing

* 文件知識問題 → `search_documents`
* 系統狀態問題 → `check_service_health`
* 文件管理問題 → `list_documents`

## 33.3 Agent 禁止能力

第一版 Agent 不允許：

* 任意執行 SQL
* 任意執行 Shell
* 修改資料
* 刪除資料
* 呼叫外部網站
* 自行上傳文件
* 存取本機任意檔案
* 使用未列入 allowlist 的工具

## 33.4 Agent 評估

Agent 若實作，至少增加：

* 工具選擇測試集
* Tool selection accuracy
* Tool execution trace
* 未知問題處理
* 工具失敗 fallback
* 不應呼叫工具時的測試

Agent 不得取代 RAG MVP 的測試與評估工作。

---

# 34. 履歷描述

## RAG MVP 完成後

智慧製造文件 RAG 知識助理：使用 FastAPI、Qdrant、sentence-transformers、SQLite 與 Docker 建置企業文件知識庫原型，完成 PDF／Markdown／TXT 文件解析、結構化 Chunking、本地 Embedding、向量檢索、LLM 問答、來源追溯與無資料拒答機制，並建立 Hit@K、負面問題拒答率與檢索延遲評估流程。

實際評估完成後，可補充：

```text
在自建智慧製造文件測試集達成 Hit@3 XX%，無關問題拒答率 XX%。
```

不得在評估完成前填寫虛構數字。

## Agent 完成後

在既有 RAG 系統上擴充受限工具型 AI Agent，整合文件檢索、知識庫文件查詢與服務健康檢查工具，並建立工具選擇測試與執行紀錄，使模型可依問題類型選擇適當工具並回傳可追溯結果。

---

# 35. Claude CLI 執行計畫輸出格式

請依照以下格式產生執行計畫：

```text
# 實作前分析

## 需求摘要
## 技術假設
## 規格衝突或待注意事項
## 主要風險
## P0、P1 與 Future Work 分界

# 執行階段

## Phase 1：專案骨架與基礎設施

### Step 1.1
- 優先級：
- 目標：
- 建立或修改檔案：
- 實作內容：
- 相依項目：
- 驗收方式：
- 風險：

### Step 1.2
...

## Phase 2：文件解析與 Chunking
...

## Phase 3：Embedding 與 Qdrant
...

## Phase 4：文件生命週期
...

## Phase 5：Retrieval 與 RAG
...

## Phase 6：前端
...

## Phase 7：測試與評估
...

## Phase 8：README 與交付
...

# 建議 Commit 切分

1. commit 名稱
   - 包含內容
   - 驗收方式

# 兩天時間分配

## 第一天上午
## 第一天下午
## 第一天晚上
## 第二天上午
## 第二天下午

# 最終驗收 Checklist

# 可能需要延後到 P1 的內容
```

執行計畫必須具體到可以逐步交給 Claude CLI 執行，但本次回答不可直接修改任何檔案。
