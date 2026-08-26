# API Reference

本文件描述 **factory-rag-assistant 目前 `main` 分支實際提供的 HTTP API**。

## Contents

- [Scope](#scope) · [Base URL](#base-url) · [Common Conventions](#common-conventions) · [Endpoint Overview](#endpoint-overview)
- [System / Health](#system--health) — `/health`、`/ready`
- [Documents](#documents) — 上傳、列表、刪除
- [Retrieval / RAG](#retrieval--rag) — `/api/retrieve`、`/api/query`、[Production vs Evaluation Retrieval](#production-vs-evaluation-retrieval)
- [Optimizer — Read-only Reporting](#optimizer--read-only-reporting) — 9 個唯讀報表端點
- [Optimizer — Deployment Admin](#optimizer--deployment-admin) — 3 個唯讀 ＋ 4 個 state-changing 端點
- [v1.2 Agent](#v12-agent) — CLI-only，無 HTTP endpoint
- [PROJECT_SPEC vs Current Code Differences](#project_spec-vs-current-code-differences)

## Scope

- 內容以 **current source code 為唯一準則**，並以 FastAPI 實際 route registry（`app.openapi()`）交叉驗證。
- 目前 application 共註冊 **23 個 API endpoint**（另有一個 `include_in_schema=False` 的靜態首頁 `GET /`，
  以及 `/static` 靜態資源掛載，兩者非 API，不列入本文件）。
- [PROJECT_SPEC.md](../PROJECT_SPEC.md) §19 為 **v1.0 歷史規格**，只涵蓋 7 個 endpoint，
  且其中部分錯誤碼目前程式碼不會產生。**與本文件不一致時一律以本文件（current code）為準。**
  差異整理見文末 [PROJECT_SPEC vs Current Code Differences](#project_spec-vs-current-code-differences)。
- v1.2 Agent 目前**沒有** HTTP endpoint，見 [v1.2 Agent](#v12-agent)。

> **本 API 目前不提供任何身分驗證機制**：沒有 API key、Bearer token、session 或 user auth dependency。
> 23 個 endpoint 全部沒有 auth dependency。Optimizer 的寫入保護是 feature flag，不是身分驗證，
> 詳見 [Admin Write Guard（非身分驗證）](#admin-write-guard非身分驗證)。

---

## Base URL

本專案為單機 / Docker Compose 部署，沒有公開的 production domain。

| 執行方式 | Base URL |
| --- | --- |
| 本機 uvicorn（預設 `HOST=0.0.0.0`、`PORT=8000`） | `http://localhost:8000` |
| Docker Compose（`docker compose up`） | `http://localhost:8000` |

FastAPI 內建互動式文件：`/docs`（Swagger UI）、`/redoc`、`/openapi.json`。

---

## Common Conventions

### Content-Type

- 除 `POST /api/documents` 使用 `multipart/form-data` 外，所有帶 body 的請求皆為 `application/json`。
- 回應除 `DELETE /api/documents/{document_id}`（204、無 body）外皆為 JSON。

### 兩種並存的錯誤格式

目前程式碼有**兩種**錯誤 body，撰寫 client 時兩種都要處理。

**(A) 應用層錯誤封包** — 由 `AppError` handler 與 optimizer router 產生：

```json
{
  "error": {
    "code": "LLM_NOT_CONFIGURED",
    "message": "LLM 服務尚未完成設定。",
    "details": {}
  }
}
```

**(B) FastAPI 驗證 / `HTTPException` 格式** — `{"detail": ...}`。
注意 `detail` 的型別**不固定**：

Pydantic 驗證失敗時為**物件陣列**（以下為實際 runtime 輸出）：

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "question"],
      "msg": "Field required",
      "input": { "top_k": 5 }
    }
  ]
}
```

而 handler 內主動拋出的 `HTTPException`（目前僅 `top_k` 超過上限）為**單一字串**：

```json
{ "detail": "top_k 不可超過 10。" }
```

### 應用層錯誤碼對照

`AppError` 子類別定義於 `app/core/exceptions.py`，各自帶固定 HTTP status：

| Code | Status | 說明 |
| --- | --- | --- |
| `UNSUPPORTED_FILE_TYPE` | 415 | 副檔名不在 `.md` / `.markdown` / `.txt` / `.pdf` |
| `EMPTY_DOCUMENT` | 400 | 上傳內容空白，或解析後無有效文字 |
| `CORRUPT_PDF` | 400 | PDF 損壞或無法解析 |
| `ENCRYPTED_PDF` | 400 | 加密 / 需密碼的 PDF |
| `NO_EXTRACTABLE_TEXT` | 400 | PDF 無可擷取文字（掃描圖片型，不支援 OCR） |
| `DUPLICATE_DOCUMENT` | 409 | 相同 SHA-256 已存在，`details.document_id` 為既有文件 |
| `DOCUMENT_NOT_FOUND` | 404 | 找不到指定文件 |
| `EMBEDDING_ERROR` | 500 | Embedding 產生失敗 |
| `VECTOR_STORE_ERROR` | 503 | Qdrant 不可用 |
| `COLLECTION_MISMATCH` | 503 | Collection 與目前 Embedding 模型不相容 |
| `LLM_NOT_CONFIGURED` | 503 | 缺少 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` |
| `LLM_TIMEOUT` | 504 | LLM 逾時 |
| `LLM_SERVICE_ERROR` | 502 | LLM 連線失敗或 Provider 回錯 |
| `LLM_EMPTY_RESPONSE` | 502 | LLM 回傳空 choices 或空內容 |
| `INTERNAL_ERROR` | 500 | 未預期例外的統一封裝，不外洩 traceback |

> `FileTooLargeError`（413）與 `DocumentParseError`（400）雖有 class 定義，
> 但**目前程式碼中沒有任何呼叫點**，因此不會出現在任何回應中。詳見文末差異章節。

### Request 驗證

- `question`：必填；自動 `strip()`；空白 → 422；長度上限 2000 字元（以 trim 後計算）→ 422。
- `top_k`：選填；schema 層 `ge=1`（0 或負數 → 422，格式 A 之外的陣列型 detail）；
  未提供時採用 `RETRIEVAL_DEFAULT_TOP_K`（預設 **5**）；
  超過 `RETRIEVAL_MAX_TOP_K`（預設 **10**）→ 422（字串型 detail）。
- Optimizer admin 的 request schema 皆為 `extra="forbid"`，出現未知欄位 → 422。
- **驗證先於授權**：body 不合法時先回 422，不會走到 admin flag 檢查。

### 外部依賴不可用時的行為

| 情境 | 行為 |
| --- | --- |
| Qdrant 不可用 | `VECTOR_STORE_ERROR` → **503**（不是回空結果） |
| Collection 與模型維度不符 | `COLLECTION_MISMATCH` → **503** |
| LLM 未設定 | 僅在檢索有結果、實際要呼叫 LLM 時才會 **503**；`/api/retrieve` 完全不受影響 |
| Optimizer reporting DB 未配置 | **200** ＋ `available: false`（視為正常狀態，不是錯誤） |

> 注意：核心服務在 lifespan 期間初始化並會呼叫 `ensure_collection`，
> 因此**啟動時** Qdrant 不可用會導致 app 無法啟動（除非設 `SKIP_STARTUP_SERVICES=1`）。

### Readiness 語意

`/ready` 檢查五個項目，**全部**為 `ok` / `loaded` 才回 200，否則回 503。
因此**未設定 LLM 時 `/ready` 一律回 503**，即使文件管理與 `/api/retrieve` 都能正常運作。
若只想確認 process 存活，請用 `/health`。

### Admin Write Guard（非身分驗證）

`OPTIMIZER_ADMIN_ENABLED`（預設 `false`）是一個 **feature flag / write guard，不是身分驗證機制**。

- 只作用於 **4 個 state-changing POST**：`/alias/initialize`、`/candidates/{candidate_id}/apply`、
  `/rollback`、`/reconcile`。停用時回 **403 `optimizer_admin_disabled`**，且**不建構、不呼叫任何 service**。
- Optimizer 的 **read-only GET endpoint 不受此 flag 影響**，永遠可讀。
- 啟用後**沒有**任何 token 或 header 保護；此時寫入端點對所有能連到該 port 的人開放。
  第二道防線是 request body 內的逐字 `confirmation` 片語與 `idempotency_key`，而非身分驗證。

---

## Endpoint Overview

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process 存活檢查，不碰外部依賴 |
| GET | `/ready` | 依賴就緒檢查（Qdrant / Collection / Embedding / SQLite / LLM 設定） |
| POST | `/api/documents` | 上傳並索引文件 |
| GET | `/api/documents` | 列出所有文件 |
| DELETE | `/api/documents/{document_id}` | 刪除文件與其全部向量 |
| POST | `/api/retrieve` | 純向量檢索，不呼叫 LLM |
| POST | `/api/query` | RAG 問答：檢索 → 雙層拒答 → Citation 驗證 |
| GET | `/api/optimizer/overview` | Optimizer 總覽（search / review / deployment 摘要） |
| GET | `/api/optimizer/searches` | 檢索搜尋列表 |
| GET | `/api/optimizer/searches/{search_id}` | 單一 search 詳情 |
| GET | `/api/optimizer/searches/{search_id}/candidates` | 該 search 的候選設定 |
| GET | `/api/optimizer/searches/{search_id}/comparison` | 候選與 baseline 比較 |
| GET | `/api/optimizer/reviews` | 人工審核 session 列表 |
| GET | `/api/optimizer/reviews/{session_id}` | 單一 review session 詳情 |
| GET | `/api/optimizer/coverage` | 評估資料集覆蓋率 |
| GET | `/api/optimizer/deployment/summary` | Read-only 部署摘要 |
| GET | `/api/optimizer/deployments` | 部署操作紀錄列表 |
| GET | `/api/optimizer/deployments/{operation_id}` | 單筆部署操作 ＋ audit |
| GET | `/api/optimizer/alias/status` | 目前 alias 與實際指向 |
| POST | `/api/optimizer/alias/initialize` | 初次建立 alias（**state-changing**） |
| POST | `/api/optimizer/candidates/{candidate_id}/apply` | 套用候選設定並切換 alias（**destructive**） |
| POST | `/api/optimizer/rollback` | 將 alias 回退到先前 collection（**destructive**） |
| POST | `/api/optimizer/reconcile` | 修復 DB 與 alias 不一致（**state-changing**） |

---

# System / Health

## GET /health

### Purpose

確認 FastAPI process 存活。**不檢查**任何外部依賴，因此不適合當作 readiness probe。

### Request

無 Header、Path、Query 或 Body 參數。

### Response

`200 OK`

```json
{ "status": "ok" }
```

### Errors

無。此端點不存取任何依賴，正常運作下不會失敗。

---

## GET /ready

### Purpose

檢查各項依賴是否就緒，供部署 readiness probe 使用。

### Request

無參數。

### Response

`200 OK` — 五項檢查全部通過：

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

| Field | Type | Description |
| --- | --- | --- |
| `status` | string | `ready` 或 `not_ready` |
| `checks.qdrant` | string | `ok` / `error` / `unknown` |
| `checks.collection` | string | `ok` / `error` / `unknown` |
| `checks.embedding` | string | `loaded` / `error` |
| `checks.metadata_db` | string | `ok` / `error` / `not_initialized` |
| `checks.llm_config` | string | `ok` / `error` |

### Errors

| Status | Condition |
| --- | --- |
| 503 | 任一檢查不是 `ok` / `loaded`；body 同上但 `status` 為 `not_ready` |

`503` 實際範例（未設定 LLM 以外的依賴亦未就緒時）：

```json
{
  "status": "not_ready",
  "checks": {
    "qdrant": "unknown",
    "collection": "unknown",
    "embedding": "error",
    "metadata_db": "not_initialized",
    "llm_config": "ok"
  }
}
```

### Notes

- 此端點**不會**拋出例外：所有檢查錯誤都被吞掉並轉為字串狀態。
- `metadata_db` 會實際執行一次 SQLite 讀寫（`ping`）才回 `ok`。

---

# Documents

## POST /api/documents

### Purpose

上傳單一文件，同步完成解析、Chunking、Embedding 與 Qdrant 索引，成功後回傳文件註冊資訊。

### Request

Headers

| Name | Required | Value |
| --- | --- | --- |
| `Content-Type` | 是 | `multipart/form-data` |

Body（multipart form fields）

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `file` | file | 是 | — | 支援 `.md` / `.markdown` / `.txt` / `.pdf` |

### Response

`201 Created`

```json
{
  "document_id": "3f2b7c10-0000-4000-8000-000000000001",
  "filename": "equipment_alarm_manual.md",
  "chunk_count": 5,
  "status": "indexed",
  "created_at": "2026-08-22T03:18:17+00:00"
}
```

| Field | Type | Description |
| --- | --- | --- |
| `document_id` | string | 後端產生的 UUID |
| `filename` | string | 原始檔名 |
| `chunk_count` | integer | 實際寫入的 chunk 數 |
| `status` | string | 成功時為 `indexed` |
| `created_at` | string | ISO-8601 時間字串 |

### Errors

| Status | Condition |
| --- | --- |
| 400 | `EMPTY_DOCUMENT`：檔案空白或解析後無有效文字 |
| 400 | `CORRUPT_PDF` / `ENCRYPTED_PDF` / `NO_EXTRACTABLE_TEXT`：PDF 無法處理 |
| 409 | `DUPLICATE_DOCUMENT`：相同 SHA-256 已存在，`details.document_id` 為既有文件 |
| 415 | `UNSUPPORTED_FILE_TYPE`：不支援的副檔名 |
| 422 | 未提供 `file` 欄位 |
| 500 | `EMBEDDING_ERROR`：Embedding 失敗（含單一 chunk 超過 token 上限） |
| 503 | `VECTOR_STORE_ERROR` / `COLLECTION_MISMATCH`：Qdrant 不可用或不相容 |

### Notes

- **目前沒有檔案大小檢查**：`MAX_UPLOAD_SIZE_MB` 設定值與 `FileTooLargeError`（413）
  在程式碼中沒有任何呼叫點，handler 會直接把整個檔案讀入記憶體。**413 不會發生。**
- 重複判定依據是**原始 bytes 的 SHA-256**，與檔名無關。
- 索引失敗時會回滾 Qdrant points，但 SQLite 註冊列會保留為 `failed` 並記錄 `error_message`。

---

## GET /api/documents

### Purpose

列出目前所有已註冊文件（含索引失敗者）。

### Request

無參數。**目前不支援分頁或過濾。**

### Response

`200 OK`

```json
{
  "documents": [
    {
      "document_id": "3f2b7c10-0000-4000-8000-000000000001",
      "filename": "equipment_alarm_manual.md",
      "file_type": "text/markdown",
      "file_size": 8420,
      "status": "indexed",
      "chunk_count": 5,
      "error_message": null,
      "created_at": "2026-08-22T03:18:17+00:00",
      "updated_at": "2026-08-22T03:18:19+00:00"
    }
  ],
  "total": 1
}
```

| Field | Type | Description |
| --- | --- | --- |
| `documents[].document_id` | string | UUID |
| `documents[].filename` | string | 原始檔名 |
| `documents[].file_type` | string | MIME：`text/markdown` / `text/plain` / `application/pdf` / `application/octet-stream` |
| `documents[].file_size` | integer | 原始 bytes 大小 |
| `documents[].status` | string | `indexing` / `indexed` / `failed` |
| `documents[].chunk_count` | integer | chunk 數 |
| `documents[].error_message` | string \| null | 索引失敗訊息（最多 500 字元） |
| `documents[].created_at` | string | ISO-8601 |
| `documents[].updated_at` | string | ISO-8601 |
| `total` | integer | 陣列長度 |

無資料時回傳 `{"documents": [], "total": 0}`。

### Errors

正常運作下無錯誤回應（僅讀取 SQLite）。

### Notes

路徑為 `/api/documents`，**結尾不加斜線**；`/api/documents/` 會觸發 307 轉址。

---

## DELETE /api/documents/{document_id}

### Purpose

刪除指定文件：先刪 Qdrant 中該文件的所有向量，再刪本地檔案，最後刪 SQLite 註冊列。

### Request

Path Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `document_id` | string | 是 | 由上傳或列表取得的文件 ID |

無 Body。

### Response

`204 No Content` — 無回應 body。

### Errors

| Status | Condition |
| --- | --- |
| 404 | `DOCUMENT_NOT_FOUND`：`document_id` 不存在 |
| 503 | `VECTOR_STORE_ERROR`：Qdrant 刪除失敗 |

### Notes

- **具破壞性且不可復原。**
- 刪除順序刻意為 Qdrant → 檔案 → SQLite：Qdrant 刪除失敗時回 503 並**保留** SQLite 記錄，
  避免產生無法追蹤的孤兒向量。

---

# Retrieval / RAG

> 本節描述的是 **Production 檢索行為**。離線 evaluation / optimizer 使用另一套實作，
> 兩者不可混為一談，見 [Production vs Evaluation Retrieval](#production-vs-evaluation-retrieval)。

## POST /api/retrieve

### Purpose

只執行向量檢索並回傳命中的 chunk，**不呼叫 LLM**。可用於單獨驗證檢索品質，
或在 LLM 不可用時檢查系統前半段。

### Request

Headers

| Name | Required | Value |
| --- | --- | --- |
| `Content-Type` | 是 | `application/json` |

Body

```json
{
  "question": "E102 警報代表什麼？",
  "top_k": 5
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `question` | string | 是 | — | 自動 trim；不可空白；trim 後上限 2000 字元 |
| `top_k` | integer \| null | 否 | `null` → 使用 `RETRIEVAL_DEFAULT_TOP_K`（5） | 最小 1；上限 `RETRIEVAL_MAX_TOP_K`（10） |

### Response

`200 OK`

```json
{
  "results": [
    {
      "source_id": "S1",
      "document_id": "3f2b7c10-0000-4000-8000-000000000001",
      "filename": "equipment_alarm_manual.md",
      "page": null,
      "section": "E102 感測器通訊異常",
      "chunk_index": 4,
      "score": 0.8712,
      "text": "E102 表示感測器通訊異常，請先檢查感測器連線狀態與通訊線材。"
    },
    {
      "source_id": "S2",
      "document_id": "3f2b7c10-0000-4000-8000-000000000002",
      "filename": "equipment_maintenance_manual.pdf",
      "page": 12,
      "section": "D900 定期保養",
      "chunk_index": 2,
      "score": 0.6431,
      "text": "D900 為定期保養提醒代碼。"
    }
  ],
  "retrieval_time_ms": 52
}
```

| Field | Type | Description |
| --- | --- | --- |
| `results[].source_id` | string | 後端依排序配發，`S1` 為最高分 |
| `results[].document_id` | string | 來源文件 UUID |
| `results[].filename` | string | 來源檔名 |
| `results[].page` | integer \| null | PDF 為 1-based 頁碼；MD / TXT 為 `null` |
| `results[].section` | string \| null | 章節標題（可能為 `null`） |
| `results[].chunk_index` | integer | 文件內 chunk 序號 |
| `results[].score` | number | 原始向量相似度，四捨五入至小數 4 位 |
| `results[].text` | string | chunk 全文 |
| `retrieval_time_ms` | integer | 含 query embedding 的耗時 |

無命中時回傳 `{"results": [], "retrieval_time_ms": <n>}`（仍為 200）。

### Errors

| Status | Condition |
| --- | --- |
| 422 | `question` 缺少 / 空白 / 超過 2000 字元；`top_k < 1`（detail 為陣列） |
| 422 | `top_k` 超過上限（detail 為字串 `"top_k 不可超過 10。"`） |
| 500 | `EMBEDDING_ERROR` |
| 503 | `VECTOR_STORE_ERROR` / `COLLECTION_MISMATCH` |

### Notes

- 檢索流程：query embedding（會加上 `EMBEDDING_QUERY_INSTRUCTION` 前綴）
  → 取候選 `max(top_k, min(top_k × 3, 50))` 筆
  → 過濾 `score >= RETRIEVAL_SCORE_THRESHOLD`（預設 **0.55**）
  → 由高到低排序 → 取前 `top_k` → 依序配發 `S1`…`Sn`。
- 先過度取候選再過濾，是為了避免「先取 top_k 才過濾」導致結果過少。
- **不受 LLM 設定影響**，未設定 LLM 時仍可正常使用。
- Qdrant payload 內的 v1.1 欄位（`file_hash`、`provenance_spans`、`identifiers`、
  `canonical_text_version`）**刻意不對外輸出**，API 契約僅含上表 8 個欄位。

---

## POST /api/query

### Purpose

完整 RAG 問答：檢索 → 第一層 threshold 拒答 → 呼叫 LLM → 第二層語意拒答 →
Citation 驗證（fail-closed）→ 依首次引用順序組裝 sources。

### Request

Headers

| Name | Required | Value |
| --- | --- | --- |
| `Content-Type` | 是 | `application/json` |

Body

```json
{
  "question": "設備發生 E102 警報時應如何處理？",
  "top_k": 5
}
```

欄位與驗證規則與 `POST /api/retrieve` **完全相同**（`QueryRequest` 直接繼承 `RetrieveRequest`）。

### Response

`200 OK` — 有效回答：

```json
{
  "answer": "依據設備警報手冊，應先檢查感測器連線狀態與通訊線材。[S1]",
  "sources": [
    {
      "source_id": "S1",
      "document_id": "3f2b7c10-0000-4000-8000-000000000001",
      "filename": "equipment_alarm_manual.md",
      "page": null,
      "section": "E102 感測器通訊異常",
      "chunk_index": 4,
      "score": 0.8712,
      "text": "E102 表示感測器通訊異常，請先檢查感測器連線狀態與通訊線材。"
    }
  ],
  "retrieval_time_ms": 52,
  "generation_time_ms": 1210
}
```

| Field | Type | Description |
| --- | --- | --- |
| `answer` | string | LLM 回答，內含 `[S1]` 形式的 citation |
| `sources` | array | 欄位結構與 `/api/retrieve` 的 `results[]` 完全相同 |
| `retrieval_time_ms` | integer | 檢索耗時 |
| `generation_time_ms` | integer | LLM 生成耗時；第一層拒答時為 `0` |

`200 OK` — 拒答（**不是錯誤**）：

```json
{
  "answer": "目前提供的文件中沒有足夠資訊可以回答此問題。",
  "sources": [],
  "retrieval_time_ms": 41,
  "generation_time_ms": 0
}
```

### Errors

| Status | Condition |
| --- | --- |
| 422 | 同 `/api/retrieve` 的兩種 422 |
| 500 | `EMBEDDING_ERROR` |
| 502 | `LLM_SERVICE_ERROR`：連線失敗或 Provider 回錯 |
| 502 | `LLM_EMPTY_RESPONSE`：LLM 回傳空 choices 或空內容 |
| 503 | `VECTOR_STORE_ERROR` / `COLLECTION_MISMATCH` |
| 503 | `LLM_NOT_CONFIGURED`：缺少 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` |
| 504 | `LLM_TIMEOUT` |

### Notes

**拒答一律回 200**，共有四種觸發路徑，回應 body 完全相同（固定訊息 ＋ `sources: []`）：

1. **第一層（threshold）**：過濾後無合格 chunk → **不呼叫 LLM**，`generation_time_ms` 為 `0`。
2. **第二層（語意）**：LLM 判定 context 無答案並回傳固定拒答句。
3. **無 citation**：LLM 回答未包含任何 `[Sn]` → 整體拒答。
4. **無效 citation**：引用了不存在的 source id → 整體 fail-closed，**不做部分修補**。

其他重點：

- **LLM 未設定時不一定回 503**：若檢索結果為空，會走第一層拒答直接回 **200**；
  只有在檢索有結果、實際需要呼叫 LLM 時才會回 503 `LLM_NOT_CONFIGURED`。
- `sources` 只包含**實際被引用**的 chunk，並依答案中首次出現順序排列、去重，
  因此 `sources` 可能少於 `top_k`，順序也未必等同分數排序。
- 所有 source metadata 一律來自後端檢索結果，**不採信 LLM 生成的任何 metadata**。

---

## Production vs Evaluation Retrieval

`/api/retrieve` 與 `/api/query` 使用的是 `app/services/retrieval_service.py`，
與離線 evaluation / optimizer 的檢索策略是**不同的實作**。文件與量測結果不可互相套用。

| 面向 | Production（`/api/retrieve`、`/api/query`） | Evaluation / Optimizer Trial（離線） |
| --- | --- | --- |
| 實作位置 | `app/services/retrieval_service.py` | `evaluation/v11/retrieval/`（`dense_strategy` / `identifier_aware_strategy`） |
| 候選來源 | 單一 dense 向量檢索 | dense，或 dense ∪ identifier payload filter（雙來源） |
| 排序分數 | 原始 vector score | `ranking_score = vector_score + exact_match_boost − confusable_penalty` |
| Threshold | 單一 `RETRIEVAL_SCORE_THRESHOLD`（0.55） | 雙層：exact identifier 命中用較低的 `identifier_vector_floor`，其餘用一般 threshold |
| 同分處理 | 依 Qdrant 回傳順序的穩定排序 | 5 鍵 deterministic tiebreak |
| Cross-Encoder rerank | **無** —— `/api/retrieve` 不做 rerank | **選用**（v1.2.1）：`rerank_strategy=cross_encoder` 時，於 admission 之後對 `A[:rerank_top_n]` 以 raw logit 重排 |
| 對外欄位 | 固定 8 欄 | 另含 evidence / relevance 判定所需欄位 |

要點：

- `identifier_aware` 策略**在 production 沒有對應實作**；其 boost / penalty / 雙層 admission
  等行為**不會**出現在 `/api/retrieve`。當查詢不含 identifier 時，該策略會退化為與 dense 相同的行為，
  這也是兩者可比較的前提。
- **Cross-Encoder rerank 只存在於 evaluation 層**（v1.2.1，`rerank_strategy` /
  `rerank_top_n`，模型 `BAAI/bge-reranker-base`）。`/api/retrieve` 與 `/api/query`
  **不支援、也不執行 rerank**，其排序仍完全由 raw vector score 決定。
  rerank 位於 admission **之後**，不具 admission 或過濾權限（結構性保證，見
  [v1.2.1 rerank design](v1.2.1-rerank-design.md)）。該能力在既有 86 ms latency
  constraint 下不具 production 可行性，且品質改善尚未正式成立。
- Production 目前可調的檢索參數只有 `top_k`、`RETRIEVAL_SCORE_THRESHOLD`
  與候選數策略；`strategy`、`exact_match_boost`、`confusable_penalty`、`identifier_vector_floor`
  等**僅存在於離線設定**。
- `apply` / `rollback` 切換的是 alias 指向的 **collection**，**不會改變 production 的 ranking 演算法**。

---

# Optimizer — Read-only Reporting

9 個 GET endpoint，全部 **read-only**：不建立 DB、不執行 migration、不呼叫 LLM、不碰 Qdrant，
回傳一律去識別化（不含私人路徑、全文或向量）。**不受 `OPTIMIZER_ADMIN_ENABLED` 影響。**

資料來自三個**分開配置**的唯讀 SQLite：`OPTIMIZER_SEARCH_DB_PATH`、`OPTIMIZER_REVIEW_DB_PATH`、
`OPTIMIZER_DEPLOYMENT_DB_PATH`，**預設皆為未配置**。

### 共用回應慣例

**DB 未配置或檔案不存在 → `200 OK`**，並非錯誤：

```json
{
  "configured": false,
  "available": false,
  "reason": "optimizer_data_not_available",
  "search": null
}
```

`reason` 目前可能為 `optimizer_data_not_available` 或 `optimizer_schema_incompatible`。
列表型端點在此情況回傳空陣列而非 `null`。

**DB 可用但 ID 不存在 → `404`**，使用應用層錯誤封包：

```json
{
  "error": {
    "code": "search_not_found",
    "message": "search 'demo-search-001' not found",
    "details": {}
  }
}
```

### Endpoints

| Method | Path | Purpose | 資料鍵 | 404 code |
| --- | --- | --- | --- | --- |
| GET | `/api/optimizer/overview` | search / review / deployment 三區塊總覽 | `search`, `review`, `deployment` | — |
| GET | `/api/optimizer/searches` | 檢索搜尋列表 | `searches`（陣列） | — |
| GET | `/api/optimizer/searches/{search_id}` | 單一 search 詳情 | `search` | `search_not_found` |
| GET | `/api/optimizer/searches/{search_id}/candidates` | 候選設定與指標 | `candidates` | `search_not_found` |
| GET | `/api/optimizer/searches/{search_id}/comparison` | 候選 vs baseline 比較 | `comparison` | `search_not_found` |
| GET | `/api/optimizer/reviews` | 人工審核 session 列表 | `sessions`（陣列） | — |
| GET | `/api/optimizer/reviews/{session_id}` | 單一 session 詳情 | `session` | `review_session_not_found` |
| GET | `/api/optimizer/coverage` | 資料集覆蓋率 | `coverage` | — |
| GET | `/api/optimizer/deployment/summary` | Read-only 部署摘要 | `operations`, `records` | — |

Path parameter 皆為 string，且皆為必填。以上端點**都沒有 query parameter**。

### 兩個行為特殊的端點

`GET /api/optimizer/overview` 會額外注入目前設定值（`active_alias`、`admin_write_enabled`），
並以三個子區塊各自回報可用性：

```json
{
  "is_demo_fixture": false,
  "search": {
    "configured": false,
    "available": false,
    "reason": "optimizer_data_not_available",
    "dataset_version": null,
    "corpus_hash_short": null,
    "case_count": 0,
    "group_count": 0,
    "split_counts": { "tune": 0, "validation": 0, "test": 0 },
    "coverage_warnings": [],
    "latest_search": null
  },
  "review": {
    "configured": false,
    "available": false,
    "reason": "optimizer_data_not_available",
    "session_count": 0,
    "sessions": []
  },
  "deployment": {
    "configured": false,
    "available": false,
    "reason": "optimizer_data_not_available",
    "admin_write_enabled": false,
    "active_alias": "factory_knowledge_active",
    "alias_initialized": false,
    "history_count": 0
  }
}
```

`GET /api/optimizer/deployment/summary` 會在 store 結果上額外附加 `admin_write_enabled`：

```json
{
  "configured": false,
  "available": false,
  "reason": "optimizer_data_not_available",
  "active_alias": "factory_knowledge_active",
  "alias_initialized": false,
  "history_count": 0,
  "operations": [],
  "records": [],
  "admin_write_enabled": false
}
```

> 此端點只讀 reporting DB，**不會**查詢 Qdrant 的真實 alias 指向；
> 需要 live alias target 請改用 `GET /api/optimizer/alias/status`。

---

# Optimizer — Deployment Admin

7 個 endpoint，其中 **3 個 read-only GET** 與 **4 個 state-changing POST**。
只有 POST 受 `OPTIMIZER_ADMIN_ENABLED` 保護（見 [Admin Write Guard](#admin-write-guard非身分驗證)）。

| Endpoint | 性質 | Admin flag | Confirmation | 破壞性 |
| --- | --- | --- | --- | --- |
| `GET /deployments` | read-only | 否 | 否 | 無 |
| `GET /deployments/{operation_id}` | read-only | 否 | 否 | 無 |
| `GET /alias/status` | read-only | 否 | 否 | 無 |
| `POST /alias/initialize` | state-changing | 是 | 是 | 建立 alias |
| `POST /candidates/{candidate_id}/apply` | state-changing | 是 | 是 | **是**（切換 production alias） |
| `POST /rollback` | state-changing | 是 | 是 | **是**（回退 production alias） |
| `POST /reconcile` | state-changing | 是 | 否（僅 idempotency key） | 修復狀態 |

### 寫入端點共用錯誤碼對照

四個 POST 共用同一組 error code → HTTP status 對照：

| Status | Codes |
| --- | --- |
| 403 | `optimizer_admin_disabled`, `unsafe_collection` |
| 404 | `collection_not_found`, `search_not_found`, `candidate_not_found` |
| 409 | `idempotency_conflict`, `deployment_in_progress`, `stale_alias_target`, `stale_candidate`, `corpus_hash_mismatch`, `apply_rejected`, `confirmation_mismatch`, `readiness_failed`, `rollback_target_not_allowed`, `manual_reconciliation_required`, `alias_already_initialized`, `alias_not_initialized`, `collection_name_conflict` |
| 500 | `build_failed` |
| 400 | `invalid_request`，以及任何未對應的 code |

操作**已執行但未成功完成**（rejected / failed / needs_reconcile）時，回應為 `{error, result}` **雙鍵**：

```json
{
  "error": {
    "code": "apply_rejected",
    "message": "candidate is not eligible for apply",
    "details": {}
  },
  "result": {
    "operation": {
      "operation_id": "op-demo-0001",
      "status": "rejected",
      "error_code": "apply_rejected"
    }
  }
}
```

---

## GET /api/optimizer/deployments

### Purpose

列出部署操作紀錄（read-only）。

### Request

Query Parameters

| Name | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `alias` | string | 否 | 無 | 依 alias 過濾。此參數以 `request.query_params` 直接讀取，**未宣告於 OpenAPI**，因此不會出現在 `/docs` 且不做型別驗證 |

### Response

`200 OK`

```json
{ "operations": [] }
```

`operations[]` 為去識別化後的操作紀錄（`to_public_dict()`，不含私人路徑）。

### Errors

無。DB 檔案不存在時**不建立** DB，直接回 `200` ＋ 空陣列。

---

## GET /api/optimizer/deployments/{operation_id}

### Purpose

取得單筆部署操作及其 audit 事件。

### Request

Path Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `operation_id` | string | 是 | 操作 ID |

### Response

`200 OK`

```json
{
  "operation": {
    "operation_id": "op-demo-0001",
    "status": "completed"
  },
  "audit": [
    { "event_type": "apply_started", "created_at": "2026-08-22T03:18:17+00:00" },
    { "event_type": "alias_switched", "created_at": "2026-08-22T03:18:24+00:00" }
  ]
}
```

### Errors

| Status | Condition |
| --- | --- |
| 404 | DB 不存在，或 `operation_id` 查無資料 |

實際 404 body：

```json
{
  "error": {
    "code": "search_not_found",
    "message": "operation not found",
    "details": {}
  }
}
```

### Notes

此處的 404 error code 為 `search_not_found`（而非 `operation_not_found`），
是目前程式碼的實際行為。

---

## GET /api/optimizer/alias/status

### Purpose

回報 alias 名稱、Qdrant 中的**實際指向**，以及 DB 記錄的 active target。

### Request

無參數。

### Response

`200 OK`

```json
{
  "alias": "factory_knowledge_active",
  "alias_target": null,
  "db_active_target": null,
  "initialized": false
}
```

| Field | Type | Description |
| --- | --- | --- |
| `alias` | string | 來自 `OPTIMIZER_ACTIVE_ALIAS` |
| `alias_target` | string \| null | Qdrant 中 alias 實際指向的 collection |
| `db_active_target` | string \| null | Deployment DB 記錄的 active collection |
| `initialized` | boolean | 等同 `alias_target is not null` |

### Errors

無。讀取 alias 失敗時會被吞掉，`alias_target` 回 `null`。

### Notes

因為讀取失敗與「尚未初始化」都會得到 `alias_target: null`，
**`initialized: false` 無法區分兩者**（例如 Qdrant 不可用時也會回 `false`）。

---

## POST /api/optimizer/alias/initialize

### Purpose

首次建立 logical alias 並指向指定 collection。**State-changing。**

### Request

Headers：`Content-Type: application/json`

Body（`extra="forbid"`，所有欄位皆必填且 `min_length=1`）

```json
{
  "alias": "factory_knowledge_active",
  "collection": "factory_knowledge",
  "confirmation": "INITIALIZE factory_knowledge_active -> factory_knowledge",
  "idempotency_key": "demo-initialize-key-001"
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `alias` | string | 是 | — | 要建立的 alias |
| `collection` | string | 是 | — | alias 指向的實體 collection |
| `confirmation` | string | 是 | — | 必須逐字等於 `INITIALIZE <alias> -> <collection>` |
| `idempotency_key` | string | 是 | — | 重放保護 |

### Response

`200 OK` — 操作完成時回傳 `result.to_public_dict()` 內容（含 `operation`）。

### Errors

| Status | Condition |
| --- | --- |
| 403 | `optimizer_admin_disabled`：`OPTIMIZER_ADMIN_ENABLED` 為 false |
| 409 | `alias_already_initialized`、`confirmation_mismatch`、`idempotency_conflict`、`collection_name_conflict` |
| 404 | `collection_not_found` |
| 422 | 缺少必填欄位或出現未知欄位 |
| 400 / 500 | 其餘 code 依共用對照表 |

停用時的實際 403 body：

```json
{
  "error": {
    "code": "optimizer_admin_disabled",
    "message": "Optimizer admin write API 未啟用（optimizer_admin_enabled=false）。",
    "details": {}
  }
}
```

---

## POST /api/optimizer/candidates/{candidate_id}/apply

### Purpose

將某個候選設定實際套用到 production：建立新的 versioned collection 並把 alias 切過去。
**這是破壞性、會改變 production 檢索來源的操作。**

### Request

Path Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `candidate_id` | string | 是 | 候選設定 ID |

Body（`extra="forbid"`）

```json
{
  "search_id": "demo-search-001",
  "expected_config_hash": "cfg-hash-placeholder-0001",
  "expected_dataset_hash": "ds-hash-placeholder-0001",
  "expected_corpus_hash": "corpus-hash-placeholder-0001",
  "expected_code_commit": "0000000000000000000000000000000000000000",
  "expected_current_collection": "factory_knowledge_v1",
  "confirmation": "APPLY factory_knowledge_active CONFIG cfg-hash-placeholder-0001 FROM factory_knowledge_v1",
  "idempotency_key": "demo-apply-key-001"
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `search_id` | string | 是 | — | 候選所屬 search |
| `expected_config_hash` | string | 是 | — | 預期候選設定 hash |
| `expected_dataset_hash` | string | 是 | — | 預期資料集 hash |
| `expected_corpus_hash` | string | 是 | — | 預期語料 hash |
| `expected_code_commit` | string | 是 | — | 預期 code commit |
| `expected_current_collection` | string | 是 | — | 預期目前 alias 指向；不符即拒絕 |
| `confirmation` | string | 是 | — | 逐字 `APPLY <alias> CONFIG <config_hash> FROM <expected_current_collection>` |
| `idempotency_key` | string | 是 | — | 重放保護 |
| `alias` | string \| null | 否 | `OPTIMIZER_ACTIVE_ALIAS` | 目標 alias |
| `expected_canonical_text_version` | string \| null | 否 | `null` | 額外一致性檢查 |
| `expected_identifier_normalization_version` | string \| null | 否 | `null` | 額外一致性檢查 |
| `expected_embedding_fingerprint_hash` | string \| null | 否 | `null` | 額外一致性檢查 |
| `expected_recommendation_locked_at` | string \| null | 否 | `null` | 額外一致性檢查 |

### Response

`200 OK` — 僅在操作 status 為 `completed` 時；body 為 `result.to_public_dict()`。

### Errors

| Status | Condition |
| --- | --- |
| 403 | `optimizer_admin_disabled`、`unsafe_collection` |
| 404 | `candidate_not_found`、`search_not_found`、`collection_not_found` |
| 409 | `apply_rejected`（候選不符合 eligibility）、`stale_candidate`、`corpus_hash_mismatch`、`confirmation_mismatch`、`idempotency_conflict`、`stale_alias_target`、`readiness_failed`、`deployment_in_progress` |
| 422 | 缺少必填欄位或出現未知欄位 |
| 500 | `build_failed`：新 collection 建置失敗 |

### Notes

- **Eligibility 由後端從 DB 的 locked outcome 重新讀取**，API **不接受**任何
  `apply_eligible` 之類的布林旗標；送出未知欄位會直接 422。
- 所有 `expected_*` 欄位構成樂觀鎖：任一項與現況不符即拒絕，避免套用過期的候選。
- 切換前會執行 readiness 驗證，未通過回 409 `readiness_failed`。
- 未成功完成時回應為 `{error, result}` 雙鍵。

---

## POST /api/optimizer/rollback

### Purpose

將 alias 回退到先前的 collection。**破壞性操作。**

### Request

Body（`extra="forbid"`）

```json
{
  "expected_current_collection": "factory_knowledge_v2",
  "target_collection": "factory_knowledge_v1",
  "confirmation": "ROLLBACK factory_knowledge_active FROM factory_knowledge_v2 TO factory_knowledge_v1",
  "idempotency_key": "demo-rollback-key-001"
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `expected_current_collection` | string | 是 | — | 預期目前 alias 指向 |
| `target_collection` | string | 是 | — | 回退目標 collection |
| `confirmation` | string | 是 | — | 逐字 `ROLLBACK <alias> FROM <current> TO <target>` |
| `idempotency_key` | string | 是 | — | 重放保護 |
| `alias` | string \| null | 否 | `OPTIMIZER_ACTIVE_ALIAS` | 目標 alias |

### Response

`200 OK` — 操作完成時回傳 `result.to_public_dict()`。

### Errors

| Status | Condition |
| --- | --- |
| 403 | `optimizer_admin_disabled`、`unsafe_collection` |
| 404 | `collection_not_found` |
| 409 | `rollback_target_not_allowed`、`stale_alias_target`、`confirmation_mismatch`、`idempotency_conflict`、`alias_not_initialized`、`deployment_in_progress` |
| 422 | 缺少必填欄位或出現未知欄位 |
| 400 / 500 | 其餘 code 依共用對照表 |

### Notes

回退目標受白名單限制，不允許指向未受管理或不安全的 collection（`rollback_target_not_allowed` / `unsafe_collection`）。

---

## POST /api/optimizer/reconcile

### Purpose

修復 Deployment DB 記錄與 Qdrant alias 實際狀態之間的不一致。**State-changing。**

### Request

Body（`extra="forbid"`）

```json
{
  "idempotency_key": "demo-reconcile-key-001"
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `idempotency_key` | string | 是 | — | 重放保護 |
| `alias` | string \| null | 否 | `OPTIMIZER_ACTIVE_ALIAS` | 目標 alias |

此端點**不需要** `confirmation` 欄位。

### Response

`200 OK` — 操作完成時回傳 `result.to_public_dict()`。

### Errors

| Status | Condition |
| --- | --- |
| 403 | `optimizer_admin_disabled` |
| 409 | `manual_reconciliation_required`：無法自動修復，需人工介入 |
| 409 | `idempotency_conflict`、`deployment_in_progress` |
| 422 | 缺少 `idempotency_key` 或出現未知欄位 |
| 400 / 500 | 其餘 code 依共用對照表 |

---

# v1.2 Agent

**v1.2 Agent 目前透過 CLI 執行，未提供 HTTP endpoint**；使用方式見
[developer-guide.md](developer-guide.md)。

`evaluation/` 套件中沒有任何 FastAPI router，也沒有註冊到 application 的 Agent 路由。
Agent 的入口為 `python -m evaluation.v12.agent.cli`（`optimize` / `show` 兩個子命令）。

同樣為 CLI-only、**沒有** HTTP 介面的還有：v1.1 Optimizer trial runner、檢索搜尋（search）、
人工審核與 Dataset Authoring。這些流程產生的結果可透過
[Optimizer Read-only Reporting](#optimizer--read-only-reporting) 端點讀取。

---

# PROJECT_SPEC vs Current Code Differences

[PROJECT_SPEC.md](../PROJECT_SPEC.md) §19 記載 7 個 endpoint，目前程式碼註冊 23 個。
以下為稽核出的差異；**PROJECT_SPEC 保留為 v1.0 歷史規格，不因此重寫**。

### 1. PROJECT_SPEC 有、current code 不會產生的行為

| 項目 | PROJECT_SPEC | Current code |
| --- | --- | --- |
| 上傳檔案過大 `413` | §19.3 列為錯誤之一 | **不存在**。`FileTooLargeError` 與 `max_upload_size_bytes` 無任何呼叫點，上傳沒有大小檢查 |
| 文件解析失敗 `400 DOCUMENT_PARSE_ERROR` | §19.3 列出 | class 已定義但**從未 raise**；PDF 失敗改用 `CORRUPT_PDF` / `ENCRYPTED_PDF` / `NO_EXTRACTABLE_TEXT` |
| §20「至少驗證檔案大小、MIME type」 | 要求驗證 | 實際只驗證**副檔名**與是否為空；MIME 由副檔名推導而非驗證 |

### 2. current code 有、PROJECT_SPEC 沒有記錄的 endpoint

PROJECT_SPEC 完全沒有記載 **16 個 `/api/optimizer/*` endpoint**：

- Read-only Reporting（9）：`/overview`、`/searches`、`/searches/{search_id}`、
  `/searches/{search_id}/candidates`、`/searches/{search_id}/comparison`、`/reviews`、
  `/reviews/{session_id}`、`/coverage`、`/deployment/summary`
- Deployment Admin（7）：`/deployments`、`/deployments/{operation_id}`、`/alias/status`、
  `/alias/initialize`、`/candidates/{candidate_id}/apply`、`/rollback`、`/reconcile`

> [v1.1-retrieval-optimizer-architecture.md](v1.1-retrieval-optimizer-architecture.md) 中的端點表
> 是 design-time 產物，其 route 清單與目前實際註冊的 16 個 optimizer endpoint 不一致
> （包含從未實作的 route，且 path 命名亦有 drift）。請以本文件為準。

### 3. Schema 差異

| Endpoint | PROJECT_SPEC | Current code |
| --- | --- | --- |
| `GET /api/documents` | 欄位為 `content_type` | 實際欄位名為 **`file_type`** |
| `GET /api/documents` | 未列出 | 實際另有 **`error_message`**、**`updated_at`** 兩個欄位 |

### 4. Status code 差異

| 情境 | PROJECT_SPEC | Current code |
| --- | --- | --- |
| `/api/query` LLM 未設定 | 未記載 | **503 `LLM_NOT_CONFIGURED`** |
| `/api/query` LLM 逾時 / 錯誤 / 空回應 | 未記載 | **504** / **502** / **502** |
| `/api/retrieve` 錯誤碼 | §19.6 未列出任何 status | 實際有 422 / 500 / 503 |
| `DELETE /api/documents/{id}` | 只記載 204 / 404 | 另有 **503**（Qdrant 刪除失敗） |
| `GET /api/documents` | 未標示成功碼 | 200 |
| `top_k` 上限 | §19.7 寫死「1 至 10」 | 上限來自 `RETRIEVAL_MAX_TOP_K` 設定，預設 10 |

### 5. 錯誤格式差異

PROJECT_SPEC §19 未描述統一錯誤封包。實際上程式碼有**兩種**格式並存：
應用層的 `{"error": {code, message, details}}`，以及 FastAPI 的 `{"detail": ...}`
（其中 `detail` 在 Pydantic 驗證時為陣列、在 `top_k` 超限時為字串）。

### 6. Auth / dependency 差異

PROJECT_SPEC 未描述任何授權機制，current code 也**確實沒有身分驗證**。
唯一新增的是 `OPTIMIZER_ADMIN_ENABLED` feature flag（預設 `false`），
僅保護 4 個 state-changing POST，且**不是**身分驗證。
