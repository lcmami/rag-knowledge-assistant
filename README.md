# Factory RAG 知識助理

可透過 **Docker Compose** 啟動的智慧製造文件 RAG 作品集 MVP，支援文件管理、向量檢索、
**真實 LLM 回答**、來源引用（Citation）及拒答機制。

支援 Markdown / TXT / PDF 文件上傳、本地中文 Embedding、Qdrant 向量檢索、雙層拒答與
Citation 驗證，並提供一個純 Vanilla JS 的簡易前端。

> **LLM 狀態**：LLM 呼叫層（OpenAI-compatible Chat Completions）已完成，並已以真實 Provider
> 完成一次端到端驗證（見 [真實驗證結果](#真實驗證結果)）。在未設定 LLM 時，`/api/query` 仍會
> 回傳明確的 `503 LLM_NOT_CONFIGURED`，而文件管理與向量檢索（`/api/retrieve`）完全可用。

---

## v1.1 — Human-in-the-loop Retrieval Optimizer

v1.1 在 v1.0 的 RAG 之上，加入一套**離線、人為把關**的檢索優化流程與其唯讀報表 UI。

> **定位（誠實）**：這是「**Factory RAG Assistant + Human-in-the-loop Retrieval Optimizer**」，
> **不是**全自主 Agent。優化器只**提案**；chunking／retrieval 設定的選擇、dataset 的擴充、
> 以及任何實際部署，都需要**人工核准**。它不會自己決定最佳設定，也不會自動套用。

### v1.1 Workflow

```
Tracked Corpus（受版控的示範語料）
  → Canonical Evidence（正規化證據 + provenance）
  → Isolated Trials（每個設定跑隔離的 evaluation run）
  → Chunking／Retrieval Search（grid／random）
  → Hard Constraints（安全門檻）
  → Pareto（非支配前緣）
  → Recommendation Lock（鎖定固定 candidate）
  → Test（在 held-out test split 上驗收）
  → Human Review（人工逐筆審 dataset 提案）
  → Safe Apply／Rollback（typed confirmation + alias 切換，預設停用）
```

### Features

- **Canonical Evidence**：gold label 綁原始文件證據 span（版本化正規化 + anchor）。
- **Versioned Dataset**：v1.1 evaluation dataset（tune／validation／test 嚴格隔離、無 group leakage）。
- **Trial Isolation**：每個 chunk／retrieval 設定跑自己的隔離 evaluation run 與 Qdrant collection。
- **Metrics**：NDCG@5、MRR、Recall@1/3/5、exact-code Top-1、unanswerable FP、duplicate、P50/P95、
  index build time、cache hit rate（可計算性與樣本粒度誠實揭露）。
- **Chunk Strategies**：current／fixed／paragraph／heading（版本化搜尋空間）。
- **Embedding Cache**：content-hash 持久化快取，同 chunk 文字跨 trial 重用。
- **Identifier-aware Retrieval**：dense ∪ identifier-filtered、exact-match boost、confusable penalty。
- **Grid／Random Search**、**Hard Constraints**、**Pareto**、**Recommendation Lock**。
- **Human Review**：LLM 只提案，接受／發布須人工逐筆，編輯永遠產生新 revision。
- **Safe Apply／Rollback**：logical alias 切換（永不改 `factory_knowledge`）、corpus/dataset hash guard、
  typed confirmation、audit trail、idempotency——且**預設停用**。
- **UI Reporting**：唯讀前端 Tab（Overview／Search／Review／Deployment）+ read-only API。

### Honest Result（目前 demo recommendation）

目前的 demo recommendation **不具備 apply 資格**，且我們刻意如實呈現，不美化：

- Current demo recommendation is **not apply-eligible**.
- The evaluation corpus is intentionally small.
- Several validation candidates **tied on quality metrics**（8 個 feasible candidate 在品質指標上完全相同）。
- Single-run latency was **not stable enough** to prove one configuration superior（選擇實為單次 P95 雜訊）。
- The **E102 same-chunk multi-identifier ambiguity remains unresolved**（`eval-e102-001` 的 first relevant
  rank 在 baseline 與 recommendation 皆為 2）。

我們**不**宣稱：自動找到全域最佳 chunking、已適用真實工廠、已正式部署、目前 candidate 可直接
apply、95% 準確率、或完整 Active Learning 已完成。

### v1.1 Commands（皆不需提交 API key 即可測試 optimizer 核心）

```bash
# Slice 0 baseline（v1.0 retrieval baseline）
python -m evaluation.evaluate_retrieval        # 註：會覆寫 evaluation/results.json，請自行 git checkout 還原

# Trial Runner / Chunking / Identifier smoke（隔離 collection，需 Qdrant）
python -m evaluation.v11.cli run    --config evaluation/configs/slice2-smoke.json
python -m evaluation.v11.cli run    --config evaluation/configs/slice3-chunking-smoke.json
python -m evaluation.v11.cli run    --config evaluation/configs/slice4-identifier-smoke.json

# Optimizer Search（grid／random + constraints + Pareto + recommendation）
python -m evaluation.v11.search.cli --config evaluation/configs/slice5-optimizer-smoke-v2.json

# Human Review CLI（正式 dataset authoring 入口；LLM 只提案，接受須人工）
python -m evaluation.v11.review.cli --help

# Deployment Admin（預設停用；停用時 write endpoint 一律 403，不建構 service）
#   啟用需明確設 OPTIMIZER_ADMIN_ENABLED=true，且 candidate 須 apply-eligible

# UI Demo Seeder（產生清楚標示的 demo fixture；不碰真實資料 / Qdrant / LLM）
python -m scripts.seed_optimizer_ui_demo --output-dir evaluation/demo

# 測試
python -m pytest tests/unit -q                                   # 全部 unit
python -m pytest tests/integration -q -m "not llm_integration"   # 非 LLM 整合（需 Qdrant）
```

前端 E2E（可重現流程；`npm ci` 由 `package-lock.json` 鎖版，`test:e2e` 對應 `playwright test`）：

```powershell
npm ci
npx playwright install chromium
npm run test:e2e
```

啟動 API 後把 seeder 產生的三個路徑設給環境變數，即可在 UI 的 **Retrieval Optimizer** Tab 展示：

```bash
OPTIMIZER_SEARCH_DB_PATH=evaluation/demo/optimizer-demo.search.db
OPTIMIZER_REVIEW_DB_PATH=evaluation/demo/optimizer-demo.review.db
OPTIMIZER_DEPLOYMENT_DB_PATH=evaluation/demo/optimizer-demo.deployment.db
```

Docker 使用者可用**一次性** CLI profile（不新增常駐 worker、無 Redis／broker、不影響 `docker compose up`）：

```bash
docker compose --profile cli run --rm optimizer-cli \
  python -m scripts.seed_optimizer_ui_demo --output-dir storage/optimizer-demo
```

### v1.1 Safety

- **Production Admin disabled by default**（`OPTIMIZER_ADMIN_ENABLED=false`）；停用時 write endpoint 不建構、不呼叫任何 service。
- **No automatic apply／rollback／alias init**——全部需人工 typed confirmation。
- **Corpus hash guard + Dataset hash guard**——candidate 與目前 corpus／dataset 不符即拒絕。
- **Typed confirmation**（`APPLY <alias> CONFIG <hash> FROM <current>` 等，後端重新驗證）。
- **Alias switch, never rename／delete**——`factory_knowledge` 為硬編碼 protected collection，永不刪除。
- **Rollback + Audit**——每次操作 append-only 稽核；idempotency key 防重複。
- **Read-only reporting**——UI／read API 以 SQLite `mode=ro` 開啟，不建立、不遷移、不改資料，不輸出私人路徑。
- **Demo fixture isolation**——demo 資料明確標示 `demo_fixture=true`，不偽造 apply 成功、不建立真實 alias。

### v1.1 Architecture

```mermaid
flowchart LR
    subgraph Offline[離線 · CLI / Optimizer（人為把關）]
        CORPUS[Tracked Corpus] --> CANON[Canonical Evidence]
        CANON --> TRIALS[Isolated Trials]
        TRIALS --> SEARCH[Chunk/Retrieval Search]
        SEARCH --> CONS[Hard Constraints] --> PARETO[Pareto] --> LOCK[Recommendation Lock]
        LOCK --> TEST[Test Split 驗收]
        REVIEW[Human Review CLI] --> DATASET[(Versioned Dataset)]
    end

    subgraph DBs[獨立 Read-only SQLite（分開配置）]
        SDB[(Search DB)]
        RDB[(Review DB)]
        DDB[(Deployment DB)]
    end

    subgraph Online[線上 · FastAPI]
        ROAPI["/api/optimizer/* (read-only)"]
        ADMIN["/api/optimizer admin (預設停用)"]
    end

    subgraph UI[前端 Tab · Retrieval Optimizer]
        OV[Overview] --- SR[Search Results]
        HR[Human Review] --- DEP[Deployment]
    end

    TEST --> SDB
    REVIEW --> RDB
    ADMIN -. alias switch（人工核准後）.-> DDB
    SDB & RDB & DDB --> ROAPI --> UI
    ADMIN -. write（typed confirmation）.-> QDRANT[(Qdrant alias)]
```

### v1.1 Screenshots（demo fixture）

<p align="center">
  <img src="docs/screenshots/v1.1-optimizer-overview.png" alt="Optimizer Overview（demo）" width="420" />
  <img src="docs/screenshots/v1.1-search-comparison.png" alt="Search Comparison + Per-case Diff（demo）" width="420" />
</p>
<p align="center">
  <img src="docs/screenshots/v1.1-human-review.png" alt="Human Review（demo）" width="420" />
  <img src="docs/screenshots/v1.1-deployment-blocked.png" alt="Deployment — Apply blocked（demo）" width="420" />
</p>

> 截圖皆使用 demo fixture（畫面標示 Demo），不含私人路徑、API key 或文件全文；
> 不表示任何 apply 已成功。

### v1.1 Limitations

- Small dataset；coverage gaps（validation 缺 identifier-bearing unanswerable）。
- Latency noise：P95 hard constraint 低於本機量測雜訊底限，recommendation 實由單次 latency 決定。
- Demo corpus only——綁受版控示範語料，非真實工廠語料。
- No RBAC／login；single-machine local admin。
- No background worker／queue（無 Redis／Celery）。
- No sparse／BM25／hybrid／reranker；no full active sampling；**no production deployment**。

---

## 架構

```mermaid
flowchart TD
    subgraph Client[前端 · Vanilla JS / StaticFiles]
        UI[文件管理 · 向量檢索 · RAG 問答]
    end

    subgraph API[FastAPI]
        H["/health · /ready"]
        D["/api/documents (上傳 / 列表 / 刪除)"]
        R["/api/retrieve (不呼叫 LLM)"]
        Q["/api/query (RAG)"]
    end

    subgraph Core[核心服務]
        LOAD[DocumentLoader / PdfLoader]
        SPLIT[TextSplitter · Chunking]
        EMB[EmbeddingService<br/>BAAI/bge-small-zh-v1.5]
        IDX[IndexingService]
        RET[RetrievalService<br/>Threshold 過濾]
        RAG[RagService<br/>雙層拒答 + Citation 驗證]
        LLM[LLMClient<br/>OpenAI-compatible]
    end

    subgraph Store[持久化]
        SQLITE[(SQLite<br/>文件 metadata)]
        QDRANT[(Qdrant<br/>向量)]
        FILES[(Upload storage)]
        HF[(HF model cache)]
    end

    UI --> H & D & R & Q
    D --> LOAD --> SPLIT --> EMB --> IDX
    IDX --> SQLITE & QDRANT & FILES
    R --> RET --> EMB & QDRANT
    Q --> RAG --> RET
    RAG --> LLM
    EMB --> HF
```

`LLM` 為 OpenAI-compatible 端點，**不綁定特定供應商**；未設定時 `/api/query` 回 `503 LLM_NOT_CONFIGURED`。

### 上傳與檢索資料流

```
上傳 Markdown / TXT / PDF
  → SQLite 文件 Registry
  → DocumentLoader / PdfLoader 解析
  → TextSplitter 切 Chunk
  → EmbeddingService 產生向量
  → Qdrant 索引
  → Retrieval（Top-K + Threshold 過濾）
  → [RagService：雙層拒答 → 真實 LLM → Citation 驗證 → 組裝 Sources]
  → Response
```

---

## 技術棧

| 層面 | 技術 |
| --- | --- |
| Web / API | FastAPI + Uvicorn |
| 前端 | HTML + CSS + Vanilla JavaScript（FastAPI StaticFiles，無 React/Vue） |
| Embedding | `BAAI/bge-small-zh-v1.5`（512 維，本地 sentence-transformers） |
| 向量資料庫 | Qdrant（Cosine） |
| 文件 metadata | SQLite |
| PDF 解析 | PyMuPDF |
| LLM 介面 | OpenAI Python SDK（OpenAI-compatible，可指向 OpenAI / Groq / vLLM / Ollama 等） |
| 容器化 | Docker Compose（FastAPI + Qdrant + 持久化 volumes） |

---

## 快速開始

### 方式一：Docker Compose（建議）

```powershell
# 1. 建立 .env（可先複製範本；未設定 LLM 時檢索仍可用）
Copy-Item .env.example .env

# 2.（可選）驗證 compose 檔語法，且**不**輸出展開後的設定
docker compose config --quiet

# 3. 建置並啟動 FastAPI + Qdrant
docker compose up --build -d
docker compose ps

# 4. 開啟前端
#   http://localhost:8000/
```

> **安全**：Avoid publishing the output of `docker compose config` when a local
> `.env` contains secrets, because Compose may print interpolated values.
> Use `docker compose config --quiet` for validation.（`--quiet` 只回傳 exit code，不印出任何展開內容。）

- `GET /health` → 200
- `GET /ready` → 依賴就緒檢查（設定 LLM 後 `llm_config=ok`；未設定時 `llm_config=error` 並回 503，屬**預期行為**）。
- 文件上傳、列表、刪除、`/api/retrieve` 皆可正常使用。
- `/api/query` 在未設定 LLM 時回 `503 LLM_NOT_CONFIGURED`。

> **不要**為了讓 `/ready` 回 200 而填入假的 LLM 設定。

### 方式二：本地開發（venv）

需自行啟動一個 Qdrant（例如 `docker run -p 6333:6333 qdrant/qdrant`），並將
`.env` 的 `QDRANT_URL` 指向它（本地通常為 `http://localhost:6333`）。

```powershell
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

---

## 環境變數

以 `.env`（不進版控）設定，範本見 [.env.example](.env.example)。重點：

| 變數 | 說明 |
| --- | --- |
| `QDRANT_URL` | Qdrant 位址。Compose 內用 service name：`http://qdrant:6333` |
| `QDRANT_COLLECTION` | 正式 Collection，預設 `factory_knowledge` |
| `EMBEDDING_MODEL` | 預設 `BAAI/bge-small-zh-v1.5` |
| `RETRIEVAL_SCORE_THRESHOLD` | 拒答門檻，預設 `0.55`（依現有資料集校準） |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI-compatible 端點；**未設定時 `/api/query` 回 503** |

> **安全**：真實 API Key 只放本機 `.env`（已被 `.gitignore` 忽略），
> 不得寫入 `.env.example`、程式碼、前端或任何 log。

若 LLM 跑在 Docker 宿主機，Container 內應以 `host.docker.internal` 連回，
**不可**在 Container 內用 `localhost` 指向宿主機服務。

---

## API 一覽

| Method | Path | 說明 |
| --- | --- | --- |
| GET | `/health` | Process 存活檢查 |
| GET | `/ready` | 依賴就緒檢查（qdrant / collection / embedding / metadata_db / llm_config） |
| POST | `/api/documents` | 上傳並索引文件，成功回 201 |
| GET | `/api/documents` | 列出所有文件 |
| DELETE | `/api/documents/{id}` | 刪除文件與其全部向量，成功回 204 |
| POST | `/api/retrieve` | 純向量檢索（**不呼叫 LLM**） |
| POST | `/api/query` | RAG 問答（需 LLM；未設定回 `503 LLM_NOT_CONFIGURED`） |
| GET | `/` · `/static/*` | 簡易前端 |

錯誤統一格式：`{"error": {"code": "...", "message": "...", "details": {}}}`。

---

## 前端

純 HTML + CSS + Vanilla JS（[app/static/](app/static/)），三個分頁：

1. **文件管理** — 上傳、列表、刪除。
2. **向量檢索** — 呼叫 `/api/retrieve`，即使 LLM 未設定仍可展示。
3. **RAG 問答** — 呼叫 `/api/query`；收到 `503 LLM_NOT_CONFIGURED` 時顯示：
   > 「LLM 服務尚未設定，目前仍可使用文件上傳與向量檢索功能。」

前端不含任何金鑰、不偽造回答、不使用 Fake LLM 當作 Demo 回應；來源 metadata 一律由後端組裝。

---

## 已完成能力

- PDF、Markdown、TXT 文件處理
- 文件 SHA-256 去重
- SQLite 文件 Registry
- Chunking 與 Token hard limit
- `BAAI/bge-small-zh-v1.5` 本地 Embedding
- Qdrant 向量搜尋
- Top-K 與 Threshold 過濾
- OpenAI-compatible LLM Client
- 真實 LLM RAG 回答
- Citation 驗證
- 無效 Citation fail-closed
- Threshold 前置拒答
- 語意拒答流程
- FastAPI
- 靜態前端
- Docker Compose
- Health／Readiness
- Unit／Integration Tests
- Retrieval Evaluation

---

## 真實驗證結果

以下為一次以真實 OpenAI-compatible Provider 完成的端到端驗證紀錄。Provider / Model 僅作為
**「已驗證環境」的紀錄**，並未寫死於應用程式架構中；系統可指向任一 OpenAI-compatible 端點。

```text
Verified provider: Groq OpenAI-compatible API
Verified model: llama-3.3-70b-versatile
```

真實 RAG 驗證：

```text
E102（設備出現 E102 警報時，應該如何處理？）：
- HTTP 200
- 真實模型回答（非 fallback）
- Citation [S1] [S2] 有效
- Sources 皆對應 equipment_alarm_manual.md

台灣首都：
- Threshold 前置拒答
- Sources 為空
- generation_ms = 0
- 未呼叫 LLM

CX-500 市場售價：
- 拒答
- 沒有猜測價格
- 本次由第一層 Threshold 拒答
```

延遲（**單次實際樣本，非穩定 Benchmark**）：

```text
Retrieval：約 98 ms
Generation：約 2047 ms
Total：約 2145 ms
```

---

## Retrieval Evaluation

檢索評估（20 題題庫：12 可回答 / 8 不可回答，raw threshold=0.0，top_k=10）：

```text
Hit@1 = 1.0
Hit@3 = 1.0
Section Hit@1 = 0.909
Page Hit@1 = 1.0
```

完整結果與 threshold sweep 見 [evaluation/results.json](evaluation/results.json)。

---

## 測試結果

以下為本次收尾實際重新執行的結果：

```powershell
# 單元 + 應用層（Fake LLM），不需外部服務
.venv/Scripts/python.exe -m pytest -m "not integration and not llm_integration"
# → 75 passed, 9 deselected

# 整合測試（需真實 Qdrant）
.venv/Scripts/python.exe -m pytest -m integration
# → 5 passed, 79 deselected

# 真實 LLM smoke test marker（未設定 LLM 環境變數時自動 skip）
.venv/Scripts/python.exe -m pytest -m llm_integration
```

另外對 `/api/query` 執行了一次最小真實 LLM smoke test（E102，見上方 [真實驗證結果](#真實驗證結果)），
回傳 HTTP 200、有效 Citation、Source 對應正式文件、無 `<think>` 內容。

---

## 已知限制

1. 目前是純向量 Retrieval baseline（尚未加入 keyword / hybrid / reranker）。
2. 語意相近但代碼不同的警報文件，可能干擾精確代碼查詢的排序。
3. Chunking 與 Retrieval 品質仍可在後續版本調校。
4. 真實 Provider 已驗證正常回答與第一層 Threshold 拒答。
5. 第二層語意拒答已由測試覆蓋，但本次真實 CX-500 查詢在第一層 Threshold 就被拒，
   因此尚未以真實 Provider 實際走過第二層語意拒答分支。
6. 目前不處理會輸出 `<think>` 的 reasoning model；已驗證模型為 `llama-3.3-70b-versatile`。

---

## Demo 圖片

<p>
  <img src="docs/screenshots/demo-run/01-home.png" alt="首頁" width="420" />
  <img src="docs/screenshots/demo-run/04-retrieval-e102.png" alt="E102 向量檢索" width="420" />
</p>
<p>
  <img src="docs/screenshots/demo-run/05-retrieval-no-result.png" alt="無結果拒答" width="420" />
  <img src="docs/screenshots/demo-run/08-real-rag-e102.png" alt="真實 LLM RAG（E102）" width="420" />
</p>

---

## 專案結構

```
app/
  api/           health / documents / retrieve / query 路由
  core/          config / constants / exceptions / logging
  services/      loader / splitter / embedding / indexing / retrieval / rag / llm_client / vector_store
  repositories/  document_repository（SQLite）
  schemas/       API 請求/回應模型
  prompts/       RAG prompt 組裝
  static/        Vanilla JS 前端（index.html / styles.css / app.js）
  main.py        FastAPI app 與 lifespan
sample_docs/     範例文件（md / txt / pdf）
evaluation/      Retrieval 評估腳本與結果
tests/           unit / integration
```
