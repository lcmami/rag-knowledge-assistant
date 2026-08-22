# Developer Guide

> 實作者導向的操作手冊：環境設定、專案結構、前端、v1.1 optimizer CLI、v1.2 agent CLI、
> Docker CLI profile 與 Optimizer UI 示範資料。
> 語言：繁體中文。第一次啟動請先看 README 的「快速開始」。

## 1. 環境變數

以 `.env`（**不進版控**）設定，範本見 [`.env.example`](../.env.example)。
完整清單見 [PROJECT_SPEC.md §26](../PROJECT_SPEC.md)。重點變數：

| 變數 | 說明 |
| --- | --- |
| `QDRANT_URL` | Qdrant 位址。Compose 內用 service name：`http://qdrant:6333`；本機通常 `http://localhost:6333` |
| `QDRANT_COLLECTION` | 正式 Collection，預設 `factory_knowledge` |
| `EMBEDDING_MODEL` | 預設 `BAAI/bge-small-zh-v1.5` |
| `EMBEDDING_QUERY_INSTRUCTION` | Query 專用 instruction（文件 chunk 不加） |
| `RETRIEVAL_SCORE_THRESHOLD` | 拒答門檻，預設 `0.55`（依現有資料集校準，非任意假設） |
| `RETRIEVAL_DEFAULT_TOP_K` / `RETRIEVAL_MAX_TOP_K` | Top-K 預設與上限 |
| `MAX_UPLOAD_SIZE_MB` | 上傳大小限制 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI-compatible 端點；**未設定時 `/api/query` 回 `503 LLM_NOT_CONFIGURED`** |
| `OPTIMIZER_ADMIN_ENABLED` | Optimizer 管理寫入 API 開關，**預設 `false`** |
| `OPTIMIZER_SEARCH_DB_PATH` / `OPTIMIZER_REVIEW_DB_PATH` / `OPTIMIZER_DEPLOYMENT_DB_PATH` | Optimizer 唯讀報表的三個 SQLite 來源（見 §6） |

### 安全規則

- 正式 API Key **只放本機 `.env`**（已被 `.gitignore` 忽略），不得寫入 `.env.example`、程式碼、
  前端或任何 log。
- **不要**為了讓 `/ready` 回 200 而填入假的 LLM 設定。未設定 LLM 時
  `llm_config=error` 並回 503 是**預期行為**，文件管理與 `/api/retrieve` 仍可正常使用。
- `docker compose config` 可能輸出由 `.env` 展開後的內容。驗證 Compose 設定時使用
  `docker compose config --quiet`（只回傳 exit code），並避免公開貼出完整輸出。
- 若 LLM 跑在 Docker 宿主機，Container 內應以 `host.docker.internal` 連回，
  **不可**在 Container 內用 `localhost` 指向宿主機服務。
- 更換 Embedding 模型後必須**重新 indexing**（維度／距離不相容時 `/ready` 回 503，
  系統不會默默重建或刪除資料）。

## 2. 專案結構

```
app/
  api/           health / documents / retrieve / query 路由 + optimizer 唯讀報表與 admin
  core/          config / constants / exceptions / logging
  services/      loader / splitter / embedding / indexing / retrieval / rag / llm_client / vector_store
  repositories/  document_repository（SQLite）
  schemas/       API 請求/回應模型
  prompts/       RAG prompt 組裝
  static/        Vanilla JS 前端（index.html / styles.css / app.js / optimizer.js）
  main.py        FastAPI app 與 lifespan
sample_docs/     範例文件（md / txt / pdf）
evaluation/      Retrieval 評估與 optimizer
  v11/           v1.1：chunking / retrieval / search / review / deploy / reporting
  v12/agent/     v1.2：context / planner / proposal / validator / executor / probes /
                 orchestrator / repository / cli
  datasets/ configs/ fixtures/    評估資料集、run config 與測試資料
scripts/         示範與驗證腳本（含 optimizer UI demo seeder、screenshot 擷取）
docs/            v1.0/v1.2 驗收、v1.1 需求／架構／計畫、v1.2 設計、測試與本指南
tests/           unit / integration / e2e（Playwright）
```

> [PROJECT_SPEC.md §7](../PROJECT_SPEC.md) 的目錄樹是 v1.0 原始草案，**已過時**；以本節為準。

## 3. 前端

純 HTML + CSS + Vanilla JS（[`app/static/`](../app/static/)，無 React／Vue），四個分頁：

1. **文件管理** — 上傳、列表、刪除。
2. **向量檢索** — 呼叫 `/api/retrieve`，即使 LLM 未設定仍可展示。
3. **RAG 問答** — 呼叫 `/api/query`；收到 `503 LLM_NOT_CONFIGURED` 時顯示：
   > 「LLM 服務尚未設定，目前仍可使用文件上傳與向量檢索功能。」
4. **Retrieval Optimizer** — v1.1 的唯讀報表與受控管理操作介面，讀取 `/api/optimizer/*`；
   管理寫入操作預設停用。

前端**不含任何金鑰、不偽造回答、不使用 Fake LLM 當作 Demo 回應**；來源 metadata 一律由後端組裝。
關鍵驗證、Threshold、拒答與引用組裝邏輯**不得**放在 JavaScript。

> 詳細 HTTP API（各 endpoint 的 request／response schema、錯誤碼與實際行為）見
> [api-reference.md](api-reference.md)。

## 4. v1.1 Optimizer CLI

以下指令可在**不提供 API key** 的情況下執行 optimizer 核心流程。

```bash
# Slice 0 baseline（v1.0 檢索 baseline；會覆寫 evaluation/results.json，請自行 git checkout 還原）
python -m evaluation.evaluate_retrieval

# Trial Runner / Chunking / Identifier smoke（隔離 collection，需 Qdrant）
python -m evaluation.v11.cli run --config evaluation/configs/slice2-smoke.json
python -m evaluation.v11.cli run --config evaluation/configs/slice3-chunking-smoke.json
python -m evaluation.v11.cli run --config evaluation/configs/slice4-identifier-smoke.json

# 檢索搜尋（grid／random search + constraints + Pareto + recommendation；不呼叫 LLM）
python -m evaluation.v11.search.cli --config evaluation/configs/slice5-optimizer-smoke-v2.json

# Human Review / LLM Dataset Authoring CLI
python -m evaluation.v11.review.cli --help

# 建立 UI 示範資料；不建立 Qdrant collection，也不呼叫 LLM
python -m scripts.seed_optimizer_ui_demo --output-dir evaluation/demo
```

### 4.1 LLM-assisted Dataset Authoring

以下流程用 LLM Candidate Generator 建立並發布一版新的評估資料集。指令的 flag 皆來自
`review.cli` 目前的 `--help`；`<...>` 為需自行填入的實際值（session id、candidate id、
文件 SHA-256 等），**請勿填入假值，也不要把 API key 寫進指令**。

```bash
# 1. 建立 Review Session（固定 base dataset 與 corpus hash）
python -m evaluation.v11.review.cli session create \
  --name my-review --base-dataset evaluation/datasets/<base>.json

# 2. 用 LLM Generator 依指定文件片段產生候選（需已設定 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）
#    未設定時可改用 --generator fake，或 candidate add 手動新增
python -m evaluation.v11.review.cli candidate generate \
  --session <session-id> --document-sha <document-sha> --generator llm --count 3

# 3. 檢視候選
python -m evaluation.v11.review.cli candidate list --session <session-id>
python -m evaluation.v11.review.cli candidate show --candidate <candidate-id>

# 4. 逐筆 Accept／Edit／Reject（沒有批次自動接受）
python -m evaluation.v11.review.cli candidate edit   --candidate <candidate-id> --question "..."
python -m evaluation.v11.review.cli candidate accept --candidate <candidate-id> --split validation

# 5. 確認證據：先定位候選 span，人工確認後再加為 evidence
python -m evaluation.v11.review.cli evidence find --anchor "E102" --document-sha <document-sha>
python -m evaluation.v11.review.cli evidence add  --candidate <candidate-id> \
  --document-sha <document-sha> --canonical-version <ver> --char-start <n> --char-end <n>

# 6. 檢視 coverage，並發布新版資料集（--out 已存在即拒絕）
python -m evaluation.v11.review.cli coverage show --session <session-id>
python -m evaluation.v11.review.cli dataset publish \
  --session <session-id> --version v1.1-eval-2 --out evaluation/datasets/<new>.json

# 7. 用新版資料集執行檢索搜尋（Loop B；此階段不呼叫 LLM）
python -m evaluation.v11.search.cli search --config <search.json> --db storage/optimizer.db
```

Candidate 的 `answerable`、`group`、`split`、evidence 與 `unanswerable` 確認**一律由人工在
Review 流程決定**，LLM 只提供建議值。所有候選一律以 `proposed` 狀態落庫，不會自動成為評估案例；
每次編輯、加證據或接受都會產生新的遞增 revision，**不覆寫歷史**。

### 4.2 LLM API 共用關係

`/api/query` 與 v1.1 的 LLM Candidate Generator 共用同一組 OpenAI-compatible provider 設定
（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`），且都經由同一個 `LLMClient`
（[`app/services/llm_client.py`](../app/services/llm_client.py)）呼叫 provider。

兩者用途與請求彼此獨立：

- `/api/query`：依向量檢索結果回答使用者問題。
- Candidate Generator：閱讀指定文件片段，提出評估案例候選。

它們**不共享 conversation history**，也不是同一次 HTTP request；Generator 只把本次請求的
segment 放進 prompt，**不提供整份 corpus**，且不保存 prompt 全文、response 全文或 API key。

### 4.3 Deployment Admin

Deployment Admin **預設停用**；停用時所有 write endpoint 一律回 **403**，不會建構或呼叫 service。
啟用需明確設定 `OPTIMIZER_ADMIN_ENABLED=true`，且候選設定須符合套用條件。

Apply／Rollback 的完整流程、typed confirmation、alias 原子切換與稽核設計見
[v1.1-retrieval-optimizer-architecture.md §8](v1.1-retrieval-optimizer-architecture.md#8-apply--rollback-與首次-alias-migration)
與 [v1.1-retrieval-optimizer-requirements.md §9](v1.1-retrieval-optimizer-requirements.md#9-safety-rules)。

> 詳細 HTTP API（實際註冊的 admin endpoint、request schema、error code 對照與 403／409 條件）見
> [api-reference.md](api-reference.md#optimizer--deployment-admin)。

## 5. v1.2 Agent CLI

以下指令可在**不呼叫任何 Provider** 的情況下跑完 Fake Planner 迴圈（需可連線的 Qdrant）：

```powershell
# Fake Planner（不呼叫 LLM）
python -m evaluation.v12.agent.cli optimize `
  --config evaluation/configs/v12-agent-smoke.json `
  --planner fake `
  --max-rounds 2

# 檢視匿名化 Audit（<agent_run_id> 與 <agent.db> 皆由上一步的 Run Summary 印出）
python -m evaluation.v12.agent.cli show <agent_run_id> --agent-db <agent.db>

# Real LLM：只有明確指定 --planner llm 才會呼叫 Provider
python -m evaluation.v12.agent.cli optimize `
  --config evaluation/configs/v12-agent-smoke.json `
  --planner llm `
  --max-rounds 1
```

`optimize` 的 Run Summary 會印出 `Run ID` 與 `Agent DB` 路徑，供 `show` 使用；可加
`--workdir <dir>` 固定輸出位置。真實 LLM 路徑需要下列環境（**不會**因環境存在金鑰就自動呼叫，
一律需明確 `--planner llm`）：

```text
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
Qdrant（--qdrant-url 或 QDRANT_URL）
```

> 真實 Provider 設定只放本機 `.env`（已被 `.gitignore` 忽略）；請勿在指令、log 或文件中顯示金鑰。

Agent 的設計、白名單參數空間、budget 預設值與 promotion 語義見
[v1.2-agent-design.md](v1.2-agent-design.md)。

### 建議：用獨立的 ephemeral Qdrant

v1.2 runtime 驗收採用**獨立、ephemeral 的 Qdrant 容器**（對外 port 6350、**不掛任何 volume**），
以 `--qdrant-url` 明確指向，正式 compose 容器全程停止。這樣可保證正式 `factory_knowledge`
不可能被觸碰。做法見
[v1.2-runtime-acceptance.md §Environment Isolation](v1.2-runtime-acceptance.md#environment-isolation)。

## 6. Optimizer UI 示範資料

啟動 API 後，將 seeder 產生的三個路徑設給環境變數，即可在 UI 的 **Retrieval Optimizer** 分頁
檢視示範資料：

```bash
OPTIMIZER_SEARCH_DB_PATH=evaluation/demo/optimizer-demo.search.db
OPTIMIZER_REVIEW_DB_PATH=evaluation/demo/optimizer-demo.review.db
OPTIMIZER_DEPLOYMENT_DB_PATH=evaluation/demo/optimizer-demo.deployment.db
```

報表為唯讀：UI 與 read API 以 SQLite `mode=ro` 開啟，不建立、不遷移、不修改資料，
也不輸出私人路徑。示範資料標示 `demo_fixture=true`，**不會偽造 apply 成功，也不會建立實際 alias**。

## 7. Docker CLI profile

Docker 使用者可用一次性 CLI profile（**不新增常駐 worker，無 Redis／broker，不影響
`docker compose up`**）：

```bash
docker compose --profile cli run --rm optimizer-cli \
  python -m scripts.seed_optimizer_ui_demo --output-dir storage/optimizer-demo
```

## 8. 測試

見 [testing.md](testing.md)。
