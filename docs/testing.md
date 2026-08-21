# Testing

> 測試範圍、執行方式，以及**哪些數字是實際跑出來的、哪些只是收集到的測試數**。
> 語言：繁體中文。

## 1. 測試規模（v1.2）

```text
880 unit tests            # 804（v1.0 / v1.1）+ 76（v1.2 agent）
36 non-LLM integration    # 需可連線的 Qdrant
7  Playwright E2E         # 需已啟動的 API
```

### 誠實度說明（重要）

- **880 個 unit test 為最近一次實際重新執行並全數通過**，且**不需要任何外部服務**
  （Fake Embedding / Fake VectorStore / Fake LLM + 暫存 SQLite）。
- **36 個 integration 與 7 個 E2E 的數字是「目前收集到的測試數」**，不是本輪實跑結果 ——
  它們需要可連線的 Qdrant／已啟動的 API 才能執行。
- 另有 **5 個 `llm_integration` 測試**在非 LLM 驗證流程中被 **deselect**（未設定 LLM 環境變數時
  自動 skip），其中包含 v1.2 的
  [`tests/integration/test_v12_agent_llm_smoke.py`](../tests/integration/test_v12_agent_llm_smoke.py)。

> 本專案不以「測試數字」代替實際驗證。真實 Provider 的端到端結果另見
> [v1.0-runtime-verification.md](v1.0-runtime-verification.md) 與
> [v1.2-runtime-acceptance.md](v1.2-runtime-acceptance.md)。

## 2. 測試層級與外部依賴

| 層級 | 路徑 | 外部依賴 | 說明 |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | **無** | Fake Embedding / VectorStore / LLM + 暫存 SQLite 與上傳目錄 |
| Integration（非 LLM） | `tests/integration/` | **Qdrant** + 本機 sentence-transformers 模型 | 真實索引／檢索／trial runner／optimizer search／deploy alias |
| Integration（LLM） | `tests/integration/test_rag_llm_smoke.py`、`test_v12_agent_llm_smoke.py` | **外部 OpenAI-compatible Provider** | 標記 `llm_integration`；未設定 `LLM_*` 時自動 skip |
| E2E | `tests/e2e/` | 已啟動的 API（Playwright + Chromium） | Retrieval Optimizer UI |

## 3. pytest markers

定義於 [`pyproject.toml`](../pyproject.toml)：

| Marker | 意義 |
| --- | --- |
| `integration` | 需要真實 Qdrant 與 sentence-transformers 模型的整合測試 |
| `llm_integration` | 需要真實 OpenAI-compatible LLM endpoint 的 smoke test（**未設定環境變數時 skip**） |

`llm_integration` 是**明確的成本與可重現性邊界**：預設驗證流程一律以 `-m "not llm_integration"`
排除它，避免在無 API key 的環境失敗，也避免對限額 Provider 產生非預期請求。

## 4. 執行方式

### 4.1 Unit（不需外部服務）

```powershell
.venv/Scripts/python.exe -m pytest tests/unit -q
```

只跑 v1.2 agent 的部分：

```powershell
.venv/Scripts/python.exe -m pytest tests/unit -q -k v12
```

### 4.2 Integration（需 Qdrant）

```powershell
.venv/Scripts/python.exe -m pytest tests/integration -q -m "not llm_integration"
```

先確保有可連線的 Qdrant（`docker run -p 6333:6333 qdrant/qdrant`，或 compose 的 qdrant 服務），
並讓 `QDRANT_URL` 指向它。整合測試使用**獨立的 `optimizer_` 前綴 collection**，
不會寫入正式的 `factory_knowledge`。

### 4.3 LLM smoke（需外部 Provider，會實際計費／計額）

```powershell
# 需先設定 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
.venv/Scripts/python.exe -m pytest tests/integration/test_rag_llm_smoke.py -m llm_integration
.venv/Scripts/python.exe -m pytest tests/integration/test_v12_agent_llm_smoke.py -m llm_integration
```

> `test_v12_agent_llm_smoke.py` **刻意寬鬆**，因此**不是** v1.2 的唯一驗收依據；
> 真正的關卡是 [v1.2-runtime-acceptance.md](v1.2-runtime-acceptance.md) 的逐項 runtime 檢查。

### 4.4 前端 E2E（Playwright）

可重現流程；`npm ci` 由 `package-lock.json` 鎖版：

```powershell
npm ci
npx playwright install chromium
npm run test:e2e
```

需要已啟動的 API；Optimizer 分頁的示範資料來源見
[developer-guide.md](developer-guide.md#optimizer-ui-示範資料)。

## 5. 測試設計原則

沿用 v1.0 規格（[PROJECT_SPEC.md](../PROJECT_SPEC.md) §28）與 v1.1 各 Slice 的停止條件
（[v1.1-retrieval-optimizer-plan.md](v1.1-retrieval-optimizer-plan.md)）：

- **不得讓所有 API 測試都依賴真實 Embedding 模型、真實 Qdrant 與真實 LLM。**
- Unit test 使用 Fake 元件，必須可重複執行且不需網路。
- LLM 相關測試一律可 mock；真實 Provider 只用於少量 smoke，不做大量請求。
- v1.2 另補了兩個 **deterministic（不呼叫 LLM）回歸測試**，鎖定
  「hard constraints 有被攤給 LLM」與「照抄 `editable_current_config` 並只改一個欄位可通過 Validator」
  （見 [v1.2-runtime-acceptance.md §Fix Applied](v1.2-runtime-acceptance.md#fix-applied)）。

## 6. 檢索評估（不是 pytest）

檢索品質評估是獨立的評估腳本，不在 pytest 流程內：

```powershell
# 會覆寫 evaluation/results.json，請自行 git checkout 還原
.venv/Scripts/python.exe -m evaluation.evaluate_retrieval
```

結果與方法論見 [v1.0-runtime-verification.md](v1.0-runtime-verification.md#相關檢索評估)、
[`evaluation/baseline-v1.1.md`](../evaluation/baseline-v1.1.md) 與
[v1.1-retrieval-optimizer-requirements.md §8](v1.1-retrieval-optimizer-requirements.md#8-metrics)。
