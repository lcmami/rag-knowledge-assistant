# 外部結構化 API Corpus 來源 — 實作計畫（MOJ 全國法規資料庫）

> 語言：繁體中文。程式名稱、class / function / config field、command 保留英文。
>
> **範圍：** 讓既有的 **offline evaluation pipeline** 能夠可重現地取得一份大型、真實、且具混淆性的外部 corpus。這是 evaluation-enablement infrastructure，**不改變任何 production 行為**。
>
> **目前唯一批准的 execution scope 是 Slice A–D。** Slice E（Legal Identifier Support）僅為 pilot 完成後的可能決策，**不屬於目前 scope**，本文不設計、不實作。

相關文件：
- [docs/v1.1-retrieval-optimizer-architecture.md](v1.1-retrieval-optimizer-architecture.md) — v1.1 optimizer 架構、corpus/trial 隔離
- [docs/v1.2.1-rerank-design.md](v1.2.1-rerank-design.md) — Cross-Encoder rerank 設計與結果
- [docs/testing.md](testing.md) — 測試分層、marker、執行指令
- [docs/developer-guide.md](developer-guide.md) — 環境變數、目錄結構、安全規則

---

## 1. Goal

讓既有 RAG evaluation 系統能夠：

1. 從外部結構化 API（MOJ 全國法規資料庫）取得顯著更大的真實世界 corpus；
2. 正規化為專案既有的文件表示（磁碟上的 Markdown 檔）；
3. 產生**可重現、可稽核、以 hash 版本化**的 corpus snapshot；
4. 以既有 chunking / embedding / Qdrant / retrieval 管線完成 ingest 與 index；
5. 產出 corpus statistics 並執行初步 retrieval sanity check。

用以回答：**「現有 retrieval / identifier-aware / reranker 系統，在更大、更混淆的真實 corpus 上是否仍表現良好？」**

本階段**不**建立最終 hard evaluation dataset。

---

## 2. Non-Goals

明確排除，不得在 Slice A–D 出現：

HTML crawler · 通用 web crawler · PDF ingestion · OCR · Vision / VLM · 影像理解 · multimodal embeddings · table / image extraction · BM25 · sparse retrieval · RRF · reranker optimization · ONNX · OpenVINO · quantization · GPU optimization · 新的 LLM answer-generation 行為 · production API 變更 · 自動 scheduler / background sync · admin UI。

同時明確禁止新增下列抽象（§4.2 詳述）：provider registry · plugin framework · generic ingestion framework · scheduler · background worker · crawler abstraction · PDF/OCR/Vision abstraction。

亦不包含：修改 `IdentifierExtractor`（見 §13 Q1）、建立最終 hard evaluation questions。

---

## 3. Current Repository Integration Points

以下為實際盤點結果（repo-relative 路徑）。

### 3.1 關鍵發現：source adapter 接點已經存在

[`evaluation/v11/hashing.py`](../evaluation/v11/hashing.py) 的 `collect_corpus_files(corpus_paths, base_dir)` 會遞迴走訪目錄，收錄 `SUPPORTED_EXTENSIONS = (".md", ".markdown", ".txt", ".pdf")`；而 `RunConfig.corpus_paths`（[`evaluation/v11/run_models.py`](../evaluation/v11/run_models.py)）就是一組目錄字串。

其後所有環節 — canonical builder、chunking strategy、embedding cache、Qdrant upsert、retrieval、metrics — 全部由此驅動。

**因此 pipeline 內部不需要任何新抽象。** 新程式碼只是一個「生產者」，把正規化後的 Markdown 寫進一個 corpus 目錄。`app/**` 與 `evaluation/v11/**` 完全不動。

### 3.2 直接沿用（不修改）

| 用途 | 位置 | 使用方式 |
|---|---|---|
| Corpus 檔案列舉 | `evaluation/v11/hashing.py::collect_corpus_files` | snapshot 目錄被指定為 `corpus_paths` |
| **Corpus fingerprint** | `evaluation/v11/hashing.py::corpus_hash` / `corpus_manifest` | 直接呼叫，**不自寫 hash** |
| Canonical JSON | `evaluation/v11/hashing.py::canonical_json` | manifest 序列化 |
| Git 狀態 | `evaluation/v11/hashing.py::git_state` | manifest provenance |
| Offline index 建立 | `evaluation/v11/real_services.py::RealTrialServiceProvider.build_index` | 吃 `list[tuple[rel, abs]]` |
| Markdown 載入 | `app/services/document_loader.py::load_markdown` | 產生 `DocumentElement`（含 `section`） |
| Canonical 文件 | `app/services/canonical_builder.py::build_canonical_document` | 每 element 一 segment |
| Chunking | `app/services/text_splitter.py::TextSplitter` / `evaluation/v11/chunking/` | 依 `(page, section)` 分組 |
| Chunk / payload 契約 | `app/models/chunk.py::Chunk.to_payload()` | 13 個 payload key，單一真相來源 |
| Embedding | `app/services/embedding_service.py::EmbeddingService` | `BAAI/bge-small-zh-v1.5`，dim 384 |
| Qdrant | `app/services/vector_store.py::VectorStore` | `ensure_collection` / `upsert_chunks` |
| Retrieval strategy | `evaluation/v11/retrieval/` | dense / identifier_aware |
| Rerank | `evaluation/v11/retrieval/rerank_stage.py` | 由 `RetrievalConfig.rerank_strategy` 開關 |
| Metrics | `evaluation/v11/retrieval_metrics.py::evaluate_trial` | Slice D 之後才用 |
| 測試隔離 | `tests/integration/conftest.py` | `unique_token`、`wait_collection_absent`、`drop_collections` |

> **注意：** `evaluation/runner.py::_corpus_manifest` 是較舊、較弱的 hash 實作（sequential digest），**不得**沿用。可重現性的單一真相來源是 `evaluation/v11/hashing.py`。

### 3.3 既有 Markdown loader 的關鍵行為（決定 §5.5 的正規化表示）

[`app/services/document_loader.py::load_markdown`](../app/services/document_loader.py) 的實測行為：

- Regex 為 `^(#{1,6})\s+(.*)$` — **任何層級**的 heading（`#` 到 `######`）都會：
  1. 把 `current_section` 設為該標題文字；
  2. **額外產生一個 `element_type=ELEMENT_HEADING` 的獨立 element**。
- 其後的段落沿用該 `current_section`，直到下一個 heading。
- `TextSplitter` 依 `(page, section)` 分組，overlap 不跨 section。

**推論：** 只要把章節（`ArticleType == "C"`）寫成任何 Markdown heading，它就會成為獨立 section 與獨立 element，也就會成為可被視為獨立法律條文 evidence 的單位。這是不被允許的（見 §5.5 與 §10 Slice A 的 C1 要求）。

### 3.4 慣例

- **Offline config 一律為 JSON**，放在 `evaluation/configs/`（受追蹤資產）；**不使用 `.env`**。
- **CLI 慣例**：`argparse` + `build_parser()` + subparsers + `set_defaults(func=...)` + `main(argv=None) -> int` + `raise SystemExit(main())`，以 `python -m ...` 執行。
- **測試慣例**：unit 測試絕不碰網路 / 模型 / Qdrant；repo 無任何 HTTP mocking library（無 `respx` / `responses` / `httpx.MockTransport`），慣例是 **constructor injection + 手寫 fake**。marker 以 module-level `pytestmark` 套用。
  - 實測 `pyproject.toml` **未設定 `addopts`，亦未啟用 `--strict-markers`**，故未註冊的 marker 只產生警告、不會失敗。
- **文件慣例**：`# 標題` → `>` blockquote 範圍聲明 → 相關文件連結 → `---` → 編號 `## N.` 章節；必須自陳 Limitations。新增 `docs/*.md` 需在 `README.md` 的 `## Documentation` 表格加一列。
- **`.gitignore` 慣例**：每個產物家族一條明確規則，附註 slice / 條款來源。

---

## 4. Proposed Minimal Architecture

### 4.1 資料流

```
MOJ Open API (單一 bulk ZIP)
        ↓  moj_law.py        fetch / 解壓 / 解析 / 決定性挑選
   LawRecord（記憶體）
        ↓  normalize.py      一部法規 → 一個 Markdown 檔
   NormalizedDocument
        ↓  snapshot.py       原子寫入 + manifest + corpus_hash
   snapshot 目錄（normalized/*.md）
        ↓  ← 既有 corpus_paths 接點，以下全部不改 →
   collect_corpus_files → load_document → build_canonical_document
        → TextSplitter → EmbeddingService → VectorStore(Qdrant)
        → retrieval strategy → rerank stage → metrics
```

### 4.2 新增模組

僅 7 個檔案。每個檔案都有明確且**目前實際需要**的責任；不為未來功能預建空泛抽象。

```
evaluation/corpus/
  __init__.py
  models.py       Slice A：LawRecord / NormalizedDocument
                  Slice B：SnapshotManifest（不得在 Slice A 提前建立）
                  被 moj_law / normalize / snapshot / stats 共用，避免循環 import。
  moj_law.py      唯一的 provider：HTTP fetch、ZIP 解壓、utf-8-sig 解碼、
                  JSON 解析、決定性挑選（filter + 排序 + 上限）。
  normalize.py    LawRecord -> Markdown 文字 + 每文件 metadata。純函式，無 I/O。
  snapshot.py     Slice B：原子寫入 normalized/ 與 manifest.json；呼叫既有 corpus_hash。
  stats.py        Slice D：由 snapshot 產生 corpus statistics 報告。
  cli.py          Slice B/D：python -m evaluation.corpus.cli fetch | snapshot | stats
  fixtures/chlaw_sample.json   真實法規子集，受追蹤，供 offline 測試。
```

**明確不新增**（Owner 指示）：

- ❌ `source.py` / `ExternalCorpusSource` Protocol — 目前只有一個 provider，Protocol 屬過早抽象。
- ❌ `providers/` 子套件、provider registry、plugin framework。
- ❌ generic ingestion framework、scheduler、background worker、crawler / PDF / OCR / Vision 抽象。

未來若真的出現第二個 provider，接縫是 `normalize.py` 的輸入型別（`LawRecord`）；屆時再抽 Protocol 是小幅重構，現在不做。

### 4.3 修改的既有檔案（極少，且皆為附加）

| 檔案 | 變更 | Slice |
|---|---|---|
| `.gitignore` | 忽略 `evaluation/corpora/*/*/raw/`，附 slice 註解 | B |
| `pyproject.toml` | 註冊 `external_api` marker（繁體中文說明，比照現有兩則） | 首次新增 external API test 的 slice（**非 Slice A**） |
| `README.md` | `## Documentation` 表格新增一列 | D |

### 4.4 不得修改

`app/**` 全樹（特別是 `identifier_extractor.py`、`text_splitter.py`、`document_loader.py`、`vector_store.py`、`api/query.py`、`api/retrieve.py`）、`evaluation/v11/**` 全樹（特別是 `hashing.py`、`retrieval/**`、`chunking/**`）、`evaluation/datasets/**`、既有的 `evaluation/configs/*.json`、`evaluation/runner.py`。

---

## 5. External Source Contract

### 5.1 已驗證的 API 事實（規劃期實際請求一次確認）

`GET https://law.moj.gov.tw/api/Ch/Law/JSON`

| 項目 | 實測值 |
|---|---|
| 認證 | 無（無 API key、無 token） |
| 回應 | `application/octet-stream`，ZIP，約 5.8 MB |
| ZIP 成員 | `ChLaw.json`（26 MB）、`schema.csv`、`manifest.csv` |
| 編碼 | **UTF-8 with BOM** → 必須 `utf-8-sig` 解碼 |
| 頂層結構 | `{"UpdateDate": "...", "Laws": [...]}` |
| 法規數 | 1,346 |
| **Pagination** | **無**。單一 bulk 下載，無 cursor、無增量端點 |

授權：Open Government Data License, version 1.0（OGDL-Taiwan-1.0）。

> **Slice D 實測確認（2026-08）：** 上表全部欄位與 §5.2 / §5.3 的數值皆與規劃期記錄**完全相符**（1,346 部 / 47,257 條 type-`A` / 4,062 筆 type-`C` / 322 部已廢止），**無契約漂移**。實際回應為 6,115,185 bytes。此後由 `tests/integration/test_corpus_external_api.py`（marker `external_api`，預設不執行）持續偵測。

`schema.csv` 為來源自帶的欄位字典，實作時一併保存於 `raw/` 以利稽核。

### 5.2 Law record 欄位（實測）

`LawLevel`, `LawName`, `LawURL`, `LawCategory`, `LawModifiedDate`, `LawEffectiveDate`, `LawEffectiveNote`, `LawAbandonNote`, `LawHasEngVersion`, `EngLawName`, `LawAttachements[]`, `LawHistories`, `LawForeword`, `LawArticles[]`

`LawArticles[]` 元素：`{ArticleType, ArticleNo, ArticleContent}`
- `ArticleType`：`"A"` = 條文（47,257 筆）、`"C"` = 編章節標題（4,062 筆）
- `ArticleNo` 型態：`第 N 條`（43,830）、`第 N-N 條`（3,399）、純數字 `N`（28）；type-`A` 無空值
- **type-`C` 的 `ArticleNo` 一律為空字串**（實測 4,062 筆全數為空）
- **type-`C` 的巢狀深度以行首空白編碼，且為 3 的倍數**：0 空白（71 筆）、3 空白（3,248 筆，如 `第 一 章 總綱`）、6 空白（684 筆，如 `第 一 節 省`）、9 空白（47 筆）、12 空白（12 筆）。故 `depth = leading_spaces // 3`。

`LawAttachements[]` 元素：`{FileName, FileURL}`，皆為 PDF → **正規化時丟棄**（PDF 明確 out of scope）。

### 5.3 語料規模與混淆性（實測）

| 指標 | 值 |
|---|---|
| 條文總字數 | 5,909,118 |
| 條文長度 | min 4 / median 80 / p90 277 / p99 654 / max 12,383 |
| **已廢止法規** | **322 / 1,346**（`LawAbandonNote` 非空） |
| 近似標題字幹碰撞 | 26 組（如 `內政部警政署組織法` vs `…組織條例`） |
| 共用 6 字標題前綴 | 193 部 |
| `pcode` 可抽取 / 唯一 | 1,346 / 1,346，全數唯一 |
| `LawName` 重複 | 0 |

已廢止法規與近似標題正是我們想測的混淆來源，**予以保留**並於 metadata 標記。

### 5.4 模組契約（以函式簽章定義，非 ABC/Protocol）

```
# moj_law.py
def fetch_raw(config, *, fetcher=None, sleeper=None) -> bytes
    # fetcher: Callable[[str, float], bytes]，預設為 httpx 實作；測試注入 fake
    # sleeper: Callable[[float], None]，預設 time.sleep；測試注入以避免真實等待
def parse_raw(raw_zip: bytes, *, max_skip_ratio=0.05)
        -> tuple[str, list[LawRecord], list[SkippedRecordDiagnostic]]
    # 回傳 (source_update_date, valid_records, skipped_record_diagnostics)
    # SkippedRecordDiagnostic = TypedDict{index: int, law_name: str|None, reason: str}
    # 逐筆跳過是有損的：Slice B manifest 需要可機器讀取的 skipped_records 數量
    # 與 reason 分佈，才能分辨「來源本來就這樣」與「來源契約悄悄改變了」。
def select(records, config) -> list[LawRecord]
    # 決定性：先套 filter，再依 pcode 升冪排序，再取前 max_laws 筆

# normalize.py
def to_markdown(record: LawRecord) -> NormalizedDocument   # 純函式，無 I/O
```

`LawRecord` 為來源欄位的忠實對映，**不臆造來源沒有的欄位**；`pcode` 由 `LawURL` 的 query 參數解析而得，解析失敗即為 malformed record（見 §8）。

### 5.5 正規化後的文件結構（C1 決議）

一部法規 → 一個 `<pcode>.md`。

```markdown
# {LawName}

- 法規類別：{LawCategory}
- 法規位階：{LawLevel}
- 最後修正日期：{LawModifiedDate}
- 廢止狀態：{is_repealed}
- 來源網址：{LawURL}

{LawForeword}（若存在）

## {chapter_path}｜{ArticleNo}

{ArticleContent}
```

**`ArticleType == "C"` 的處理（關鍵）：**

因 §3.3 已證實任何 Markdown heading 都會成為獨立 section 與獨立 element，**C 一律不輸出為 heading**。改採：

- 以 `depth = leading_spaces // 3` 維護一個章節路徑堆疊；遇到新的 C 時，截斷至該深度後推入。
- 章節路徑僅**併入其後每一條 A 條文自己的 `##` heading**，形式為 `{chapter_path}｜{ArticleNo}`（多層以 `｜` 連接，如 `第一章 總綱｜第 1 條`；`第一章 地方制度｜第一節 省｜第 112 條`）。
- 無章節脈絡時退化為單純 `## {ArticleNo}`。

如此同時滿足：
1. C 從不成為獨立 section／獨立 element → **不可能被視為獨立法律條文 evidence**；
2. C 的章節 context **完整保留**在其後 A 條文的 section 名稱中；
3. **零 loader 修改**；
4. `TextSplitter` 依 `(page, section)` 分組後，每條文各成一組 → 短條文一條一 chunk。

`LawAttachements` 於 Slice A 明確忽略、不下載。`is_repealed`（由 `LawAbandonNote` 是否非空決定）保留於 metadata 與 Markdown 標頭。

---

## 6. Snapshot and Reproducibility Design

### 6.1 目錄配置

```
evaluation/corpora/moj-law/<snapshot_id>/
  manifest.json        ← COMMITTED（約 2 KB）
  normalized/*.md      ← COMMITTED（100 部法規約 1.3 MB）
  raw/ChLaw.zip        ← GITIGNORED（5.8 MB）
  raw/ChLaw.json       ← GITIGNORED（26 MB）
  raw/schema.csv       ← GITIGNORED（來源自帶欄位字典，稽核用）
```

### 6.2 snapshot_id（Correction 1）

```
snapshot_id = moj-chlaw-<UpdateDate:YYYYMMDD>-<selection_hash[:8]>-<raw_sha256[:8]>
```

**必須包含 `raw_sha256`。** 理由：`UpdateDate` 不足以單獨保證來源內容 identity — 若來源 bytes 改變但 `UpdateDate` 未同步更新，**不得**產生相同的 snapshot_id。manifest 仍保留**完整**的 `raw_sha256`（64 字元），id 中僅取前 8 字元作為可讀短碼。

### 6.3 Raw 與 Normalized 政策

- **Raw**：保留於本機供稽核與重跑正規化，但**不入庫**（體積）。以 SHA-256 記入 manifest，故仍可驗證。
- **Normalized**：入庫。evaluation 只依賴 normalized snapshot，**不直接依賴 live API**，因此跨執行完全可重現。

### 6.4 決定性序列化

- Markdown 一律以 `\n` 換行、UTF-8 無 BOM 寫出（避免 repo `core.autocrlf=true` 造成的 hash 漂移）。
- 檔名 = `<pcode>.md`（ASCII，跨 OS 安全，且與唯一識別碼一致）。
- `manifest.json` 以 `evaluation/v11/hashing.py::canonical_json` 序列化（key 排序、穩定 separators）。
- 挑選順序固定為 pcode 升冪，與來源陣列順序無關。

### 6.5 Corpus hash

**直接呼叫既有 `evaluation/v11/hashing.py::corpus_hash([snapshot/normalized])`。** 該實作已為 canonical（排序後 POSIX 相對路徑 + 內容 SHA-256，不含絕對路徑 / mtime / OS 分隔符），且已被 `deploy/readiness.py` 與 `eligibility.py` 信任。**不另寫第二套 hash。**

### 6.6 Provenance metadata（`manifest.json` 欄位）

| 欄位 | 說明 |
|---|---|
| `provider` | `"moj_law"` |
| `source_endpoint` | 實際請求的 URL |
| `source_update_date` | 來源 `UpdateDate` 原字串 |
| `fetched_at` | ISO-8601 UTC 抓取時間 |
| `raw_sha256` / `raw_bytes` | 原始 ZIP 的**完整** SHA-256 與位元組數 |
| `selection` | 完整挑選設定（filter / 上限 / 排序） |
| `selection_hash` | 挑選設定的 canonical hash |
| `source_record_count` | 來源法規總數 |
| `selected_record_count` | 挑選後法規數 |
| `normalized_document_count` | 產出 `.md` 檔數 |
| `article_count` | type-`A` 條文總數 |
| `skipped_record_count` / `skipped_record_reason_counts` / `skipped_records` | 被跳過的**來源 record** 筆數、原因統計與逐筆明細 |
| `skipped_article_count` | 已挑選的 record 中，因正規化規則而未輸出的 type-`A` 條文總數 |
| `corpus_hash` | 由 `hashing.corpus_hash` 產生 |
| `git_state` | 由 `hashing.git_state()` 產生 |
| `schema_version` | manifest 自身 schema 版本 |

### 6.7 入庫與否

| 產物 | 政策 |
|---|---|
| `manifest.json` | **入庫** |
| `normalized/*.md` | **入庫**（見下方 size trigger） |
| `raw/**` | **gitignore** |
| `evaluation/corpus/fixtures/chlaw_sample.json` | **入庫**（測試資產） |
| `evaluation/configs/corpus-moj-law-pilot.json` | **入庫**（受追蹤 config 資產） |
| statistics 報告 | **入庫**（小型 Markdown / JSON，屬證據文件） |

> **Size trigger（Owner Decision Q2）：** 100-law snapshot（估計約 1.3 MB）允許 commit。但**任何單一 normalized snapshot 預估或實際超過約 10 MB，在 commit 前必須重新取得 Repository Owner 決定**。不得因目前允許 1.3 MB，即推論完整 1,346-law（約 18 MB）corpus 亦自動允許入庫。

---

## 7. Configuration

單一 JSON 檔：`evaluation/configs/corpus-moj-law-pilot.json`（受追蹤，Slice B 建立）。**不含任何 secret / API key — 此 API 不需認證，亦不得為此新增 `.env` 欄位。**

### 7.1 Provider / source 設定

| 欄位 | 預設 | 說明 |
|---|---|---|
| `provider` | `"moj_law"` | 目前唯一值 |
| `endpoint` | `https://law.moj.gov.tw/api/Ch/Law/JSON` | 僅 Ch/Law；Ch/Order 不在本階段 |
| `timeout_seconds` | `120` | 單次請求逾時（回應約 5.8 MB） |
| `max_retries` | `3` | 指數退避 |
| `retry_backoff_seconds` | `2.0` | 起始退避 |

### 7.2 Fetch limits / 挑選

| 欄位 | 預設 | 說明 |
|---|---|---|
| `max_laws` | `100` | Pilot 上限；設 `null` 取全部 |
| `category_prefix` | `null` | 選用的 `LawCategory` 前綴過濾 |
| `include_repealed` | `true` | **保留已廢止法規**，並於 metadata 標記 |
| `order_by` | `"pcode"` | 決定性排序鍵 |

### 7.3 Snapshot 位置

| 欄位 | 預設 |
|---|---|
| `output_root` | `evaluation/corpora/moj-law` |

### 7.4 既有 retrieval / chunking 設定（沿用，不新增機制）

Pilot 使用既有 `ChunkingConfig` 欄位，僅調整數值（理由見 §11 L2）：

```
target_chars: 450        # 而非預設 600
overlap_chars: 80        # 而非預設 100
max_tokens: 480          # 不變，等於 embedding_max_tokens
```

retrieval / rerank 設定沿用既有 `evaluation/configs/` 慣例，本計畫不引入新的 retrieval 參數。

---

## 8. Error Handling and Safety

| 情境 | 處理 |
|---|---|
| **API 無法連線** | 退避重試至 `max_retries`；仍失敗則以明確錯誤訊息（endpoint + 嘗試次數 + 最後錯誤）非零離開。不留下部分 snapshot。 |
| **Timeout** | 同上，計入重試。 |
| **非 ZIP / ZIP 損毀** | 立即失敗並回報實際前導 bytes；不嘗試猜測格式。 |
| **缺少 `ChLaw.json`** | 立即失敗並列出 ZIP 實際成員。 |
| **頂層結構不符** | 缺 `UpdateDate` 或 `Laws`、`Laws` 非 list → 立即失敗。 |
| **Malformed record** | 缺 `LawArticles`、`LawName` 空、`LawURL` 無法解析 `pcode` → 跳過該筆，累計於 `skipped_records`。若跳過比例超過門檻（預設 5%）則整體失敗（代表來源結構改變）。 |
| **條文內容為空** | 跳過該條，計入 `skipped_article_count`。若一部法規所有條文皆空 → 該法規正規化後無 document；它**仍是 valid selected record**（不計入 `skipped_record_count`），但其條文全數計入 `skipped_article_count`。 |
| **Partial fetch** | 回應位元組數與 `Content-Length` 不符 → 視為失敗並重試。 |
| **Duplicate record** | `pcode` 重複 → 立即失敗（違反已驗證的唯一性假設，代表來源契約變動）。 |
| **Empty result** | 挑選後為 0 筆 → 明確失敗，不產生空 snapshot。 |
| **Snapshot 產生中斷** | **原子寫入**：先寫入同層 `.<snapshot_id>.tmp/`，全部成功後才 `os.replace` 更名。中斷只會留下 tmp 目錄，永不產生半成品 snapshot。CLI 啟動時清掉自己的舊 tmp。（Slice B） |
| **抓取期間來源變動** | 單一 bulk 請求，不存在跨頁不一致問題。來源版本以 `UpdateDate` + `raw_sha256` 記錄；兩者任一改變即為不同 snapshot（見 §6.2）。 |

### Qdrant production 安全（Owner Decision Q3）

**不得** hardcode 任何固定的 production point count（如 `factory_knowledge == 10`）。10 只是當下狀態，**不是 invariant**。

改採 **before/after snapshot 語意**：

```
測試前：
  - 偵測 production collection 是否存在
  - 若存在：記錄其 point count 與必要 identity/state
  - 若不存在：記錄為 absent

測試後：
  - 若 before 存在 → after 必須與 before 完全相同
  - 若 before 不存在 → after 必須仍不存在
```

測試只建立、只清除**自己 unique-name 的 resources**（沿用 `tests/integration/conftest.py` 的 `unique_token()` / `TEST_COLLECTION_PREFIXES` / `drop_collections()` / `wait_collection_absent()`）。**絕不**對整個 Qdrant namespace 做斷言，**絕不**依賴固定的 production point count。

其他安全規則：本功能全程不呼叫 production `/api/query`、`/api/retrieve`，不寫入 `factory_knowledge`，不呼叫 LLM。

---

## 9. Testing Strategy

### 9.1 Unit tests（`tests/unit/`，完全 offline、無網路、無模型、無 Qdrant）

以 `evaluation/corpus/fixtures/chlaw_sample.json`（真實法規子集）為輸入，測試中以 `zipfile` 在記憶體組出 ZIP，**不入庫二進位 golden 檔**。

**Source + Normalization（Slice A）**
- **Normalization**：Markdown golden 文字；**`ArticleType == "C"` 不成為獨立 section／獨立 article evidence，但其 context 保留於後續 A 條文**；`LawAttachements` 被丟棄；`is_repealed` metadata 正確。
- **挑選決定性**：打亂輸入順序 → 挑選結果不變；`include_repealed` 行為。
- **Malformed responses**：缺欄位、空條文、`LawURL` 無 pcode、pcode 重複、頂層結構不符、缺 `ChLaw.json`、非 ZIP、ZIP 損毀、挑選結果為空。
- **BOM**：`utf-8-sig` 正確解碼。
- **Retry / timeout**：以注入的 fake fetcher 驗證重試次數與 backoff sequence。**必須注入 sleeper（或等價 deterministic 手段），wall-clock 等待須接近 0**（Clarification C3）。

**Snapshot（Slice B，不得於 Slice A 提前實作）**
- canonical snapshot / `corpus_hash` 一致性、manifest determinism、原子寫入、中斷不留半成品。

**Fetcher injection 而非 HTTP mocking。** repo 無任何 HTTP mocking library，dev 相依只有 `pytest`；house style 是 constructor injection（比照 `LLMClient(settings, client=...)`）與手寫 fake。`httpx` 已是既有相依，**不新增任何套件**。

### 9.2 Integration tests（`tests/integration/`，`pytestmark = pytest.mark.integration`，Slice C）

- 需真實 Qdrant；沿用「偵測不到就 **skip、不 fail**」的 readiness fixture。
- 使用 unique-name 測試 collection（`factory_test_*` 前綴），只清除自己建立的資源。
- 套用 §8 的 before/after production snapshot 語意。
- 驗證：`collect_corpus_files` 能收到 snapshot 的 `.md`；每條文對應一個 canonical segment；Qdrant payload 保有 `filename` / `section`，可還原「法規 + 條號」。

### 9.3 External API drift test（`pytestmark = pytest.mark.external_api`，Correction 2）

MOJ endpoint 本身即為單一 bulk ZIP，**一次 request 必然取得完整 ZIP**，故不存在「只下載一部分」的選項。

此測試**只做一次 bulk download**，且**僅驗證**：
- response contract（可取得位元組、為 ZIP）
- ZIP structure（成員清單）
- `ChLaw.json` 存在
- 頂層 `UpdateDate` 存在
- 頂層 `Laws` 存在且為 list

**不得**在此測試進行：normalize 全 corpus、ingest、embedding、Qdrant index、retrieval evaluation。

此為**契約漂移偵測**。預設不隨主測試套件執行（需顯式 `-m external_api`），因此**不會讓 unit suite 變得 flaky**。marker 於**首次新增此測試的 slice** 才註冊至 `pyproject.toml`（**非 Slice A**；且因 §3.4 實測未啟用 `--strict-markers`，提前註冊無技術必要）。

### 9.4 End-to-end smoke

external source → snapshot → normalized docs → chunks → Qdrant → retrieval 回傳非空結果。屬 Slice C，走 integration 分層。

### 9.5 決定性 / offline 歸屬

| 層級 | 決定性 | 需要網路 | 隨預設套件執行 |
|---|---|---|---|
| Unit | 是 | 否 | 是 |
| Integration | 是（給定 Qdrant） | 否 | 是（`-m "not llm_integration"`） |
| External API drift | 否（依來源） | 是 | **否**（需 `-m external_api`） |

---

## 10. Execution Slices

**Slice A–D 是目前唯一批准的 execution scope。** 每個 slice 獨立可審，結束後 **STOP 等待 Repository Owner review**。

---

### Slice A — Source contract + fixture 正規化

**目標**：把已驗證的來源契約固化為程式碼，並在完全 offline 的前提下完成解析與正規化。

**可能變更的檔案**
`evaluation/corpus/__init__.py`、`models.py`、`moj_law.py`、`normalize.py`、`fixtures/chlaw_sample.json`、`tests/unit/test_corpus_moj_law.py`、`tests/unit/test_corpus_normalize.py`

**不得新增**
`snapshot.py`、`stats.py`、`cli.py`、`SnapshotManifest`、真實 corpus snapshot、Qdrant integration test

**不得變更**
`app/**`、`evaluation/v11/**`、`evaluation/runner.py`、`evaluation/datasets/**`、既有 `evaluation/configs/*.json`、`README.md`、`.gitignore`、`pyproject.toml`

**實作任務**
1. `LawRecord` / `NormalizedDocument` dataclass（忠實對映來源欄位，不臆造；**不建立 `SnapshotManifest`**）。
2. `fetch_raw(config, *, fetcher=None, sleeper=None)`：預設 `httpx` 實作，可注入；timeout / retry 可測。
3. `parse_raw`：ZIP → `ChLaw.json` → `utf-8-sig` → JSON → `(UpdateDate, [LawRecord], [SkippedRecordDiagnostic])`；`pcode` 由 `LawURL` 解析；驗證 ZIP 格式、`ChLaw.json` 存在、頂層結構、必要欄位、duplicate pcode。跳過的 record 以 `SkippedRecordDiagnostic`（`models.py`）保留 provenance，供 Slice B manifest 使用。
4. `select`：filter → pcode 升冪 → `max_laws`；輸入順序不影響輸出；`include_repealed` 預設 `true`。
5. `to_markdown`：依 §5.5，章節路徑堆疊 + `{chapter_path}｜{ArticleNo}` heading；C 絕不獨立成 heading；丟棄 `LawAttachements`；`is_repealed` 進 metadata。

**測試**：只做 §9.1 的 **Source + Normalization** 部分 — fixture parse、UTF-8 BOM、pcode extraction、deterministic selection、`include_repealed`、Markdown golden、C handling（雙向斷言）、attachments ignored、malformed law record、missing/invalid `ChLaw.json`、duplicate pcode、empty `ArticleContent`、retry、timeout。
**不得測**：snapshot、`corpus_hash`、manifest、atomic write、Qdrant、embedding、retrieval。

**產出**：`evaluation/corpus/` 前四個模組 + 受追蹤 fixture + unit 測試。

**驗收**
- `pytest tests/unit -q -k corpus` 全綠；完整 `pytest tests/unit -q` 無退化。
- unit suite 全程無網路、無模型載入；retry 測試 wall-clock 接近 0。
- `git status --porcelain app/` 為空；diff 不含任何禁改路徑。

**停止條件**：以上達成即 **STOP**，等待 review。不 commit、不 push、不進入 Slice B。

---

### Slice B — Snapshot + corpus fingerprint

**目標**：產生可重現、可稽核、以 hash 版本化的 snapshot。

**可能變更的檔案**
`evaluation/corpus/snapshot.py`、`cli.py`、`models.py`（新增 `SnapshotManifest`）、`evaluation/configs/corpus-moj-law-pilot.json`、`.gitignore`、`tests/unit/test_corpus_snapshot.py`

**不得變更**
同 Slice A，並額外強調 **`evaluation/v11/hashing.py` 不得修改**。

**實作任務**
1. `SnapshotManifest` dataclass。
2. `build_snapshot`：原子寫入（`.tmp` → `os.replace`）。
3. `manifest.json`：§6.6 全部欄位；以 `canonical_json` 序列化；`snapshot_id` 依 §6.2 含 `raw_sha256[:8]`。
4. `corpus_hash`：**呼叫既有 `hashing.corpus_hash`**。
5. CLI `fetch` / `snapshot` 子指令（house argparse 慣例）。
6. `.gitignore` 加入 `evaluation/corpora/*/*/raw/`，附 slice 註解。

**測試**：manifest 決定性、`corpus_hash` 一致性、中斷不留半成品、空結果明確失敗、snapshot_id 對 raw bytes 變動敏感（皆 offline）。

**驗收**
- 同 fixture 連跑兩次 → `corpus_hash` 相同、`manifest.json` 除 `fetched_at` 外位元組相同。
- 相同 `UpdateDate` 但不同 raw bytes → **不同 snapshot_id**。
- 模擬中斷後無半成品 snapshot。
- `git status` 確認 `raw/` 被忽略。

**停止條件**：**STOP**，等待 review。不進入 Slice C。

---

### Slice C — 既有 pipeline 整合

**目標**：證明 snapshot 目錄能直接作為 `corpus_paths`，**且 pipeline 內部零新增程式碼**。

**可能變更的檔案**
`tests/integration/test_corpus_pipeline.py`（新增）、pilot config 的 chunking 數值

**不得變更**
`app/**`、`evaluation/v11/**`（本 slice 的價值正在於證明不必改）

**實作任務**
1. 以小型 snapshot 為 `corpus_paths`，走 `RealTrialServiceProvider.build_index`。
2. 驗證 canonical segment 與條文的對應關係，且無任何 segment 來自 C heading。
3. 驗證 Qdrant payload 保有 `filename` / `section`，可還原法規 + 條號 + 章節。
4. 確認 §11 L2 的 chunking 數值（`target_chars` 450）確實降低 hard token split 比例。
5. 套用 §8 的 before/after production snapshot 語意。

**測試**：integration 分層；Qdrant 不可用則 skip 不 fail；唯一命名 collection；只清除自己建立的資源。

**驗收**
- 小型 snapshot 完成 index，point count 與預期 chunk 數相符。
- retrieval 回傳非空結果。
- production collection 的 before/after 狀態一致（存在則相同、不存在則仍不存在）。
- 無 `optimizer_*` / `factory_test_*` 殘留。

**停止條件**：**STOP**，等待 review。不進入 Slice D。

---

### Slice D — 真實 corpus pilot（約 100 部法規）

**目標**：取得有界的真實 corpus，完成 ingest/index，產出 corpus statistics 與 retrieval sanity check。

**可能變更的檔案**
`evaluation/corpus/stats.py`、`cli.py`（`stats` 子指令）、`evaluation/corpora/moj-law/<snapshot_id>/**`（產物）、statistics 報告、`README.md`（Documentation 一列）、`docs/` 內的結果紀錄

**不得變更**
`app/**`、`evaluation/v11/**`、`evaluation/datasets/**`

**實作任務**
1. 依 pilot config 執行 fetch → snapshot（約 100 部法規）。
2. Ingest / index 至隔離 collection。
3. 產出 corpus statistics：來源文件數、正規化文件數、chunk 總數、chunk 長度分布、**hard token split 比例**、唯一標題 / pcode 數、**已廢止法規數**、近似標題重複指標、metadata 完整度、ingest/index 耗時。
4. 少量人工挑選的 retrieval sanity probes（確認回傳合理條文，非最終評測）。

**明確不做**：**不建立 hard evaluation dataset，不撰寫最終評測問題。** 這些必須等 Owner 讀過 statistics 後另行決定。

**測試**：statistics 計算本身的 unit 測試（以 fixture 為輸入，決定性）。

**產出**：真實 snapshot（manifest + normalized 入庫）、statistics 報告、sanity check 紀錄。

**驗收**
- statistics 報告產出且數值可由 snapshot 重現。
- retrieval 回傳非空且主題相關的結果。
- 明確回答：「此 corpus 是否夠大、夠混淆，足以支撐下一階段的 hard holdout？」
- production collection before/after 一致。
- 未觸發 §6.7 的 10 MB size trigger（若觸發，已取得 Owner 決定）。

**停止條件**：**STOP**，等待 review。**不得**自行開啟 Slice E 或建立評測資料集。

**Slice D 執行結果（實測，見 [docs/reports/corpus-moj-law-pilot-report.md](reports/corpus-moj-law-pilot-report.md)）**

- snapshot `moj-chlaw-20260821-2e815ced-9629909b`：100 部法規、3,437 條文、
  3,643 chunk；入庫體積約 1.28 MB，**未**觸發 §6.7 的 10 MB size trigger。
- 來源契約與 §5.1–§5.3 規劃期記錄**完全相符**（1,346 部 / 47,257 條 / 4,062 編章節 /
  322 廢止），無漂移。契約偵測已固化為 `external_api` marker 測試。
- L2 確認：hard token split 68 / 3,437 = **1.98%**（估計約 1.5%）；0 chunk 超過
  token 上限、0 strategy fallback。
- L1 確認：3,643 個 chunk 中 **0 個**帶有 identifier（0.00%），identifier-aware
  100% 退化為 dense。
- Slice C 發現的法規標頭 chunk 在 100 部規模占 **2.74%**；一般條文查詢 top-5 中
  僅占 **0.67%**，**未觀察到 retrieval pollution**。
- **Pilot 結論：NOT YET SUITABLE。** 條號歧義與跨法規語意歧義充足，但
  `max_laws=100` + `order_by=pcode` 的決定性截斷落在憲政／院級組織法規區段，
  **0 組標題詞幹碰撞、0 組現行／廢止交錯**（完整 corpus 分別有 26 組與 19 組）。
  下一階段 holdout 前需 Owner 先決定 selection policy。詳見結果文件 §7.F。

---

## 11. Final Acceptance Criteria

本功能視為完成，需全部成立：

1. `python -m evaluation.corpus.cli snapshot --config evaluation/configs/corpus-moj-law-pilot.json` 可從零重現出相同 `corpus_hash` 的 snapshot。
2. `manifest.json` 具備 §6.6 全部 provenance 欄位；`snapshot_id` 依 §6.2 含 `raw_sha256[:8]`；`corpus_hash` 由既有 `evaluation/v11/hashing.py` 產生。
3. snapshot 的 `normalized/` 可直接作為 `corpus_paths`，完成 chunk → embed → Qdrant → retrieval，**且 `app/**` 與 `evaluation/v11/**` 零變更**。
4. Unit 測試完全 offline、決定性；external API drift 測試以獨立 marker 隔離，不影響預設套件穩定度。
5. Corpus statistics 報告已產出並可重現。
6. Production Qdrant collection 的 before/after 狀態一致（**不以固定 point count 斷言**）；無測試殘留 collection。
7. 入庫產物符合 §6.7，且未觸發 10 MB size trigger（若觸發，已取得 Owner 決定）。
8. 未建立任何最終 hard evaluation dataset。
9. 已知限制 L1 / L2 明確記錄於本文與結果文件。

### 已知限制（必須保留，不得移除）

**L1 — identifier-aware retrieval 在此 corpus 上退化為 dense。**
實測 `app/services/identifier_extractor.py` 的 `extract_normalized()` 對本 corpus 的識別碼全數回傳空陣列：

```
'第 1 條'         -> []
'第 15-1 條'      -> []
'A0000001'       -> []      # pcode
'pcode=A0000001' -> []
'D-900 警報'      -> ['D900']   # 既有 demo corpus 正常
```

原因：regex `(?<![A-Za-z0-9])([A-Za-z]{1,5})[-_]?(\d{2,6})(?![A-Za-z0-9])` 要求 ASCII 字母 + **2–6** 位數字。CJK「第 N 條」永不匹配；pcode 為 1 字母 + **7** 位數字，亦超出位數上限。

後果：chunk 的 `identifiers` payload 幾乎全為空，`IdentifierAwareRetrievalStrategy` 走「查詢無識別碼」路徑而委派給 `DenseRetrievalStrategy`。**因此 pilot 實際量測的是 dense + cross-encoder rerank。**

依 Owner Decision Q1，Slice A–D **不修改** `IdentifierExtractor`；此退化本身即為必須明確記錄的發現。

**L2 — 預設 chunk 大小在中文語料上超出 embedding token 上限。**
以真實 `BAAI/bge-small-zh-v1.5` tokenizer 對 400 條真實條文實測：**0.97 tokens/char**（median 1.00）。

`chunk_size_chars=600` → 約 **583 tokens**，超過 `embedding_max_tokens=480`。`TextSplitter._enforce_token_limit` / `_hard_split_by_tokens` 會頻繁觸發，切點落在 token 邊界而非語意邊界。另有約 1.5%（6/400）的條文單獨即超過 480 tokens，無論如何都會被硬切。

處置：pilot 以 **config 值**（`target_chars` 450 / `overlap_chars` 80）因應，**不修改任何程式碼**；`target_chars` 本就是既有的 search dimension。實際 hard split 比例列為 Slice D 的 statistics 項目。

**Slice D 實測：** 68 / 3,437 = **1.98%** 的條文本身即超過 480 tokens（估計約 1.5%）；切分後 0 chunk 超過上限、0 strategy fallback。

**L3 — pilot 的 100 部並非完整 corpus 的代表性樣本（Slice D 實測發現）。**

`max_laws=100` + `order_by=pcode` 是決定性截斷，取到 pcode `A0000001`–`A0030243`，即憲政／院級組織法規這一個狹窄區段。以同一組確定性規則量測：

| 指標 | pilot（100 部） | 完整 corpus（1,346 部） |
|---|---|---|
| 標題詞幹碰撞 | 0 群 / 0 部 | 26 群 / 52 部 |
| 現行／已廢止交錯詞幹 | 0 群 | 19 群 |
| 共用 6 字前綴 | 4 群 / 8 部 | 95 群 / 288 部 |
| 已廢止法規 | 23 | 322 |

後果：本 snapshot 的條號歧義與跨法規語意歧義充足，但**標題混淆與現行／廢止混淆幾乎不存在**。任何基於本 snapshot 的結論**不得**外推至完整 corpus；建立 hard holdout 前需先決定 selection policy（見 [docs/reports/corpus-moj-law-pilot-report.md](reports/corpus-moj-law-pilot-report.md) §7.F）。

---

## 12. Future Extensions — Not Part of This Plan

僅列出，**不設計、不實作**：

- HTML crawler source
- PDF source（`LawAttachements` 的 `FileURL` 是最自然的第一個目標）
- OCR fallback
- Vision / multimodal source
- Scheduled incremental synchronization（來源 `UpdateDate` 是天然的 cursor）

**Slice E — Legal Identifier Support（pilot 後的可能決策，非目前 scope）**
若 Slice D 的 statistics / retrieval findings 支持，Owner 可能另開獨立 slice，處理「第 N 條」、「第 N-N 條」、pcode、`IDENTIFIER_NORMALIZATION_VERSION` bump，以及 config / baseline 相容性。**本文不設計其內容，目前不得實作。**

---

## 13. Open Questions / Owner Decisions

### 已決議

**Q1 — identifier-aware retrieval 在此 corpus 上失效，如何處理？**
**Owner Decision：採選項 (a)。** Pilot 階段**不修改** `IdentifierExtractor`。此退化作為已知 limitation 明確記錄（見 L1）。Slice A–D 只要求能正確評估**現有**系統，不得為法律資料修改 identifier normalization。Slice D 完成後，才依 corpus statistics / retrieval findings 決定是否另開 Slice E。

**Q2 — normalized snapshot 入庫的體積界線？**
**Owner Decision：** 100-law snapshot（估計約 1.3 MB）允許 commit。設定明確 trigger：**任何單一 normalized snapshot 預估或實際超過約 10 MB，在 commit 前必須重新取得 Repository Owner 決定。** 不得因目前允許 1.3 MB，即推論完整 1,346-law（約 18 MB）corpus 亦自動允許入庫。

**Q3 — Qdrant production 安全斷言方式？**
**Owner Decision：** 移除任何 hardcoded `factory_knowledge == 10 points`。改採 §8 的 before/after snapshot 語意。10 只是目前狀態，不是 invariant。

**C1 — `ArticleType == "C"` 的表示方式？**
**Owner Clarification：** C 只作為後續 A 條文的章節 context，不得輸出成既有 loader 會視為獨立法律條文 evidence 的 section。依 §3.3 的 loader 實測行為，決議為 §5.5 的「章節路徑併入 A 條文 heading」表示；unit test 須雙向斷言。**不得因此修改 `app/` loader。**

**C2 — `external_api` marker 何時註冊？**
**Owner Clarification：** Slice A 無 external network test，故不修改 `pyproject.toml`；marker 延後至首次新增該測試的 slice。實測 `pyproject.toml` 未啟用 `--strict-markers`，故提前註冊無技術必要。

**C3 — retry/backoff 測試如何避免空等？**
**Owner Clarification：** unit test 不得真的等待 backoff；以 injected sleeper / `monkeypatch` 等 deterministic 方式驗證重試次數與 backoff sequence，wall-clock 等待須接近 0。

### 尚未決議

無。其餘事項（模組位置、config 格式、CLI 形態、測試分層、marker 命名、文件結構）均由既有 repository 慣例決定，不需 Owner 輸入。

---

## 安全聲明

本文件不含 API key、Authorization header、完整 prompt、私人路徑或文件全文。MOJ 全國法規資料庫 Open API 為公開資料，無需認證，授權為 Open Government Data License, version 1.0（OGDL-Taiwan-1.0）。規劃期間僅對來源發出一次請求以確認契約，未下載完整語料至 repository，未接觸或變更任何 production 服務。
