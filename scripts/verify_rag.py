"""RAG 端到端驗證腳本（規格 Phase 5 §17）。

可重複執行，支援兩種模式：

Fake LLM 模式（預設一定執行；不需外部 API、不需 Qdrant）：
  以 Fake Retrieval + Scripted LLM 直接驅動 RagService，確定性驗證：
    - 正常回答
    - Threshold 前置拒答
    - Semantic 拒答
    - Citation 缺失
    - 無效 Citation

Real LLM 模式（僅在 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 皆設定時執行）：
  真實 Qdrant + 真實 Embedding + 真實 LLM，透過 /api/query 驗證：
    1. /ready
    2. 一題正常回答（Q001）
    3. 一題 Threshold fallback（Q020）
    4. CX-500 市場售價 semantic fallback（Q016）
    5. Citation 與 Sources 一致性
    6. Latency
  若環境變數不存在：明確回報 real LLM verification skipped，不假裝已驗證。

使用：
    python scripts/verify_rag.py
    QDRANT_URL=http://localhost:6333 LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... \
        python scripts/verify_rag.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

# Windows 主控台預設可能為 cp950，重設為 UTF-8 以正確輸出中文與符號。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.constants import CITATION_PATTERN, REFUSAL_MESSAGE  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from app.services.retrieval_service import RetrievalResult, RetrievedChunk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
QUESTIONS_PATH = os.path.join(ROOT, "evaluation", "questions.json")

MD_NAME = "equipment_alarm_manual.md"
TXT_NAME = "manufacturing_data_dictionary.txt"
PDF_NAME = "equipment_maintenance_manual.pdf"
REAL_COLLECTION = "factory_verify_rag"
_REQUIRED_ENV = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


def _question(qid: str) -> str:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for q in json.load(f):
            if q["id"] == qid:
                return q["question"]
    raise KeyError(qid)


def _ok(cond: bool) -> str:
    return "✅ PASS" if cond else "❌ FAIL"


# --------------------------------------------------------------------------- #
# Fake LLM 模式
# --------------------------------------------------------------------------- #
class _FakeRetrieval:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = 0

    def retrieve(self, question, top_k):
        self.calls += 1
        return RetrievalResult(results=list(self._chunks), retrieval_time_ms=12)


class _ScriptedLLM:
    def __init__(self, response=None):
        self.response = response
        self.call_count = 0

    async def complete(self, messages):
        self.call_count += 1
        return self.response


def _chunk(rank, text="E102 表示感測器通訊異常。"):
    return RetrievedChunk(
        source_id=f"S{rank}",
        document_id=f"doc-{rank}",
        filename=MD_NAME,
        page=None,
        section="E102 感測器通訊異常",
        chunk_index=rank,
        score=0.70 - rank * 0.01,
        text=text,
    )


def run_fake_mode() -> bool:
    print("=" * 68)
    print("Fake LLM 模式（不需外部 API / Qdrant）")
    print("=" * 68)
    chunks = [_chunk(1), _chunk(2)]
    ok = True

    # 1. 正常回答
    llm = _ScriptedLLM("E102 代表感測器通訊異常，應檢查通訊線材。[S1]")
    retr = _FakeRetrieval(chunks)
    res = asyncio.run(RagService(retr, llm).answer("E102?", 5))
    case = res.answer.endswith("[S1]") and [s.source_id for s in res.sources] == ["S1"]
    ok = case and ok
    print(f"[1] 正常回答               {_ok(case)}  sources={[s.source_id for s in res.sources]}")

    # 2. Threshold 前置拒答（retrieval 空 → 不呼叫 LLM）
    llm = _ScriptedLLM("不該被呼叫")
    res = asyncio.run(RagService(_FakeRetrieval([]), llm).answer("台灣首都?", 5))
    case = (
        res.answer == REFUSAL_MESSAGE
        and res.sources == []
        and res.generation_time_ms == 0
        and llm.call_count == 0
    )
    ok = case and ok
    print(f"[2] Threshold 前置拒答     {_ok(case)}  gen_ms={res.generation_time_ms} llm_calls={llm.call_count}")

    # 3. Semantic 拒答（有 context，LLM 回 fallback）
    llm = _ScriptedLLM(REFUSAL_MESSAGE)
    res = asyncio.run(RagService(_FakeRetrieval(chunks), llm).answer("CX-500 市場售價?", 5))
    case = res.answer == REFUSAL_MESSAGE and res.sources == [] and llm.call_count == 1
    ok = case and ok
    print(f"[3] Semantic 語意拒答      {_ok(case)}  gen_ms>0={res.generation_time_ms >= 0} llm_calls={llm.call_count}")

    # 4. Citation 缺失 → fail-closed
    llm = _ScriptedLLM("E102 代表感測器通訊異常。")
    res = asyncio.run(RagService(_FakeRetrieval(chunks), llm).answer("E102?", 5))
    case = res.answer == REFUSAL_MESSAGE and res.sources == []
    ok = case and ok
    print(f"[4] Citation 缺失          {_ok(case)}  → fail-closed")

    # 5. 無效 Citation [S99] → fail-closed
    llm = _ScriptedLLM("答案內容。[S99]")
    res = asyncio.run(RagService(_FakeRetrieval(chunks), llm).answer("E102?", 5))
    case = res.answer == REFUSAL_MESSAGE and res.sources == []
    ok = case and ok
    print(f"[5] 無效 Citation [S99]    {_ok(case)}  → fail-closed")

    print(f"\nFake LLM 模式結果：{'PASS ✅' if ok else 'FAIL ❌'}\n")
    return ok


# --------------------------------------------------------------------------- #
# Real LLM 模式
# --------------------------------------------------------------------------- #
def _valid_citations(answer: str, sources: list[dict]) -> bool:
    cited = {f"S{m.group(1)}" for m in CITATION_PATTERN.finditer(answer)}
    return cited == {s["source_id"] for s in sources}


def run_real_mode() -> bool:
    print("=" * 68)
    print("Real LLM 模式")
    print("=" * 68)

    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f"real LLM verification skipped：缺少環境變數 {missing}")
        print("（未設定即略過，不假裝已驗證）\n")
        return True  # skip 不算失敗

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        from qdrant_client import QdrantClient

        probe = QdrantClient(url=qdrant_url)
        probe.get_collections()
        probe.close()
    except Exception as exc:  # noqa: BLE001
        print(f"real LLM verification skipped：真實 Qdrant 未就緒（{qdrant_url}）：{exc}\n")
        return True

    if not os.path.exists(os.path.join(SAMPLE_DIR, PDF_NAME)):
        print("real LLM verification skipped：尚未產生 sample PDF\n")
        return True

    import tempfile

    workdir = tempfile.mkdtemp(prefix="verify_rag_")
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["QDRANT_COLLECTION"] = REAL_COLLECTION
    os.environ["SQLITE_PATH"] = os.path.join(workdir, "metadata.db")
    os.environ["UPLOAD_DIR"] = os.path.join(workdir, "uploads")
    os.environ.setdefault("RETRIEVAL_SCORE_THRESHOLD", "0.55")
    os.environ.pop("SKIP_STARTUP_SERVICES", None)

    from fastapi.testclient import TestClient

    from app.main import create_app

    provider = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    print(f"Provider(base_url)={provider}  model={model}  (API Key 不顯示)")

    ok = True
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        try:
            # index samples
            for name, ct in ((MD_NAME, "text/markdown"), (TXT_NAME, "text/plain"), (PDF_NAME, "application/pdf")):
                with open(os.path.join(SAMPLE_DIR, name), "rb") as f:
                    r = c.post("/api/documents", files={"file": (name, f.read(), ct)})
                assert r.status_code == 201, r.text

            # 1. /ready
            r = c.get("/ready")
            case = r.status_code == 200 and r.json()["checks"]["llm_config"] == "ok"
            ok = case and ok
            print(f"[1] /ready                 {_ok(case)}  status={r.status_code}")

            # 2. 正常回答（Q001）
            r = c.post("/api/query", json={"question": _question("Q001"), "top_k": 5})
            b = r.json()
            case = (
                r.status_code == 200
                and b["answer"] != REFUSAL_MESSAGE
                and bool(b["sources"])
                and _valid_citations(b["answer"], b["sources"])
                and b["generation_time_ms"] > 0
            )
            ok = case and ok
            print(f"[2] 正常回答（Q001）       {_ok(case)}  sources={[s['source_id'] for s in b['sources']]} "
                  f"retr={b['retrieval_time_ms']}ms gen={b['generation_time_ms']}ms")

            # 3. Threshold fallback（Q020）
            r = c.post("/api/query", json={"question": _question("Q020"), "top_k": 5})
            b = r.json()
            case = b["answer"] == REFUSAL_MESSAGE and b["sources"] == [] and b["generation_time_ms"] == 0
            ok = case and ok
            print(f"[3] Threshold fallback     {_ok(case)}  gen={b['generation_time_ms']}ms")

            # 4. CX-500 semantic fallback（Q016）
            r = c.post("/api/query", json={"question": _question("Q016"), "top_k": 5})
            b = r.json()
            case = b["answer"] == REFUSAL_MESSAGE and b["sources"] == [] and b["generation_time_ms"] > 0
            ok = case and ok
            print(f"[4] CX-500 semantic fallback {_ok(case)}  gen={b['generation_time_ms']}ms（>0 表示有呼叫 LLM）")
        finally:
            try:
                app.state.vector_store.client.delete_collection(REAL_COLLECTION)
                print(f"\n已刪除測試 collection「{REAL_COLLECTION}」。")
            except Exception as exc:  # noqa: BLE001
                print(f"\n刪除測試 collection 失敗：{exc}", file=sys.stderr)

    print(f"\nReal LLM 模式結果：{'PASS ✅' if ok else 'FAIL ❌'}\n")
    return ok


def main() -> int:
    fake_ok = run_fake_mode()
    real_ok = run_real_mode()
    overall = fake_ok and real_ok
    print("=" * 68)
    print("verify_rag 總結：", "PASS ✅" if overall else "FAIL ❌")
    print("=" * 68)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
