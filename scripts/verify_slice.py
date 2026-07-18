"""垂直切片端到端驗證腳本（支援真實 Qdrant Server 與 :memory: 兩種模式）。

流程：
  一份 Markdown → Loader → Chunking → Embedding → 建立 Collection → 寫入 Points → Top-K 檢索

環境變數：
  QDRANT_URL   Qdrant 連線位址；預設 ":memory:"（內嵌測試模式）。
               指定 http://localhost:6333 即使用真實 Qdrant Server。
  SLICE_MODE   "write"（預設）：跑完整切片並寫入資料。
               "check"：不寫入，僅驗證既有 Collection 與 Points（用於重啟後持久性檢查）。

範例：
  # :memory: 測試模式
  python scripts/verify_slice.py
  # 真實 Server：寫入 + 驗證
  QDRANT_URL=http://localhost:6333 SLICE_MODE=write  python scripts/verify_slice.py
  # 真實 Server：重啟後只驗證資料仍在
  QDRANT_URL=http://localhost:6333 SLICE_MODE=check  python scripts/verify_slice.py
"""
from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import Settings
from app.dependencies import create_qdrant_client
from app.services.document_loader import load_document
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.text_splitter import TextSplitter
from app.services.vector_store import VectorStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_NAME = "equipment_alarm_manual.md"
# 穩定 document_id：跨程序 / 跨 container 重啟保持一致，方便驗證持久性
STABLE_DOC_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "factory-rag-slice-demo"))
QUESTION = "設備出現 E102 警報時，應該先檢查什麼？"


def build_chunks(settings: Settings, embedding: EmbeddingService):
    raw = open(os.path.join(ROOT, "sample_docs", DOC_NAME), "rb").read()
    elements = load_document(DOC_NAME, raw)
    splitter = TextSplitter(
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        max_tokens=settings.embedding_max_tokens,
        count_tokens=embedding.count_tokens,
    )
    chunks = splitter.split(
        elements,
        document_id=STABLE_DOC_ID,
        filename=DOC_NAME,
        file_hash="demo-hash",
    )
    return elements, chunks


def verify_collection(store: VectorStore, expected_dim: int) -> bool:
    from qdrant_client import models

    info = store.client.get_collection(store.collection)
    size = info.config.params.vectors.size
    distance = info.config.params.vectors.distance
    print(f"    Collection='{store.collection}' 向量維度={size} 距離函數={distance}")
    ok = True
    if size != expected_dim:
        print(f"    ❌ 維度不符：期望 {expected_dim}")
        ok = False
    if distance != models.Distance.COSINE:
        print(f"    ❌ 距離函數不符：期望 Cosine")
        ok = False
    if ok:
        print("    ✅ 維度為動態取得的 512 且距離函數為 Cosine")
    return ok


def report_ready_and_health(settings: Settings, embedding, store) -> None:
    os.environ["SKIP_STARTUP_SERVICES"] = "1"
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as tc:
        # 注入已建立的服務，避免 /ready 重載模型
        app.state.embedding_service = embedding
        app.state.vector_store = store
        app.state.settings = settings

        h = tc.get("/health")
        print(f"    GET /health  -> {h.status_code} {h.json()}")

        r = tc.get("/ready")
        print(f"    GET /ready   -> {r.status_code}")
        body = r.json()
        print(f"    status: {body['status']}")
        for k, v in body["checks"].items():
            print(f"      - {k:<12}: {v}")


def main() -> int:
    qdrant_url = os.getenv("QDRANT_URL", ":memory:")
    mode = os.getenv("SLICE_MODE", "write")
    settings = Settings(qdrant_url=qdrant_url)
    ok = True

    print("=" * 64)
    print(f"QDRANT_URL = {qdrant_url}    SLICE_MODE = {mode}")
    print("=" * 64)

    print("[1] 載入 Embedding 模型：", settings.embedding_model)
    embedding = EmbeddingService(
        model_name=settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        max_tokens=settings.embedding_max_tokens,
    )
    print("    向量維度（動態取得）：", embedding.dimension)

    client = create_qdrant_client(qdrant_url)
    store = VectorStore(client, settings.qdrant_collection)

    _, chunks = build_chunks(settings, embedding)
    expected_count = len(chunks)

    if mode == "write":
        print(f"[2] 解析 + Chunking：{expected_count} 個 Chunk")
        max_tok = max(embedding.count_tokens(c.text) for c in chunks)
        print(f"    最大 chunk token 數：{max_tok}（硬限制 {settings.embedding_max_tokens}）")
        assert max_tok <= settings.embedding_max_tokens
        assert all(c.text.strip() for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(expected_count))

        print("[3] 建立 Collection 並寫入 Points")
        store.ensure_collection(embedding.dimension)
        store.delete_by_document_id(STABLE_DOC_ID)  # 清掉舊資料確保冪等
        vectors = embedding.embed_documents([c.text for c in chunks])
        store.upsert_chunks(chunks, vectors)
    else:
        print("[2] check 模式：不寫入，僅驗證既有資料")

    print("[4] 驗證 Collection")
    ok = verify_collection(store, embedding.dimension) and ok

    print("[5] 驗證 Point 數 == Chunk 數")
    point_count = store.count_by_document_id(STABLE_DOC_ID)
    print(f"    document_id={STABLE_DOC_ID}")
    print(f"    point_count={point_count}  chunk_count={expected_count}")
    if point_count == expected_count and point_count > 0:
        print("    ✅ point count 等於 chunk count")
    else:
        print("    ❌ point count 與 chunk count 不符")
        ok = False

    print("[6] Top-K 檢索")
    retrieval = RetrievalService(embedding, store, settings.retrieval_score_threshold)
    result = retrieval.retrieve(QUESTION, top_k=3)
    print(f"    問題：{QUESTION}")
    print(f"    retrieval_time_ms={result.retrieval_time_ms}  Top-{len(result.results)}：")
    for r in result.results:
        preview = r.text.replace("\n", " ")[:36]
        print(f"      {r.source_id} score={r.score:.4f} section={r.section} | {preview}")
    if result.results:
        top1 = result.results[0]
        hit = "E102" in (top1.text or "") or "E102" in (top1.section or "")
        print(f"    Top-1 命中 E102 段落：{hit}")
        ok = hit and ok
    else:
        print("    ❌ 無檢索結果")
        ok = False

    print("[7] FastAPI /health 與 /ready 實際結果")
    report_ready_and_health(settings, embedding, store)

    print("=" * 64)
    print("切片驗證結果：", "PASS ✅" if ok else "FAIL ❌")
    print("（註：/ready 因 LLM 尚未設定預期回 503，屬正常，非切片失敗）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
