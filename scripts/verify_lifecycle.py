"""切片二「文件生命週期」端到端驗證（真實 Qdrant Server，非 :memory:）。

透過 FastAPI TestClient 驅動完整 ASGI 應用（lifespan 會載入真實 Embedding 模型、
連線 localhost Qdrant、建立全新暫存 SQLite），逐一執行規格 §六 的 14 項驗證案例。

執行：
  QDRANT_URL=http://localhost:6333 python scripts/verify_lifecycle.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Windows 主控台預設 cp950，強制 UTF-8 以輸出中文與符號
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

# 模型已由切片一快取，離線載入以避免 HF 網路噪音
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --- 於匯入 app 前設定環境：真實 Qdrant + 全新暫存 SQLite / 上傳目錄 ---
_TMP = tempfile.mkdtemp(prefix="lifecycle_verify_")
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL", "http://localhost:6333")
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "metadata.db")
os.environ["UPLOAD_DIR"] = os.path.join(_TMP, "uploads")
os.environ.pop("SKIP_STARTUP_SERVICES", None)

import logging  # noqa: E402

for _noisy in ("httpx", "sentence_transformers", "huggingface_hub", "transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

MD_NAME = "equipment_alarm_manual.md"
MD_PATH = os.path.join(ROOT, "sample_docs", MD_NAME)
TXT_NAME = "quality_note.txt"
TXT_BYTES = (
    "本文件為作品集示範所建立的虛構資料。\n\n"
    "AOI 檢測結果為 NG 時，必須記錄批號、機台編號與缺陷座標。\n\n"
    "每日保養包含清潔鏡頭與校正光源。\n"
).encode("utf-8")

_PASSED: list[str] = []
_FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "✅" if ok else "❌"
    print(f"  {tag} {name}" + (f" — {detail}" if detail else ""))
    (_PASSED if ok else _FAILED).append(name)
    return ok


def dump(label: str, resp) -> None:
    try:
        body = json.dumps(resp.json(), ensure_ascii=False)
    except Exception:  # noqa: BLE001 - 204 無 body
        body = "<no body>"
    print(f"    {label} -> HTTP {resp.status_code}  {body}")


def main() -> int:
    print("=" * 70)
    print(f"QDRANT_URL   = {os.environ['QDRANT_URL']}")
    print(f"SQLITE_PATH  = {os.environ['SQLITE_PATH']}")
    print(f"UPLOAD_DIR   = {os.environ['UPLOAD_DIR']}")
    print("=" * 70)

    app = create_app()
    created_doc_ids: list[str] = []

    # raise_server_exceptions=False：注入失敗案例回傳 500 回應而非向測試端重拋
    with TestClient(app, raise_server_exceptions=False) as client:
        store = app.state.vector_store
        repo = app.state.document_repository

        md_bytes = open(MD_PATH, "rb").read()

        # [1~3] 上傳 Markdown → 201 → SQLite status=indexed
        print("\n[案例 1-3] 上傳 Markdown")
        r = client.post(
            "/api/documents",
            files={"file": (MD_NAME, md_bytes, "text/markdown")},
        )
        dump("POST /api/documents", r)
        check("案例2：Markdown 上傳回 201", r.status_code == 201)
        md_doc_id = r.json().get("document_id") if r.status_code == 201 else None
        if md_doc_id:
            created_doc_ids.append(md_doc_id)
            rec = repo.get(md_doc_id)
            check("案例3：SQLite status=indexed", rec is not None and rec.status == "indexed",
                  f"status={rec.status if rec else None}")

            # [4] chunk_count == Qdrant point count
            print("\n[案例 4] chunk_count 一致性")
            point_count = store.count_by_document_id(md_doc_id)
            api_chunk = r.json().get("chunk_count")
            check("案例4：chunk_count == Qdrant point count",
                  rec.chunk_count == point_count == api_chunk and point_count > 0,
                  f"sqlite={rec.chunk_count} qdrant={point_count} api={api_chunk}")

            # [5] Top-K 可檢索到剛上傳內容
            print("\n[案例 5] Top-K 檢索")
            result = app.state.retrieval_service.retrieve(
                "設備出現 E102 警報時，應該先檢查什麼？", top_k=10
            )
            hit = any(rc.document_id == md_doc_id for rc in result.results)
            top = result.results[0] if result.results else None
            check("案例5：Top-K 檢索到剛上傳文件", hit,
                  f"top1 score={top.score:.4f} section={top.section}" if top else "無結果")

        # [6] 重複上傳相同 bytes → 409 + 既有 document_id
        print("\n[案例 6] 重複上傳去重")
        r_dup = client.post(
            "/api/documents",
            files={"file": (MD_NAME, md_bytes, "text/markdown")},
        )
        dump("POST /api/documents (相同 bytes)", r_dup)
        dup_id = r_dup.json().get("error", {}).get("details", {}).get("document_id")
        check("案例6：重複文件回 409", r_dup.status_code == 409)
        check("案例6：409 附既有 document_id", dup_id == md_doc_id, f"document_id={dup_id}")

        # [7] 上傳 TXT → 成功
        print("\n[案例 7] 上傳 TXT")
        r_txt = client.post(
            "/api/documents",
            files={"file": (TXT_NAME, TXT_BYTES, "text/plain")},
        )
        dump("POST /api/documents (txt)", r_txt)
        check("案例7：TXT 上傳成功 201", r_txt.status_code == 201)
        if r_txt.status_code == 201:
            created_doc_ids.append(r_txt.json()["document_id"])

        # [8] 上傳空白文件 → 拒絕 400
        print("\n[案例 8] 空白文件")
        r_empty = client.post(
            "/api/documents",
            files={"file": ("blank.txt", b"   \n  \t\n", "text/plain")},
        )
        dump("POST /api/documents (空白)", r_empty)
        check("案例8：空白文件被拒（400）", r_empty.status_code == 400)

        # [9] 上傳不支援格式 → 415（本階段 PDF 亦拒絕）
        print("\n[案例 9] 不支援格式")
        r_bad = client.post(
            "/api/documents",
            files={"file": ("scan.pdf", b"%PDF-1.4 dummy", "application/pdf")},
        )
        dump("POST /api/documents (.pdf 本階段不支援)", r_bad)
        check("案例9：不支援格式被拒（415）", r_bad.status_code == 415)

        # [10] GET /api/documents 列出文件
        print("\n[案例 10] 文件列表")
        r_list = client.get("/api/documents")
        dump("GET /api/documents", r_list)
        listed = r_list.json()
        check("案例10：列表回 200 且列出文件",
              r_list.status_code == 200 and listed["total"] >= 2,
              f"total={listed['total']}")

        # [11] DELETE → Qdrant points / 本地檔案 / SQLite 記錄皆刪除
        print("\n[案例 11] 刪除文件")
        if md_doc_id:
            rec_before = repo.get(md_doc_id)
            doc_dir = os.path.dirname(rec_before.storage_path)
            r_del = client.delete(f"/api/documents/{md_doc_id}")
            print(f"    DELETE /api/documents/{md_doc_id} -> HTTP {r_del.status_code}")
            check("案例11：刪除回 204", r_del.status_code == 204)
            check("案例11：Qdrant points 已刪除",
                  store.count_by_document_id(md_doc_id) == 0)
            check("案例11：本地檔案已刪除", not os.path.isdir(doc_dir), doc_dir)
            check("案例11：SQLite 記錄已刪除", repo.get(md_doc_id) is None)
            created_doc_ids = [d for d in created_doc_ids if d != md_doc_id]

            # 附加：刪除不存在文件 → 404
            r_del404 = client.delete(f"/api/documents/{md_doc_id}")
            dump("DELETE (已不存在)", r_del404)
            check("案例11：不存在文件回 404", r_del404.status_code == 404)

        # [12] 人為製造 indexing 失敗 → status=failed / error_message / 無殘留 points
        print("\n[案例 12] indexing 失敗回滾")
        svc = app.state.indexing_service
        original = svc.repository.mark_indexed
        captured: dict = {}

        def boom(document_id, chunk_count):  # noqa: ANN001 - 於 upsert 之後才失敗
            captured["document_id"] = document_id
            raise RuntimeError("注入的 indexing 失敗（測試）")

        svc.repository.mark_indexed = boom  # type: ignore[assignment]
        fail_bytes = (
            "# 注入失敗測試\n\n這份文件會在寫入 Qdrant 之後被人為中斷。\n"
        ).encode("utf-8")
        try:
            r_fail = client.post(
                "/api/documents",
                files={"file": ("inject_fail.md", fail_bytes, "text/markdown")},
            )
        finally:
            svc.repository.mark_indexed = original  # type: ignore[assignment]
        dump("POST /api/documents (注入失敗)", r_fail)
        fail_id = captured.get("document_id")
        check("案例12：失敗不回傳成功（非 2xx）", not (200 <= r_fail.status_code < 300),
              f"status={r_fail.status_code}")
        if fail_id:
            frec = repo.get(fail_id)
            check("案例12：status=failed", frec is not None and frec.status == "failed",
                  f"status={frec.status if frec else None}")
            check("案例12：error_message 有內容",
                  bool(frec and frec.error_message), frec.error_message if frec else "")
            check("案例12：Qdrant 無殘留 points",
                  store.count_by_document_id(fail_id) == 0)
            repo.delete(fail_id)  # 清理暫存 SQLite 中的失敗記錄

        # [13] /health 仍回 200
        print("\n[案例 13] /health")
        r_health = client.get("/health")
        dump("GET /health", r_health)
        check("案例13：/health 回 200", r_health.status_code == 200)

        # [14] /ready 的 metadata_db 經真實 SQLite 讀寫後回 ok
        print("\n[案例 14] /ready metadata_db 真實檢查")
        r_ready = client.get("/ready")
        dump("GET /ready", r_ready)
        checks = r_ready.json().get("checks", {})
        check("案例14：metadata_db == ok（真實讀寫）", checks.get("metadata_db") == "ok",
              f"metadata_db={checks.get('metadata_db')}")

        # 清理殘餘上傳文件的 Qdrant points（保持共用 collection 乾淨）
        for d in created_doc_ids:
            try:
                store.delete_by_document_id(d)
            except Exception:  # noqa: BLE001
                pass

    # 附加：SQLite 尚未初始化前，metadata_db 不得為 ok、/ready 回 503
    print("\n[附加] SQLite 未初始化時 /ready 行為")
    os.environ["SKIP_STARTUP_SERVICES"] = "1"
    app2 = create_app()
    with TestClient(app2) as client2:
        r = client2.get("/ready")
        dump("GET /ready (未初始化服務)", r)
        checks = r.json().get("checks", {})
        check("附加：未初始化時 metadata_db != ok",
              checks.get("metadata_db") != "ok", f"metadata_db={checks.get('metadata_db')}")
        check("附加：未就緒時 /ready 回 503", r.status_code == 503)
    os.environ.pop("SKIP_STARTUP_SERVICES", None)

    print("\n" + "=" * 70)
    print(f"通過 {len(_PASSED)} 項，失敗 {len(_FAILED)} 項")
    if _FAILED:
        print("失敗項目：" + ", ".join(_FAILED))
    print("結果：", "PASS ✅" if not _FAILED else "FAIL ❌")
    print("=" * 70)

    shutil.rmtree(_TMP, ignore_errors=True)
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
