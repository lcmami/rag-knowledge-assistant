"""Health / Readiness API（規格 §19.1 / §19.2）。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """僅確認 FastAPI process 正常，不進行外部依賴檢查（§19.1）。"""
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request):
    """檢查各依賴是否就緒（§19.2）。"""
    state = request.app.state
    checks = {
        "qdrant": "unknown",
        "collection": "unknown",
        "embedding": "unknown",
        "metadata_db": "unknown",
        "llm_config": "unknown",
    }

    # Embedding
    checks["embedding"] = "loaded" if getattr(state, "embedding_service", None) else "error"

    # Qdrant + Collection
    vector_store = getattr(state, "vector_store", None)
    if vector_store is not None:
        try:
            vector_store.client.get_collections()
            checks["qdrant"] = "ok"
            try:
                dim = state.embedding_service.dimension
                vector_store.ensure_collection(dim)
                checks["collection"] = "ok"
            except Exception:  # noqa: BLE001
                checks["collection"] = "error"
        except Exception:  # noqa: BLE001
            checks["qdrant"] = "error"

    # SQLite：必須經過真實讀寫檢查才回 ok；尚未初始化不得顯示 ok（本切片 §一 / §八）
    repository = getattr(state, "document_repository", None)
    if repository is None:
        checks["metadata_db"] = "not_initialized"
    else:
        try:
            repository.ping()
            checks["metadata_db"] = "ok"
        except Exception:  # noqa: BLE001
            checks["metadata_db"] = "error"

    # LLM 設定
    settings = getattr(state, "settings", None)
    checks["llm_config"] = "ok" if settings and settings.llm_configured else "error"

    all_ok = all(v in ("ok", "loaded") for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )
