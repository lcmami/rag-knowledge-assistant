"""Read-only Optimizer Reporting API（Slice 8 §五 / §六）。

全部為 **GET / read-only**：不執行 migration、不建立 DB、不呼叫 LLM、不碰 Qdrant。
資料來自三個**分開配置**的 read-only SQLite（search / review / deployment）。

回應規則：
- DB 未配置或不存在 → 回 200 且 ``{configured, available, reason}``（不是錯誤，§五.3）。
- DB 可用但 ID 不存在 → 穩定 **404**（§五.13）。
- 一律去識別化（store 已移除私人路徑 / 全文 / 向量；只留短欄位，§五.4/.5）。
- Read API **不依賴** admin write enabled（§五.10）；admin 狀態只是被動顯示。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.dependencies import get_settings
from evaluation.v11.reporting import OptimizerReportingStore

router = APIRouter(prefix="/api/optimizer", tags=["optimizer-reporting"])


def get_reporting_store(request: Request) -> OptimizerReportingStore:
    """由 settings 的三個 read-only 路徑建立 store（未配置的路徑為 None）。"""
    settings: Settings = request.app.state.settings
    return OptimizerReportingStore(
        search_db_path=settings.optimizer_search_db_path,
        review_db_path=settings.optimizer_review_db_path,
        deployment_db_path=settings.optimizer_deployment_db_path,
    )


def _not_found(kind: str, ident: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": {"code": f"{kind}_not_found",
                           "message": f"{kind} '{ident}' not found", "details": {}}},
    )


def _resolve(result: dict, kind: str, ident: str, data_key: str):
    """detail 端點共用：available=False → 200 unavailable；found=False → 404；否則資料。"""
    if not result.get("available", False):
        return {"configured": result.get("configured", False), "available": False,
                "reason": result.get("reason"), data_key: None}
    if not result.get("found", False):
        return _not_found(kind, ident)
    return result


@router.get("/overview")
def overview(
    store: OptimizerReportingStore = Depends(get_reporting_store),
    settings: Settings = Depends(get_settings),
) -> dict:
    return store.overview(
        active_alias=settings.optimizer_active_alias,
        admin_write_enabled=settings.optimizer_admin_enabled,
    )


@router.get("/searches")
def list_searches(store: OptimizerReportingStore = Depends(get_reporting_store)) -> dict:
    return store.list_searches()


@router.get("/searches/{search_id}")
def get_search(search_id: str, store: OptimizerReportingStore = Depends(get_reporting_store)):
    return _resolve(store.get_search(search_id), "search", search_id, "search")


@router.get("/searches/{search_id}/candidates")
def list_candidates(search_id: str, store: OptimizerReportingStore = Depends(get_reporting_store)):
    return _resolve(store.list_candidates(search_id), "search", search_id, "candidates")


@router.get("/searches/{search_id}/comparison")
def comparison(search_id: str, store: OptimizerReportingStore = Depends(get_reporting_store)):
    return _resolve(store.comparison(search_id), "search", search_id, "comparison")


@router.get("/reviews")
def list_reviews(store: OptimizerReportingStore = Depends(get_reporting_store)) -> dict:
    return store.list_reviews()


@router.get("/reviews/{session_id}")
def get_review(session_id: str, store: OptimizerReportingStore = Depends(get_reporting_store)):
    return _resolve(store.get_review(session_id), "review_session", session_id, "session")


@router.get("/coverage")
def coverage(store: OptimizerReportingStore = Depends(get_reporting_store)) -> dict:
    return store.coverage()


@router.get("/deployment/summary")
def deployment_summary(
    store: OptimizerReportingStore = Depends(get_reporting_store),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Read-only deployment 摘要（history / records / eligibility 顯示用，§十一）。

    真實 alias 的 live target 與 write 操作仍由 Slice 7 admin API 提供（``/alias/status`` 等）。
    """
    summary = store.deployment_summary(settings.optimizer_active_alias)
    summary["admin_write_enabled"] = settings.optimizer_admin_enabled
    return summary
