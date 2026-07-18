"""FastAPI 應用進入點與 lifespan（規格 §21）。"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import documents, health, query, retrieve
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("啟動 %s（env=%s）", settings.app_name, settings.app_env)

    app.state.settings = settings

    # 是否於啟動時載入重量級服務（切片 / 測試可用 SKIP_STARTUP_SERVICES=1 略過）
    if os.getenv("SKIP_STARTUP_SERVICES") != "1":
        from app.dependencies import build_services

        services = build_services(settings)
        for name, svc in services.items():
            setattr(app.state, name, svc)
        logger.info("核心服務已初始化：%s", ", ".join(services))
    else:
        logger.info("已略過核心服務初始化（SKIP_STARTUP_SERVICES=1）")

    yield
    logger.info("關閉 %s", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(title="factory-rag-assistant", lifespan=lifespan)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        # 未預期錯誤轉為一致格式，不外洩 traceback（§22）；
        # 單一請求失敗不得使服務停止。
        logger.exception("未處理的例外：%s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "伺服器內部錯誤。",
                    "details": {},
                }
            },
        )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(retrieve.router)
    app.include_router(query.router)

    # 簡易前端：Vanilla JS，靜態資源掛在 /static，首頁回傳 index.html。
    # API 路由先註冊、先比對，因此不會被靜態掛載遮蔽。
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

    return app


app = create_app()
