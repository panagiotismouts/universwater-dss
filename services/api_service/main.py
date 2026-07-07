"""
API service entry point.

Starts independently of ml_engine and ingestion (Amendment C.6).
Returns DATA_UNAVAILABLE responses when no predictions exist yet.

Lifespan:
  startup:  MongoDB probe, logging setup
  shutdown: close MongoDB client

Routes:
  POST /auth/token          — issue bearer token
  GET  /results/latest      — latest predictions per pipeline (auth required)
  GET  /results/history     — paginated history (auth required)
  GET  /health              — liveness check (no auth)
  GET  /admin/models        — model registry list (admin key required)
  POST /admin/bootstrap     — stub, 501 in v1
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from dss_shared.config import get_settings
from dss_shared.db import close_motor_client, get_database, probe_mongo
from dss_shared.logging import get_logger, setup_logging
from services.api_service.middleware.delivery_logger import DeliveryLoggerMiddleware
from services.api_service.routers import admin, auth, results

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format, service="api_service")
    log.info("api_service_starting", env=settings.env)

    db = get_database()
    await probe_mongo(db)
    log.info("mongodb_connected", db=settings.mongo_db_name)

    yield

    log.info("api_service_stopping")
    close_motor_client()
    log.info("api_service_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DSS Result API",
        version="1.0.0",
        description="Decision Support System — external prediction and XAI result API.",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────
    app.add_middleware(DeliveryLoggerMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(results.router, prefix="/results", tags=["results"])
    app.include_router(results.wqi_router, prefix="/wqi", tags=["wqi"])
    app.include_router(admin.router, prefix="", tags=["admin"])

    # ── Global exception handlers ─────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error_code": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "services.api_service.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
