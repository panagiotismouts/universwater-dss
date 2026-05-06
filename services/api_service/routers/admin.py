"""
Admin router.

GET /health        — service liveness (no auth)
GET /admin/models  — list model registry entries (X-Admin-Key required)
POST /admin/bootstrap — stub, 501 in v1 (Amendment C.7)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from dss_shared.db.repositories.model_registry import ModelRegistryRepository
from dss_shared.schemas.api_response import AdminModelsResponse, ModelAdminSummary
from services.api_service.dependencies.auth import verify_admin_key
from services.api_service.dependencies.db import get_db

router = APIRouter()


@router.get("/health")
async def health_check(db=Depends(get_db)) -> dict:
    """
    Liveness check.  Pings MongoDB to confirm connectivity.
    Returns HTTP 200 if healthy, 503 if MongoDB is unreachable.
    """
    try:
        await db.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False

    if not mongo_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "SERVICE_UNAVAILABLE", "message": "MongoDB unreachable."},
        )
    return {"status": "ok", "service": "api_service", "mongo": "ok"}


@router.get("/admin/models", response_model=AdminModelsResponse)
async def list_models(
    pipeline: str = Query(default=None, description="Filter by pipeline"),
    model_status: str = Query(default=None, alias="status", description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200),
    _: None = Depends(verify_admin_key),
    db=Depends(get_db),
) -> AdminModelsResponse:
    """List model registry entries.  Requires X-Admin-Key header."""
    repo = ModelRegistryRepository(db)
    docs = await repo.list_all(pipeline=pipeline, status=model_status, limit=limit)

    summaries = [
        ModelAdminSummary(
            model_id=d.model_id,
            pipeline=d.pipeline,
            model_type=d.model_type,
            model_family=d.model_family,
            status=d.status,
            trained_at=d.trained_at,
            activated_at=d.activated_at,
            metrics_summary=d.metrics_summary.model_dump(),
        )
        for d in docs
    ]
    return AdminModelsResponse(models=summaries, total=len(summaries))


@router.post("/admin/bootstrap", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def bootstrap_stub() -> dict:
    """
    Bootstrap trigger — stub in v1.
    Use scripts/bootstrap.py for manual bootstrap (Amendment C.7).
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error_code": "NOT_IMPLEMENTED",
            "message": "Bootstrap is triggered manually via scripts/bootstrap.py in v1.",
        },
    )
