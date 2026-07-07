"""
Results router.

GET /results/latest   — latest prediction per sensor for a pipeline
GET /results/history  — paginated historical predictions
GET /wqi/current      — latest computed WQI values per water sensor (wqi_router)

All endpoints require a valid Bearer token.
Returns HTTP 404 with error_code=NO_RESULTS_AVAILABLE when no predictions
exist yet (expected during initial bootstrap).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from dss_shared.schemas.api_response import (
    CurrentWQIResponse,
    HistoricalResultsResponse,
    LatestResultsResponse,
)
from services.api_service.dependencies.auth import verify_bearer_token
from services.api_service.dependencies.db import get_db
from services.api_service.services.result_service import (
    get_current_wqi,
    get_latest_results,
    get_result_history,
)

router = APIRouter()
wqi_router = APIRouter()


@wqi_router.get("/current", response_model=CurrentWQIResponse)
async def endpoint_current_wqi(
    client_id: str = Depends(verify_bearer_token),
    db=Depends(get_db),
) -> CurrentWQIResponse:
    """Latest computed WQI values (deterministic, not predictions) per water sensor."""
    response = await get_current_wqi(db)
    if not response.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "NO_RESULTS_AVAILABLE",
                "message": "No water feature documents available yet.",
            },
        )
    return response


@router.get("/latest", response_model=LatestResultsResponse)
async def endpoint_latest(
    pipeline: str = Query(..., description="Pipeline: water | soil"),
    sensor_id: Optional[str] = Query(default=None, description="Filter to a specific sensor"),
    include_full_xai: bool = Query(default=False, description="Include full SHAP values in response"),
    client_id: str = Depends(verify_bearer_token),
    db=Depends(get_db),
) -> LatestResultsResponse:
    """Return the latest prediction for each sensor in the requested pipeline."""
    response = await get_latest_results(
        pipeline=pipeline,
        db=db,
        sensor_id=sensor_id,
        include_full_xai=include_full_xai,
    )
    if not response.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "NO_RESULTS_AVAILABLE",
                "message": f"No predictions available yet for pipeline={pipeline!r}.",
            },
        )
    return response


@router.get("/history", response_model=HistoricalResultsResponse)
async def endpoint_history(
    pipeline: str = Query(..., description="Pipeline: water | soil"),
    sensor_id: Optional[str] = Query(default=None),
    from_time: Optional[datetime] = Query(default=None, alias="from", description="ISO 8601 UTC start time"),
    to_time: Optional[datetime] = Query(default=None, alias="to", description="ISO 8601 UTC end time"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=200),
    include_full_xai: bool = Query(default=False),
    client_id: str = Depends(verify_bearer_token),
    db=Depends(get_db),
) -> HistoricalResultsResponse:
    """Return paginated historical predictions for the requested pipeline."""
    return await get_result_history(
        pipeline=pipeline,
        db=db,
        sensor_id=sensor_id,
        from_time=from_time,
        to_time=to_time,
        page=page,
        page_size=page_size,
        include_full_xai=include_full_xai,
    )
