"""
Result service.

Assembles API response payloads from prediction_results + xai_results +
model_registry.  Handles the DATA_UNAVAILABLE case transparently — callers
receive an empty results list and decide whether to return 404.

For each PredictionDocument returned:
  - Model provenance (ModelInfo) is fetched from model_registry by model_id.
    Results are cached per request (dict keyed by model_id) to avoid repeated
    MongoDB round-trips when multiple predictions share the same model.
  - XAI data (explainer_type, base_value, full_shap_values) is fetched from
    xai_results when needed.  top_features are already embedded on
    PredictionDocument.top_shap_features.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.repositories.features import FeatureRepository
from dss_shared.db.repositories.model_registry import ModelRegistryRepository
from dss_shared.db.repositories.predictions import PredictionRepository
from dss_shared.db.repositories.xai_results import XAIResultRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.api_response import (
    CurrentWQIEntry,
    CurrentWQIResponse,
    DataQuality,
    HistoricalResultsResponse,
    LatestResultsResponse,
    ModelInfo,
    PredictionInterval,
    PredictionObject,
    XAIBlock,
    XAITopFeature,
)
from dss_shared.schemas.prediction import PredictionDocument

log = get_logger(__name__)

_VALID_PIPELINES = {
    "water", "soil",
    # Retired nowcast pipelines — kept queryable for historical records
    "water_wqi_brown", "water_wqi_ccme", "water_wqi_entropy",
    # Forecast pipelines (+7 / +14 days)
    "water_wqi_brown_7d", "water_wqi_brown_14d",
    "water_wqi_ccme_7d", "water_wqi_ccme_14d",
    "water_wqi_entropy_7d", "water_wqi_entropy_14d",
}


def _validate_pipeline(pipeline: str) -> None:
    if pipeline not in _VALID_PIPELINES:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "INVALID_PIPELINE", "message": f"Unknown pipeline: {pipeline!r}. Valid: {sorted(_VALID_PIPELINES)}"},
        )


async def _build_prediction_object(
    doc: PredictionDocument,
    db: AsyncIOMotorDatabase,
    model_cache: dict,
    include_full_xai: bool,
) -> PredictionObject:
    # ── Model info (cached) ────────────────────────────────────────────────
    model_info = model_cache.get(doc.model_id)
    if model_info is None:
        reg_repo = ModelRegistryRepository(db)
        reg_doc = await reg_repo.find_by_model_id(doc.model_id)
        if reg_doc:
            model_info = ModelInfo(
                model_id=reg_doc.model_id,
                model_type=reg_doc.model_type,
                model_family=reg_doc.model_family,
                trained_at=reg_doc.trained_at,
            )
        else:
            # Fallback if model_registry doc somehow missing
            model_info = ModelInfo(
                model_id=doc.model_id,
                model_type=doc.model_type,
                model_family=doc.model_family,
                trained_at=doc.prediction_generated_at,
            )
        model_cache[doc.model_id] = model_info

    # ── XAI block ─────────────────────────────────────────────────────────
    xai_repo = XAIResultRepository(db)
    xai_doc = await xai_repo.find_by_prediction_id(str(doc.id)) if doc.id else None

    top_features = [
        XAITopFeature(
            feature=f.feature,
            shap_value=f.shap_value,
            input_value=f.input_value,
        )
        for f in doc.top_shap_features
    ]

    full_shap_values: Optional[dict[str, float]] = None
    base_value = 0.0
    explainer_type = "unknown"

    if xai_doc:
        base_value = xai_doc.base_value
        explainer_type = xai_doc.explainer_type
        if include_full_xai:
            full_shap_values = xai_doc.shap_values

    xai_block = XAIBlock(
        explainer_type=explainer_type,
        base_value=base_value,
        top_features=top_features,
        full_shap_values=full_shap_values,
    )

    # ── Prediction interval ────────────────────────────────────────────────
    interval: Optional[PredictionInterval] = None
    if doc.prediction_interval_low is not None and doc.prediction_interval_high is not None:
        interval = PredictionInterval(
            low=doc.prediction_interval_low,
            high=doc.prediction_interval_high,
        )

    return PredictionObject(
        pipeline=doc.pipeline,
        sensor_id=doc.sensor_id,
        predicted_variable=doc.target_variable,
        predicted_value=doc.predicted_value,
        predicted_delta=doc.predicted_delta,
        prediction_interval=interval,
        prediction_timestamp=doc.prediction_generated_at,
        input_feature_timestamp=doc.input_feature_timestamp,
        target_timestamp=doc.target_timestamp,
        model=model_info,
        xai=xai_block,
        data_quality=DataQuality(input_had_filled_values=doc.input_had_filled_values),
    )


async def get_latest_results(
    pipeline: str,
    db: AsyncIOMotorDatabase,
    sensor_id: Optional[str] = None,
    include_full_xai: bool = False,
) -> LatestResultsResponse:
    """
    Return the most recent prediction per sensor for the pipeline.

    Returns LatestResultsResponse with empty results list if no predictions
    exist yet (DATA_UNAVAILABLE — caller decides on HTTP 404 vs 200).
    """
    _validate_pipeline(pipeline)
    repo = PredictionRepository(db)
    docs = await repo.find_latest(pipeline, sensor_id)

    model_cache: dict = {}
    results = []
    for doc in docs:
        try:
            obj = await _build_prediction_object(doc, db, model_cache, include_full_xai)
            results.append(obj)
        except Exception as exc:
            log.error("result_service_build_failed", sensor_id=doc.sensor_id, error=str(exc))

    return LatestResultsResponse(pipeline=pipeline, results=results)


async def get_current_wqi(db: AsyncIOMotorDatabase) -> CurrentWQIResponse:
    """
    Return the latest computed WQI values per water sensor, read from the most
    recent engineered feature document.  These are deterministic computations
    from sensor data (not model predictions).
    """
    repo = FeatureRepository(db)
    stages = [
        {"$match": {"pipeline": "water", "superseded": False}},
        {"$sort": {"feature_timestamp": -1}},
        {"$group": {
            "_id": "$sensor_id",
            "feature_timestamp": {"$first": "$feature_timestamp"},
            "features": {"$first": "$features"},
        }},
        {"$sort": {"_id": 1}},
    ]
    entries: list[CurrentWQIEntry] = []
    async for row in repo.col.aggregate(stages):
        feats = row.get("features") or {}
        entries.append(CurrentWQIEntry(
            sensor_id=row["_id"],
            feature_timestamp=row["feature_timestamp"],
            wqi_brown=feats.get("wqi_brown"),
            wqi_ccme=feats.get("wqi_ccme"),
            wqi_entropy=feats.get("wqi_entropy"),
        ))
    return CurrentWQIResponse(results=entries)


async def get_result_history(
    pipeline: str,
    db: AsyncIOMotorDatabase,
    sensor_id: Optional[str] = None,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 50,
    include_full_xai: bool = False,
) -> HistoricalResultsResponse:
    """Return paginated historical predictions for the pipeline."""
    _validate_pipeline(pipeline)

    if from_time and to_time and from_time > to_time:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "INVALID_DATE_RANGE", "message": "'from' must be before 'to'."},
        )

    repo = PredictionRepository(db)
    docs, total = await repo.find_history(
        pipeline=pipeline,
        sensor_id=sensor_id,
        from_time=from_time,
        to_time=to_time,
        page=page,
        page_size=page_size,
    )

    model_cache: dict = {}
    results = []
    for doc in docs:
        try:
            obj = await _build_prediction_object(doc, db, model_cache, include_full_xai)
            results.append(obj)
        except Exception as exc:
            log.error("result_service_build_failed", sensor_id=doc.sensor_id, error=str(exc))

    return HistoricalResultsResponse(
        pipeline=pipeline,
        sensor_id=sensor_id,
        total_count=total,
        page=page,
        page_size=page_size,
        results=results,
    )
