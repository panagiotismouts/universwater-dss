"""
Feature engineering coordinator.

Receives a batch of successfully processed MeasurementDocuments, determines
which (pipeline, sensor_id) pairs need feature vectors computed, dispatches
to the correct pipeline-specific engineer, and writes FeatureDocuments.

Only documents with processing_status="processed" are passed to feature
engineering.  Failed readings are skipped (§H.1 boundary).

Feature timestamps:
  For each (pipeline, sensor_id) pair, features are computed at 1-hour
  intervals across the full time range of the batch, starting 6h after
  the earliest measurement (to ensure enough data exists in the rolling
  window).  This handles both the initial historical backfill (large batch
  spanning months) and ongoing incremental polls (small batch spanning
  15–60 min, producing 1 feature doc).

Pipeline dispatch:
  "water"     → compute_water_features
  "soil"      → compute_soil_features
  "met_water" → skipped (met variables contribute via water_features' window reads)
  "met_soil"  → skipped (met variables contribute via soil_features' window reads)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.repositories.features import FeatureRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import ProcessingStatus
from dss_shared.schemas.measurement import MeasurementDocument
from services.ingestion.feature_engineering.soil_features import compute_soil_features
from services.ingestion.feature_engineering.water_features import compute_water_features

log = get_logger(__name__)

# Pipelines that own feature computation.
# Met variables are consumed by the water/soil engineers as co-features.
_FEATURE_PIPELINES = {"water", "soil"}


async def coordinate_feature_engineering(
    processed_docs: list[MeasurementDocument],
    db: AsyncIOMotorDatabase,
) -> None:
    """
    Run feature engineering for all successfully processed documents.

    Groups documents by (pipeline, sensor_id), computes one feature vector
    per group using the most recent measured_at as feature_timestamp, then
    upserts the resulting FeatureDocument.
    """
    feature_repo = FeatureRepository(db)

    # Only process successfully preprocessed docs
    ok_docs = [
        d for d in processed_docs
        if d.processing_status == ProcessingStatus.PROCESSED
    ]
    if not ok_docs:
        return

    # Find time range (min, max measured_at) per (pipeline, sensor_id)
    _UNSET = datetime.min.replace(tzinfo=timezone.utc)
    time_range: dict[tuple[str, str], list[datetime]] = {}
    for doc in ok_docs:
        if str(doc.pipeline) not in _FEATURE_PIPELINES:
            continue
        key = (str(doc.pipeline), doc.sensor_id)
        if key not in time_range:
            time_range[key] = [doc.measured_at, doc.measured_at]
        else:
            if doc.measured_at < time_range[key][0]:
                time_range[key][0] = doc.measured_at
            if doc.measured_at > time_range[key][1]:
                time_range[key][1] = doc.measured_at

    _1H = timedelta(hours=1)
    _6H = timedelta(hours=6)

    for (pipeline, sensor_id), (batch_start, batch_end) in time_range.items():
        # For large historical batches: iterate hourly from batch_start+6h to batch_end.
        # The 6h offset ensures the rolling window has enough historical data on first run.
        ts = batch_start + _6H
        while ts <= batch_end:
            await _compute_and_persist(pipeline, sensor_id, ts, db, feature_repo)
            ts += _1H
        # Always compute for batch_end itself.
        # For small incremental batches (< 6h), this is the only feature generated.
        # Historical data already in MongoDB satisfies the rolling window requirement.
        if batch_end != ts - _1H:
            await _compute_and_persist(pipeline, sensor_id, batch_end, db, feature_repo)


async def _compute_and_persist(
    pipeline: str,
    sensor_id: str,
    feature_timestamp: datetime,
    db: AsyncIOMotorDatabase,
    repo: FeatureRepository,
) -> None:
    """Compute a feature vector for one (pipeline, sensor_id) and upsert it."""
    try:
        if pipeline == "water":
            feature_doc = await compute_water_features(db, sensor_id, feature_timestamp)
        elif pipeline == "soil":
            feature_doc = await compute_soil_features(db, sensor_id, feature_timestamp)
        else:
            return

        if feature_doc is None:
            log.debug(
                "feature_engineering_skipped_no_data",
                pipeline=pipeline,
                sensor_id=sensor_id,
                feature_timestamp=feature_timestamp.isoformat(),
            )
            return

        inserted = await repo.upsert(feature_doc)
        log.info(
            "feature_document_written" if inserted else "feature_document_already_exists",
            pipeline=pipeline,
            sensor_id=sensor_id,
            feature_timestamp=feature_timestamp.isoformat(),
        )

    except Exception as exc:
        # Feature engineering failure is non-fatal per §A.5 (fail partial not total)
        log.error(
            "feature_engineering_error",
            pipeline=pipeline,
            sensor_id=sensor_id,
            feature_timestamp=feature_timestamp.isoformat(),
            error=str(exc),
        )
