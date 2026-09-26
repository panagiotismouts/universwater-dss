"""
Feature engineering coordinator.

Receives a batch of successfully processed MeasurementDocuments, determines
which (pipeline, sensor_id) pairs need feature vectors computed, dispatches
to the correct pipeline-specific engineer, and writes FeatureDocuments.

Only documents with processing_status="processed" are passed to feature
engineering.  Failed readings are skipped (§H.1 boundary).

Feature timestamps:
  For each (pipeline, sensor_id) pair, features are computed at 1-hour
  intervals across the time range of the batch, starting 6h after the
  earliest measurement (to ensure enough data exists in the rolling
  window), plus the batch end itself.  This handles both a multi-day
  catch-up (after an outage) and ongoing incremental polls (small batch
  spanning 15–60 min, producing 1 feature doc).

Bounded work per run:
  A batch with no prior checkpoint can span a year.  Walking that hourly
  inside a scheduler tick would hold the job's single slot for hours, so
  the walk is capped at settings.feature_backfill_max_hours most-recent
  hours.  Anything older is logged as "feature_backfill_truncated" and
  must be backfilled explicitly (scripts/).

Skip-if-exists:
  Before computing a vector the coordinator checks whether one already
  exists for (pipeline, sensor_id, timestamp, schema_version).  Computing
  is ~20 window queries; the check is one index lookup.  Re-runs over
  already-covered ranges therefore cost almost nothing.

Pipeline dispatch:
  "water"     → compute_water_features
  "soil"      → compute_soil_features
  "met_water" → skipped (met variables contribute via water_features' window reads)
  "met_soil"  → skipped (met variables contribute via soil_features' window reads)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_settings
from dss_shared.db.repositories.features import FeatureRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import ProcessingStatus
from dss_shared.schemas.measurement import MeasurementDocument
from services.ingestion.feature_engineering.soil_features import (
    FEATURE_SCHEMA_VERSION as SOIL_SCHEMA_VERSION,
    compute_soil_features,
)
from services.ingestion.feature_engineering.water_features import (
    FEATURE_SCHEMA_VERSION as WATER_SCHEMA_VERSION,
    compute_water_features,
)

log = get_logger(__name__)

# Pipelines that own feature computation.
# Met variables are consumed by the water/soil engineers as co-features.
_FEATURE_PIPELINES = {"water", "soil"}

_SCHEMA_VERSIONS: dict[str, str] = {
    "water": WATER_SCHEMA_VERSION,
    "soil": SOIL_SCHEMA_VERSION,
}

_1H = timedelta(hours=1)
_6H = timedelta(hours=6)


def plan_feature_timestamps(
    batch_start: datetime,
    batch_end: datetime,
    max_hours: int,
) -> tuple[list[datetime], datetime | None]:
    """
    Decide which feature timestamps a batch should produce.

    Returns (timestamps, truncated_from):
      timestamps      — ascending, hourly from batch_start+6h through batch_end,
                        always ending with batch_end itself; at most max_hours
                        entries, keeping the most recent ones.
      truncated_from  — the earliest timestamp that was dropped by the cap,
                        or None if nothing was dropped.

    Pure function so the scheduling policy can be unit-tested without Mongo.
    """
    if max_hours < 1:
        raise ValueError("max_hours must be >= 1")

    timestamps: list[datetime] = []
    ts = batch_start + _6H
    while ts <= batch_end:
        timestamps.append(ts)
        ts += _1H
    # Always compute for batch_end itself (small incremental batches < 6h
    # produce only this one).  Historical data already in MongoDB satisfies
    # the rolling-window requirement.
    if not timestamps or timestamps[-1] != batch_end:
        timestamps.append(batch_end)

    if len(timestamps) <= max_hours:
        return timestamps, None
    dropped = timestamps[:-max_hours]
    return timestamps[-max_hours:], dropped[0]


async def coordinate_feature_engineering(
    processed_docs: list[MeasurementDocument],
    db: AsyncIOMotorDatabase,
    *,
    max_hours: int | None = None,
) -> None:
    """
    Run feature engineering for all successfully processed documents.

    Groups documents by (pipeline, sensor_id), plans the hourly feature
    timestamps for each group (bounded by max_hours, default from
    settings.feature_backfill_max_hours), skips timestamps that already
    have a vector, computes and upserts the rest.
    """
    feature_repo = FeatureRepository(db)
    if max_hours is None:
        max_hours = get_settings().feature_backfill_max_hours

    # Only process successfully preprocessed docs
    ok_docs = [
        d for d in processed_docs
        if d.processing_status == ProcessingStatus.PROCESSED
    ]
    if not ok_docs:
        return

    # Find time range (min, max measured_at) per (pipeline, sensor_id)
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

    for (pipeline, sensor_id), (batch_start, batch_end) in time_range.items():
        timestamps, truncated_from = plan_feature_timestamps(batch_start, batch_end, max_hours)
        if truncated_from is not None:
            log.warning(
                "feature_backfill_truncated",
                pipeline=pipeline,
                sensor_id=sensor_id,
                batch_start=batch_start.isoformat(),
                batch_end=batch_end.isoformat(),
                truncated_from=truncated_from.isoformat(),
                computing_from=timestamps[0].isoformat(),
                max_hours=max_hours,
                hint="Run the explicit backfill script for the older range",
            )

        written = skipped = failed = 0
        for ts in timestamps:
            outcome = await _compute_and_persist(pipeline, sensor_id, ts, db, feature_repo)
            if outcome == "written":
                written += 1
            elif outcome == "skipped":
                skipped += 1
            elif outcome == "error":
                failed += 1

        log.info(
            "feature_engineering_group_done",
            pipeline=pipeline,
            sensor_id=sensor_id,
            planned=len(timestamps),
            written=written,
            skipped_existing=skipped,
            errors=failed,
        )


async def _compute_and_persist(
    pipeline: str,
    sensor_id: str,
    feature_timestamp: datetime,
    db: AsyncIOMotorDatabase,
    repo: FeatureRepository,
) -> str:
    """
    Compute a feature vector for one (pipeline, sensor_id, timestamp) and
    upsert it.  Returns one of "written", "exists", "skipped", "no_data",
    "error" for the caller's summary counters.
    """
    schema_version = _SCHEMA_VERSIONS.get(pipeline)
    if schema_version is None:
        return "skipped"

    try:
        # Cheap index lookup before the ~20 window queries a computation costs.
        if await repo.exists(pipeline, sensor_id, feature_timestamp, schema_version):
            log.debug(
                "feature_document_already_exists",
                pipeline=pipeline,
                sensor_id=sensor_id,
                feature_timestamp=feature_timestamp.isoformat(),
            )
            return "skipped"

        if pipeline == "water":
            feature_doc = await compute_water_features(db, sensor_id, feature_timestamp)
        else:
            feature_doc = await compute_soil_features(db, sensor_id, feature_timestamp)

        if feature_doc is None:
            log.debug(
                "feature_engineering_skipped_no_data",
                pipeline=pipeline,
                sensor_id=sensor_id,
                feature_timestamp=feature_timestamp.isoformat(),
            )
            return "no_data"

        inserted = await repo.upsert(feature_doc)
        log.info(
            "feature_document_written" if inserted else "feature_document_already_exists",
            pipeline=pipeline,
            sensor_id=sensor_id,
            feature_timestamp=feature_timestamp.isoformat(),
        )
        return "written" if inserted else "exists"

    except Exception as exc:
        # Feature engineering failure is non-fatal per §A.5 (fail partial not total)
        log.error(
            "feature_engineering_error",
            pipeline=pipeline,
            sensor_id=sensor_id,
            feature_timestamp=feature_timestamp.isoformat(),
            error=str(exc),
        )
        return "error"
