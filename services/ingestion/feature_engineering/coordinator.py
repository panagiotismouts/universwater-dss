"""
Feature engineering coordinator.

Receives a batch of successfully processed MeasurementDocuments, determines
which (pipeline, sensor_id) pairs need feature vectors computed, dispatches
to the correct pipeline-specific engineer, and writes FeatureDocuments.

Only documents with processing_status="processed" are passed to feature
engineering.  Failed readings are skipped (§H.1 boundary).

Feature timestamp:
  For each (pipeline, sensor_id) pair, the feature timestamp is the
  measured_at of the most recent processed document in the batch.
  This represents "we now have data up to this point in time".

Pipeline dispatch:
  "water"     → compute_water_features
  "soil"      → compute_soil_features
  "met_water" → skipped (met variables contribute via water_features' window reads)
  "met_soil"  → skipped (met variables contribute via soil_features' window reads)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

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

    # Group by (pipeline, sensor_id) → latest measured_at
    latest_ts: dict[tuple[str, str], datetime] = defaultdict(
        lambda: datetime.min.replace(tzinfo=None)
    )
    for doc in ok_docs:
        if str(doc.pipeline) not in _FEATURE_PIPELINES:
            continue
        key = (str(doc.pipeline), doc.sensor_id)
        if doc.measured_at > latest_ts[key]:
            latest_ts[key] = doc.measured_at

    for (pipeline, sensor_id), feature_timestamp in latest_ts.items():
        await _compute_and_persist(
            pipeline=pipeline,
            sensor_id=sensor_id,
            feature_timestamp=feature_timestamp,
            db=db,
            repo=feature_repo,
        )


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
