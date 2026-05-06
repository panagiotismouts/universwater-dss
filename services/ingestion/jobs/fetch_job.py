"""
Core ingestion job function — the full inline pipeline.

This function is called once per (source, variable) pair on each scheduler
tick.  It executes the complete data pipeline inline:

  [1] Source client fetch
  [2] Checkpoint filter  → de-duplicated, new-only NormalizedReadings
  [3] Preprocessing      → 6-stage inline pipeline → MeasurementDocuments
  [4] MongoDB write      → preprocessed_measurements (ordered=False, idempotent)
  [5] Feature engineering→ per (pipeline, sensor_id) FeatureDocuments
  [6] Checkpoint advance → only on successful write

Per-job atomicity (Amendment B.6):
  Checkpoint is NOT advanced if the MongoDB write fails.
  The next scheduler tick will re-fetch the same data.
  Idempotency is preserved by the unique index on preprocessed_measurements.

Failure isolation (§A.5):
  SourceAPIError   → log WARNING, update checkpoint as failed, continue
  PreprocessingError per reading → failure document written, pipeline continues
  MongoDB write failure → log ERROR, record partial, do NOT advance checkpoint
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.repositories.measurements import MeasurementRepository
from dss_shared.exceptions import SourceAPIError, SourceAPITimeoutError
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import Pipeline, ProcessingStatus
from services.ingestion.checkpoint_manager import CheckpointManager
from services.ingestion.clients.base_client import BaseAPIClient
from services.ingestion.feature_engineering.coordinator import coordinate_feature_engineering
from services.ingestion.preprocessing.pipeline import run_preprocessing_pipeline

log = get_logger(__name__)


def make_fetch_job(
    client: BaseAPIClient,
    variable_name: str,
    pipeline: str,
    db: AsyncIOMotorDatabase,
) -> Callable[[], Coroutine[Any, Any, None]]:
    """
    Factory that returns an async job coroutine bound to a specific
    (client, variable_name, pipeline, db) combination.

    The returned coroutine is registered with APScheduler as the job callable.

    Args:
        client:         Instantiated source API client (WingsWaterClient, etc.)
        variable_name:  Canonical variable name (e.g. "ph", "rainfall")
        pipeline:       Target pipeline string (e.g. "water", "met_water")
        db:             Async Motor database handle
    """
    source = client.source_name
    cm = CheckpointManager(db)
    meas_repo = MeasurementRepository(db)

    async def run() -> None:
        log.info(
            "fetch_job_started",
            source=source,
            variable=variable_name,
            pipeline=pipeline,
        )

        # ── [1] Get checkpoint watermark ──────────────────────────────────────
        try:
            since = await cm.get_since(source, pipeline, variable_name)
        except Exception as exc:
            log.error(
                "fetch_job_checkpoint_read_failed",
                source=source,
                variable=variable_name,
                error=str(exc),
            )
            return  # Cannot proceed without knowing the watermark

        # ── [2] Fetch from source API ─────────────────────────────────────────
        try:
            all_readings = await client.fetch(variable_name, since)
        except (SourceAPIError, SourceAPITimeoutError) as exc:
            log.warning(
                "fetch_job_source_api_failed",
                source=source,
                variable=variable_name,
                error=str(exc),
            )
            await cm.record_failure(source, pipeline, variable_name)
            return

        # ── [3] Checkpoint filter (deduplication) ─────────────────────────────
        new_readings = CheckpointManager.filter_new(all_readings, since)
        if not new_readings:
            log.debug(
                "fetch_job_no_new_data",
                source=source,
                variable=variable_name,
                total_fetched=len(all_readings),
            )
            return

        log.info(
            "fetch_job_new_readings",
            source=source,
            variable=variable_name,
            count=len(new_readings),
        )

        # ── [4] Preprocessing pipeline ────────────────────────────────────────
        measurement_docs = await run_preprocessing_pipeline(new_readings, db)

        # ── [5] MongoDB write — preprocessed_measurements ─────────────────────
        try:
            inserted, duplicates = await meas_repo.insert_many(measurement_docs)
            log.info(
                "fetch_job_measurements_written",
                source=source,
                variable=variable_name,
                inserted=inserted,
                duplicates=duplicates,
                failed=sum(
                    1 for d in measurement_docs
                    if d.processing_status == ProcessingStatus.FAILED
                ),
            )
        except Exception as exc:
            log.error(
                "fetch_job_write_failed",
                source=source,
                variable=variable_name,
                error=str(exc),
            )
            await cm.record_partial(source, pipeline, variable_name)
            return  # Do not advance checkpoint; do not run feature engineering

        # ── [6] Feature engineering ───────────────────────────────────────────
        try:
            await coordinate_feature_engineering(measurement_docs, db)
        except Exception as exc:
            # Feature engineering failure is non-fatal (§A.5)
            log.error(
                "fetch_job_feature_engineering_error",
                source=source,
                variable=variable_name,
                error=str(exc),
            )

        # ── [7] Advance checkpoint ─────────────────────────────────────────────
        new_watermark = CheckpointManager.max_measured_at(new_readings)
        if new_watermark is not None:
            await cm.advance(source, pipeline, variable_name, new_watermark)

        log.info(
            "fetch_job_completed",
            source=source,
            variable=variable_name,
            watermark=new_watermark.isoformat() if new_watermark else None,
        )

    run.__name__ = f"fetch_{source}_{pipeline}_{variable_name}"
    return run
