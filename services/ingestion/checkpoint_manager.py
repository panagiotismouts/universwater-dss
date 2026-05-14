"""
Checkpoint manager.

Thin wrapper around CheckpointRepository that:
  - Provides get_since() — the watermark datetime the client should fetch from
  - Provides advance() — called ONLY after successful MongoDB write
  - Provides record_failure() — called on job failure (does NOT advance watermark)

Bootstrap mode (§D.4):
  When DSS_BOOTSTRAP_MODE=true, get_since() returns the configured historical
  start date regardless of any stored checkpoint.  This triggers a full
  historical backfill on first deployment.

The checkpoint key is (source, pipeline, variable_name) per CheckpointRepository.
Met variables write two checkpoints — one per pipeline (met_water, met_soil) —
because they produce two NormalizedReadings per reading (§E.3).
"""

from __future__ import annotations

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_settings
from dss_shared.db.repositories.checkpoints import CheckpointRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import IngestionRunStatus

log = get_logger(__name__)

# Historical start date used in bootstrap mode (§D.4).
_BOOTSTRAP_EPOCH = datetime(2022, 1, 1, tzinfo=timezone.utc)

# Fallback when no checkpoint exists and bootstrap mode is off.
_DEFAULT_SINCE = datetime(2025, 10, 1, tzinfo=timezone.utc)


class CheckpointManager:
    """
    Per-job checkpoint manager.

    Each ingestion job instantiates one CheckpointManager (or receives a shared
    instance) and uses it for the lifetime of that job execution.
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._repo = CheckpointRepository(db)

    async def get_since(
        self,
        source: str,
        pipeline: str,
        variable_name: str,
    ) -> datetime:
        """
        Return the datetime the client should fetch from.

        Priority:
          1. Bootstrap mode → return _BOOTSTRAP_EPOCH
          2. Stored checkpoint → return last_fetched_at
          3. No checkpoint → return _DEFAULT_SINCE
        """
        settings = get_settings()
        if settings.bootstrap_mode:
            log.info(
                "checkpoint_bootstrap_mode",
                source=source,
                pipeline=pipeline,
                variable=variable_name,
            )
            return _BOOTSTRAP_EPOCH

        last_fetched = await self._repo.get_last_fetched_at(source, pipeline, variable_name)
        if last_fetched is None:
            log.info(
                "checkpoint_first_run",
                source=source,
                pipeline=pipeline,
                variable=variable_name,
                since=_DEFAULT_SINCE.isoformat(),
            )
            return _DEFAULT_SINCE

        return last_fetched

    @staticmethod
    def filter_new(readings: list, since: datetime) -> list:
        """
        Return only readings with measured_at > since.

        This is the checkpoint-based deduplication filter (§D.2, §D.3).
        Applied after get_since(), before preprocessing.
        """
        return [r for r in readings if r.measured_at > since]

    @staticmethod
    def max_measured_at(readings: list) -> datetime | None:
        """
        Return the maximum measured_at from a list of NormalizedReadings.
        Returns None for an empty list.
        """
        if not readings:
            return None
        return max(r.measured_at for r in readings)

    async def advance(
        self,
        source: str,
        pipeline: str,
        variable_name: str,
        last_fetched_at: datetime,
    ) -> None:
        """
        Advance the checkpoint watermark after a successful MongoDB write.
        MUST only be called after the write has completed successfully.
        """
        await self._repo.advance_watermark(source, pipeline, variable_name, last_fetched_at)
        log.debug(
            "checkpoint_advanced",
            source=source,
            pipeline=pipeline,
            variable=variable_name,
            last_fetched_at=last_fetched_at.isoformat(),
        )

    async def record_failure(
        self,
        source: str,
        pipeline: str,
        variable_name: str,
    ) -> None:
        """Record a job failure without advancing the watermark."""
        await self._repo.record_run_outcome(
            source=source,
            pipeline=pipeline,
            variable_name=variable_name,
            status=IngestionRunStatus.FAILED,
            increment_failure=True,
        )

    async def record_partial(
        self,
        source: str,
        pipeline: str,
        variable_name: str,
    ) -> None:
        """Record a partial run without advancing the watermark."""
        await self._repo.record_run_outcome(
            source=source,
            pipeline=pipeline,
            variable_name=variable_name,
            status=IngestionRunStatus.PARTIAL,
            increment_failure=False,
        )
