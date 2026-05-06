"""
Repository for the ingestion_checkpoints collection.

Upserted (not inserted) on each ingestion job execution.
One document per (source, pipeline, variable_name) for the lifetime of the deployment.

The checkpoint watermark (last_fetched_at) is ONLY advanced after a successful
MongoDB write in the same job execution.  The repository enforces this by
never updating last_fetched_at unless explicitly requested.

Retrieval patterns (per Persistence Blueprint §C.9):
  - Upsert a checkpoint record
  - Get the current watermark for a (source, pipeline, variable)
  - Record job run outcome without advancing watermark (on partial/failed run)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.collections import CHECKPOINTS
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.schemas.checkpoint import CheckpointDocument
from dss_shared.schemas.enums import DataSource, IngestionRunStatus, Pipeline


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class CheckpointRepository(BaseRepository):
    """Repository for ingestion_checkpoints."""

    COLLECTION_NAME = CHECKPOINTS

    def _key(self, source: str, pipeline: str, variable_name: str) -> dict:
        return {
            "source": source,
            "pipeline": pipeline,
            "variable_name": variable_name,
        }

    async def get_checkpoint(
        self,
        source: DataSource | str,
        pipeline: Pipeline | str,
        variable_name: str,
    ) -> Optional[CheckpointDocument]:
        """
        Return the checkpoint document for a (source, pipeline, variable).

        Returns None if no checkpoint exists yet (first run for this variable).
        """
        raw = await self.col.find_one(
            self._key(str(source), str(pipeline), variable_name)
        )
        if raw is None:
            return None
        return self._from_doc(raw, CheckpointDocument)

    async def get_last_fetched_at(
        self,
        source: DataSource | str,
        pipeline: Pipeline | str,
        variable_name: str,
    ) -> Optional[datetime]:
        """
        Return the last_fetched_at watermark for a source+variable.

        Returns None if no checkpoint exists (triggers full historical fetch).
        This is the primary accessor used by the ingestion job before each fetch.
        """
        doc = await self.get_checkpoint(source, pipeline, variable_name)
        if doc is None:
            return None
        return doc.last_fetched_at

    async def advance_watermark(
        self,
        source: DataSource | str,
        pipeline: Pipeline | str,
        variable_name: str,
        last_fetched_at: datetime,
    ) -> None:
        """
        Advance the last_fetched_at watermark after a successful write.

        Also records the run as a success and resets consecutive_failures.
        Upserts — creates the document if it does not exist yet.

        MUST only be called after successful MongoDB write in the same job.
        """
        now = _utc_now()
        await self.col.update_one(
            self._key(str(source), str(pipeline), variable_name),
            {"$set": {
                "last_fetched_at": last_fetched_at,
                "last_run_at": now,
                "last_run_status": IngestionRunStatus.SUCCESS,
                "consecutive_failures": 0,
                "updated_at": now,
            }},
            upsert=True,
        )

    async def record_run_outcome(
        self,
        source: DataSource | str,
        pipeline: Pipeline | str,
        variable_name: str,
        status: IngestionRunStatus,
        increment_failure: bool = False,
    ) -> None:
        """
        Record the outcome of a job run WITHOUT advancing the watermark.

        Used when:
          - The job succeeded partially (status=PARTIAL) but the write failed
          - The job failed entirely (status=FAILED)

        increment_failure=True increments the consecutive_failures counter.
        Upserts so that a first-ever failed run still creates a checkpoint record.
        """
        now = _utc_now()
        update: dict = {"$set": {
            "last_run_at": now,
            "last_run_status": str(status),
            "updated_at": now,
        }}
        if increment_failure:
            update["$inc"] = {"consecutive_failures": 1}

        await self.col.update_one(
            self._key(str(source), str(pipeline), variable_name),
            update,
            upsert=True,
        )
