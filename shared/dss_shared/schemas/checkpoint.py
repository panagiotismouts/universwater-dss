"""
Ingestion checkpoint schema.

CheckpointDocument
    Persistence model for the ingestion_checkpoints collection.
    One document per (source, pipeline, variable_name).  Upserted on each
    ingestion job execution.

    last_fetched_at is the most recent measured_at timestamp that was
    successfully staged (written to preprocessed_measurements).  The next
    ingestion job will fetch readings with measured_at > last_fetched_at.

    The checkpoint is NOT advanced if the MongoDB write fails (per-job
    atomicity guarantee described in Amendment B.6).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from dss_shared.schemas.base import MongoDocument, utc_now
from dss_shared.schemas.enums import DataSource, IngestionRunStatus, Pipeline


class CheckpointDocument(MongoDocument):
    """
    MongoDB document shape for the ingestion_checkpoints collection.

    Natural unique key: (source, pipeline, variable_name).
    Enforced by a unique compound index.

    Upserted (not inserted) on each job run — one document lives here for
    the lifetime of the service deployment.

    Timestamp discipline:
      last_fetched_at — domain: latest measured_at successfully staged
      last_run_at     — domain: when the ingestion job last executed
      updated_at      — system: when this document was last written
    """

    source: DataSource
    pipeline: Pipeline
    variable_name: str = Field(..., min_length=1)

    # Watermark — advanced only on successful write
    last_fetched_at: Optional[datetime] = None   # None until first successful run

    # Last run metadata
    last_run_at: Optional[datetime] = None
    last_run_status: Optional[IngestionRunStatus] = None
    consecutive_failures: int = Field(default=0, ge=0)

    # System timestamp
    updated_at: datetime = Field(default_factory=utc_now)
