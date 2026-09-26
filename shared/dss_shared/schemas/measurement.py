"""
Measurement schemas.

NormalizedReading
    In-memory contract between source API clients and the preprocessing
    pipeline.  Never persisted directly.  This is the output of a source
    client's fetch() method and the input to the inline preprocessing stages.

MeasurementDocument
    Application-layer representation of a preprocessed_measurements document.
    One document per (pipeline, sensor_id, variable_name, measured_at).
    Immutable after creation per the Persistence Blueprint §A.4.

    processing_status values (Amendment C.3):
        "processed" — inline preprocessing succeeded
        "failed"    — inline preprocessing failed; failure_reason is set
    The "unprocessed" state is NOT written in v1.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field, field_validator, model_validator

from dss_shared.schemas.base import DSSBaseModel, MongoDocument, utc_now
from dss_shared.schemas.enums import (
    DataSource,
    Pipeline,
    ProcessingStatus,
    QualityFlag,
)


# ---------------------------------------------------------------------------
# NormalizedReading — internal service contract, never persisted
# ---------------------------------------------------------------------------

class NormalizedReading(DSSBaseModel):
    """
    Single sensor reading after source normalization.

    This is the output contract of BaseAPIClient.fetch() and the input
    contract of the preprocessing pipeline.  All source-specific field names
    and units are resolved to canonical DSS names before this object is
    constructed.

    Field discipline:
      measured_at  — domain timestamp: when the physical measurement occurred
      fetched_at   — system timestamp: when the API call returned this data
    """

    pipeline: Pipeline
    source: DataSource
    sensor_id: str = Field(..., min_length=1)
    variable_name: str = Field(..., min_length=1)
    raw_value: float
    unit: str = Field(..., min_length=1)
    measured_at: datetime   # UTC — domain timestamp
    fetched_at: datetime    # UTC — system timestamp

    @field_validator("measured_at", "fetched_at", mode="before")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if isinstance(v, datetime) and v.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware (UTC).")
        return v


# ---------------------------------------------------------------------------
# MeasurementDocument — persistence model for preprocessed_measurements
# ---------------------------------------------------------------------------

class FillMetadata(DSSBaseModel):
    """
    Forward-fill provenance metadata, embedded on MeasurementDocument when
    filled=True.

    fill_source_measured_at — the measured_at of the prior reading that was
                              carried forward
    fill_gap_seconds        — seconds between fill_source_measured_at and
                              this document's measured_at
    """

    fill_source_measured_at: datetime
    fill_gap_seconds: int = Field(..., ge=0)


class MeasurementDocument(MongoDocument):
    """
    MongoDB document shape for the preprocessed_measurements collection.

    Natural unique key: (pipeline, sensor_id, variable_name, measured_at).
    Enforced by the unique compound index uq_measurement_observation, defined in
    dss_shared.db.collections and created by dss_shared.db.bootstrap_db at every
    service startup (also runnable on demand via scripts/create_indexes.py).

    Immutability: only processing_status and processed_at are updated after
    initial insert.  All other fields are written once.

    Timestamp discipline:
      measured_at  — domain timestamp: when the physical reading occurred
      processed_at — domain-ish timestamp: when inline preprocessing ran
      created_at   — system timestamp: when the document was inserted
    """

    # Identity
    pipeline: Pipeline
    source: DataSource
    sensor_id: str = Field(..., min_length=1)
    variable_name: str = Field(..., min_length=1)

    # Value
    value: float
    unit: str = Field(..., min_length=1)

    # Timestamps
    measured_at: datetime           # UTC — domain timestamp
    created_at: datetime = Field(default_factory=utc_now)
    processed_at: Optional[datetime] = None  # Set when processing_status transitions

    # Preprocessing outcome (Amendment C.3: only "processed" | "failed" in v1)
    processing_status: ProcessingStatus = ProcessingStatus.PROCESSED
    failure_reason: Optional[str] = None  # Required when processing_status="failed"

    # Forward-fill traceability
    filled: bool = False
    fill_metadata: Optional[FillMetadata] = None  # Required when filled=True

    # Quality annotation
    quality_flag: QualityFlag = QualityFlag.OK

    # Correction chain
    superseded: bool = False
    supersedes_id: Optional[str] = None  # ObjectId string of replaced document

    @model_validator(mode="after")
    def _validate_fill_consistency(self) -> "MeasurementDocument":
        if self.filled and self.fill_metadata is None:
            raise ValueError("fill_metadata is required when filled=True.")
        if not self.filled and self.fill_metadata is not None:
            raise ValueError("fill_metadata must be None when filled=False.")
        return self

    @model_validator(mode="after")
    def _validate_failure_consistency(self) -> "MeasurementDocument":
        if self.processing_status == ProcessingStatus.FAILED and not self.failure_reason:
            raise ValueError("failure_reason is required when processing_status='failed'.")
        return self
