"""
Engineered feature schemas.

FeatureDocument
    Application-layer representation of an engineered_features document.
    One document per (pipeline, sensor_id, feature_timestamp, feature_schema_version).
    Immutable after creation.

    The feature_timestamp is the "as of" point the feature vector represents —
    for rolling-window features this is the window end time.

    features is a flat dict {feature_name: float}.  The schema does not
    enumerate individual feature names because feature engineering evolves
    across versions.  The feature_schema_version field is the stable version
    anchor that ties a feature vector to a trained model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field, field_validator, model_validator

from dss_shared.schemas.base import DSSBaseModel, MongoDocument, utc_now
from dss_shared.schemas.enums import Pipeline


# ---------------------------------------------------------------------------
# FeatureDocument — persistence model for engineered_features
# ---------------------------------------------------------------------------

class FeatureDocument(MongoDocument):
    """
    MongoDB document shape for the engineered_features collection.

    Natural unique key: (pipeline, sensor_id, feature_timestamp, feature_schema_version).
    Enforced by a unique compound index.

    Immutable after creation (Persistence Blueprint §A.4).

    Timestamp discipline:
      feature_timestamp    — domain timestamp: the "as of" time this vector represents
      source_window_start  — earliest measured_at from preprocessed_measurements used
      source_window_end    — latest measured_at used (should equal feature_timestamp)
      created_at           — system timestamp: when this document was written

    fill_fraction field:
      Fraction of input measurements that were forward-filled (0.0–1.0).
      Used to optionally filter high-imputation feature vectors from training.
      Null is acceptable if the ingestion pipeline does not compute it.
    """

    # Identity
    pipeline: Pipeline
    sensor_id: str = Field(..., min_length=1)
    feature_timestamp: datetime         # UTC — domain "as of" timestamp
    feature_schema_version: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_\-\.]+$")

    # Feature vector
    features: dict[str, float] = Field(..., min_length=1)

    # Source window provenance
    source_window_start: datetime       # UTC
    source_window_end: datetime         # UTC

    # Fill traceability
    has_filled_inputs: bool = False
    fill_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # System timestamp
    created_at: datetime = Field(default_factory=utc_now)

    # Correction chain
    superseded: bool = False
    supersedes_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_window_order(self) -> "FeatureDocument":
        if self.source_window_start > self.source_window_end:
            raise ValueError("source_window_start must be <= source_window_end.")
        return self

    @model_validator(mode="after")
    def _validate_fill_fraction_consistency(self) -> "FeatureDocument":
        if self.fill_fraction is not None and self.fill_fraction > 0.0:
            if not self.has_filled_inputs:
                raise ValueError(
                    "has_filled_inputs must be True when fill_fraction > 0."
                )
        return self
