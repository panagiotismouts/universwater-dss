"""
Prediction result schemas.

TopShapFeature
    Embedded SHAP summary per feature, stored inline on PredictionDocument
    for fast API serving without a secondary collection lookup.

PredictionDocument
    Persistence model for the prediction_results collection.
    One document per (pipeline, sensor_id, input_feature_timestamp).
    Immutable after creation.

    Cross-collection references:
      model_id       → model_registry.model_id  (string key)
      xai_result_id  → xai_results._id           (ObjectId string)

    Denormalized fields (copied from model_registry at prediction time to
    avoid joins on the hot read path):
      model_type
      model_family
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field, model_validator

from dss_shared.schemas.base import DSSBaseModel, MongoDocument, PyObjectId, utc_now
from dss_shared.schemas.enums import ModelFamily, ModelType, Pipeline


# ---------------------------------------------------------------------------
# TopShapFeature — embedded on PredictionDocument
# ---------------------------------------------------------------------------

class TopShapFeature(DSSBaseModel):
    """
    Single feature's SHAP contribution, embedded in prediction_results.

    top_shap_features stores the top-N features by absolute SHAP value.
    N is configured in config.yaml under api.results.shap_top_n (default 5).
    Both positive and negative contributors are included.
    """

    feature: str
    shap_value: float
    input_value: float


# ---------------------------------------------------------------------------
# PredictionDocument
# ---------------------------------------------------------------------------

class PredictionDocument(MongoDocument):
    """
    MongoDB document shape for the prediction_results collection.

    Natural unique key: (pipeline, sensor_id, input_feature_timestamp).
    Enforced by a unique compound index.

    Immutable after creation (Persistence Blueprint §A.4).

    Timestamp discipline:
      input_feature_timestamp — domain: the "as of" time of the input features
      prediction_generated_at — domain: when the ML engine produced this prediction
      created_at              — system: when this document was written to MongoDB
    """

    # Identity / classification
    pipeline: Pipeline
    sensor_id: str = Field(..., min_length=1)
    target_variable: str = Field(..., min_length=1)

    # Prediction output
    predicted_value: float
    prediction_interval_low: Optional[float] = None
    prediction_interval_high: Optional[float] = None

    # Input provenance
    input_feature_timestamp: datetime       # UTC — domain
    input_had_filled_values: bool = False   # Copied from FeatureDocument.has_filled_inputs

    # Forecast target time (input_feature_timestamp + horizon).
    # None for nowcast pipelines (soil, retired wqi nowcasts).
    target_timestamp: Optional[datetime] = None    # UTC — domain

    # Model reference (denormalized for join-free reads)
    model_id: str = Field(..., min_length=1)
    model_type: ModelType
    model_family: ModelFamily

    # XAI reference
    xai_result_id: PyObjectId           # FK → xai_results._id

    # Embedded SHAP summary (top-N by |shap_value|)
    top_shap_features: list[TopShapFeature] = Field(default_factory=list)

    # Timestamps
    prediction_generated_at: datetime   # UTC — domain
    created_at: datetime = Field(default_factory=utc_now)

    # Correction chain
    superseded: bool = False
    supersedes_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_interval_consistency(self) -> "PredictionDocument":
        low = self.prediction_interval_low
        high = self.prediction_interval_high
        both_set = low is not None and high is not None
        neither_set = low is None and high is None
        if not both_set and not neither_set:
            raise ValueError(
                "prediction_interval_low and prediction_interval_high must "
                "both be set or both be None."
            )
        if both_set and low > high:  # type: ignore[operator]
            raise ValueError(
                "prediction_interval_low must be <= prediction_interval_high."
            )
        return self
