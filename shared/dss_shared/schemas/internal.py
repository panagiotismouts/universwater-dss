"""
Internal service contracts.

These models are never persisted and never appear in API responses.
They define the data contracts between layers within a single service,
making function signatures self-documenting and type-safe.

TrainingRunSummary
    Summary returned by bootstrap/recalibration orchestrators to their callers.
    Communicates outcome without requiring callers to re-query MongoDB.

PredictionPayload
    In-memory prediction result assembled by predictor.py before being split
    into PredictionDocument + XAIResultDocument for separate persistence.

FeatureRow
    A single feature vector as a typed Python object, produced by dataset_builder
    and consumed by model wrappers.  Bridges the gap between MongoDB documents
    and numpy array construction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from dss_shared.schemas.base import DSSBaseModel
from dss_shared.schemas.enums import ModelFamily, ModelStatus, ModelType, Pipeline


# ---------------------------------------------------------------------------
# TrainingRunSummary — returned by bootstrap/recalibration orchestrators
# ---------------------------------------------------------------------------

class TrainingRunSummary(DSSBaseModel):
    """
    Outcome summary returned by run_bootstrap_if_needed() and run_recalibration().

    Allows callers to log outcomes and make branching decisions without
    re-querying the model_registry collection.

    activated_model_id — set if a new model was activated; None if no change
    rejected_reason    — set if metric gate failed
    """

    pipeline: Pipeline
    model_type: ModelType
    model_id: str
    outcome: ModelStatus   # "active" (promoted), "rejected", "candidate" (training failed)
    activated_model_id: Optional[str] = None
    retired_model_id: Optional[str] = None   # The model retired during promotion
    rejected_reason: Optional[str] = None
    r2: Optional[float] = None
    mae: Optional[float] = None


# ---------------------------------------------------------------------------
# PredictionPayload — assembled by predictor.py before persistence split
# ---------------------------------------------------------------------------

class PredictionPayload(DSSBaseModel):
    """
    Complete prediction result (prediction value + XAI) assembled in-memory
    by predictor.py before splitting into two separate MongoDB documents.

    This object is never serialized to JSON or stored in MongoDB directly.
    The predictor constructs this, then persistence code splits it into:
      - PredictionDocument  → prediction_results
      - XAIResultDocument   → xai_results
    """

    pipeline: Pipeline
    sensor_id: str
    model_id: str
    model_type: ModelType
    model_family: ModelFamily
    target_variable: str

    # Prediction output
    predicted_value: float
    prediction_interval_low: Optional[float] = None
    prediction_interval_high: Optional[float] = None

    # Feature input provenance
    input_feature_timestamp: datetime
    input_had_filled_values: bool
    feature_input_values: dict[str, float]

    # XAI output
    explainer_type: str
    base_value: float
    shap_values: dict[str, float]     # Full SHAP vector for all features
    top_shap_features: list[dict]     # Pre-computed top-N [{feature, shap_value, input_value}]

    # Timing
    prediction_generated_at: datetime


# ---------------------------------------------------------------------------
# FeatureRow — typed feature vector produced by dataset_builder
# ---------------------------------------------------------------------------

class FeatureRow(DSSBaseModel):
    """
    Single feature vector with its associated target value and metadata.

    Produced by dataset_builder.build_dataset() and consumed by model
    wrappers during training.  This is an intermediate in-memory contract —
    not stored in MongoDB.

    feature_values is ordered consistently with model's expected feature_names.
    target_value is the label for supervised training.  Set to None for
    prediction-only rows (the current feature vector).
    """

    pipeline: Pipeline
    sensor_id: str
    feature_timestamp: datetime
    feature_schema_version: str
    feature_values: dict[str, float] = Field(..., min_length=1)
    target_value: Optional[float] = None
    has_filled_inputs: bool = False
