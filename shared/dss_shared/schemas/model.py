"""
Model lifecycle schemas.

ModelRegistryDocument
    Persistence model for the model_registry collection.
    One document per training run.  The authoritative source for which model
    is currently active per pipeline.

    model_id format: "{pipeline}_{model_type}_{YYYYMMDD}_{HHMMSS}"
    Example:         "water_xgboost_20260315_020134"

    Status transitions (managed exclusively by registry_manager.py):
        candidate → active    (metric gate passed)
        active    → retired   (newer model activated)
        candidate → rejected  (metric gate failed)

    At most one document per pipeline may have status="active".
    Enforced by a partial unique index and by registry_manager.

ModelMetricsDocument
    Persistence model for the model_metrics collection.
    Full evaluation outputs per model evaluation run.  Kept separate from
    model_registry to keep the registry document lean and to allow multiple
    evaluation passes (e.g. bootstrap eval + post-deployment check).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

from pydantic import Field, model_validator

from dss_shared.schemas.base import DSSBaseModel, MongoDocument, utc_now
from dss_shared.schemas.enums import (
    EvaluationType,
    ModelFamily,
    ModelStatus,
    ModelType,
    Pipeline,
)


# ---------------------------------------------------------------------------
# EmbeddedMetricsSummary — embedded in ModelRegistryDocument
# ---------------------------------------------------------------------------

class EmbeddedMetricsSummary(DSSBaseModel):
    """
    Compact copy of key evaluation metrics embedded on the model_registry
    document.  Avoids a join on the hot active-model lookup path.

    Full metrics are stored in the corresponding ModelMetricsDocument.
    """

    r2: float
    mae: float
    rmse: float
    mse: Optional[float] = None
    mape: Optional[float] = None
    # Persistence-baseline MAE (mean |Δ| on the validation set) — present only
    # for delta-target forecast models; mae < baseline_mae ⇔ beats persistence.
    baseline_mae: Optional[float] = None


# ---------------------------------------------------------------------------
# ModelRegistryDocument
# ---------------------------------------------------------------------------

class ModelRegistryDocument(MongoDocument):
    """
    MongoDB document shape for the model_registry collection.

    Immutability: only status, activated_at, retired_at, and rejection_reason
    are updated during lifecycle transitions.  All other fields are written once.

    Timestamp discipline:
      training_data_start  — domain: earliest feature_timestamp in training set
      training_data_end    — domain: latest feature_timestamp in training set
      trained_at           — domain: when training completed
      activated_at         — domain: when status changed to "active"
      retired_at           — domain: when status changed to "retired"
      created_at           — system: when this document was first written
    """

    # Stable human-readable identifier
    model_id: str = Field(
        ...,
        min_length=1,
        description='Format: "{pipeline}_{model_type}_{YYYYMMDD}_{HHMMSS}"',
    )

    # Classification
    pipeline: Pipeline
    model_type: ModelType
    model_family: ModelFamily

    # Training data provenance
    feature_schema_version: str = Field(..., min_length=1)
    target_variable: str = Field(..., min_length=1)
    # True when the model was trained on horizon deltas (WQI(t+h) − WQI(t)).
    # The predictor anchors such models: forecast = current WQI + predicted Δ.
    target_is_delta: bool = False
    training_data_start: datetime       # UTC — domain
    training_data_end: datetime         # UTC — domain
    training_sample_count: int = Field(..., ge=1)
    feature_names: list[str] = Field(..., min_length=1)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    # Artifact
    artifact_path: str = Field(..., min_length=1)

    # Feature pipeline override (WQI sub-pipelines store features under "water")
    feature_pipeline: Optional[str] = None

    # Embedded metrics (fast read path)
    metrics_summary: EmbeddedMetricsSummary

    # Lifecycle
    status: ModelStatus = ModelStatus.CANDIDATE
    trained_at: datetime                # UTC — domain
    activated_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    # System timestamp
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate_training_window(self) -> "ModelRegistryDocument":
        if self.training_data_start > self.training_data_end:
            raise ValueError("training_data_start must be <= training_data_end.")
        return self

    @model_validator(mode="after")
    def _validate_status_fields(self) -> "ModelRegistryDocument":
        if self.status == ModelStatus.ACTIVE and self.activated_at is None:
            raise ValueError("activated_at is required when status='active'.")
        if self.status == ModelStatus.RETIRED and self.retired_at is None:
            raise ValueError("retired_at is required when status='retired'.")
        if self.status == ModelStatus.REJECTED and not self.rejection_reason:
            raise ValueError("rejection_reason is required when status='rejected'.")
        return self


# ---------------------------------------------------------------------------
# ThresholdCheckResult — embedded in ModelMetricsDocument
# ---------------------------------------------------------------------------

class ThresholdCheckResult(DSSBaseModel):
    """
    Result of a single metric threshold check.
    Embedded in ModelMetricsDocument.threshold_checks.
    """

    threshold: float
    actual: float
    passed: bool
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# ModelMetrics — the core metric values
# ---------------------------------------------------------------------------

class ModelMetrics(DSSBaseModel):
    """
    Evaluation metric values for a trained model.

    All fields except r2 are non-negative by definition.
    r2 can be negative for very poor models.
    """

    r2: float
    mae: float = Field(..., ge=0.0)
    mse: float = Field(..., ge=0.0)
    rmse: float = Field(..., ge=0.0)
    mape: Optional[float] = Field(default=None, ge=0.0)
    baseline_mae: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_rmse_consistency(self) -> "ModelMetrics":
        expected_rmse = math.sqrt(self.mse)
        tolerance = max(0.01 * self.rmse, 1e-6)
        if abs(self.rmse - expected_rmse) > tolerance:
            raise ValueError(
                f"rmse ({self.rmse:.6f}) is inconsistent with sqrt(mse) "
                f"({expected_rmse:.6f}). Check metric computation."
            )
        return self


# ---------------------------------------------------------------------------
# ModelMetricsDocument
# ---------------------------------------------------------------------------

class ModelMetricsDocument(MongoDocument):
    """
    MongoDB document shape for the model_metrics collection.

    One document per evaluation run.  Multiple evaluation runs may exist for
    the same model_id (e.g. bootstrap eval + post-deployment check).

    Timestamp discipline:
      validation_window_start — domain: start of the validation data window
      validation_window_end   — domain: end of the validation data window
      evaluated_at            — domain: when evaluation ran
      created_at              — system: when this document was written

    threshold_checks maps check name → ThresholdCheckResult.
    Example keys: "r2_minimum", "mae_relative_tolerance".
    """

    model_id: str = Field(..., min_length=1)
    pipeline: Pipeline
    evaluation_type: EvaluationType

    # Validation window
    validation_window_start: datetime   # UTC — domain
    validation_window_end: datetime     # UTC — domain
    validation_sample_count: int = Field(..., ge=1)

    # Metrics
    metrics: ModelMetrics
    passed_threshold: bool
    threshold_checks: dict[str, ThresholdCheckResult] = Field(default_factory=dict)
    rejection_reason: Optional[str] = None

    # Automated feature selection audit trail (cross-validated RFE, one run
    # per pipeline shared across the whole candidate pool; empty when
    # selection was skipped, e.g. insufficient rows). selected_features is
    # the column subset RFECV kept; feature_importance is the SHAP-based
    # cross-check ranking (mean |SHAP value|) on the same fitted estimator.
    selected_features: list[str] = Field(default_factory=list)
    feature_importance: dict[str, float] = Field(default_factory=dict)

    # Comparison baseline (null for initial bootstrap)
    baseline_model_id: Optional[str] = None

    # Timestamps
    evaluated_at: datetime              # UTC — domain
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate_window_order(self) -> "ModelMetricsDocument":
        if self.validation_window_start > self.validation_window_end:
            raise ValueError("validation_window_start must be <= validation_window_end.")
        return self

    @model_validator(mode="after")
    def _validate_rejection_reason(self) -> "ModelMetricsDocument":
        if not self.passed_threshold and not self.rejection_reason:
            raise ValueError("rejection_reason is required when passed_threshold=False.")
        return self
