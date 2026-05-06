"""
External API response and request contract models.

These are the Pydantic models for the external API contract defined in the
API Contract Blueprint.  They are what FastAPI serializes to JSON and what
API clients receive.  They are NOT persistence models.

Design rules (from API Contract Blueprint §A, §D):
  - No MongoDB ObjectIds in any response field
  - No model diagnostic metrics (R², MAE, etc.) in responses
  - prediction_interval is always present (null when unavailable, never absent)
  - xai.full_shap_values is always present (null unless include_full_xai=true)
  - All timestamps are ISO 8601 UTC strings (Pydantic serializes datetime → str)
  - Error responses use a consistent envelope

Canonical prediction object (API Contract Blueprint §D.3):
  See PredictionObject below.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from dss_shared.schemas.base import DSSBaseModel


# ---------------------------------------------------------------------------
# Sub-objects within the canonical prediction object
# ---------------------------------------------------------------------------

class PredictionInterval(DSSBaseModel):
    """Prediction interval bounds.  Null on the parent object when unavailable."""
    low: float
    high: float


class ModelInfo(DSSBaseModel):
    """
    Model provenance block within a prediction object.

    Only fields a client needs to understand the prediction source.
    No diagnostic metrics (Blueprint §A.2).
    """
    model_id: str
    model_type: str
    model_family: str
    trained_at: datetime


class XAITopFeature(DSSBaseModel):
    """Single feature's SHAP summary in the API XAI block."""
    feature: str
    shap_value: float
    input_value: float


class XAIBlock(DSSBaseModel):
    """
    XAI block within a prediction object (API Contract Blueprint §D.3, §G).

    top_features — always present; top-N features by |shap_value|
    full_shap_values — null unless include_full_xai=true was requested
    base_value — SHAP expected value; always present
    explainer_type — e.g. "tree_shap"
    """
    explainer_type: str
    base_value: float
    top_features: list[XAITopFeature]
    full_shap_values: Optional[dict[str, float]] = None  # always present, null by default


class DataQuality(DSSBaseModel):
    """Data quality annotations on a prediction result."""
    input_had_filled_values: bool


# ---------------------------------------------------------------------------
# PredictionObject — the canonical prediction unit
# ---------------------------------------------------------------------------

class PredictionObject(DSSBaseModel):
    """
    Canonical prediction object returned in both /results/latest and
    /results/history responses (API Contract Blueprint §D.3).

    prediction_interval is always present as null when unavailable.
    xai.full_shap_values is always present as null unless requested.
    """

    pipeline: str
    sensor_id: str
    predicted_variable: str
    predicted_value: float
    prediction_interval: Optional[PredictionInterval] = None   # null when not available

    prediction_timestamp: datetime      # When the ML engine generated this prediction
    input_feature_timestamp: datetime   # The "as of" time of the input features

    model: ModelInfo
    xai: XAIBlock
    data_quality: DataQuality


# ---------------------------------------------------------------------------
# GET /results/latest  — response model
# ---------------------------------------------------------------------------

class LatestResultsResponse(DSSBaseModel):
    """
    Response body for GET /results/latest (200 OK).

    results is always an array, even for a single sensor (Blueprint §E.3).
    """

    pipeline: str
    results: list[PredictionObject]


# ---------------------------------------------------------------------------
# GET /results/history — request params and response model
# ---------------------------------------------------------------------------

class HistoryQueryParams(DSSBaseModel):
    """
    Validated query parameters for GET /results/history.

    Used by the result_service to receive typed params from the router.
    Not the FastAPI Query(...) model itself — the router extracts params
    and constructs this model for validation.
    """

    pipeline: str
    sensor_id: Optional[str] = None
    from_time: Optional[datetime] = Field(default=None, alias="from")
    to_time: Optional[datetime] = Field(default=None, alias="to")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)
    include_full_xai: bool = False


class HistoricalResultsResponse(DSSBaseModel):
    """
    Response body for GET /results/history (200 OK).

    total_count is the total matching records, not just this page.
    Results are sorted by prediction_timestamp ascending (oldest first).
    """

    pipeline: str
    sensor_id: Optional[str] = None
    from_time: Optional[datetime] = Field(default=None, alias="from")
    to_time: Optional[datetime] = Field(default=None, alias="to")
    total_count: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1)
    results: list[PredictionObject]


# ---------------------------------------------------------------------------
# Error response envelope
# ---------------------------------------------------------------------------

class ErrorDetail(DSSBaseModel):
    """
    Structured error details for client-debuggable errors.
    field: which request field caused the error (optional).
    """
    field: Optional[str] = None
    message: str


class ErrorResponse(DSSBaseModel):
    """
    Standard error response envelope for all DSS API error responses.

    error_code — machine-readable string for programmatic handling.
    Defined codes (API Contract Blueprint §I):
      INVALID_CREDENTIALS     — 401 auth failures
      TOKEN_EXPIRED           — 401 expired token
      INVALID_TOKEN           — 401 bad token
      NO_RESULTS_AVAILABLE    — 404 no predictions for pipeline
      INVALID_PIPELINE        — 400 unknown pipeline value
      INVALID_DATE_RANGE      — 400 from > to or bad date format
      INVALID_PAGE_PARAMS     — 400 page/page_size out of range
      SERVICE_UNAVAILABLE     — 500 MongoDB unreachable
      INTERNAL_ERROR          — 500 unexpected server error
    """

    error_code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GET /admin/models — response model (admin-only, not external contract)
# ---------------------------------------------------------------------------

class ModelAdminSummary(DSSBaseModel):
    """
    Single model entry in the /admin/models response.
    Includes metrics summary — permitted because this is an admin endpoint,
    not the external client contract.
    """
    model_id: str
    pipeline: str
    model_type: str
    model_family: str
    status: str
    trained_at: datetime
    activated_at: Optional[datetime] = None
    metrics_summary: dict[str, Any]


class AdminModelsResponse(DSSBaseModel):
    """Response body for GET /admin/models."""
    models: list[ModelAdminSummary]
    total: int
