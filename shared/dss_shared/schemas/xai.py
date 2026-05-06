"""
XAI result schemas.

XAIResultDocument
    Persistence model for the xai_results collection.
    One document per prediction event.  Written immediately after the
    corresponding PredictionDocument in the same ML engine cycle.

    Separated from prediction_results to avoid bloating prediction documents
    and to allow independent querying/analysis of SHAP explanations.

    Cross-collection references:
      prediction_id → prediction_results._id  (ObjectId string — back-reference)
      model_id      → model_registry.model_id (string key — denormalized)

    shap_values maps feature_name → SHAP value for ALL features in the model.
    feature_input_values maps feature_name → actual input value used.

    shap_sum_check is an optional self-consistency audit field:
      sum(shap_values.values()) + base_value ≈ predicted_value
    The ML engine may omit it; it should never be used for correctness decisions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from dss_shared.schemas.base import MongoDocument, PyObjectId, utc_now
from dss_shared.schemas.enums import ExplainerType, Pipeline


class XAIResultDocument(MongoDocument):
    """
    MongoDB document shape for the xai_results collection.

    Immutable after creation.

    Timestamp discipline:
      input_feature_timestamp — domain: matches PredictionDocument.input_feature_timestamp
      generated_at            — domain: when SHAP computation ran
      created_at              — system: when this document was written to MongoDB
    """

    # Back-reference to prediction_results
    prediction_id: PyObjectId           # FK → prediction_results._id

    # Denormalized for independent querying
    model_id: str = Field(..., min_length=1)
    pipeline: Pipeline
    sensor_id: str = Field(..., min_length=1)
    input_feature_timestamp: datetime   # UTC — domain

    # SHAP output
    explainer_type: ExplainerType
    base_value: float                   # SHAP expected value over training distribution
    shap_values: dict[str, float] = Field(..., min_length=1)
    feature_input_values: dict[str, float] = Field(..., min_length=1)

    # Optional self-consistency audit
    shap_sum_check: Optional[float] = None

    # Timestamps
    generated_at: datetime              # UTC — domain
    created_at: datetime = Field(default_factory=utc_now)
