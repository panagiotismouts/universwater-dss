"""
Domain enums and typed string constants for the DSS.

All enumerations use str as their base class so that:
  - Values serialize directly to/from MongoDB BSON strings
  - Values serialize directly to/from JSON without extra coercion
  - Enum members can be compared to plain strings safely

Do not import from services here. This module has no dependencies other
than the Python standard library.
"""

from __future__ import annotations

from enum import Enum


# ── Domain pipelines ──────────────────────────────────────────────────────────

class Pipeline(str, Enum):
    """
    Domain pipeline identifier.

    "water" and "soil" are the two primary prediction domains.
    "met_water" and "met_soil" are meteorological data pipelines that feed
    the water and soil models respectively.  They are ingestion-level
    identifiers, not model-level identifiers.
    """
    WATER = "water"
    SOIL = "soil"
    # Nowcast WQI pipelines (retired — kept for historical documents)
    WATER_WQI_BROWN   = "water_wqi_brown"
    WATER_WQI_CCME    = "water_wqi_ccme"
    WATER_WQI_ENTROPY = "water_wqi_entropy"
    # Forecast WQI pipelines (+7 / +14 days ahead)
    WATER_WQI_BROWN_7D    = "water_wqi_brown_7d"
    WATER_WQI_BROWN_14D   = "water_wqi_brown_14d"
    WATER_WQI_CCME_7D     = "water_wqi_ccme_7d"
    WATER_WQI_CCME_14D    = "water_wqi_ccme_14d"
    WATER_WQI_ENTROPY_7D  = "water_wqi_entropy_7d"
    WATER_WQI_ENTROPY_14D = "water_wqi_entropy_14d"
    MET_WATER = "met_water"
    MET_SOIL = "met_soil"


# ── Data source identifiers ───────────────────────────────────────────────────

class DataSource(str, Enum):
    """External API sources from which sensor readings are ingested."""
    WINGS = "wings"
    UOWM = "uowm"
    WINGS_WATER = "wings_water"
    WINGS_SOIL = "wings_soil"
    UOWM_MET = "uowm_met"


# ── Model taxonomy ────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    """
    Algorithm identifier.  New model types are added here and to the model
    class registry in services/ml_engine/models/registry.py.
    """
    # White-box (directly interpretable)
    LINEAR_REGRESSION = "linear_regression"
    ELASTIC_NET = "elastic_net"
    DECISION_TREE = "decision_tree"
    # Black-box (SHAP required for XAI)
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVR = "svr"
    # Legacy — kept for backward compat with stored documents
    RIDGE_REGRESSION = "ridge_regression"


class ModelFamily(str, Enum):
    """
    Model family.  Determines whether SHAP is applied at prediction time.

    black_box — SHAP XAI layer required (Random Forest, XGBoost, LightGBM, CatBoost, SVR)
    white_box — directly interpretable; no SHAP (Linear Regression, Elastic Net, Decision Tree)
    """
    BLACK_BOX = "black_box"
    WHITE_BOX = "white_box"


class ModelStatus(str, Enum):
    """
    Lifecycle status of a trained model instance in model_registry.

    Transitions:
        candidate  → active     (metric gate passed, promoted by registry_manager)
        active     → retired    (superseded when a newer active model is promoted)
        candidate  → rejected   (metric gate failed)
        active     → superseded (manual admin reset to force bootstrap retraining)

    At most one document per pipeline may have status="active" at any time.
    This is enforced by a partial unique index and by registry_manager logic.
    """
    ACTIVE = "active"
    CANDIDATE = "candidate"
    RETIRED = "retired"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


# ── Ingestion / preprocessing status ──────────────────────────────────────────

class ProcessingStatus(str, Enum):
    """
    Inline preprocessing outcome for a preprocessed_measurements document.

    Amendment C.3: only "processed" and "failed" are written in v1.
    The "unprocessed" variant is retained as a constant for future compatibility
    but is NEVER written by any v1 code path.
    """
    PROCESSED = "processed"
    FAILED = "failed"
    # UNPROCESSED = "unprocessed"  # reserved for future use — do not write in v1


class QualityFlag(str, Enum):
    """
    Optional data quality annotation on a preprocessed measurement.

    "ok"           — value passed all validation checks
    "suspect"      — value is within range but statistically unusual
    "sensor_fault" — source API signalled a sensor fault condition
    "no_data"      — source API returned a null/missing value; forward-fill applied
    """
    OK = "ok"
    SUSPECT = "suspect"
    SENSOR_FAULT = "sensor_fault"
    NO_DATA = "no_data"


# ── Ingestion job run status ───────────────────────────────────────────────────

class IngestionRunStatus(str, Enum):
    """
    Outcome of a single ingestion job execution, stored in ingestion_checkpoints.

    "success" — all fetched readings were written successfully
    "partial" — some readings processed, some failed (pipeline continued)
    "failed"  — job failed entirely; checkpoint was NOT advanced
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# ── Model evaluation type ─────────────────────────────────────────────────────

class EvaluationType(str, Enum):
    """
    Context in which a model was evaluated; stored in model_metrics.

    "validation"         — held-out split during training (bootstrap or recalibration)
    "recalibration_check"— threshold comparison against the current active model
    "post_deployment"    — future use; out-of-sample evaluation after activation
    """
    VALIDATION = "validation"
    RECALIBRATION_CHECK = "recalibration_check"
    POST_DEPLOYMENT = "post_deployment"


# ── XAI explainer type ────────────────────────────────────────────────────────

class ExplainerType(str, Enum):
    """
    SHAP explainer variant used to generate an xai_results document.

    Routing: model_family → explainer type is handled by xai/registry.py.
    """
    TREE_SHAP = "tree_shap"
    LINEAR_SHAP = "linear_shap"
    KERNEL_SHAP = "kernel_shap"
