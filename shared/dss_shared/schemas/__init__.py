"""
dss_shared.schemas — canonical typed contracts for the DSS.

Import map
──────────
enums          Domain enumerations and typed string constants
base           DSSBaseModel, MongoDocument, PyObjectId, utc_now

measurement    NormalizedReading (internal), MeasurementDocument (persistence)
feature        FeatureDocument (persistence)
model          ModelRegistryDocument, ModelMetricsDocument,
               EmbeddedMetricsSummary, ModelMetrics,
               ThresholdCheckResult (persistence)
prediction     PredictionDocument, TopShapFeature (persistence)
xai            XAIResultDocument (persistence)
token          ClientTokenDocument (persistence),
               TokenRequest, TokenResponse, JWTClaims (API / internal)
delivery_log   DeliveryLogDocument (persistence)
checkpoint     CheckpointDocument (persistence)

api_response   All external API request/response contracts
internal       Internal service contracts (never persisted)

Usage:
    # Preferred — import from the specific sub-module for clarity:
    from dss_shared.schemas.measurement import MeasurementDocument
    from dss_shared.schemas.enums import Pipeline, ModelStatus

    # Acceptable for scripts/tests — flat import from this package:
    from dss_shared.schemas import MeasurementDocument, Pipeline
"""

# ── Enums ─────────────────────────────────────────────────────────────────────
from dss_shared.schemas.enums import (
    DataSource,
    EvaluationType,
    ExplainerType,
    IngestionRunStatus,
    ModelFamily,
    ModelStatus,
    ModelType,
    Pipeline,
    ProcessingStatus,
    QualityFlag,
)

# ── Base ──────────────────────────────────────────────────────────────────────
from dss_shared.schemas.base import (
    DSSBaseModel,
    MongoDocument,
    PyObjectId,
    utc_now,
)

# ── Measurement ───────────────────────────────────────────────────────────────
from dss_shared.schemas.measurement import (
    FillMetadata,
    MeasurementDocument,
    NormalizedReading,
)

# ── Feature ───────────────────────────────────────────────────────────────────
from dss_shared.schemas.feature import FeatureDocument

# ── Model lifecycle ───────────────────────────────────────────────────────────
from dss_shared.schemas.model import (
    EmbeddedMetricsSummary,
    ModelMetrics,
    ModelMetricsDocument,
    ModelRegistryDocument,
    ThresholdCheckResult,
)

# ── Prediction ────────────────────────────────────────────────────────────────
from dss_shared.schemas.prediction import (
    PredictionDocument,
    TopShapFeature,
)

# ── XAI ───────────────────────────────────────────────────────────────────────
from dss_shared.schemas.xai import XAIResultDocument

# ── Auth / tokens ─────────────────────────────────────────────────────────────
from dss_shared.schemas.token import (
    ClientTokenDocument,
    JWTClaims,
    TokenRequest,
    TokenResponse,
)

# ── Delivery log ──────────────────────────────────────────────────────────────
from dss_shared.schemas.delivery_log import DeliveryLogDocument

# ── Checkpoint ────────────────────────────────────────────────────────────────
from dss_shared.schemas.checkpoint import CheckpointDocument

# ── External API contracts ────────────────────────────────────────────────────
from dss_shared.schemas.api_response import (
    AdminModelsResponse,
    DataQuality,
    ErrorDetail,
    ErrorResponse,
    HistoricalResultsResponse,
    HistoryQueryParams,
    LatestResultsResponse,
    ModelAdminSummary,
    ModelInfo,
    PredictionInterval,
    PredictionObject,
    XAIBlock,
    XAITopFeature,
)

# ── Internal service contracts ────────────────────────────────────────────────
from dss_shared.schemas.internal import (
    FeatureRow,
    PredictionPayload,
    TrainingRunSummary,
)

__all__ = [
    # enums
    "DataSource", "EvaluationType", "ExplainerType", "IngestionRunStatus",
    "ModelFamily", "ModelStatus", "ModelType", "Pipeline", "ProcessingStatus",
    "QualityFlag",
    # base
    "DSSBaseModel", "MongoDocument", "PyObjectId", "utc_now",
    # measurement
    "FillMetadata", "MeasurementDocument", "NormalizedReading",
    # feature
    "FeatureDocument",
    # model
    "EmbeddedMetricsSummary", "ModelMetrics", "ModelMetricsDocument",
    "ModelRegistryDocument", "ThresholdCheckResult",
    # prediction
    "PredictionDocument", "TopShapFeature",
    # xai
    "XAIResultDocument",
    # token
    "ClientTokenDocument", "JWTClaims", "TokenRequest", "TokenResponse",
    # delivery_log
    "DeliveryLogDocument",
    # checkpoint
    "CheckpointDocument",
    # api_response
    "AdminModelsResponse", "DataQuality", "ErrorDetail", "ErrorResponse",
    "HistoricalResultsResponse", "HistoryQueryParams", "LatestResultsResponse",
    "ModelAdminSummary", "ModelInfo", "PredictionInterval", "PredictionObject",
    "XAIBlock", "XAITopFeature",
    # internal
    "FeatureRow", "PredictionPayload", "TrainingRunSummary",
]
