"""
WQI forecast pipeline configurations (+7 / +14 days ahead).

Each WQI index (Brown 1972, CCME, Entropy) is forecast at two horizons,
giving six pipelines in total.  Feature documents are stored under
pipeline="water" and queried via feature_pipeline; model registry slots and
prediction documents use pipeline_name.

Target construction: for a feature vector at time t, the training target is
the WQI value from the feature document nearest t + horizon_days (handled by
dataset_builder.build_dataset with horizon_days > 0).

excluded_features is deliberately empty: unlike the retired nowcast pipelines
(where the current WQI value IS the target and must be excluded), for
forecasting every time-t quantity — including the current WQI itself — is a
legitimate predictor of the index one or two weeks ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_MODEL_TYPES = [
    "linear_regression", "elastic_net", "decision_tree",
    "random_forest", "xgboost", "lightgbm", "catboost", "svr",
]


@dataclass
class WqiHorizonPipelineConfig:
    pipeline_name: str
    target_variable: str
    horizon_days: int
    feature_pipeline: str = "water"
    feature_schema_version: str = "water_v2"
    model_types: list[str] = field(default_factory=lambda: list(_MODEL_TYPES))
    excluded_features: list[str] = field(default_factory=list)
    # Forecast skill is inherently lower than nowcast skill; bootstrap always
    # activates the best candidate, so the gate only matters at recalibration.
    min_r2_threshold: float = -2.0
    # Train on the CHANGE over the horizon (Δ = WQI(t+h) − WQI(t)) rather than
    # the absolute future value; the predictor anchors the published forecast
    # as current WQI + predicted Δ.
    delta_target: bool = True
    sensor_ids: list[str] = field(default_factory=list)


WATER_WQI_BROWN_7D_PIPELINE = WqiHorizonPipelineConfig(
    pipeline_name="water_wqi_brown_7d", target_variable="wqi_brown", horizon_days=7,
)
WATER_WQI_BROWN_14D_PIPELINE = WqiHorizonPipelineConfig(
    pipeline_name="water_wqi_brown_14d", target_variable="wqi_brown", horizon_days=14,
)
WATER_WQI_CCME_7D_PIPELINE = WqiHorizonPipelineConfig(
    pipeline_name="water_wqi_ccme_7d", target_variable="wqi_ccme", horizon_days=7,
)
WATER_WQI_CCME_14D_PIPELINE = WqiHorizonPipelineConfig(
    pipeline_name="water_wqi_ccme_14d", target_variable="wqi_ccme", horizon_days=14,
)
WATER_WQI_ENTROPY_7D_PIPELINE = WqiHorizonPipelineConfig(
    pipeline_name="water_wqi_entropy_7d", target_variable="wqi_entropy", horizon_days=7,
)
WATER_WQI_ENTROPY_14D_PIPELINE = WqiHorizonPipelineConfig(
    pipeline_name="water_wqi_entropy_14d", target_variable="wqi_entropy", horizon_days=14,
)

WQI_HORIZON_PIPELINES = [
    WATER_WQI_BROWN_7D_PIPELINE,
    WATER_WQI_BROWN_14D_PIPELINE,
    WATER_WQI_CCME_7D_PIPELINE,
    WATER_WQI_CCME_14D_PIPELINE,
    WATER_WQI_ENTROPY_7D_PIPELINE,
    WATER_WQI_ENTROPY_14D_PIPELINE,
]
