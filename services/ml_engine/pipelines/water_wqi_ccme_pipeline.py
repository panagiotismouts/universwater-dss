"""
Water WQI CCME pipeline configuration.

Predicts wqi_ccme — the CCME-WQI over a 7-day rolling window with EU WFD
thresholds calibrated for a eutrophic freshwater lake.  CCME has a structural
R² ceiling (~0.07) because the 7-day window makes it nearly constant hour-to-hour
(lag-1 autocorrelation ≈ 0.9995).  The metric gate threshold is set accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WaterWqiCcmePipelineConfig:
    pipeline_name: str = "water_wqi_ccme"
    feature_pipeline: str = "water"
    target_variable: str = "wqi_ccme"
    feature_schema_version: str = "water_v1"
    model_types: list[str] = field(default_factory=lambda: [
        "linear_regression", "elastic_net", "decision_tree",
        "random_forest", "xgboost", "lightgbm", "catboost", "svr",
    ])
    excluded_features: list[str] = field(default_factory=lambda: [
        "dissolved_oxygen", "ph", "temperature_water", "conductivity", "orp",
        "wqi_brown", "wqi_ccme", "wqi_entropy",
    ])
    min_r2_threshold: float = -2.0  # structural ceiling ~0.07 due to 7-day rolling window; 80/20 split gives poor holdout R²
    sensor_ids: list[str] = field(default_factory=list)


WATER_WQI_CCME_PIPELINE = WaterWqiCcmePipelineConfig()
