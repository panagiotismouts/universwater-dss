"""
Water WQI Brown (1972) pipeline configuration.

Predicts wqi_brown — the Brown (1972) weighted arithmetic WQI computed from
the 5 HCMR water parameters.  Feature documents are stored under pipeline="water"
and queried via feature_pipeline; model registry slots use pipeline_name.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WaterWqiBrownPipelineConfig:
    pipeline_name: str = "water_wqi_brown"
    feature_pipeline: str = "water"
    target_variable: str = "wqi_brown"
    feature_schema_version: str = "water_v1"
    model_types: list[str] = field(default_factory=lambda: ["xgboost", "ridge_regression"])
    excluded_features: list[str] = field(default_factory=lambda: [
        "dissolved_oxygen", "ph", "temperature_water", "conductivity", "orp",
        "wqi_brown", "wqi_ccme", "wqi_entropy",
    ])
    min_r2_threshold: float = 0.60
    sensor_ids: list[str] = field(default_factory=list)


WATER_WQI_BROWN_PIPELINE = WaterWqiBrownPipelineConfig()
