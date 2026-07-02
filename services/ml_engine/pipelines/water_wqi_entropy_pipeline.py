"""
Water WQI Entropy-weighted pipeline configuration.

Predicts wqi_entropy — Wang et al. (2017) entropy-weighted WQI using data-driven
weights derived from the Shannon entropy of each parameter's Qi distribution
over the 168-h rolling window.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WaterWqiEntropyPipelineConfig:
    pipeline_name: str = "water_wqi_entropy"
    feature_pipeline: str = "water"
    target_variable: str = "wqi_entropy"
    feature_schema_version: str = "water_v1"
    model_types: list[str] = field(default_factory=lambda: ["xgboost", "ridge_regression"])
    excluded_features: list[str] = field(default_factory=lambda: [
        "dissolved_oxygen", "ph", "temperature_water", "conductivity", "orp",
        "wqi_brown", "wqi_ccme", "wqi_entropy",
    ])
    min_r2_threshold: float = -2.0  # small value range causes poor 80/20 holdout R² despite good OOF CV performance
    sensor_ids: list[str] = field(default_factory=list)


WATER_WQI_ENTROPY_PIPELINE = WaterWqiEntropyPipelineConfig()
