"""
Water pipeline configuration.

Dissolved oxygen is the primary prediction target (confirmed: Aquaread sensors).
model_types drives which model classes are trained in bootstrap and recalibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WaterPipelineConfig:
    pipeline_name: str = "water"
    target_variable: str = "dissolved_oxygen"
    feature_schema_version: str = "water_v1"
    model_types: list[str] = field(default_factory=lambda: [
        "linear_regression", "elastic_net", "decision_tree",
        "random_forest", "xgboost", "lightgbm", "catboost", "svr",
    ])
    # sensor_ids is intentionally empty — populated dynamically from feature docs
    sensor_ids: list[str] = field(default_factory=list)


WATER_PIPELINE = WaterPipelineConfig()
