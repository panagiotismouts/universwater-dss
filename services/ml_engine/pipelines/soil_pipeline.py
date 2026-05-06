"""
Soil pipeline configuration.

Volumetric water content (soil_moisture) is the primary target.
N/P/K sensors have ~55% missing data due to sensor malfunction and are not
targeted in v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SoilPipelineConfig:
    pipeline_name: str = "soil"
    target_variable: str = "soil_moisture"
    feature_schema_version: str = "soil_v1"
    model_types: list[str] = field(default_factory=lambda: ["xgboost", "ridge_regression"])
    sensor_ids: list[str] = field(default_factory=list)


SOIL_PIPELINE = SoilPipelineConfig()
