"""
Soil pipeline feature engineer (soil_v1).

Computes a feature vector for the soil quality pipeline.

Feature set (soil_v1):
  Soil variables (pipeline="soil"):
    soil_moisture_mean_3h, soil_moisture_std_3h, soil_moisture_lag_1, soil_moisture_lag_2
    soil_temp_mean_3h, soil_temp_lag_1

  Meteorological co-features (pipeline="met_soil"):
    rainfall_sum_3h, rainfall_sum_6h
    air_temp_mean_3h, air_temp_lag_1
    humidity_mean_3h

  Time encodings:
    hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, month_sin, month_cos

Returns None if fewer than MIN_SOIL_READINGS of 'soil_moisture' exist in the 3h window.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.repositories.measurements import MeasurementRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import Pipeline
from dss_shared.schemas.feature import FeatureDocument
from dss_shared.schemas.measurement import MeasurementDocument

log = get_logger(__name__)

FEATURE_SCHEMA_VERSION = "soil_v1"

MIN_SOIL_READINGS = 1   # minimum 'soil_moisture' readings in 12h window (data arrives ~every 4h)

# UOWM meteorological sensor IDs (numeric string, as stored in MongoDB)
_MET_SENSOR_AIR_TEMP = "142"
_MET_SENSOR_RAINFALL = "144"
_MET_SENSOR_HUMIDITY = "145"

_3H  = timedelta(hours=3)
_6H  = timedelta(hours=6)
_12H = timedelta(hours=12)


async def compute_soil_features(
    db: AsyncIOMotorDatabase,
    sensor_id: str,
    feature_timestamp: datetime,
) -> Optional[FeatureDocument]:
    """
    Compute the soil pipeline feature vector as of feature_timestamp.

    Reads a 12h window of measurements from MongoDB.
    Returns None if there is insufficient data.
    """
    repo = MeasurementRepository(db)
    window_start = feature_timestamp - _12H

    moist_docs = await repo.find_window("soil", sensor_id, "soil_moisture",    window_start, feature_timestamp)
    temp_docs  = await repo.find_window("soil", sensor_id, "soil_temperature", window_start, feature_timestamp)

    rain_docs    = await repo.find_window("met_soil", _MET_SENSOR_RAINFALL, "rainfall",        window_start, feature_timestamp)
    airtemp_docs = await repo.find_window("met_soil", _MET_SENSOR_AIR_TEMP, "air_temperature", window_start, feature_timestamp)
    humid_docs   = await repo.find_window("met_soil", _MET_SENSOR_HUMIDITY, "humidity",        window_start, feature_timestamp)

    moist_12h = _in_window(moist_docs, feature_timestamp, _12H)
    if len(moist_12h) < MIN_SOIL_READINGS:
        log.debug(
            "soil_features_insufficient_data",
            sensor_id=sensor_id,
            moist_12h_count=len(moist_12h),
        )
        return None

    features: dict[str, Optional[float]] = {}

    _add_rolling(features, moist_docs, feature_timestamp, "soil_moisture", _3H)
    _add_lags(features, moist_docs, "soil_moisture", n=2)
    _add_rolling(features, temp_docs,  feature_timestamp, "soil_temp",     _3H)
    _add_lags(features, temp_docs, "soil_temp", n=1)

    # Raw current values for target variable candidates (required by dataset_builder)
    _add_current_value(features, moist_docs, "soil_moisture")
    _add_current_value(features, temp_docs,  "soil_temperature")

    _add_sum(features, rain_docs,    feature_timestamp, "rainfall",  _3H, _6H)
    _add_rolling(features, airtemp_docs, feature_timestamp, "air_temp", _3H)
    _add_lags(features, airtemp_docs, "air_temp", n=1)
    _add_rolling(features, humid_docs,   feature_timestamp, "humidity", _3H)

    _add_time_encodings(features, feature_timestamp)

    final_features: dict[str, float] = {k: v for k, v in features.items() if v is not None}
    if not final_features:
        return None

    all_docs = moist_docs + temp_docs
    fill_fraction = _fill_fraction(all_docs)
    all_ts = [d.measured_at for d in all_docs]

    return FeatureDocument(
        pipeline=Pipeline.SOIL,
        sensor_id=sensor_id,
        feature_timestamp=feature_timestamp,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features=final_features,
        source_window_start=min(all_ts) if all_ts else window_start,
        source_window_end=max(all_ts) if all_ts else feature_timestamp,
        has_filled_inputs=fill_fraction > 0.0,
        fill_fraction=fill_fraction if fill_fraction > 0.0 else None,
    )


# ── Helpers (same as water_features.py) ───────────────────────────────────────

def _in_window(docs: list[MeasurementDocument], end: datetime, window: timedelta) -> list[MeasurementDocument]:
    start = end - window
    return [d for d in docs if start < d.measured_at <= end]


def _ok_values(docs: list[MeasurementDocument]) -> list[float]:
    return [d.value for d in docs if d.quality_flag == "ok"]


def _add_rolling(features: dict, docs: list[MeasurementDocument], end: datetime, prefix: str, *windows: timedelta) -> None:
    for window in windows:
        h = int(window.total_seconds() // 3600)
        vals = _ok_values(_in_window(docs, end, window))
        features[f"{prefix}_mean_{h}h"] = statistics.mean(vals) if vals else None
        features[f"{prefix}_std_{h}h"]  = statistics.stdev(vals) if len(vals) >= 2 else 0.0


def _add_sum(features: dict, docs: list[MeasurementDocument], end: datetime, prefix: str, *windows: timedelta) -> None:
    for window in windows:
        h = int(window.total_seconds() // 3600)
        vals = _ok_values(_in_window(docs, end, window))
        features[f"{prefix}_sum_{h}h"] = sum(vals) if vals else 0.0


def _add_lags(features: dict, docs: list[MeasurementDocument], prefix: str, n: int) -> None:
    recent = sorted(
        [d for d in docs if d.quality_flag == "ok"],
        key=lambda d: d.measured_at,
        reverse=True,
    )
    for i in range(1, n + 1):
        features[f"{prefix}_lag_{i}"] = recent[i - 1].value if len(recent) >= i else None


def _add_time_encodings(features: dict, ts: datetime) -> None:
    features["hour_sin"]      = math.sin(2 * math.pi * ts.hour / 24)
    features["hour_cos"]      = math.cos(2 * math.pi * ts.hour / 24)
    features["dayofweek_sin"] = math.sin(2 * math.pi * ts.weekday() / 7)
    features["dayofweek_cos"] = math.cos(2 * math.pi * ts.weekday() / 7)
    features["month_sin"]     = math.sin(2 * math.pi * (ts.month - 1) / 12)
    features["month_cos"]     = math.cos(2 * math.pi * (ts.month - 1) / 12)


def _add_current_value(features: dict, docs: list[MeasurementDocument], key: str) -> None:
    """Store the most recent ok-quality reading under its canonical variable name."""
    ok = [d for d in docs if d.quality_flag == "ok"]
    if ok:
        features[key] = sorted(ok, key=lambda d: d.measured_at)[-1].value


def _fill_fraction(docs: list[MeasurementDocument]) -> float:
    if not docs:
        return 0.0
    return sum(1 for d in docs if d.filled) / len(docs)
