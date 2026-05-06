"""
Water pipeline feature engineer (water_v1).

Computes a feature vector for the water quality pipeline given a reference
timestamp and a sensor's measurements stored in preprocessed_measurements.

Feature set (water_v1):
  Water variables (pipeline="water"):
    ph_mean_1h, ph_std_1h, ph_mean_3h, ph_lag_1, ph_lag_2
    do_mean_1h, do_std_1h, do_mean_3h, do_lag_1
    temp_water_mean_1h, temp_water_lag_1
    turbidity_mean_1h, turbidity_lag_1
    conductivity_mean_1h, conductivity_lag_1
    orp_mean_1h, orp_lag_1

  Meteorological co-features (pipeline="met_water"):
    rainfall_sum_1h, rainfall_sum_3h
    air_temp_mean_1h, air_temp_mean_3h
    humidity_mean_1h

  Time encodings:
    hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, month_sin, month_cos

Returns None if the water variable window has insufficient data
(fewer than MIN_WATER_READINGS observations of 'ph' in the 1h window).
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

FEATURE_SCHEMA_VERSION = "water_v1"

MIN_WATER_READINGS = 2   # minimum 'ph' readings in 1h window

_1H = timedelta(hours=1)
_3H = timedelta(hours=3)
_6H = timedelta(hours=6)


async def compute_water_features(
    db: AsyncIOMotorDatabase,
    sensor_id: str,
    feature_timestamp: datetime,
) -> Optional[FeatureDocument]:
    """
    Compute the water pipeline feature vector as of feature_timestamp.

    Reads a 6h window of measurements from MongoDB for each variable.
    Returns None if there is insufficient data.
    """
    repo = MeasurementRepository(db)
    window_start = feature_timestamp - _6H

    ph_docs   = await repo.find_window("water", sensor_id, "ph",               window_start, feature_timestamp)
    do_docs   = await repo.find_window("water", sensor_id, "dissolved_oxygen",  window_start, feature_timestamp)
    temp_docs = await repo.find_window("water", sensor_id, "temperature_water", window_start, feature_timestamp)
    turb_docs = await repo.find_window("water", sensor_id, "turbidity",         window_start, feature_timestamp)
    cond_docs = await repo.find_window("water", sensor_id, "conductivity",      window_start, feature_timestamp)
    orp_docs  = await repo.find_window("water", sensor_id, "orp",               window_start, feature_timestamp)

    rain_docs    = await repo.find_window("met_water", sensor_id, "rainfall",        window_start, feature_timestamp)
    airtemp_docs = await repo.find_window("met_water", sensor_id, "air_temperature", window_start, feature_timestamp)
    humid_docs   = await repo.find_window("met_water", sensor_id, "humidity",        window_start, feature_timestamp)

    ph_1h = _in_window(ph_docs, feature_timestamp, _1H)
    if len(ph_1h) < MIN_WATER_READINGS:
        log.debug(
            "water_features_insufficient_data",
            sensor_id=sensor_id,
            ph_1h_count=len(ph_1h),
        )
        return None

    features: dict[str, Optional[float]] = {}

    _add_rolling(features, ph_docs,   feature_timestamp, "ph",           _1H, _3H)
    _add_lags(features, ph_docs, "ph", n=2)
    _add_rolling(features, do_docs,   feature_timestamp, "do",           _1H, _3H)
    _add_lags(features, do_docs, "do", n=1)
    _add_rolling(features, temp_docs, feature_timestamp, "temp_water",   _1H)
    _add_lags(features, temp_docs, "temp_water", n=1)
    _add_rolling(features, turb_docs, feature_timestamp, "turbidity",    _1H)
    _add_lags(features, turb_docs, "turbidity", n=1)
    _add_rolling(features, cond_docs, feature_timestamp, "conductivity", _1H)
    _add_lags(features, cond_docs, "conductivity", n=1)
    _add_rolling(features, orp_docs,  feature_timestamp, "orp",          _1H)
    _add_lags(features, orp_docs, "orp", n=1)
    _add_sum(features, rain_docs,    feature_timestamp, "rainfall",    _1H, _3H)
    _add_rolling(features, airtemp_docs, feature_timestamp, "air_temp",  _1H, _3H)
    _add_rolling(features, humid_docs,   feature_timestamp, "humidity",  _1H)
    _add_time_encodings(features, feature_timestamp)

    # Drop non-computable features (None values) before writing
    final_features: dict[str, float] = {k: v for k, v in features.items() if v is not None}
    if not final_features:
        return None

    all_docs = ph_docs + do_docs + temp_docs + turb_docs + cond_docs + orp_docs
    fill_fraction = _fill_fraction(all_docs)
    all_ts = [d.measured_at for d in all_docs]

    return FeatureDocument(
        pipeline=Pipeline.WATER,
        sensor_id=sensor_id,
        feature_timestamp=feature_timestamp,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features=final_features,
        source_window_start=min(all_ts) if all_ts else window_start,
        source_window_end=max(all_ts) if all_ts else feature_timestamp,
        has_filled_inputs=fill_fraction > 0.0,
        fill_fraction=fill_fraction if fill_fraction > 0.0 else None,
    )


# ── Feature computation helpers ────────────────────────────────────────────────

def _in_window(docs: list[MeasurementDocument], end: datetime, window: timedelta) -> list[MeasurementDocument]:
    start = end - window
    return [d for d in docs if start < d.measured_at <= end]


def _ok_values(docs: list[MeasurementDocument]) -> list[float]:
    return [d.value for d in docs if d.quality_flag == "ok"]


def _add_rolling(
    features: dict,
    docs: list[MeasurementDocument],
    end: datetime,
    prefix: str,
    *windows: timedelta,
) -> None:
    for window in windows:
        h = int(window.total_seconds() // 3600)
        vals = _ok_values(_in_window(docs, end, window))
        features[f"{prefix}_mean_{h}h"] = statistics.mean(vals) if vals else None
        features[f"{prefix}_std_{h}h"]  = statistics.stdev(vals) if len(vals) >= 2 else 0.0


def _add_sum(
    features: dict,
    docs: list[MeasurementDocument],
    end: datetime,
    prefix: str,
    *windows: timedelta,
) -> None:
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


def _fill_fraction(docs: list[MeasurementDocument]) -> float:
    if not docs:
        return 0.0
    return sum(1 for d in docs if d.filled) / len(docs)
