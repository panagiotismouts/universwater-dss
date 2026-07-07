"""
Water pipeline feature engineer (water_v1).

Computes a feature vector for the water quality pipeline given a reference
timestamp and a sensor's measurements stored in preprocessed_measurements.

Feature set (water_v1):
  Water variables (pipeline="water"):
    ph_mean_1h, ph_std_1h, ph_mean_3h, ph_lag_1, ph_lag_2
    do_mean_1h, do_std_1h, do_mean_3h, do_lag_1
    temp_water_mean_1h, temp_water_lag_1
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

MIN_WATER_READINGS = 1   # minimum 'ph' readings in 1h window (aquaread_1/hcmr send ≤1/hour)

# UOWM meteorological sensor IDs (numeric string, as stored in MongoDB)
_MET_SENSOR_AIR_TEMP = "142"
_MET_SENSOR_RAINFALL = "144"
_MET_SENSOR_HUMIDITY = "145"

_1H   = timedelta(hours=1)
_3H   = timedelta(hours=3)
_6H   = timedelta(hours=6)
_168H = timedelta(hours=168)  # 7-day window required by CCME-WQI

# ── CCME-WQI EU WFD thresholds (eutrophic freshwater lake) ────────────────────
# Same values as hcmr_analysis/wqi.py _CCME_OBJECTIVES
_CCME_MIN_OBS = 6  # minimum complete observations needed to compute CCME


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
    window_start = feature_timestamp - _168H  # extended for CCME 7-day window

    ph_docs   = await repo.find_window("water", sensor_id, "ph",               window_start, feature_timestamp)
    do_docs   = await repo.find_window("water", sensor_id, "dissolved_oxygen",  window_start, feature_timestamp)
    temp_docs = await repo.find_window("water", sensor_id, "temperature_water", window_start, feature_timestamp)
    cond_docs = await repo.find_window("water", sensor_id, "conductivity",      window_start, feature_timestamp)
    orp_docs  = await repo.find_window("water", sensor_id, "orp",               window_start, feature_timestamp)

    rain_docs    = await repo.find_window("met_water", _MET_SENSOR_RAINFALL, "rainfall",        window_start, feature_timestamp)
    airtemp_docs = await repo.find_window("met_water", _MET_SENSOR_AIR_TEMP, "air_temperature", window_start, feature_timestamp)
    humid_docs   = await repo.find_window("met_water", _MET_SENSOR_HUMIDITY, "humidity",        window_start, feature_timestamp)

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
    _add_rolling(features, cond_docs, feature_timestamp, "conductivity", _1H)
    _add_lags(features, cond_docs, "conductivity", n=1)
    _add_rolling(features, orp_docs,  feature_timestamp, "orp",          _1H)
    _add_lags(features, orp_docs, "orp", n=1)

    # Raw current values for target variable candidates (required by dataset_builder)
    _add_current_value(features, ph_docs,   "ph")
    _add_current_value(features, do_docs,   "dissolved_oxygen")
    _add_current_value(features, temp_docs, "temperature_water")
    _add_current_value(features, cond_docs, "conductivity")
    _add_current_value(features, orp_docs,  "orp")

    _add_sum(features, rain_docs,    feature_timestamp, "rainfall",    _1H, _3H)
    _add_rolling(features, airtemp_docs, feature_timestamp, "air_temp",  _1H, _3H)
    _add_rolling(features, humid_docs,   feature_timestamp, "humidity",  _1H)
    _add_time_encodings(features, feature_timestamp)

    # ── WQI targets (stored as features so ML pipelines can use them as y) ──
    features["wqi_brown"]   = _compute_wqi_brown(ph_docs, do_docs, temp_docs, cond_docs, orp_docs, feature_timestamp)
    features["wqi_ccme"]    = _compute_wqi_ccme(ph_docs, do_docs, temp_docs, cond_docs, orp_docs)
    features["wqi_entropy"] = _compute_wqi_entropy(ph_docs, do_docs, temp_docs, cond_docs, orp_docs, feature_timestamp)

    # Drop non-computable features (None values) before writing
    final_features: dict[str, float] = {k: v for k, v in features.items() if v is not None}
    if not final_features:
        return None

    all_docs = ph_docs + do_docs + temp_docs + cond_docs + orp_docs
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


def _add_current_value(features: dict, docs: list[MeasurementDocument], key: str) -> None:
    """Store the most recent ok-quality reading under its canonical variable name."""
    ok = [d for d in docs if d.quality_flag == "ok"]
    if ok:
        features[key] = sorted(ok, key=lambda d: d.measured_at)[-1].value


def _fill_fraction(docs: list[MeasurementDocument]) -> float:
    if not docs:
        return 0.0
    return sum(1 for d in docs if d.filled) / len(docs)


# ── WQI computation helpers ────────────────────────────────────────────────────
# Formulas mirror hcmr_analysis/wqi.py exactly.  All return float | None.
# None is written when inputs are insufficient; dataset_builder skips those rows.

def _latest_ok(docs: list[MeasurementDocument]) -> Optional[float]:
    ok = [d for d in docs if d.quality_flag == "ok"]
    if not ok:
        return None
    return sorted(ok, key=lambda d: d.measured_at)[-1].value


def _compute_wqi_brown(
    ph_docs, do_docs, temp_docs, cond_docs, orp_docs,
    feature_timestamp: datetime,
) -> Optional[float]:
    ph   = _latest_ok(_in_window(ph_docs,   feature_timestamp, _1H))
    do_  = _latest_ok(_in_window(do_docs,   feature_timestamp, _1H))
    temp = _latest_ok(_in_window(temp_docs, feature_timestamp, _1H))
    cond = _latest_ok(_in_window(cond_docs, feature_timestamp, _1H))
    orp  = _latest_ok(_in_window(orp_docs,  feature_timestamp, _1H))

    if any(v is None for v in (ph, do_, temp, cond, orp)):
        return None

    qi_do   = min(do_ / 9.0 * 100.0, 100.0)
    qi_ph   = max(100.0 - abs(ph - 7.0) / 1.5 * 100.0, 0.0)
    qi_temp = max(100.0 - abs(temp - 20.0) / 15.0 * 100.0, 0.0)
    qi_cond = max(100.0 - cond / 1000.0 * 100.0, 0.0)
    qi_orp  = max(100.0 - abs(orp - 300.0) / 300.0 * 100.0, 0.0)
    return (5 * qi_do + 3 * qi_ph + 2 * qi_temp + 2 * qi_cond + 1 * qi_orp) / 13.0


def _compute_wqi_ccme(
    ph_docs, do_docs, temp_docs, cond_docs, orp_docs,
) -> Optional[float]:
    """CCME-WQI over the full 168-h window already loaded."""
    # EU WFD thresholds for eutrophic freshwater lake (mirrors hcmr_analysis/wqi.py)
    objectives = {
        "ph":       ("range", 6.5, 10.0),
        "do":       ("min",   4.0),
        "cond":     ("range", 50.0, 750.0),
        "temp":     ("max",   28.0),
        "orp":      ("min",   200.0),
    }
    param_docs = {
        "ph":   ph_docs,
        "do":   do_docs,
        "cond": cond_docs,
        "temp": temp_docs,
        "orp":  orp_docs,
    }

    total_tests = 0
    total_fails = 0
    total_exc   = 0.0
    failed_vars = 0

    for key, obj in objectives.items():
        vals = [d.value for d in param_docs[key] if d.quality_flag == "ok"]
        if not vals:
            continue
        kind = obj[0]
        n_fail = 0
        exc_sum = 0.0
        for v in vals:
            if kind == "min":
                limit = obj[1]
                if v < limit:
                    n_fail += 1
                    exc_sum += (limit / max(v, 1e-9)) - 1.0
            elif kind == "max":
                limit = obj[1]
                if v > limit:
                    n_fail += 1
                    exc_sum += v / limit - 1.0
            else:  # range
                lo, hi = obj[1], obj[2]
                if v < lo:
                    n_fail += 1
                    exc_sum += (lo / max(v, 1e-9)) - 1.0
                elif v > hi:
                    n_fail += 1
                    exc_sum += v / hi - 1.0
        total_tests += len(vals)
        total_fails += n_fail
        total_exc   += exc_sum
        if n_fail > 0:
            failed_vars += 1

    complete_obs = min(
        len([d for d in ph_docs   if d.quality_flag == "ok"]),
        len([d for d in do_docs   if d.quality_flag == "ok"]),
        len([d for d in temp_docs if d.quality_flag == "ok"]),
        len([d for d in cond_docs if d.quality_flag == "ok"]),
        len([d for d in orp_docs  if d.quality_flag == "ok"]),
    )
    if complete_obs < _CCME_MIN_OBS or total_tests == 0:
        return None

    n_vars = len(objectives)
    F1 = (failed_vars / n_vars) * 100.0
    F2 = (total_fails / total_tests) * 100.0
    nse = total_exc / total_tests
    F3  = nse / (0.01 * nse + 0.01)
    ccme = 100.0 - (math.sqrt(F1**2 + F2**2 + F3**2) / 1.732)
    return max(0.0, min(100.0, ccme))


def _compute_wqi_entropy(
    ph_docs, do_docs, temp_docs, cond_docs, orp_docs,
    feature_timestamp: datetime,
) -> Optional[float]:
    """Entropy-weighted WQI (Wang et al. 2017).

    Weights are derived from Shannon entropy of each parameter's Qi distribution
    across the 168-h rolling window already loaded.  The current-hour Qi values
    are then combined using those window-derived weights.
    """
    # Current-hour Qi values (point-in-time prediction target)
    ph_cur   = _latest_ok(_in_window(ph_docs,   feature_timestamp, _1H))
    do_cur   = _latest_ok(_in_window(do_docs,   feature_timestamp, _1H))
    temp_cur = _latest_ok(_in_window(temp_docs, feature_timestamp, _1H))
    cond_cur = _latest_ok(_in_window(cond_docs, feature_timestamp, _1H))
    orp_cur  = _latest_ok(_in_window(orp_docs,  feature_timestamp, _1H))

    if any(v is None for v in (ph_cur, do_cur, temp_cur, cond_cur, orp_cur)):
        return None

    def _qi_series(docs, fn) -> list[float]:
        return [fn(d.value) for d in docs if d.quality_flag == "ok"]

    param_qi: dict[str, list[float]] = {
        "do":   _qi_series(do_docs,   lambda v: min(v / 9.0 * 100.0, 100.0)),
        "ph":   _qi_series(ph_docs,   lambda v: max(100.0 - abs(v - 7.0) / 1.5 * 100.0, 0.0)),
        "temp": _qi_series(temp_docs, lambda v: max(100.0 - abs(v - 20.0) / 15.0 * 100.0, 0.0)),
        "cond": _qi_series(cond_docs, lambda v: max(100.0 - v / 1000.0 * 100.0, 0.0)),
        "orp":  _qi_series(orp_docs,  lambda v: max(100.0 - abs(v - 300.0) / 300.0 * 100.0, 0.0)),
    }

    # Need at least 2 observations per parameter to compute entropy meaningfully
    if any(len(s) < 2 for s in param_qi.values()):
        equal_w = 1.0 / 5
        qi_cur = {
            "do":   min(do_cur / 9.0 * 100.0, 100.0),
            "ph":   max(100.0 - abs(ph_cur - 7.0) / 1.5 * 100.0, 0.0),
            "temp": max(100.0 - abs(temp_cur - 20.0) / 15.0 * 100.0, 0.0),
            "cond": max(100.0 - cond_cur / 1000.0 * 100.0, 0.0),
            "orp":  max(100.0 - abs(orp_cur - 300.0) / 300.0 * 100.0, 0.0),
        }
        return sum(equal_w * v for v in qi_cur.values())

    # Shannon entropy per parameter (Wang et al. 2017)
    eps = 1e-12
    divergences: dict[str, float] = {}
    for key, series in param_qi.items():
        if max(series) == min(series):
            # Constant Qi series (including all-zero, e.g. a clamped sub-index)
            # carries no discriminating information — divergence 0, weight 0.
            # Without this guard an all-zero series degenerates to entropy 0 /
            # divergence 1 and absorbs ~all the weight.
            divergences[key] = 0.0
            continue
        col_sum = sum(series) or eps
        p_vals  = [v / col_sum for v in series]
        n_obs   = len(p_vals)
        ln_n    = math.log(n_obs)
        entropy = -(sum(p * math.log(p + eps) for p in p_vals)) / ln_n
        divergences[key] = max(1.0 - entropy, 0.0)

    div_total = sum(divergences.values())
    if div_total < 1e-12:
        weights = {k: 1.0 / 5 for k in divergences}
    else:
        weights = {k: v / div_total for k, v in divergences.items()}

    qi_cur = {
        "do":   min(do_cur / 9.0 * 100.0, 100.0),
        "ph":   max(100.0 - abs(ph_cur - 7.0) / 1.5 * 100.0, 0.0),
        "temp": max(100.0 - abs(temp_cur - 20.0) / 15.0 * 100.0, 0.0),
        "cond": max(100.0 - cond_cur / 1000.0 * 100.0, 0.0),
        "orp":  max(100.0 - abs(orp_cur - 300.0) / 300.0 * 100.0, 0.0),
    }
    return sum(weights[k] * qi_cur[k] for k in qi_cur)
