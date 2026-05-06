"""
Source normalizer.

Converts raw API response rows into NormalizedReading objects.
Contains the ONLY place where source-specific field names, timestamp formats,
and unit conventions are known.  After normalization, all downstream code
uses canonical DSS names.

WINGS SensorThings API (OGC):
  Each row: {"phenomenon_time_start": "2025-09-01T00:00:00Z", "result_number": 7.4}
  sensor_id and variable_name are passed in from the client (derived from datastream config).

UOWM REST API:
  Each row: {"timestamp": 1728388800, "datetime": "2024-10-08T10:00:00", "value": 15.2}
  sensor_id is the numeric sensor ID string; variable_name is passed in.

Met variables (§E.3):
  Each UOWM reading produces TWO NormalizedReading objects — one for
  pipeline="met_water" and one for pipeline="met_soil".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dss_shared.exceptions import IngestionValidationError
from dss_shared.schemas.enums import DataSource, Pipeline
from dss_shared.schemas.measurement import NormalizedReading

# ── Canonical variable metadata ────────────────────────────────────────────────
# Maps canonical variable_name → (pipeline, unit)
_WINGS_WATER_VARIABLES: dict[str, tuple[str, str]] = {
    "ph":               ("water", "pH units"),
    "dissolved_oxygen": ("water", "mg/L"),
    "temperature_water": ("water", "°C"),
    "turbidity":        ("water", "NTU"),
    "conductivity":     ("water", "μS/cm"),
    "orp":              ("water", "mV"),
    "do_saturation":    ("water", "%"),
    "tds":              ("water", "mg/L"),
    "salinity":         ("water", "ppt"),
    "ammonia":          ("water", "mg/L"),
    "nitrate":          ("water", "mg/L"),
    "chlorophyll_a":    ("water", "μg/L"),
    "ammonium":         ("water", "mg/L"),
    "blue_green_algae": ("water", "cells/mL"),
    "cdom":             ("water", "ppb"),
    "sigma_t":          ("water", "kg/m³"),
}

_WINGS_SOIL_VARIABLES: dict[str, tuple[str, str]] = {
    "soil_moisture":    ("soil", "%"),
    "soil_temperature": ("soil", "°C"),
    "nitrogen":         ("soil", "mg/kg"),
    "phosphorus":       ("soil", "mg/kg"),
    "potassium":        ("soil", "mg/kg"),
    "soil_conductivity": ("soil", "μS/cm"),
}

_UOWM_MET_VARIABLES: dict[str, str] = {
    "air_temperature":     "°C",
    "rainfall":            "mm",
    "humidity":            "%",
    "wind_speed":          "m/s",
    "wind_direction":      "°",
    "solar_radiation":     "W/m²",
    "infrared_temperature": "°C",
}


def _parse_timestamp(raw_ts: Any, source: str) -> datetime:
    """
    Parse a timestamp from the API response and return a timezone-aware UTC datetime.

    Accepts:
      - ISO 8601 string (with or without Z/offset suffix)
      - Unix epoch integer/float

    Raises IngestionValidationError for unparseable inputs.
    """
    if isinstance(raw_ts, (int, float)):
        return datetime.fromtimestamp(raw_ts, tz=timezone.utc)
    if isinstance(raw_ts, str):
        normalized = raw_ts.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise IngestionValidationError(
                f"[{source}] Unparseable timestamp: {raw_ts!r}"
            ) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise IngestionValidationError(
        f"[{source}] Timestamp has unexpected type {type(raw_ts).__name__}: {raw_ts!r}"
    )


def _get_float_value(raw_val: Any, field: str, source: str) -> float | None:
    """
    Coerce a raw API value to float.  Returns None for null/missing indicators.

    Raises IngestionValidationError for values that are present but non-numeric.
    """
    if raw_val is None:
        return None
    try:
        return float(raw_val)
    except (TypeError, ValueError) as exc:
        raise IngestionValidationError(
            f"[{source}] Non-numeric value for field {field!r}: {raw_val!r}"
        ) from exc


# ── WINGS normalizers ──────────────────────────────────────────────────────────

def normalize_wings_water(
    rows: list[dict[str, Any]],
    sensor_id: str,
    variable_name: str,
    fetched_at: datetime,
) -> list[NormalizedReading]:
    """
    Convert WINGS SensorThings observation rows for a water variable into
    NormalizedReading objects.

    Each row: {"phenomenon_time_start": "...", "result_number": 7.4}
    """
    meta = _WINGS_WATER_VARIABLES.get(variable_name)
    if meta is None:
        return []
    pipeline, unit = meta

    results: list[NormalizedReading] = []
    for row in rows:
        raw_ts  = row.get("phenomenon_time_start")
        raw_val = row.get("result_number")
        if raw_ts is None:
            continue
        measured_at = _parse_timestamp(raw_ts, "wings")
        value = _get_float_value(raw_val, "result_number", "wings")
        if value is None:
            continue
        results.append(NormalizedReading(
            pipeline=Pipeline(pipeline),
            source=DataSource.WINGS,
            sensor_id=sensor_id,
            variable_name=variable_name,
            raw_value=value,
            unit=unit,
            measured_at=measured_at,
            fetched_at=fetched_at,
        ))
    return results


def normalize_wings_soil(
    rows: list[dict[str, Any]],
    sensor_id: str,
    variable_name: str,
    fetched_at: datetime,
) -> list[NormalizedReading]:
    """
    Convert WINGS SensorThings observation rows for a soil variable into
    NormalizedReading objects.

    Each row: {"phenomenon_time_start": "...", "result_number": 23.1}
    """
    meta = _WINGS_SOIL_VARIABLES.get(variable_name)
    if meta is None:
        return []
    pipeline, unit = meta

    results: list[NormalizedReading] = []
    for row in rows:
        raw_ts  = row.get("phenomenon_time_start")
        raw_val = row.get("result_number")
        if raw_ts is None:
            continue
        measured_at = _parse_timestamp(raw_ts, "wings")
        value = _get_float_value(raw_val, "result_number", "wings")
        if value is None:
            continue
        results.append(NormalizedReading(
            pipeline=Pipeline(pipeline),
            source=DataSource.WINGS,
            sensor_id=sensor_id,
            variable_name=variable_name,
            raw_value=value,
            unit=unit,
            measured_at=measured_at,
            fetched_at=fetched_at,
        ))
    return results


def normalize_uowm_met(
    rows: list[dict[str, Any]],
    variable_name: str,
    sensor_id: str,
    fetched_at: datetime,
) -> list[NormalizedReading]:
    """
    Convert UOWM API rows for a meteorological variable into NormalizedReading
    objects.

    Each row: {"timestamp": 1728388800, "datetime": "2024-10-08T10:00:00", "value": 15.2}

    Met variables produce TWO NormalizedReading objects per timestamp —
    one for pipeline="met_water" and one for pipeline="met_soil" (§E.3).
    """
    unit = _UOWM_MET_VARIABLES.get(variable_name)
    if unit is None:
        return []

    results: list[NormalizedReading] = []
    for row in rows:
        # Prefer unix timestamp for precision; fall back to "datetime" string
        raw_ts  = row.get("timestamp") or row.get("datetime")
        raw_val = row.get("value")
        if raw_ts is None:
            continue
        measured_at = _parse_timestamp(raw_ts, "uowm")
        value = _get_float_value(raw_val, "value", "uowm")
        if value is None:
            continue
        for pipeline in ("met_water", "met_soil"):
            results.append(NormalizedReading(
                pipeline=Pipeline(pipeline),
                source=DataSource.UOWM,
                sensor_id=sensor_id,
                variable_name=variable_name,
                raw_value=value,
                unit=unit,
                measured_at=measured_at,
                fetched_at=fetched_at,
            ))
    return results
