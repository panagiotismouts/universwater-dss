"""
Unit tests for placeholder (non-)readings at preprocessing
(services.ingestion.preprocessing.pipeline) and the stored-data repair
query in scripts/flag_placeholder_readings.py.

new_water_station sends whole records of -1 when it has no data, and a dead
conductivity cell reports exact 0s; both used to be stored with
quality_flag "ok" because they sit inside the variables' physical ranges.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from dss_shared.schemas.enums import DataSource, Pipeline, QualityFlag
from dss_shared.schemas.measurement import NormalizedReading
from services.ingestion.preprocessing.pipeline import _process_reading, placeholder_flag
from services.ingestion.preprocessing.variable_registry import get_variable_spec

_spec = importlib.util.spec_from_file_location(
    "flag_placeholder_readings",
    Path(__file__).resolve().parents[2] / "scripts" / "flag_placeholder_readings.py",
)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

_TS = datetime(2026, 9, 28, 18, 45, tzinfo=timezone.utc)


def _flag(sensor_id: str, variable: str, value: float) -> QualityFlag:
    spec = get_variable_spec(variable)
    reading = NormalizedReading(
        pipeline=Pipeline.WATER, source=DataSource.WINGS_WATER, sensor_id=sensor_id,
        variable_name=variable, raw_value=value, unit=spec.expected_unit,
        measured_at=_TS, fetched_at=_TS,
    )
    return QualityFlag(_process_reading(reading, _TS).quality_flag)


# ── ingestion: stage 2 ─────────────────────────────────────────────────────────

def test_new_station_minus_one_is_no_data_even_inside_range():
    # -1 is a plausible ORP / water temperature, so only the marker rule catches it.
    assert _flag("new_water_station", "orp", -1.0) == QualityFlag.NO_DATA
    assert _flag("new_water_station", "temperature_water", -1.0) == QualityFlag.NO_DATA


def test_minus_one_is_a_reading_for_other_sensors():
    assert _flag("hcmr", "orp", -1.0) == QualityFlag.OK


def test_zero_conductivity_and_tds_are_suspect_for_any_sensor():
    assert _flag("new_water_station", "conductivity", 0.0) == QualityFlag.SUSPECT
    assert _flag("hcmr", "conductivity", 0.0) == QualityFlag.SUSPECT
    assert _flag("new_water_station", "tds", 0.0) == QualityFlag.SUSPECT


def test_zero_stays_valid_where_it_is_a_real_value():
    assert _flag("new_water_station", "turbidity", 0.0) == QualityFlag.OK
    assert _flag("hcmr", "orp", 0.0) == QualityFlag.OK


def test_real_readings_are_untouched():
    assert _flag("new_water_station", "conductivity", 4.0) == QualityFlag.OK
    assert _flag("new_water_station", "orp", 436.6) == QualityFlag.OK
    assert _flag("hcmr", "conductivity", 301.9) == QualityFlag.OK


def test_range_check_still_applies():
    assert _flag("new_water_station", "nitrate", 422.3) == QualityFlag.SUSPECT


def test_placeholder_flag_without_marker_or_zero_rule_is_none():
    assert placeholder_flag("hcmr", get_variable_spec("ph"), 7.6) is None


# ── repair script: candidate query ─────────────────────────────────────────────

def test_candidate_query_for_new_station_covers_marker_and_zeros():
    q = fp.candidate_query("new_water_station")
    assert q["sensor_id"] == "new_water_station"
    assert q["superseded"] is False
    assert {"value": {"$in": [-1.0]}} in q["$or"]
    assert {"variable_name": {"$in": ["conductivity", "tds"]}, "value": 0.0} in q["$or"]


def test_candidate_query_for_sensor_without_marker_only_checks_zeros():
    assert fp.candidate_query("hcmr")["$or"] == [
        {"variable_name": {"$in": ["conductivity", "tds"]}, "value": 0.0},
    ]
