"""
Unit tests for scripts/backfill_features.py helpers (pure functions only).
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "backfill_features", Path(__file__).resolve().parents[2] / "scripts" / "backfill_features.py"
)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)


def test_parse_utc_accepts_naive_z_and_offset():
    expected = datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc)
    assert bf.parse_utc("2026-07-14T07:00") == expected
    assert bf.parse_utc("2026-07-14T07:00Z") == expected
    assert bf.parse_utc("2026-07-14T09:00+02:00") == expected


def test_hourly_range_is_inclusive():
    start = datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc)
    ts = bf.hourly_range(start, start + timedelta(hours=3))
    assert ts == [start + timedelta(hours=h) for h in range(4)]


def test_hourly_range_rejects_reversed_bounds():
    start = datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        bf.hourly_range(start, start - timedelta(hours=1))


# ── decide_action: --force / --created-before policy ──────────────────────────

_CUTOFF = datetime(2026, 9, 26, 14, 30, tzinfo=timezone.utc)


def test_decide_action_missing_vector_is_always_computed():
    assert bf.decide_action(False, None, force=False, created_before=None) == "compute"
    assert bf.decide_action(False, None, force=True, created_before=_CUTOFF) == "compute"


def test_decide_action_without_force_keeps_existing():
    old = _CUTOFF - timedelta(days=1)
    assert bf.decide_action(True, old, force=False, created_before=None) == "skip"


def test_decide_action_force_without_cutoff_recomputes_everything():
    new = _CUTOFF + timedelta(hours=1)
    assert bf.decide_action(True, new, force=True, created_before=None) == "recompute"


def test_decide_action_force_with_cutoff_only_recomputes_stale_vectors():
    old = _CUTOFF - timedelta(seconds=1)
    new = _CUTOFF + timedelta(seconds=1)
    assert bf.decide_action(True, old, force=True, created_before=_CUTOFF) == "recompute"
    assert bf.decide_action(True, _CUTOFF, force=True, created_before=_CUTOFF) == "skip"
    assert bf.decide_action(True, new, force=True, created_before=_CUTOFF) == "skip"


def test_decide_action_treats_missing_created_at_as_stale():
    assert bf.decide_action(True, None, force=True, created_before=_CUTOFF) == "recompute"


# ── stored_query: --stored selection ───────────────────────────────────────────

def test_stored_query_selects_range_and_schema():
    start = datetime(2025, 9, 23, tzinfo=timezone.utc)
    end = datetime(2026, 9, 26, tzinfo=timezone.utc)
    q = bf.stored_query("soil", "soil_station_1", "soil_v1", start, end, None)
    assert q == {
        "pipeline": "soil",
        "sensor_id": "soil_station_1",
        "feature_schema_version": "soil_v1",
        "feature_timestamp": {"$gte": start, "$lte": end},
    }


def test_stored_query_with_cutoff_includes_legacy_docs_without_created_at():
    start = datetime(2025, 9, 23, tzinfo=timezone.utc)
    q = bf.stored_query("soil", "soil_station_1", "soil_v1", start, _CUTOFF, _CUTOFF)
    assert q["$or"] == [
        {"created_at": {"$lt": _CUTOFF}},
        {"created_at": {"$exists": False}},
    ]
