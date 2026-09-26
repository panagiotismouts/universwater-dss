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
