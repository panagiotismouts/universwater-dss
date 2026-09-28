"""
Unit tests for the ML engine's cron parsing (services/ml_engine/scheduler.py).

Standard cron numbers weekdays with 0/7 = Sunday; APScheduler 3.x uses
0 = Monday. These tests pin the translation so "0 2 * * 1" fires on Monday.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from apscheduler.triggers.cron import CronTrigger

from services.ml_engine.scheduler import _cron_dow_to_apscheduler, _parse_cron

# Sunday 2026-09-27 12:00 UTC
_SUNDAY_NOON = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def _next_fire(cron_expr: str, now: datetime = _SUNDAY_NOON) -> datetime:
    trigger = CronTrigger(timezone="UTC", **_parse_cron(cron_expr))
    return trigger.get_next_fire_time(None, now)


def test_default_recalibration_cron_fires_on_monday():
    fire = _next_fire("0 2 * * 1")
    assert fire == datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc)
    assert fire.strftime("%a") == "Mon"


@pytest.mark.parametrize(
    "dow, weekday",
    [("0", "Sun"), ("7", "Sun"), ("1", "Mon"), ("3", "Wed"), ("6", "Sat"),
     ("mon", "Mon"), ("sun", "Sun")],
)
def test_single_weekday_matches_standard_cron(dow, weekday):
    # Monday 00:00 so the next Sunday is also reachable within the week.
    now = datetime(2026, 9, 28, 0, 0, tzinfo=timezone.utc)
    assert _next_fire(f"0 2 * * {dow}", now).strftime("%a") == weekday


@pytest.mark.parametrize(
    "field, expected",
    [
        ("*", "*"),
        ("mon", "mon"),
        ("mon-fri", "mon-fri"),
        ("1", "mon"),
        ("0", "sun"),
        ("7", "sun"),
        ("1-5", "mon,tue,wed,thu,fri"),
        ("5-7", "fri,sat,sun"),
        ("0,6", "sun,sat"),
        ("*/2", "sun,tue,thu,sat"),
        ("1-5/2", "mon,wed,fri"),
        ("0,7", "sun"),
        ("mon,3", "mon,wed"),
    ],
)
def test_cron_dow_translation(field, expected):
    assert _cron_dow_to_apscheduler(field) == expected


@pytest.mark.parametrize("field", ["8", "5-2", "1-9", "*/0", "mon/2", "x1"])
def test_cron_dow_rejects_invalid(field):
    with pytest.raises(ValueError):
        _cron_dow_to_apscheduler(field)


def test_parse_cron_rejects_wrong_field_count():
    with pytest.raises(ValueError):
        _parse_cron("0 2 * *")


def test_recalibration_job_has_misfire_grace_and_fires_monday(monkeypatch):
    from types import SimpleNamespace

    from services.ml_engine import scheduler as ml_scheduler

    monkeypatch.setattr(
        ml_scheduler,
        "get_settings",
        lambda: SimpleNamespace(recalibration_cron="0 2 * * 1", prediction_interval_seconds=604800),
    )
    sched = ml_scheduler.build_ml_scheduler(db=None)
    job = sched.get_job("recalibration")

    assert job.misfire_grace_time == 6 * 3600
    assert job.trigger.get_next_fire_time(None, _SUNDAY_NOON) == datetime(
        2026, 9, 28, 2, 0, tzinfo=timezone.utc
    )
