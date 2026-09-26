"""
Unit tests for services.ingestion.jobs.fetch_job.

Covers the ordering guarantees of one job run:
  - checkpoint is advanced as soon as measurements are written, even when
    feature engineering raises
  - checkpoint is advanced BEFORE feature engineering starts
  - checkpoint is NOT advanced when the MongoDB write fails, and feature
    engineering is not run
  - source API failure records a failure and does not advance

All collaborators are replaced with in-memory fakes via monkeypatch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dss_shared.exceptions import SourceAPIError
from services.ingestion.jobs import fetch_job as fj


def utc(h: int) -> datetime:
    return datetime(2026, 3, 1, h, 0, tzinfo=timezone.utc)


class _FakeClient:
    source_name = "wings_water"

    def __init__(self, readings=None, error: Exception | None = None):
        self._readings = readings or []
        self._error = error

    async def fetch(self, variable_name, since):
        if self._error:
            raise self._error
        return self._readings


class _FakeCheckpointManager:
    """Records the order of calls so tests can assert sequencing."""

    def __init__(self, db):
        self.calls: list[str] = []
        self.advanced_to: datetime | None = None

    async def get_since(self, source, pipeline, variable_name):
        self.calls.append("get_since")
        return utc(0)

    @staticmethod
    def filter_new(readings, since):
        return [r for r in readings if r.measured_at > since]

    @staticmethod
    def max_measured_at(readings):
        return max((r.measured_at for r in readings), default=None)

    async def advance(self, source, pipeline, variable_name, last_fetched_at):
        self.calls.append("advance")
        self.advanced_to = last_fetched_at

    async def record_failure(self, source, pipeline, variable_name):
        self.calls.append("record_failure")

    async def record_partial(self, source, pipeline, variable_name):
        self.calls.append("record_partial")


class _FakeMeasurementRepo:
    def __init__(self, db, fail: bool = False):
        self.fail = fail

    async def insert_many(self, docs):
        if self.fail:
            raise RuntimeError("mongo down")
        return len(docs), 0


@pytest.fixture
def wiring(monkeypatch):
    """Patch every collaborator; return a dict of handles for assertions."""
    state: dict = {"cm": None, "calls": [], "repo_fail": False, "fe_error": None}

    class _CM(_FakeCheckpointManager):
        """Class (not factory) so the static filter_new/max_measured_at resolve."""

        def __init__(self, db):
            super().__init__(db)
            # share one call log so feature-engineering calls interleave with it
            self.calls = state["calls"]
            state["cm"] = self

    async def fake_preprocess(readings, db):
        return [
            SimpleNamespace(processing_status="processed", measured_at=r.measured_at)
            for r in readings
        ]

    async def fake_feature_engineering(docs, db):
        state["calls"].append("feature_engineering")
        if state["fe_error"]:
            raise state["fe_error"]

    monkeypatch.setattr(fj, "CheckpointManager", _CM)
    monkeypatch.setattr(fj, "MeasurementRepository", lambda db: _FakeMeasurementRepo(db, state["repo_fail"]))
    monkeypatch.setattr(fj, "run_preprocessing_pipeline", fake_preprocess)
    monkeypatch.setattr(fj, "coordinate_feature_engineering", fake_feature_engineering)
    return state


def _readings(*hours: int):
    return [SimpleNamespace(measured_at=utc(h)) for h in hours]


@pytest.mark.anyio
async def test_checkpoint_advances_before_feature_engineering(wiring):
    job = fj.make_fetch_job(_FakeClient(_readings(1, 2, 3)), "ph", "water", db=None)

    await job()

    calls = wiring["calls"]
    assert "advance" in calls and "feature_engineering" in calls
    assert calls.index("advance") < calls.index("feature_engineering")
    assert wiring["cm"].advanced_to == utc(3)


@pytest.mark.anyio
async def test_checkpoint_advances_even_if_feature_engineering_raises(wiring):
    wiring["fe_error"] = RuntimeError("slow feature walk blew up")
    job = fj.make_fetch_job(_FakeClient(_readings(1, 2)), "ph", "water", db=None)

    await job()  # must not raise

    assert wiring["cm"].advanced_to == utc(2)


@pytest.mark.anyio
async def test_write_failure_does_not_advance_or_run_features(wiring):
    wiring["repo_fail"] = True
    job = fj.make_fetch_job(_FakeClient(_readings(1, 2)), "ph", "water", db=None)

    await job()

    calls = wiring["calls"]
    assert "record_partial" in calls
    assert "advance" not in calls
    assert "feature_engineering" not in calls


@pytest.mark.anyio
async def test_source_api_failure_records_failure_only(wiring):
    job = fj.make_fetch_job(_FakeClient(error=SourceAPIError("503")), "ph", "water", db=None)

    await job()

    assert wiring["calls"] == ["get_since", "record_failure"]


@pytest.mark.anyio
async def test_no_new_readings_is_a_noop(wiring):
    # All readings at/before the watermark (utc(0)) are filtered out.
    job = fj.make_fetch_job(_FakeClient(_readings(0)), "ph", "water", db=None)

    await job()

    assert wiring["calls"] == ["get_since"]
