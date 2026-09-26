"""
Unit tests for services.ml_engine.training.dataset_builder.build_dataset:
sensor whitelist and global chronological ordering (the 80/20 split in the
trainers takes the last rows as validation, so order matters).

FeatureRepository is replaced with an in-memory fake; no MongoDB needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.ml_engine.training import dataset_builder as dbld

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _doc(sensor: str, hour: int) -> SimpleNamespace:
    return SimpleNamespace(
        sensor_id=sensor,
        feature_timestamp=_T0 + timedelta(hours=hour),
        has_filled_inputs=False,
        # "hour" column lets the test read back the row order from X
        features={"target": float(hour), "hour": float(hour), "other": 1.0},
    )


class _FakeRepo:
    def __init__(self, docs):
        self._docs = docs

    async def find_training_window(self, pipeline, start, end, schema):
        # Mirror the real repository: grouped by sensor, then by time.
        return sorted(self._docs, key=lambda d: (d.sensor_id, d.feature_timestamp))


@pytest.fixture
def two_stations(monkeypatch):
    # station "a" covers hours 0..11, station "b" hours 6..17 (overlapping, later)
    docs = [_doc("a", h) for h in range(12)] + [_doc("b", h) for h in range(6, 18)]
    monkeypatch.setattr(dbld, "FeatureRepository", lambda db: _FakeRepo(docs))
    return docs


@pytest.mark.anyio
async def test_rows_are_in_global_time_order_not_grouped_by_station(two_stations):
    X, y, names = await dbld.build_dataset(None, "water", _T0, _T0 + timedelta(days=1), "v", "target")
    hour_col = names.index("hour")
    hours = list(X[:, hour_col])
    assert hours == sorted(hours)
    # the last 20% of rows must be the LATEST hours, not "all of station b"
    tail = hours[-int(len(hours) * 0.2):]
    assert min(tail) >= 14


@pytest.mark.anyio
async def test_sensor_ids_whitelist_filters_rows(two_stations):
    X, y, names = await dbld.build_dataset(
        None, "water", _T0, _T0 + timedelta(days=1), "v", "target", sensor_ids=["a"],
    )
    assert X.shape[0] == 12
    assert list(y) == [float(h) for h in range(12)]


@pytest.mark.anyio
async def test_empty_sensor_ids_means_all_stations(two_stations):
    X, _, _ = await dbld.build_dataset(
        None, "water", _T0, _T0 + timedelta(days=1), "v", "target", sensor_ids=[],
    )
    assert X.shape[0] == 24
