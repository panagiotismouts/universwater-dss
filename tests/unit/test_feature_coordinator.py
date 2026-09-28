"""
Unit tests for services.ingestion.feature_engineering.coordinator.

Covers:
  - plan_feature_timestamps: small batch, multi-hour batch, cap truncation
  - coordinate_feature_engineering:
      * skips timestamps whose feature vector already exists (no compute)
      * computes and upserts the rest
      * ignores met_* and failed documents
      * applies the max_hours cap and logs truncation

No MongoDB is needed: the repository and the engineers are replaced with
in-memory fakes via monkeypatch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.ingestion.feature_engineering import coordinator as coord
from services.ingestion.feature_engineering.coordinator import plan_feature_timestamps


def utc(h: int, d: int = 1) -> datetime:
    return datetime(2026, 3, d, h, 0, tzinfo=timezone.utc)


# ── plan_feature_timestamps ────────────────────────────────────────────────────

def test_plan_small_incremental_batch_yields_only_batch_end():
    start, end = utc(10), utc(10) + timedelta(minutes=45)
    ts, truncated = plan_feature_timestamps(start, end, max_hours=168)
    assert ts == [end]
    assert truncated is None


def test_plan_multi_hour_batch_walks_hourly_from_start_plus_6h():
    start, end = utc(0), utc(12)
    ts, truncated = plan_feature_timestamps(start, end, max_hours=168)
    assert ts == [utc(h) for h in range(6, 13)]
    assert ts[-1] == end
    assert truncated is None


def test_plan_appends_batch_end_when_not_on_the_hour():
    start, end = utc(0), utc(9) + timedelta(minutes=30)
    ts, truncated = plan_feature_timestamps(start, end, max_hours=168)
    assert ts[:-1] == [utc(6), utc(7), utc(8), utc(9)]
    assert ts[-1] == end
    assert truncated is None


def test_plan_cap_keeps_most_recent_hours_and_reports_truncation():
    start, end = utc(0, d=1), utc(0, d=10)  # 9 days → 211 hourly steps
    ts, truncated = plan_feature_timestamps(start, end, max_hours=24)
    assert len(ts) == 24
    assert ts[-1] == end
    assert ts[0] == end - timedelta(hours=23)
    assert truncated == utc(6, d=1)  # the first dropped timestamp


def test_plan_rejects_non_positive_cap():
    with pytest.raises(ValueError):
        plan_feature_timestamps(utc(0), utc(1), max_hours=0)


# ── coordinate_feature_engineering ─────────────────────────────────────────────

class _FakeRepo:
    """Stands in for FeatureRepository: tracks existing keys and upserts."""

    def __init__(self, existing: set[tuple[str, str, datetime, str]] | None = None):
        self.existing = set(existing or ())
        self.upserted: list[tuple[str, str, datetime]] = []

    async def exists(self, pipeline, sensor_id, feature_timestamp, feature_schema_version):
        return (pipeline, sensor_id, feature_timestamp, feature_schema_version) in self.existing

    async def upsert(self, doc):
        key = (doc.pipeline, doc.sensor_id, doc.feature_timestamp)
        self.upserted.append(key)
        return True


def _doc(pipeline: str, sensor_id: str, measured_at: datetime, status: str = "processed"):
    return SimpleNamespace(
        pipeline=pipeline,
        sensor_id=sensor_id,
        measured_at=measured_at,
        processing_status=status,
    )


@pytest.fixture
def fake_env(monkeypatch):
    """Wire the coordinator to a fake repo and recording engineers."""
    repo = _FakeRepo()
    computed: list[tuple[str, str, datetime]] = []

    async def fake_water(db, sensor_id, ts):
        computed.append(("water", sensor_id, ts))
        return SimpleNamespace(pipeline="water", sensor_id=sensor_id, feature_timestamp=ts)

    async def fake_soil(db, sensor_id, ts):
        computed.append(("soil", sensor_id, ts))
        return SimpleNamespace(pipeline="soil", sensor_id=sensor_id, feature_timestamp=ts)

    monkeypatch.setattr(coord, "FeatureRepository", lambda db: repo)
    monkeypatch.setattr(coord, "compute_water_features", fake_water)
    monkeypatch.setattr(coord, "compute_soil_features", fake_soil)
    return repo, computed


@pytest.mark.anyio
async def test_coordinator_skips_existing_and_computes_missing(fake_env):
    repo, computed = fake_env
    water_version = coord._SCHEMA_VERSIONS["water"]
    # Batch 00:00 → 08:00 plans 06:00, 07:00, 08:00.  Pre-seed 06:00 and 07:00.
    repo.existing = {
        ("water", "hcmr", utc(6), water_version),
        ("water", "hcmr", utc(7), water_version),
    }
    docs = [_doc("water", "hcmr", utc(0)), _doc("water", "hcmr", utc(8))]

    await coord.coordinate_feature_engineering(docs, db=None, max_hours=168)

    assert computed == [("water", "hcmr", utc(8))]
    assert repo.upserted == [("water", "hcmr", utc(8))]


@pytest.mark.anyio
async def test_coordinator_ignores_met_pipelines_and_failed_docs(fake_env):
    repo, computed = fake_env
    docs = [
        _doc("met_water", "142", utc(8)),
        _doc("met_soil", "142", utc(8)),
        _doc("water", "hcmr", utc(8), status="failed"),
    ]

    await coord.coordinate_feature_engineering(docs, db=None, max_hours=168)

    assert computed == []
    assert repo.upserted == []


@pytest.mark.anyio
async def test_coordinator_groups_by_pipeline_and_sensor(fake_env):
    repo, computed = fake_env
    docs = [
        _doc("water", "hcmr", utc(8)),
        _doc("water", "new_water_station", utc(8)),
        _doc("soil", "soil_station_1", utc(8)),
    ]

    await coord.coordinate_feature_engineering(docs, db=None, max_hours=168)

    assert sorted(computed) == sorted([
        ("water", "hcmr", utc(8)),
        ("water", "new_water_station", utc(8)),
        ("soil", "soil_station_1", utc(8)),
    ])


@pytest.mark.anyio
async def test_coordinator_applies_backfill_cap(fake_env):
    repo, computed = fake_env
    # 10-day batch would plan ~235 hourly steps; cap at 5.
    docs = [_doc("soil", "soil_station_1", utc(0, d=1)), _doc("soil", "soil_station_1", utc(0, d=11))]

    await coord.coordinate_feature_engineering(docs, db=None, max_hours=5)

    assert len(computed) == 5
    assert computed[-1] == ("soil", "soil_station_1", utc(0, d=11))
    assert computed[0][2] == utc(0, d=11) - timedelta(hours=4)


@pytest.mark.anyio
async def test_coordinator_engineer_error_is_non_fatal(fake_env, monkeypatch):
    repo, computed = fake_env

    async def boom(db, sensor_id, ts):
        raise RuntimeError("engineer exploded")

    monkeypatch.setattr(coord, "compute_water_features", boom)
    docs = [_doc("water", "hcmr", utc(8)), _doc("soil", "soil_station_1", utc(8))]

    # Must not raise; soil still gets computed.
    await coord.coordinate_feature_engineering(docs, db=None, max_hours=168)

    assert computed == [("soil", "soil_station_1", utc(8))]


# ── _compute_and_persist(force=...) ────────────────────────────────────────────

class _OverwritingRepo(_FakeRepo):
    """Upsert reports 'matched, not inserted' for keys that already exist."""

    async def upsert(self, doc):
        key = (doc.pipeline, doc.sensor_id, doc.feature_timestamp)
        self.upserted.append(key)
        return (doc.pipeline, doc.sensor_id, doc.feature_timestamp, coord._SCHEMA_VERSIONS[doc.pipeline]) not in self.existing


@pytest.mark.anyio
async def test_compute_and_persist_force_recomputes_existing_vector(fake_env, monkeypatch):
    repo, computed = fake_env
    repo = _OverwritingRepo(existing={("soil", "soil_station_1", utc(6), coord._SCHEMA_VERSIONS["soil"])})
    monkeypatch.setattr(coord, "FeatureRepository", lambda db: repo)

    # Default: the existing vector is skipped, nothing is computed.
    outcome = await coord._compute_and_persist("soil", "soil_station_1", utc(6), None, repo)
    assert outcome == "skipped"
    assert computed == []

    # force=True: computed again and replaced in place.
    outcome = await coord._compute_and_persist("soil", "soil_station_1", utc(6), None, repo, force=True)
    assert outcome == "overwritten"
    assert computed == [("soil", "soil_station_1", utc(6))]
    assert repo.upserted == [("soil", "soil_station_1", utc(6))]


@pytest.mark.anyio
async def test_compute_and_persist_force_on_missing_vector_is_a_plain_write(fake_env):
    repo, computed = fake_env
    outcome = await coord._compute_and_persist("water", "hcmr", utc(9), None, repo, force=True)
    assert outcome == "written"
    assert computed == [("water", "hcmr", utc(9))]


# ── settle window: recent vectors are recomputed when late readings arrive ─────

class _SettleRepo(_OverwritingRepo):
    """Adds find_timestamps_since over the existing keys and records calls."""

    def __init__(self, existing=None):
        super().__init__(existing)
        self.since_calls: list[datetime] = []

    async def find_timestamps_since(self, pipeline, sensor_id, feature_schema_version, since):
        self.since_calls.append(since)
        return sorted(
            ts for (p, s, ts, v) in self.existing
            if (p, s, v) == (pipeline, sensor_id, feature_schema_version) and ts >= since
        )


@pytest.fixture
def settle_env(fake_env, monkeypatch):
    _, computed = fake_env
    repo = _SettleRepo()
    monkeypatch.setattr(coord, "FeatureRepository", lambda db: repo)
    return repo, computed


def _existing(pipeline, sensor_id, *hours):
    version = coord._SCHEMA_VERSIONS[pipeline]
    return {(pipeline, sensor_id, utc(h), version) for h in hours}


@pytest.mark.anyio
async def test_settle_window_recomputes_recent_existing_vectors(settle_env):
    repo, computed = settle_env
    # Vectors 06:00-13:00 exist; now 14:10 with a 6 h window → cutoff 08:10.
    repo.existing = _existing("water", "hcmr", *range(6, 14))
    docs = [_doc("water", "hcmr", utc(14))]  # new 14:00 reading

    await coord.coordinate_feature_engineering(
        docs, db=None, max_hours=168, settle_hours=6, now=utc(14) + timedelta(minutes=10),
    )

    # 14:00 is new; 09:00-13:00 sit inside the window and are recomputed;
    # 06:00-08:00 are older than the window and left alone.
    assert sorted(ts for _, _, ts in computed) == [utc(h) for h in range(9, 15)]
    assert repo.since_calls == [utc(8) + timedelta(minutes=10)]


@pytest.mark.anyio
async def test_settle_window_forces_planned_timestamp_that_already_exists(settle_env):
    repo, computed = settle_env
    repo.existing = _existing("water", "hcmr", 13)
    docs = [_doc("water", "hcmr", utc(13))]  # a second variable's reading for 13:00

    await coord.coordinate_feature_engineering(
        docs, db=None, max_hours=168, settle_hours=6, now=utc(13) + timedelta(minutes=40),
    )

    assert computed == [("water", "hcmr", utc(13))]


@pytest.mark.anyio
async def test_settle_window_leaves_other_sensors_alone(settle_env):
    repo, computed = settle_env
    repo.existing = _existing("water", "hcmr", 12, 13) | _existing("water", "new_water_station", 12, 13)
    docs = [_doc("water", "hcmr", utc(14))]

    await coord.coordinate_feature_engineering(
        docs, db=None, max_hours=168, settle_hours=6, now=utc(14) + timedelta(minutes=10),
    )

    assert {s for _, s, _ in computed} == {"hcmr"}


@pytest.mark.anyio
async def test_old_batch_does_not_touch_the_settle_window(settle_env):
    repo, computed = settle_env
    # Catch-up batch entirely older than the window: plain skip-if-exists.
    repo.existing = _existing("water", "hcmr", 6, 7)
    docs = [_doc("water", "hcmr", utc(0)), _doc("water", "hcmr", utc(8))]

    await coord.coordinate_feature_engineering(
        docs, db=None, max_hours=168, settle_hours=6, now=utc(8, d=5),
    )

    assert computed == [("water", "hcmr", utc(8))]
    assert repo.since_calls == []


@pytest.mark.anyio
async def test_settle_hours_zero_restores_plain_skip_if_exists(settle_env):
    repo, computed = settle_env
    repo.existing = _existing("water", "hcmr", 12, 13)
    docs = [_doc("water", "hcmr", utc(13))]

    await coord.coordinate_feature_engineering(
        docs, db=None, max_hours=168, settle_hours=0, now=utc(13) + timedelta(minutes=40),
    )

    assert computed == []
    assert repo.since_calls == []
