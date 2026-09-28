"""
Unit tests for how the prediction cycle picks its input feature vector
(services/ml_engine/prediction/predictor.py::_select_feature_vector) and the
require_feature filter on FeatureRepository.find_latest_for_prediction.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from dss_shared.db.repositories.features import FeatureRepository
from services.ml_engine.prediction.predictor import _backfill_end, _feature_row, _select_feature_vector

_NOW = datetime(2026, 9, 28, 14, 32, tzinfo=timezone.utc)


def _doc(hour: int, **features):
    return SimpleNamespace(
        feature_timestamp=datetime(2026, 9, 28, hour, tzinfo=timezone.utc),
        features=features,
    )


class _FakeFeatureRepo:
    """Serves docs newest-first, honouring require_feature like the Mongo query."""

    def __init__(self, docs):
        self.docs = sorted(docs, key=lambda d: d.feature_timestamp, reverse=True)
        self.calls: list[dict] = []

    async def find_latest_for_prediction(
        self, pipeline, sensor_id, feature_schema_version, require_feature=None, not_after=None,
    ):
        self.calls.append({"require_feature": require_feature, "not_after": not_after})
        for d in self.docs:
            if not_after is not None and d.feature_timestamp > not_after:
                continue
            if require_feature is None or d.features.get(require_feature) is not None:
                return d
        return None


def _active(target_is_delta=True, target="wqi_brown"):
    return SimpleNamespace(feature_schema_version="water_v2", target_is_delta=target_is_delta, target_variable=target)


def _select(repo, active, now=_NOW, settle_hours=0.0):
    return asyncio.run(_select_feature_vector(
        repo, "water_wqi_brown_7d", "water", "hcmr", active, now, settle_hours=settle_hours,
    ))


def test_uses_newest_vector_when_anchor_present():
    repo = _FakeFeatureRepo([_doc(13, wqi_brown=64.8), _doc(14, wqi_brown=60.0)])
    assert _select(repo, _active()).feature_timestamp.hour == 14
    assert len(repo.calls) == 1


def test_falls_back_to_newest_vector_with_anchor():
    # 14:00 was built before DO / water temperature arrived: no WQI yet.
    repo = _FakeFeatureRepo([_doc(12, wqi_brown=72.5), _doc(13, wqi_brown=64.8), _doc(14, ph=7.7)])
    chosen = _select(repo, _active())
    assert chosen.feature_timestamp.hour == 13
    assert repo.calls[-1]["require_feature"] == "wqi_brown"


def test_null_anchor_also_triggers_fallback():
    repo = _FakeFeatureRepo([_doc(13, wqi_brown=64.8), _doc(14, wqi_brown=None)])
    assert _select(repo, _active()).feature_timestamp.hour == 13


def test_skips_when_no_vector_has_anchor():
    repo = _FakeFeatureRepo([_doc(13, ph=7.6), _doc(14, ph=7.7)])
    assert _select(repo, _active()) is None


def test_skips_when_no_vectors():
    assert _select(_FakeFeatureRepo([]), _active()) is None


def test_non_delta_model_never_needs_anchor():
    repo = _FakeFeatureRepo([_doc(13, wqi_brown=64.8), _doc(14, ph=7.7)])
    assert _select(repo, _active(target_is_delta=False)).feature_timestamp.hour == 14
    assert len(repo.calls) == 1


def test_stale_input_is_still_used():
    repo = _FakeFeatureRepo([_doc(1, wqi_brown=64.8)])
    chosen = _select(repo, _active(), now=_NOW + timedelta(days=2))
    assert chosen is not None


# ── settled input: prefer vectors old enough for late readings to be in ──────

def test_settle_uses_newest_vector_old_enough():
    # now 14:32, settle 3 h → newest vector at or before 11:32 is 11:00.
    repo = _FakeFeatureRepo([_doc(h, wqi_brown=60.0) for h in range(9, 15)])
    chosen = _select(repo, _active(), settle_hours=3)
    assert chosen.feature_timestamp.hour == 11
    assert repo.calls[0]["not_after"] == _NOW - timedelta(hours=3)


def test_settle_falls_back_to_newest_when_nothing_old_enough():
    repo = _FakeFeatureRepo([_doc(13, wqi_brown=64.8), _doc(14, wqi_brown=60.0)])
    chosen = _select(repo, _active(), settle_hours=3)
    assert chosen.feature_timestamp.hour == 14
    assert [c["not_after"] for c in repo.calls] == [_NOW - timedelta(hours=3), None]


def test_anchor_fallback_stays_within_settle_cutoff():
    # 11:00 (newest settled) lacks the WQI; 10:00 has it; 13:00 has it but is too new.
    repo = _FakeFeatureRepo([_doc(10, wqi_brown=70.0), _doc(11, ph=7.6), _doc(13, wqi_brown=64.8)])
    chosen = _select(repo, _active(), settle_hours=3)
    assert chosen.feature_timestamp.hour == 10
    assert repo.calls[-1] == {"require_feature": "wqi_brown", "not_after": _NOW - timedelta(hours=3)}


# ── backfill: leaves unsettled vectors to the prediction cycle ────────────────

def test_backfill_end_stops_at_settle_cutoff():
    # now 14:32, settle 3 h → backfill covers up to 11:32 (11:00 vector); the
    # cycle predicts 12:00 onward once each has settled, with SHAP.
    assert _backfill_end(_NOW, 3) == _NOW - timedelta(hours=3)


def test_backfill_end_without_settle_is_now():
    assert _backfill_end(_NOW, 0) == _NOW


# ── _feature_row: missing features are zero-filled and reported ──────────────

def test_feature_row_reports_missing_and_null_features():
    doc = _doc(13, ph=7.6, humidity_mean_1h=None, orp=190.0)
    row, missing = _feature_row(doc, ["ph", "air_temp_mean_1h", "humidity_mean_1h", "orp"])
    assert row.tolist() == [[7.6, 0.0, 0.0, 190.0]]
    assert missing == ["air_temp_mean_1h", "humidity_mean_1h"]


def test_feature_row_complete_vector_has_no_missing():
    row, missing = _feature_row(_doc(13, ph=7.6, orp=190.0), ["orp", "ph"])
    assert row.tolist() == [[190.0, 7.6]]
    assert missing == []


class _CapturingCollection:
    def __init__(self):
        self.query = None

    async def find_one(self, query, sort=None):
        self.query = query
        return None


def _repo_query(**kwargs):
    col = _CapturingCollection()
    repo = FeatureRepository({FeatureRepository.COLLECTION_NAME: col})
    result = asyncio.run(repo.find_latest_for_prediction(
        pipeline="water", sensor_id="hcmr", feature_schema_version="water_v2", **kwargs,
    ))
    assert result is None
    return col.query


def test_repo_query_without_require_feature_is_unchanged():
    assert _repo_query() == {
        "pipeline": "water",
        "sensor_id": "hcmr",
        "feature_schema_version": "water_v2",
        "superseded": False,
    }


def test_repo_query_with_require_feature_filters_missing_and_null():
    assert _repo_query(require_feature="wqi_brown")["features.wqi_brown"] == {"$ne": None}


def test_repo_query_with_not_after_bounds_the_timestamp():
    cutoff = _NOW - timedelta(hours=3)
    assert _repo_query(not_after=cutoff)["feature_timestamp"] == {"$lte": cutoff}
