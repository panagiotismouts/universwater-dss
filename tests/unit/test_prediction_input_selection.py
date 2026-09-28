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
from services.ml_engine.prediction.predictor import _select_feature_vector

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

    async def find_latest_for_prediction(self, pipeline, sensor_id, feature_schema_version, require_feature=None):
        self.calls.append({"require_feature": require_feature})
        for d in self.docs:
            if require_feature is None or d.features.get(require_feature) is not None:
                return d
        return None


def _active(target_is_delta=True, target="wqi_brown"):
    return SimpleNamespace(feature_schema_version="water_v2", target_is_delta=target_is_delta, target_variable=target)


def _select(repo, active, now=_NOW):
    return asyncio.run(_select_feature_vector(repo, "water_wqi_brown_7d", "water", "hcmr", active, now))


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
