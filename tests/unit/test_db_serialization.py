"""
Unit tests for dss_shared.db.serialization.

Tests that:
  - ObjectId values are coerced to strings on read
  - Nested ObjectIds in dicts and lists are coerced
  - to_document removes _id=None for new documents
  - to_document preserves _id when set
  - doc_to_model round-trips through a schema model correctly
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from bson import ObjectId

from dss_shared.db.serialization import coerce_object_ids, doc_to_model, to_document


def utc(y=2026, m=3, d=1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# ── coerce_object_ids ──────────────────────────────────────────────────────────

def test_coerce_top_level_object_id():
    oid = ObjectId()
    result = coerce_object_ids({"_id": oid, "name": "test"})
    assert result["_id"] == str(oid)
    assert len(result["_id"]) == 24


def test_coerce_nested_dict():
    oid = ObjectId()
    result = coerce_object_ids({"outer": {"inner_id": oid}})
    assert result["outer"]["inner_id"] == str(oid)


def test_coerce_list_of_object_ids():
    ids = [ObjectId(), ObjectId()]
    result = coerce_object_ids({"ids": ids})
    assert all(isinstance(v, str) for v in result["ids"])


def test_coerce_leaves_non_oid_values():
    result = coerce_object_ids({"value": 42, "name": "hello", "flag": True})
    assert result == {"value": 42, "name": "hello", "flag": True}


def test_coerce_none_values_preserved():
    result = coerce_object_ids({"token_hash": None, "expiry": None})
    assert result["token_hash"] is None
    assert result["expiry"] is None


# ── to_document ────────────────────────────────────────────────────────────────

def test_to_document_omits_none_id():
    """New documents (no _id) should not include _id key in the output dict."""
    from dss_shared.schemas.checkpoint import CheckpointDocument
    doc = CheckpointDocument(
        source="wings",
        pipeline="water",
        variable_name="ph",
    )
    result = to_document(doc)
    assert "_id" not in result


def test_to_document_preserves_set_id():
    """Documents with an existing _id string should include it."""
    from dss_shared.schemas.checkpoint import CheckpointDocument
    doc = CheckpointDocument(
        id="a" * 24,
        source="wings",
        pipeline="water",
        variable_name="ph",
    )
    result = to_document(doc)
    assert result["_id"] == "a" * 24


def test_to_document_uses_aliases():
    """The _id alias (not the Python 'id' field name) must appear in the dict."""
    from dss_shared.schemas.delivery_log import DeliveryLogDocument
    doc = DeliveryLogDocument(
        client_id="test_client",
        endpoint="/results/latest",
        http_method="GET",
        request_timestamp=utc(),
        response_status=200,
        latency_ms=12,
    )
    raw = to_document(doc)
    assert "client_id" in raw
    assert "request_timestamp" in raw
    assert "_id" not in raw  # no id set → omitted


def test_to_document_includes_none_fields():
    """None-valued optional fields must be present (not excluded) for query correctness."""
    from dss_shared.schemas.token import ClientTokenDocument
    doc = ClientTokenDocument(
        client_id="c1",
        client_name="Client One",
        hashed_secret="$2b$hash",
    )
    raw = to_document(doc)
    # token_hash=None must be present so {token_hash: None} queries work
    assert "token_hash" in raw
    assert raw["token_hash"] is None


# ── doc_to_model ───────────────────────────────────────────────────────────────

def test_doc_to_model_coerces_object_id():
    """doc_to_model should convert ObjectId _id to a string field on the model."""
    from dss_shared.schemas.checkpoint import CheckpointDocument
    oid = ObjectId()
    raw = {
        "_id": oid,
        "source": "wings",
        "pipeline": "water",
        "variable_name": "ph",
        "consecutive_failures": 0,
        "updated_at": utc(),
    }
    doc = doc_to_model(raw, CheckpointDocument)
    assert doc.id == str(oid)
    assert len(doc.id) == 24


def test_doc_to_model_string_id_passthrough():
    """String _id values should pass through unchanged."""
    from dss_shared.schemas.checkpoint import CheckpointDocument
    oid_str = "a" * 24
    raw = {
        "_id": oid_str,
        "source": "uowm",
        "pipeline": "soil",
        "variable_name": "soil_moisture",
        "consecutive_failures": 2,
        "updated_at": utc(),
    }
    doc = doc_to_model(raw, CheckpointDocument)
    assert doc.id == oid_str


def test_doc_to_model_nested_object_ids_in_prediction():
    """Nested ObjectId FK fields (xai_result_id) should be coerced to string."""
    from dss_shared.schemas.prediction import PredictionDocument
    pred_oid = ObjectId()
    xai_oid = ObjectId()
    raw = {
        "_id": pred_oid,
        "pipeline": "water",
        "sensor_id": "s001",
        "target_variable": "dissolved_oxygen",
        "predicted_value": 8.3,
        "prediction_interval_low": None,
        "prediction_interval_high": None,
        "input_feature_timestamp": utc(),
        "input_had_filled_values": False,
        "model_id": "water_xgboost_20260315_020134",
        "model_type": "xgboost",
        "model_family": "black_box",
        "xai_result_id": xai_oid,
        "top_shap_features": [],
        "prediction_generated_at": utc(),
        "created_at": utc(),
        "superseded": False,
        "supersedes_id": None,
    }
    doc = doc_to_model(raw, PredictionDocument)
    assert doc.id == str(pred_oid)
    assert doc.xai_result_id == str(xai_oid)
