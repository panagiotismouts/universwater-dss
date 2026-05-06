"""
Unit tests for dss_shared.schemas.

Tests that:
  - All document models accept valid data
  - All model validators reject structurally invalid data
  - Enum values serialize correctly
  - Cross-field invariants are enforced

These tests do not require a running MongoDB or network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

# ── Helpers ───────────────────────────────────────────────────────────────────

def utc(year=2026, month=3, day=1, hour=0, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ── Enum sanity ───────────────────────────────────────────────────────────────

def test_pipeline_values():
    from dss_shared.schemas.enums import Pipeline
    assert Pipeline.WATER.value == "water"
    assert Pipeline.SOIL.value == "soil"


def test_model_status_values():
    from dss_shared.schemas.enums import ModelStatus
    assert ModelStatus.ACTIVE.value == "active"
    assert ModelStatus.CANDIDATE.value == "candidate"


def test_processing_status_values():
    from dss_shared.schemas.enums import ProcessingStatus
    assert ProcessingStatus.PROCESSED.value == "processed"
    assert ProcessingStatus.FAILED.value == "failed"


# ── NormalizedReading ─────────────────────────────────────────────────────────

def test_normalized_reading_valid():
    from dss_shared.schemas.measurement import NormalizedReading
    r = NormalizedReading(
        pipeline="water",
        source="wings",
        sensor_id="s001",
        variable_name="ph",
        raw_value=7.2,
        unit="pH_units",
        measured_at=utc(hour=10),
        fetched_at=utc(hour=10, minute=1),
    )
    assert r.variable_name == "ph"


def test_normalized_reading_rejects_naive_timestamp():
    from dss_shared.schemas.measurement import NormalizedReading
    with pytest.raises(ValidationError):
        NormalizedReading(
            pipeline="water",
            source="wings",
            sensor_id="s001",
            variable_name="ph",
            raw_value=7.2,
            unit="pH_units",
            measured_at=datetime(2026, 3, 1, 10, 0),  # naive — no tzinfo
            fetched_at=utc(hour=10, minute=1),
        )


# ── MeasurementDocument ───────────────────────────────────────────────────────

def _base_measurement(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": "water",
        "source": "wings",
        "sensor_id": "s001",
        "variable_name": "ph",
        "value": 7.2,
        "unit": "pH_units",
        "measured_at": utc(hour=10),
        "processing_status": "processed",
    }
    base.update(overrides)
    return base


def test_measurement_document_valid():
    from dss_shared.schemas.measurement import MeasurementDocument
    doc = MeasurementDocument(**_base_measurement())
    assert doc.processing_status == "processed"
    assert doc.filled is False
    assert doc.fill_metadata is None


def test_measurement_document_filled_requires_metadata():
    from dss_shared.schemas.measurement import MeasurementDocument
    with pytest.raises(ValidationError, match="fill_metadata is required"):
        MeasurementDocument(**_base_measurement(filled=True, fill_metadata=None))


def test_measurement_document_failed_requires_reason():
    from dss_shared.schemas.measurement import MeasurementDocument
    with pytest.raises(ValidationError, match="failure_reason is required"):
        MeasurementDocument(**_base_measurement(
            processing_status="failed",
            failure_reason=None,
        ))


def test_measurement_document_failed_with_reason():
    from dss_shared.schemas.measurement import MeasurementDocument
    doc = MeasurementDocument(**_base_measurement(
        processing_status="failed",
        failure_reason="outlier: value 999.9 exceeds range [0, 14]",
    ))
    assert doc.processing_status == "failed"


# ── FeatureDocument ───────────────────────────────────────────────────────────

def test_feature_document_valid():
    from dss_shared.schemas.feature import FeatureDocument
    doc = FeatureDocument(
        pipeline="water",
        sensor_id="s001",
        feature_timestamp=utc(hour=12),
        feature_schema_version="water_v1",
        features={"ph_lag_1h": 7.1, "do_rolling_6h": 8.3},
        source_window_start=utc(hour=6),
        source_window_end=utc(hour=12),
    )
    assert doc.has_filled_inputs is False


def test_feature_document_window_order():
    from dss_shared.schemas.feature import FeatureDocument
    with pytest.raises(ValidationError, match="source_window_start"):
        FeatureDocument(
            pipeline="water",
            sensor_id="s001",
            feature_timestamp=utc(hour=12),
            feature_schema_version="water_v1",
            features={"ph_lag_1h": 7.1},
            source_window_start=utc(hour=13),   # after end — invalid
            source_window_end=utc(hour=12),
        )


# ── ModelRegistryDocument ─────────────────────────────────────────────────────

def _base_registry(**overrides) -> dict[str, Any]:
    from dss_shared.schemas.model import EmbeddedMetricsSummary
    base: dict[str, Any] = {
        "model_id": "water_xgboost_20260315_020134",
        "pipeline": "water",
        "model_type": "xgboost",
        "model_family": "black_box",
        "feature_schema_version": "water_v1",
        "target_variable": "dissolved_oxygen",
        "training_data_start": utc(2025, 9, 1),
        "training_data_end": utc(2026, 3, 1),
        "training_sample_count": 4320,
        "feature_names": ["ph_lag_1h", "temp_rolling_6h"],
        "artifact_path": "/app/models/water_xgboost_20260315_020134.joblib",
        "metrics_summary": EmbeddedMetricsSummary(r2=0.87, mae=0.12, rmse=0.35),
        "status": "candidate",
        "trained_at": utc(2026, 3, 15, 2, 1),
    }
    base.update(overrides)
    return base


def test_model_registry_candidate_valid():
    from dss_shared.schemas.model import ModelRegistryDocument
    doc = ModelRegistryDocument(**_base_registry())
    assert doc.status == "candidate"
    assert doc.activated_at is None


def test_model_registry_active_requires_activated_at():
    from dss_shared.schemas.model import ModelRegistryDocument
    with pytest.raises(ValidationError, match="activated_at is required"):
        ModelRegistryDocument(**_base_registry(status="active", activated_at=None))


def test_model_registry_active_valid():
    from dss_shared.schemas.model import ModelRegistryDocument
    doc = ModelRegistryDocument(**_base_registry(
        status="active",
        activated_at=utc(2026, 3, 15, 2, 5),
    ))
    assert doc.status == "active"


def test_model_registry_rejected_requires_reason():
    from dss_shared.schemas.model import ModelRegistryDocument
    with pytest.raises(ValidationError, match="rejection_reason is required"):
        ModelRegistryDocument(**_base_registry(status="rejected", rejection_reason=None))


# ── ModelMetrics ──────────────────────────────────────────────────────────────

def test_model_metrics_rmse_consistency():
    from dss_shared.schemas.model import ModelMetrics
    import math
    mse = 0.1225
    rmse = math.sqrt(mse)
    m = ModelMetrics(r2=0.87, mae=0.12, mse=mse, rmse=rmse)
    assert abs(m.rmse - rmse) < 1e-6


def test_model_metrics_rmse_inconsistent_raises():
    from dss_shared.schemas.model import ModelMetrics
    with pytest.raises(ValidationError, match="inconsistent"):
        ModelMetrics(r2=0.87, mae=0.12, mse=0.1225, rmse=0.5)  # wrong rmse


# ── PredictionDocument ────────────────────────────────────────────────────────

def _base_prediction(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": "water",
        "sensor_id": "s001",
        "target_variable": "dissolved_oxygen",
        "predicted_value": 8.3,
        "input_feature_timestamp": utc(2026, 3, 28, 13, 45),
        "input_had_filled_values": False,
        "model_id": "water_xgboost_20260315_020134",
        "model_type": "xgboost",
        "model_family": "black_box",
        "xai_result_id": "a" * 24,
        "prediction_generated_at": utc(2026, 3, 28, 14, 0),
    }
    base.update(overrides)
    return base


def test_prediction_document_valid():
    from dss_shared.schemas.prediction import PredictionDocument
    doc = PredictionDocument(**_base_prediction())
    assert doc.predicted_value == 8.3
    assert doc.prediction_interval_low is None


def test_prediction_document_interval_both_or_neither():
    from dss_shared.schemas.prediction import PredictionDocument
    with pytest.raises(ValidationError, match="both be set or both be None"):
        PredictionDocument(**_base_prediction(
            prediction_interval_low=7.9,
            prediction_interval_high=None,   # low set but high missing
        ))


def test_prediction_document_interval_low_must_be_lte_high():
    from dss_shared.schemas.prediction import PredictionDocument
    with pytest.raises(ValidationError, match="low.*<=.*high"):
        PredictionDocument(**_base_prediction(
            prediction_interval_low=9.0,
            prediction_interval_high=7.0,    # high < low
        ))


# ── XAIResultDocument ─────────────────────────────────────────────────────────

def test_xai_result_document_valid():
    from dss_shared.schemas.xai import XAIResultDocument
    doc = XAIResultDocument(
        prediction_id="b" * 24,
        model_id="water_xgboost_20260315_020134",
        pipeline="water",
        sensor_id="s001",
        input_feature_timestamp=utc(2026, 3, 28, 13, 45),
        explainer_type="tree_shap",
        base_value=7.21,
        shap_values={"ph_lag_1h": 0.18, "temp_rolling": -0.11},
        feature_input_values={"ph_lag_1h": 7.2, "temp_rolling": 18.3},
        generated_at=utc(2026, 3, 28, 14, 0),
    )
    assert doc.explainer_type == "tree_shap"


# ── ClientTokenDocument ───────────────────────────────────────────────────────

def test_client_token_document_valid():
    from dss_shared.schemas.token import ClientTokenDocument
    doc = ClientTokenDocument(
        client_id="client_uowm_dashboard",
        client_name="UOWM Dashboard",
        hashed_secret="$2b$12$somehash",
    )
    assert doc.active is True
    assert doc.token_hash is None


# ── CheckpointDocument ────────────────────────────────────────────────────────

def test_checkpoint_document_valid():
    from dss_shared.schemas.checkpoint import CheckpointDocument
    doc = CheckpointDocument(
        source="wings",
        pipeline="water",
        variable_name="ph",
    )
    assert doc.last_fetched_at is None
    assert doc.consecutive_failures == 0


# ── API response models ───────────────────────────────────────────────────────

def test_error_response_valid():
    from dss_shared.schemas.api_response import ErrorResponse
    err = ErrorResponse(error_code="NO_RESULTS_AVAILABLE", message="No predictions yet.")
    assert err.error_code == "NO_RESULTS_AVAILABLE"


def test_prediction_object_builds():
    from dss_shared.schemas.api_response import (
        DataQuality, ModelInfo, PredictionObject, XAIBlock, XAITopFeature,
    )
    obj = PredictionObject(
        pipeline="water",
        sensor_id="s001",
        predicted_variable="dissolved_oxygen",
        predicted_value=8.3,
        prediction_interval=None,
        prediction_timestamp=utc(2026, 3, 28, 14),
        input_feature_timestamp=utc(2026, 3, 28, 13, 45),
        model=ModelInfo(
            model_id="water_xgboost_20260315_020134",
            model_type="xgboost",
            model_family="black_box",
            trained_at=utc(2026, 3, 15, 2, 1),
        ),
        xai=XAIBlock(
            explainer_type="tree_shap",
            base_value=7.21,
            top_features=[
                XAITopFeature(feature="ph_lag_1h", shap_value=0.18, input_value=7.2),
            ],
            full_shap_values=None,
        ),
        data_quality=DataQuality(input_had_filled_values=False),
    )
    assert obj.predicted_value == 8.3
    assert obj.xai.full_shap_values is None
    assert len(obj.xai.top_features) == 1
