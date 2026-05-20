"""
Prediction cycle.

Runs on the IntervalTrigger schedule (default: every hour).

For each enabled pipeline:
  1. Find the active model in model_registry
  2. Load its artifact from disk via ArtifactStore
  3. Find the latest feature vector for each sensor in the pipeline
  4. Run model.predict on the feature vector
  5. Run SHAP explanation (if enable_xai=True)
  6. Persist XAIResultDocument → get xai_result_id
  7. Build top-N SHAP summary (embedded)
  8. Persist PredictionDocument

Sensor IDs for each pipeline are discovered by querying the latest feature
documents (no hard-coded list).  New sensors added to the pipeline are picked
up automatically on the next prediction cycle.

Skips silently if:
  - No active model for a pipeline
  - No feature vectors available yet (ingestion hasn't run)
  - XAI computation fails (non-fatal, prediction is still written)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_settings
from dss_shared.db.repositories.features import FeatureRepository
from dss_shared.db.repositories.predictions import PredictionRepository
from dss_shared.db.repositories.xai_results import XAIResultRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import ExplainerType, ModelFamily, ModelType, Pipeline
from dss_shared.schemas.prediction import PredictionDocument, TopShapFeature
from dss_shared.schemas.xai import XAIResultDocument
from services.ml_engine.artifact_store import ArtifactStore
from services.ml_engine.models.registry import get_model_class
from services.ml_engine.pipelines.soil_pipeline import SOIL_PIPELINE
from services.ml_engine.pipelines.water_pipeline import WATER_PIPELINE
from services.ml_engine.registry_manager import find_active_model
from services.ml_engine.xai.registry import get_explainer

log = get_logger(__name__)

_ENABLED_PIPELINES = [WATER_PIPELINE, SOIL_PIPELINE]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


async def _discover_sensors(
    db: AsyncIOMotorDatabase,
    pipeline: str,
    feature_schema_version: str,
) -> list[str]:
    """Return all sensor_ids that have at least one feature document."""
    repo = FeatureRepository(db)
    # Use a small aggregation to find distinct sensor_ids in the collection
    col = repo.col
    pipeline_stages = [
        {"$match": {"pipeline": pipeline, "feature_schema_version": feature_schema_version, "superseded": False}},
        {"$group": {"_id": "$sensor_id"}},
    ]
    cursor = col.aggregate(pipeline_stages)
    return [doc["_id"] async for doc in cursor]


async def run_prediction_cycle(db: AsyncIOMotorDatabase) -> None:
    """Run one prediction cycle for all enabled pipelines."""
    settings = get_settings()
    store = ArtifactStore()
    pred_repo = PredictionRepository(db)
    xai_repo = XAIResultRepository(db)
    feat_repo = FeatureRepository(db)

    log.info("prediction_cycle_started")

    for pipeline_cfg in _ENABLED_PIPELINES:
        pipeline = pipeline_cfg.pipeline_name

        if pipeline == "water" and not settings.enable_water_pipeline:
            continue
        if pipeline == "soil" and not settings.enable_soil_pipeline:
            continue

        active = await find_active_model(db, pipeline)
        if active is None:
            log.warning("prediction_no_active_model", pipeline=pipeline)
            continue

        # Load model artifact
        try:
            model_cls = get_model_class(active.model_type)
            model = model_cls()
            model.load(Path(store._resolve(active.artifact_path)))
        except Exception as exc:
            log.error("prediction_artifact_load_failed", pipeline=pipeline, model_id=active.model_id, error=str(exc))
            continue

        sensor_ids = await _discover_sensors(db, pipeline, active.feature_schema_version)
        if not sensor_ids:
            log.warning("prediction_no_feature_vectors", pipeline=pipeline)
            continue

        now = _utc_now()
        top_n = settings.api_shap_top_n

        for sensor_id in sensor_ids:
            feat_doc = await feat_repo.find_latest_for_prediction(
                pipeline=pipeline,
                sensor_id=sensor_id,
                feature_schema_version=active.feature_schema_version,
            )
            if feat_doc is None:
                log.debug("prediction_no_features_for_sensor", pipeline=pipeline, sensor_id=sensor_id)
                continue

            # Build feature vector aligned to the model's feature_names
            try:
                x_row = np.array(
                    [feat_doc.features.get(fname, 0.0) for fname in active.feature_names],
                    dtype=np.float64,
                ).reshape(1, -1)
            except Exception as exc:
                log.error("prediction_feature_vector_failed", sensor_id=sensor_id, error=str(exc))
                continue

            # Predict
            try:
                predicted_value = float(model.predict(x_row)[0])
            except Exception as exc:
                log.error("prediction_model_predict_failed", sensor_id=sensor_id, error=str(exc))
                continue

            # XAI — compute raw explanation values first (before prediction insert)
            # The XAIResultDocument is built after insert so we have the real pred_id.
            xai_explanation: Optional[dict] = None
            top_shap: list[TopShapFeature] = []

            if settings.enable_xai:
                try:
                    explainer = get_explainer(active.model_family)
                    xai_explanation = explainer.explain(model, x_row, active.feature_names)

                    shap_vals: dict[str, float] = xai_explanation["shap_values"]
                    feat_vals: dict[str, float] = xai_explanation["feature_values"]
                    sorted_shap = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)
                    top_shap = [
                        TopShapFeature(
                            feature=fname,
                            shap_value=fval,
                            input_value=feat_vals.get(fname, 0.0),
                        )
                        for fname, fval in sorted_shap[:top_n]
                    ]
                except Exception as exc:
                    log.warning("prediction_xai_failed", sensor_id=sensor_id, error=str(exc))

            # Persist prediction to get its real _id
            pred_doc = PredictionDocument(
                pipeline=Pipeline(pipeline),
                sensor_id=sensor_id,
                target_variable=active.target_variable,
                predicted_value=predicted_value,
                input_feature_timestamp=feat_doc.feature_timestamp,
                input_had_filled_values=feat_doc.has_filled_inputs,
                model_id=active.model_id,
                model_type=ModelType(active.model_type),
                model_family=ModelFamily(active.model_family),
                xai_result_id="000000000000000000000000",
                top_shap_features=top_shap,
                prediction_generated_at=now,
            )
            pred_id = await pred_repo.insert(pred_doc)

            # Build and persist XAI doc now that we have the real prediction _id
            if pred_id and settings.enable_xai and xai_explanation is not None:
                try:
                    xai_doc = XAIResultDocument(
                        prediction_id=pred_id,
                        model_id=active.model_id,
                        pipeline=Pipeline(pipeline),
                        sensor_id=sensor_id,
                        input_feature_timestamp=feat_doc.feature_timestamp,
                        explainer_type=ExplainerType(xai_explanation["explainer_type"]),
                        base_value=xai_explanation["base_value"],
                        shap_values=xai_explanation["shap_values"],
                        feature_input_values=xai_explanation["feature_values"],
                        shap_sum_check=sum(xai_explanation["shap_values"].values()) + xai_explanation["base_value"],
                        generated_at=now,
                    )
                    xai_result_id = await xai_repo.insert(xai_doc)
                    if xai_result_id:
                        await pred_repo.update_xai_result_id(pred_id, xai_result_id)
                except Exception as exc:
                    log.warning("prediction_xai_persist_failed", sensor_id=sensor_id, error=str(exc))

            log.debug(
                "prediction_written",
                pipeline=pipeline,
                sensor_id=sensor_id,
                predicted_value=predicted_value,
                feature_timestamp=feat_doc.feature_timestamp.isoformat(),
            )

    log.info("prediction_cycle_completed")
