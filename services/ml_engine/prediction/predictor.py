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

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_raw_yaml, get_settings
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
from services.ml_engine.pipelines.water_pipeline import WATER_PIPELINE  # kept for reference
from services.ml_engine.pipelines.wqi_horizon_pipelines import WQI_HORIZON_PIPELINES
from services.ml_engine.registry_manager import find_active_model
from services.ml_engine.xai.registry import get_explainer

log = get_logger(__name__)

_ENABLED_PIPELINES = [
    *WQI_HORIZON_PIPELINES,
    SOIL_PIPELINE,
]


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


def _restrict_sensors(discovered: list[str], pipeline_cfg) -> list[str]:
    """
    Apply the pipeline config's sensor_ids whitelist to the stations found in
    the feature collection.  An empty whitelist means "all discovered".
    """
    wanted = getattr(pipeline_cfg, "sensor_ids", None) or []
    if not wanted:
        return list(discovered)
    return [s for s in discovered if s in set(wanted)]


# Input vectors older than this are still used, but logged as stale.
_STALE_INPUT_HOURS = 6


def _feature_row(feat_doc, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Build the model input row aligned to feature_names.

    Features absent (or null) in the vector are filled with 0.0 and returned
    in `missing`, so the prediction can be flagged as built on filled inputs.
    """
    values: list[float] = []
    missing: list[str] = []
    for fname in feature_names:
        v = feat_doc.features.get(fname)
        if v is None:
            missing.append(fname)
            v = 0.0
        values.append(v)
    return np.array(values, dtype=np.float64).reshape(1, -1), missing


def _backfill_end(now: datetime, settle_hours: float) -> datetime:
    """
    Newest feature timestamp the historical backfill may predict from.

    Vectors younger than settle_hours are left to the prediction cycle: they
    may still be missing late readings, and a backfilled row for their hour
    would block (duplicate key) the cycle's settled, explained prediction.
    """
    return now - timedelta(hours=settle_hours) if settle_hours > 0 else now


async def _select_feature_vector(
    feat_repo: FeatureRepository,
    pipeline: str,
    feature_pl: str,
    sensor_id: str,
    active,
    now: datetime,
    settle_hours: float = 0.0,
):
    """
    Return the feature vector to predict from, or None to skip this sensor.

    Normally the newest vector at least settle_hours old: newer vectors may
    still be missing readings that arrive late (met data, a station's slower
    variables) and are recomputed by ingestion once they land.  If no vector
    is that old, the newest one is used.

    For delta-target models the chosen vector can still lack the current WQI
    (the anchor) because its hour was incomplete, so fall back to the newest
    vector that has it rather than skipping the pipeline for the whole cycle.
    """
    not_after = now - timedelta(hours=settle_hours) if settle_hours > 0 else None
    feat_doc = await feat_repo.find_latest_for_prediction(
        pipeline=feature_pl,
        sensor_id=sensor_id,
        feature_schema_version=active.feature_schema_version,
        not_after=not_after,
    )
    if feat_doc is None and not_after is not None:
        not_after = None
        feat_doc = await feat_repo.find_latest_for_prediction(
            pipeline=feature_pl,
            sensor_id=sensor_id,
            feature_schema_version=active.feature_schema_version,
        )
        if feat_doc is not None:
            log.info(
                "prediction_input_unsettled",
                pipeline=pipeline,
                sensor_id=sensor_id,
                feature_timestamp=feat_doc.feature_timestamp.isoformat(),
            )
    if feat_doc is None:
        log.debug("prediction_no_features_for_sensor", pipeline=pipeline, sensor_id=sensor_id)
        return None

    if active.target_is_delta and feat_doc.features.get(active.target_variable) is None:
        fallback = await feat_repo.find_latest_for_prediction(
            pipeline=feature_pl,
            sensor_id=sensor_id,
            feature_schema_version=active.feature_schema_version,
            require_feature=active.target_variable,
            not_after=not_after,
        )
        if fallback is None:
            log.warning("prediction_no_anchor_value", pipeline=pipeline, sensor_id=sensor_id)
            return None
        log.info(
            "prediction_anchor_fallback",
            pipeline=pipeline,
            sensor_id=sensor_id,
            latest_feature_timestamp=feat_doc.feature_timestamp.isoformat(),
            used_feature_timestamp=fallback.feature_timestamp.isoformat(),
        )
        feat_doc = fallback

    age_hours = (now - feat_doc.feature_timestamp).total_seconds() / 3600.0
    if age_hours > _STALE_INPUT_HOURS:
        log.warning(
            "prediction_input_stale",
            pipeline=pipeline,
            sensor_id=sensor_id,
            feature_timestamp=feat_doc.feature_timestamp.isoformat(),
            age_hours=round(age_hours, 1),
        )
    return feat_doc


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
        horizon_days = getattr(pipeline_cfg, "horizon_days", 0)

        if (pipeline == "water" or pipeline.startswith("water_wqi")) and not settings.enable_water_pipeline:
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

        feature_pl = active.feature_pipeline or pipeline
        sensor_ids = await _discover_sensors(db, feature_pl, active.feature_schema_version)
        sensor_ids = _restrict_sensors(sensor_ids, pipeline_cfg)
        if not sensor_ids:
            log.warning("prediction_no_feature_vectors", pipeline=pipeline)
            continue

        now = _utc_now()
        top_n = settings.api_shap_top_n

        for sensor_id in sensor_ids:
            feat_doc = await _select_feature_vector(
                feat_repo, pipeline, feature_pl, sensor_id, active, now,
                settle_hours=settings.prediction_input_settle_hours,
            )
            if feat_doc is None:
                continue

            # Build feature vector aligned to the model's feature_names
            try:
                x_row, missing_features = _feature_row(feat_doc, active.feature_names)
            except Exception as exc:
                log.error("prediction_feature_vector_failed", sensor_id=sensor_id, error=str(exc))
                continue
            if missing_features:
                log.warning(
                    "prediction_features_missing",
                    pipeline=pipeline,
                    sensor_id=sensor_id,
                    feature_timestamp=feat_doc.feature_timestamp.isoformat(),
                    missing=missing_features,
                )

            # Predict
            try:
                raw_pred = float(model.predict(x_row)[0])
            except Exception as exc:
                log.error("prediction_model_predict_failed", sensor_id=sensor_id, error=str(exc))
                continue

            # Delta-target models forecast the CHANGE; anchor on the current WQI.
            predicted_delta: Optional[float] = None
            if active.target_is_delta:
                anchor = feat_doc.features.get(active.target_variable)
                if anchor is None:
                    log.warning("prediction_no_anchor_value", pipeline=pipeline, sensor_id=sensor_id)
                    continue
                predicted_delta = raw_pred
                predicted_value = min(max(float(anchor) + raw_pred, 0.0), 100.0)
            else:
                predicted_value = raw_pred

            # XAI — compute raw explanation values first (before prediction insert)
            # The XAIResultDocument is built after insert so we have the real pred_id.
            xai_explanation: Optional[dict] = None
            top_shap: list[TopShapFeature] = []

            if settings.enable_xai:
                try:
                    explainer = get_explainer(active.model_family, model_type=active.model_type)
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
                input_had_filled_values=feat_doc.has_filled_inputs or bool(missing_features),
                target_timestamp=(
                    feat_doc.feature_timestamp + timedelta(days=horizon_days)
                    if horizon_days > 0 else None
                ),
                predicted_delta=predicted_delta,
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


async def run_historical_backfill(
    db: AsyncIOMotorDatabase,
    since: Optional[datetime] = None,
    pipelines: Optional[list[str]] = None,
) -> None:
    """
    Write predictions for all historical feature documents for any active WQI
    pipeline that has no predictions older than 7 days.  Called once at startup
    after run_bootstrap_if_needed() so the dashboard history chart is populated.
    Duplicate inserts are silently ignored by PredictionRepository.insert().

    With `since`, predictions are written for feature vectors from that time
    on, whether or not older predictions exist — used to regenerate a range
    after its predictions were deleted (scripts/force_recalibration.py).
    `pipelines` restricts the run to those pipeline names.

    Vectors newer than prediction_input_settle_hours are skipped (see
    _backfill_end); the prediction cycle covers them once they settle.
    """
    settings = get_settings()
    if since is not None:
        historical_start = since
    else:
        raw = get_raw_yaml()
        historical_start_str = raw.get("training", {}).get("historical_start_date", "2022-01-01")
        historical_start = datetime.fromisoformat(historical_start_str).replace(tzinfo=timezone.utc)

    store = ArtifactStore()
    pred_repo = PredictionRepository(db)
    feat_repo = FeatureRepository(db)
    xai_repo = XAIResultRepository(db)
    now = _utc_now()
    cutoff = now - timedelta(days=7)
    top_n = settings.api_shap_top_n

    for pipeline_cfg in _ENABLED_PIPELINES:
        pipeline = pipeline_cfg.pipeline_name
        horizon_days = getattr(pipeline_cfg, "horizon_days", 0)

        if (pipeline == "water" or pipeline.startswith("water_wqi")) and not settings.enable_water_pipeline:
            continue
        if pipeline == "soil" and not settings.enable_soil_pipeline:
            continue
        if pipelines is not None and pipeline not in pipelines:
            continue

        # Skip if historical predictions already exist
        has_old = since is None and await pred_repo.col.count_documents(
            {"pipeline": pipeline, "superseded": False, "input_feature_timestamp": {"$lt": cutoff}},
            limit=1,
        ) > 0
        if has_old:
            log.info("historical_backfill_skipped_already_done", pipeline=pipeline)
            continue

        active = await find_active_model(db, pipeline)
        if active is None:
            log.warning("historical_backfill_no_active_model", pipeline=pipeline)
            continue

        try:
            model_cls = get_model_class(active.model_type)
            model = model_cls()
            model.load(Path(store._resolve(active.artifact_path)))
        except Exception as exc:
            log.error("historical_backfill_artifact_load_failed", pipeline=pipeline, error=str(exc))
            continue

        feature_pl = active.feature_pipeline or pipeline
        feat_docs = await feat_repo.find_training_window(
            pipeline=feature_pl,
            start=historical_start,
            end=_backfill_end(now, settings.prediction_input_settle_hours),
            feature_schema_version=active.feature_schema_version,
        )
        allowed = _restrict_sensors(sorted({d.sensor_id for d in feat_docs}), pipeline_cfg)
        feat_docs = [d for d in feat_docs if d.sensor_id in allowed]
        if not feat_docs:
            log.warning("historical_backfill_no_feature_docs", pipeline=pipeline)
            continue

        log.info("historical_backfill_started", pipeline=pipeline, n_docs=len(feat_docs))
        written = 0

        for feat_doc in feat_docs:
            try:
                x_row, missing_features = _feature_row(feat_doc, active.feature_names)
                raw_pred = float(model.predict(x_row)[0])
            except Exception as exc:
                log.debug("historical_backfill_predict_failed", sensor_id=feat_doc.sensor_id, error=str(exc))
                continue

            predicted_delta: Optional[float] = None
            if active.target_is_delta:
                anchor = feat_doc.features.get(active.target_variable)
                if anchor is None:
                    continue
                predicted_delta = raw_pred
                predicted_value = min(max(float(anchor) + raw_pred, 0.0), 100.0)
            else:
                predicted_value = raw_pred

            top_shap: list[TopShapFeature] = []
            xai_result_id_for_pred = "000000000000000000000000"
            xai_explanation: Optional[dict] = None

            # KernelSHAP (SVR) is ~30s per row — skip XAI during bulk backfill
            # for SVR actives; live weekly predictions still compute it.
            # Linear models use the exact closed-form explainer (microseconds),
            # decision trees the tree explainer, so both are explained here.
            if settings.enable_xai and active.model_type != "svr":
                try:
                    explainer = get_explainer(active.model_family, model_type=active.model_type)
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
                except Exception:
                    pass

            pred_doc = PredictionDocument(
                pipeline=Pipeline(pipeline),
                sensor_id=feat_doc.sensor_id,
                target_variable=active.target_variable,
                predicted_value=predicted_value,
                input_feature_timestamp=feat_doc.feature_timestamp,
                input_had_filled_values=feat_doc.has_filled_inputs or bool(missing_features),
                target_timestamp=(
                    feat_doc.feature_timestamp + timedelta(days=horizon_days)
                    if horizon_days > 0 else None
                ),
                predicted_delta=predicted_delta,
                model_id=active.model_id,
                model_type=ModelType(active.model_type),
                model_family=ModelFamily(active.model_family),
                xai_result_id=xai_result_id_for_pred,
                top_shap_features=top_shap,
                prediction_generated_at=now,
            )
            pred_id = await pred_repo.insert(pred_doc)
            if pred_id:
                written += 1

                if settings.enable_xai and top_shap and xai_explanation is not None:
                    try:
                        xai_doc = XAIResultDocument(
                            prediction_id=pred_id,
                            model_id=active.model_id,
                            pipeline=Pipeline(pipeline),
                            sensor_id=feat_doc.sensor_id,
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
                    except Exception:
                        pass

        log.info("historical_backfill_completed", pipeline=pipeline, written=written, total=len(feat_docs))
