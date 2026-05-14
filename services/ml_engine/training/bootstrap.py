"""
Bootstrap training orchestrator.

Called at ml_engine startup before the APScheduler starts.  Checks each enabled
pipeline for an active model.  If none exists, runs initial training on the full
historical feature dataset (from historical_start_date to now).

A 80/20 chronological train/validation split is used.  If the metric gate passes,
the model is activated; otherwise it is rejected and logged.

Bootstrap is idempotent: if an active model already exists the pipeline is skipped.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_raw_yaml, get_settings
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import EvaluationType, ModelFamily, ModelStatus, Pipeline
from dss_shared.schemas.model import EmbeddedMetricsSummary, ModelRegistryDocument
from services.ml_engine.artifact_store import ArtifactStore
from services.ml_engine.models.registry import instantiate_model
from services.ml_engine.pipelines.soil_pipeline import SOIL_PIPELINE
from services.ml_engine.pipelines.water_pipeline import WATER_PIPELINE
from services.ml_engine.registry_manager import (
    activate_model,
    has_active_model,
    insert_candidate,
    reject_model,
)
from services.ml_engine.training.dataset_builder import build_dataset
from services.ml_engine.training.evaluator import evaluate_model

log = get_logger(__name__)

_ENABLED_PIPELINES = [WATER_PIPELINE, SOIL_PIPELINE]
_TRAIN_RATIO = 0.8


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_model_id(pipeline: str, model_type: str) -> str:
    ts = _utc_now().strftime("%Y%m%d_%H%M%S")
    return f"{pipeline}_{model_type}_{ts}"


async def run_bootstrap_if_needed(db: AsyncIOMotorDatabase) -> None:
    """
    For each enabled pipeline, run bootstrap training if no active model exists.

    Idempotent — safe to call on every startup.
    """
    settings = get_settings()
    raw = get_raw_yaml()
    historical_start_str = raw.get("training", {}).get("historical_start_date", "2022-01-01")
    historical_start = datetime.fromisoformat(historical_start_str).replace(tzinfo=timezone.utc)
    now = _utc_now()

    for pipeline_cfg in _ENABLED_PIPELINES:
        pipeline_name = pipeline_cfg.pipeline_name

        # Skip disabled pipelines
        if pipeline_name == "water" and not settings.enable_water_pipeline:
            continue
        if pipeline_name == "soil" and not settings.enable_soil_pipeline:
            continue

        already_active = await has_active_model(db, pipeline_name)
        if already_active:
            log.info("bootstrap_skipped_active_model_exists", pipeline=pipeline_name)
            continue

        log.info("bootstrap_training_started", pipeline=pipeline_name)
        await _run_bootstrap_for_pipeline(db, pipeline_cfg, historical_start, now)


async def _run_bootstrap_for_pipeline(db, pipeline_cfg, start: datetime, end: datetime) -> None:
    pipeline = pipeline_cfg.pipeline_name
    schema_version = pipeline_cfg.feature_schema_version
    target = pipeline_cfg.target_variable
    store = ArtifactStore()

    try:
        X, y, feature_names = await build_dataset(
            db, pipeline, start, end, schema_version, target
        )
    except ValueError as exc:
        log.warning("bootstrap_dataset_insufficient", pipeline=pipeline, error=str(exc))
        return

    # 80/20 chronological split
    split = int(len(X) * _TRAIN_RATIO)
    if split < 5 or (len(X) - split) < 2:
        log.warning("bootstrap_split_too_small", pipeline=pipeline, n=len(X))
        return

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Derive validation window timestamps from the dataset position
    # (approximate — dataset_builder sorts by feature_timestamp ascending)
    val_start = end  # placeholder; exact timestamps not tracked through numpy
    val_end = end

    # Train all model types; collect (model_id, r2, passed, model, artifact_rel) tuples.
    # Activate only the single best-performing model after all are trained.
    candidates = []
    for model_type in pipeline_cfg.model_types:
        result = await _train_and_register(
            db=db,
            pipeline=pipeline,
            model_type=model_type,
            schema_version=schema_version,
            target=target,
            feature_names=feature_names,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_start=start,
            train_end=end,
            val_start=val_start,
            val_end=val_end,
            store=store,
        )
        if result is not None:
            candidates.append(result)

    # Pick the candidate with the highest R² and activate only that one
    if candidates:
        best = max(candidates, key=lambda c: c[1])  # c = (model_id, r2, model, artifact_rel)
        best_model_id, best_r2, best_model, best_artifact_rel = best
        best_model.save(store._resolve(best_artifact_rel))
        await activate_model(db, best_model_id)
        log.info(
            "bootstrap_best_model_activated",
            pipeline=pipeline,
            model_id=best_model_id,
            r2=f"{best_r2:.4f}",
        )


async def _train_and_register(
    db,
    pipeline: str,
    model_type: str,
    schema_version: str,
    target: str,
    feature_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_start: datetime,
    train_end: datetime,
    val_start: datetime,
    val_end: datetime,
    store: ArtifactStore,
) -> "tuple[str, float, object, str] | None":
    """Train and register one model type. Returns (model_id, r2, model, artifact_rel)
    if the metric gate passes, or None if rejected."""
    now = _utc_now()
    model_id = _make_model_id(pipeline, model_type)
    artifact_rel = f"{pipeline}/{model_type}/{model_id}.joblib"

    log.info("bootstrap_model_training", pipeline=pipeline, model_type=model_type, model_id=model_id)

    model = instantiate_model(model_type)
    model.fit(X_train, y_train)

    # Evaluate on validation split
    metrics_doc = await evaluate_model(
        model=model,
        X_val=X_val,
        y_val=y_val,
        pipeline=pipeline,
        model_id=model_id,
        db=db,
        validation_window_start=val_start,
        validation_window_end=val_end,
        evaluation_type=EvaluationType.VALIDATION,
    )

    m = metrics_doc.metrics
    summary = EmbeddedMetricsSummary(r2=m.r2, mae=m.mae, rmse=m.rmse, mse=m.mse, mape=m.mape)

    # Determine hyperparameters from the inner sklearn model
    inner = getattr(model, "_model", None)
    hyperparams: dict = inner.get_params() if inner is not None else {}

    # Build and persist the candidate registry document
    candidate = ModelRegistryDocument(
        model_id=model_id,
        pipeline=Pipeline(pipeline),
        model_type=model_type,
        model_family=model.model_family,
        feature_schema_version=schema_version,
        target_variable=target,
        training_data_start=train_start,
        training_data_end=train_end,
        training_sample_count=len(X_train),
        feature_names=feature_names,
        hyperparameters=hyperparams,
        artifact_path=artifact_rel,
        metrics_summary=summary,
        status=ModelStatus.CANDIDATE,
        trained_at=now,
    )
    await insert_candidate(db, candidate)

    if metrics_doc.passed_threshold:
        log.info(
            "bootstrap_model_passed_gate",
            pipeline=pipeline,
            model_type=model_type,
            model_id=model_id,
            r2=f"{m.r2:.4f}",
        )
        return (model_id, m.r2, model, artifact_rel)
    else:
        await reject_model(db, model_id, metrics_doc.rejection_reason or "metric gate failed")
        log.warning(
            "bootstrap_model_rejected",
            pipeline=pipeline,
            model_type=model_type,
            model_id=model_id,
            reason=metrics_doc.rejection_reason,
        )
        return None
