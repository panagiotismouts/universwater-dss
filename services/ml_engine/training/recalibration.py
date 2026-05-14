"""
Weekly recalibration orchestrator.

Runs on the CronTrigger schedule (default: Monday 02:00 UTC).

For each enabled pipeline:
  1. Load the full expanding-window feature dataset (historical_start_date → now)
  2. Chronological 80/20 train/validation split
  3. Train a candidate model for each configured model type
  4. Evaluate candidate against the metric gate
  5. If gate passed: activate candidate, retire previous active model
  6. If gate failed: keep existing active model, log rejection

The full expanding-window strategy means every recalibration uses all available
data up to now, not a rolling window.

Sliding window (config training.window_strategy = "sliding") is also supported:
  uses only the most recent training.sliding_window_days of feature data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_raw_yaml, get_settings
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import EvaluationType, ModelStatus, Pipeline
from dss_shared.schemas.model import EmbeddedMetricsSummary, ModelRegistryDocument
from services.ml_engine.artifact_store import ArtifactStore
from services.ml_engine.models.registry import instantiate_model
from services.ml_engine.pipelines.soil_pipeline import SOIL_PIPELINE
from services.ml_engine.pipelines.water_pipeline import WATER_PIPELINE
from services.ml_engine.registry_manager import (
    activate_model,
    find_active_model,
    insert_candidate,
    reject_model,
)
from services.ml_engine.training.bootstrap import _make_model_id
from services.ml_engine.training.dataset_builder import build_dataset
from services.ml_engine.training.evaluator import evaluate_model

log = get_logger(__name__)

_ENABLED_PIPELINES = [WATER_PIPELINE, SOIL_PIPELINE]
_TRAIN_RATIO = 0.8


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


async def run_recalibration(db: AsyncIOMotorDatabase) -> None:
    """Run the weekly recalibration cycle for all enabled pipelines."""
    settings = get_settings()
    raw = get_raw_yaml()
    training_cfg = raw.get("training", {})
    window_strategy = training_cfg.get("window_strategy", "full")
    sliding_days = int(training_cfg.get("sliding_window_days", 365))
    historical_start_str = training_cfg.get("historical_start_date", "2022-01-01")

    now = _utc_now()

    if window_strategy == "sliding":
        start = now - timedelta(days=sliding_days)
    else:
        start = datetime.fromisoformat(historical_start_str).replace(tzinfo=timezone.utc)

    log.info("recalibration_started", window_strategy=window_strategy, start=start.isoformat())

    for pipeline_cfg in _ENABLED_PIPELINES:
        pipeline_name = pipeline_cfg.pipeline_name
        if pipeline_name == "water" and not settings.enable_water_pipeline:
            continue
        if pipeline_name == "soil" and not settings.enable_soil_pipeline:
            continue

        log.info("recalibration_pipeline_started", pipeline=pipeline_name)
        try:
            await _recalibrate_pipeline(db, pipeline_cfg, start, now)
        except Exception as exc:
            log.error(
                "recalibration_pipeline_failed",
                pipeline=pipeline_name,
                error=str(exc),
                exc_info=True,
            )

    log.info("recalibration_completed")


async def _recalibrate_pipeline(db, pipeline_cfg, start: datetime, end: datetime) -> None:
    pipeline = pipeline_cfg.pipeline_name
    schema_version = pipeline_cfg.feature_schema_version
    target = pipeline_cfg.target_variable
    store = ArtifactStore()

    try:
        X, y, feature_names = await build_dataset(
            db, pipeline, start, end, schema_version, target
        )
    except ValueError as exc:
        log.warning("recalibration_dataset_insufficient", pipeline=pipeline, error=str(exc))
        return

    split = int(len(X) * _TRAIN_RATIO)
    if split < 5 or (len(X) - split) < 2:
        log.warning("recalibration_split_too_small", pipeline=pipeline, n=len(X))
        return

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Get current active model for baseline comparison
    active = await find_active_model(db, pipeline)
    baseline_model_id = active.model_id if active else None

    # Train all model types; collect passing candidates and activate only the best.
    candidates = []
    for model_type in pipeline_cfg.model_types:
        model_id = _make_model_id(pipeline, model_type)
        artifact_rel = f"{pipeline}/{model_type}/{model_id}.joblib"

        log.info("recalibration_model_training", pipeline=pipeline, model_type=model_type, model_id=model_id)

        model = instantiate_model(model_type)
        model.fit(X_train, y_train)

        metrics_doc = await evaluate_model(
            model=model,
            X_val=X_val,
            y_val=y_val,
            pipeline=pipeline,
            model_id=model_id,
            db=db,
            validation_window_start=end,
            validation_window_end=end,
            evaluation_type=EvaluationType.RECALIBRATION_CHECK,
            baseline_model_id=baseline_model_id,
        )

        m = metrics_doc.metrics
        summary = EmbeddedMetricsSummary(r2=m.r2, mae=m.mae, rmse=m.rmse, mse=m.mse, mape=m.mape)
        inner = getattr(model, "_model", None)
        hyperparams: dict = inner.get_params() if inner is not None else {}

        candidate = ModelRegistryDocument(
            model_id=model_id,
            pipeline=Pipeline(pipeline),
            model_type=model_type,
            model_family=model.model_family,
            feature_schema_version=schema_version,
            target_variable=target,
            training_data_start=start,
            training_data_end=end,
            training_sample_count=len(X_train),
            feature_names=feature_names,
            hyperparameters=hyperparams,
            artifact_path=artifact_rel,
            metrics_summary=summary,
            status=ModelStatus.CANDIDATE,
            trained_at=end,
        )
        await insert_candidate(db, candidate)

        if metrics_doc.passed_threshold:
            log.info(
                "recalibration_model_passed_gate",
                pipeline=pipeline,
                model_type=model_type,
                model_id=model_id,
                r2=f"{m.r2:.4f}",
            )
            candidates.append((model_id, m.r2, model, artifact_rel))
        else:
            await reject_model(db, model_id, metrics_doc.rejection_reason or "metric gate failed")
            log.info(
                "recalibration_model_rejected",
                pipeline=pipeline,
                model_type=model_type,
                model_id=model_id,
                reason=metrics_doc.rejection_reason,
            )

    # Activate only the single best-performing candidate
    if candidates:
        best_model_id, best_r2, best_model, best_artifact_rel = max(candidates, key=lambda c: c[1])
        best_model.save(store._resolve(best_artifact_rel))
        await activate_model(db, best_model_id)
        log.info(
            "recalibration_best_model_activated",
            pipeline=pipeline,
            model_id=best_model_id,
            r2=f"{best_r2:.4f}",
        )
