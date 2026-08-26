"""
Model evaluator.

Computes evaluation metrics for a trained model candidate and checks them
against configured thresholds (the metric gate).

Metric gate rules (from config.yaml training.metric_thresholds):
  - R² ≥ min_r2 (default 0.75)
  - MAE / (max(y) - min(y)) ≤ max_mae_relative (default 0.15)

A ModelMetricsDocument is always written — both for passed and failed candidates.
On failure, rejection_reason is populated and passed_threshold=False.

The doc is written to the model_metrics collection but NOT returned to the caller
via repository; the caller receives the document and decides whether to activate
or reject the candidate.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_raw_yaml
from dss_shared.db.repositories.model_metrics import ModelMetricsRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import EvaluationType, Pipeline
from dss_shared.schemas.model import (
    ModelMetrics,
    ModelMetricsDocument,
    ThresholdCheckResult,
)

log = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


async def evaluate_model(
    model,               # BaseModel instance (already trained)
    X_val: np.ndarray,
    y_val: np.ndarray,
    pipeline: str,
    model_id: str,
    db: AsyncIOMotorDatabase,
    validation_window_start: datetime,
    validation_window_end: datetime,
    evaluation_type: EvaluationType = EvaluationType.VALIDATION,
    baseline_model_id: str | None = None,
    min_r2_override: float | None = None,
    delta_target: bool = False,
    selected_features: list[str] | None = None,
    feature_importance: dict[str, float] | None = None,
) -> ModelMetricsDocument:
    """
    Compute metrics, check gate thresholds, write model_metrics document.

    delta_target: the y values are horizon deltas (WQI(t+h) − WQI(t)).
      - baseline_mae = mean|y| is recorded: the MAE of the persistence
        scenario "no change" (Δ=0).  Because |(cur+Δ̂)−(cur+Δ)| = |Δ̂−Δ|,
        the model's MAE on deltas equals its MAE on the final forecast
        value, so mae < baseline_mae ⇔ the model beats persistence.
      - MAPE is skipped (deltas cross zero; the ratio is meaningless).

    Returns the persisted ModelMetricsDocument (with passed_threshold set).
    """
    # ── Compute predictions ────────────────────────────────────────────────
    y_pred = model.predict(X_val)

    # ── Compute metrics ────────────────────────────────────────────────────
    n = len(y_val)
    ss_res = float(np.sum((y_val - y_pred) ** 2))
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae  = float(np.mean(np.abs(y_val - y_pred)))
    mse  = float(np.mean((y_val - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mape: float | None = None
    baseline_mae: float | None = None
    if delta_target:
        baseline_mae = float(np.mean(np.abs(y_val)))
    else:
        nonzero = y_val != 0
        if np.any(nonzero):
            mape = float(np.mean(np.abs((y_val[nonzero] - y_pred[nonzero]) / y_val[nonzero])) * 100)

    metrics = ModelMetrics(r2=r2, mae=mae, mse=mse, rmse=rmse, mape=mape, baseline_mae=baseline_mae)

    # ── Load thresholds from config ────────────────────────────────────────
    raw = get_raw_yaml()
    thresholds = raw.get("training", {}).get("metric_thresholds", {})
    min_r2 = min_r2_override if min_r2_override is not None else float(thresholds.get("min_r2", 0.75))
    max_mae_relative = float(thresholds.get("max_mae_relative", 0.15))

    value_range = float(np.max(y_val) - np.min(y_val)) if len(y_val) > 1 else 1.0
    mae_relative = mae / value_range if value_range > 0 else float("inf")

    r2_check = ThresholdCheckResult(
        threshold=min_r2,
        actual=r2,
        passed=r2 >= min_r2,
        description=f"R² ≥ {min_r2}",
    )
    mae_check = ThresholdCheckResult(
        threshold=max_mae_relative,
        actual=mae_relative,
        passed=mae_relative <= max_mae_relative,
        description=f"MAE/range ≤ {max_mae_relative}",
    )
    passed = r2_check.passed and mae_check.passed

    rejection_reason: str | None = None
    if not passed:
        parts = []
        if not r2_check.passed:
            parts.append(f"R²={r2:.4f} < {min_r2}")
        if not mae_check.passed:
            parts.append(f"MAE_rel={mae_relative:.4f} > {max_mae_relative}")
        rejection_reason = "; ".join(parts)

    now = _utc_now()
    doc = ModelMetricsDocument(
        model_id=model_id,
        pipeline=Pipeline(pipeline),
        evaluation_type=evaluation_type,
        validation_window_start=validation_window_start,
        validation_window_end=validation_window_end,
        validation_sample_count=n,
        metrics=metrics,
        passed_threshold=passed,
        threshold_checks={
            "r2_minimum":         r2_check,
            "mae_relative_tolerance": mae_check,
        },
        rejection_reason=rejection_reason,
        baseline_model_id=baseline_model_id,
        evaluated_at=now,
        selected_features=selected_features or [],
        feature_importance=feature_importance or {},
    )

    # ── Persist ────────────────────────────────────────────────────────────
    repo = ModelMetricsRepository(db)
    await repo.insert(doc)

    log.info(
        "model_evaluated",
        model_id=model_id,
        pipeline=pipeline,
        r2=f"{r2:.4f}",
        mae=f"{mae:.4f}",
        mae_relative=f"{mae_relative:.4f}",
        baseline_mae=f"{baseline_mae:.4f}" if baseline_mae is not None else None,
        beats_baseline=(mae < baseline_mae) if baseline_mae is not None else None,
        passed=passed,
        rejection_reason=rejection_reason,
    )
    return doc
