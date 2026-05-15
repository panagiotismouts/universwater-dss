"""
APScheduler setup for the ML engine service.

Jobs:
  - recalibration:   CronTrigger (default: Monday 02:00 UTC)
  - prediction_cycle: IntervalTrigger (default: every 3600 seconds)

Both jobs use max_instances=1, coalesce=True to prevent overlap.
"""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_settings
from dss_shared.logging import get_logger
from services.ml_engine.prediction.predictor import run_prediction_cycle
from services.ml_engine.training.bootstrap import run_bootstrap_if_needed
from services.ml_engine.training.recalibration import run_recalibration

log = get_logger(__name__)


def _parse_cron(cron_expr: str) -> dict:
    """
    Parse a 5-field cron expression into APScheduler CronTrigger kwargs.

    Standard cron: "minute hour dom month dow"
    Example: "0 2 * * 1" → Monday at 02:00 UTC.
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5-field cron expression, got: {cron_expr!r}")
    return {
        "minute":      parts[0],
        "hour":        parts[1],
        "day":         parts[2],
        "month":       parts[3],
        "day_of_week": parts[4],
    }


def build_ml_scheduler(db: AsyncIOMotorDatabase) -> AsyncIOScheduler:
    """
    Create and configure the ML engine AsyncIOScheduler.

    Returns a configured-but-not-started scheduler.
    """
    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="UTC")

    # ── Recalibration job ──────────────────────────────────────────────────
    cron_kwargs = _parse_cron(settings.recalibration_cron)
    scheduler.add_job(
        func=run_recalibration,
        args=[db],
        trigger=CronTrigger(timezone="UTC", **cron_kwargs),
        id="recalibration",
        name="Weekly recalibration",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    log.info("ml_scheduler_job_registered", job="recalibration", cron=settings.recalibration_cron)

    # ── Prediction cycle job ───────────────────────────────────────────────
    scheduler.add_job(
        func=run_prediction_cycle,
        args=[db],
        trigger=IntervalTrigger(seconds=settings.prediction_interval_seconds),
        id="prediction_cycle",
        name="Prediction cycle",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    log.info(
        "ml_scheduler_job_registered",
        job="prediction_cycle",
        interval_seconds=settings.prediction_interval_seconds,
    )

    # ── Bootstrap check job ────────────────────────────────────────────────
    scheduler.add_job(
        func=run_bootstrap_if_needed,
        args=[db],
        trigger=IntervalTrigger(seconds=3600),
        id="bootstrap_check",
        name="Bootstrap check",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    log.info("ml_scheduler_job_registered", job="bootstrap_check", interval_seconds=3600)

    log.info("ml_scheduler_built")
    return scheduler
