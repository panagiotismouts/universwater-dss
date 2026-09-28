"""
APScheduler setup for the ML engine service.

Jobs:
  - recalibration:   CronTrigger (default: Monday 02:00 UTC)
  - prediction_cycle: IntervalTrigger (config.yaml: prediction_interval_seconds)

Both jobs use max_instances=1, coalesce=True to prevent overlap.
"""

from __future__ import annotations

from datetime import datetime, timezone

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

# A recalibration delayed by a busy event loop still runs if it starts within this window.
_RECALIBRATION_MISFIRE_GRACE_SECONDS = 6 * 3600


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
        "day_of_week": _cron_dow_to_apscheduler(parts[4]),
    }


# Standard cron numbers weekdays 0-7 with 0 and 7 = Sunday; APScheduler 3.x
# numbers them 0-6 with 0 = Monday, so a numeric "1" passed straight through
# fires on Tuesday. Numeric weekdays are translated to names, which both agree on.
_CRON_DOW_NAMES = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")


def _cron_dow_to_apscheduler(field: str) -> str:
    """
    Translate a standard-cron day-of-week field into an APScheduler one.

    Numeric items ("1", "1-5", "*/2", "0,6") are expanded to explicit day names
    ("mon", "mon,tue,wed,thu,fri", ...). Name-only items ("mon", "mon-fri") and
    "*" are passed through unchanged.
    """
    names: list[str] = []
    for item in field.split(","):
        rng, _, step_str = item.partition("/")
        if not step_str and not any(c.isdigit() for c in rng):
            names.append(item)
            continue
        try:
            step = int(step_str) if step_str else 1
            if rng == "*":
                lo, hi = 0, 6
            elif "-" in rng:
                lo, hi = (int(x) for x in rng.split("-", 1))
            else:
                lo = int(rng)
                hi = 7 if step_str else lo
        except ValueError:
            raise ValueError(f"Unsupported cron day-of-week field: {field!r}") from None
        if not (0 <= lo <= hi <= 7) or step < 1:
            raise ValueError(f"Unsupported cron day-of-week field: {field!r}")
        names.extend(_CRON_DOW_NAMES[d % 7] for d in range(lo, hi + 1, step))
    return ",".join(dict.fromkeys(names))


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
        # Training runs on the event loop, so a prediction cycle in progress at
        # the scheduled time delays the wake-up; APScheduler's default grace of
        # 1 s would then skip the whole week's run.
        misfire_grace_time=_RECALIBRATION_MISFIRE_GRACE_SECONDS,
        replace_existing=True,
    )
    log.info("ml_scheduler_job_registered", job="recalibration", cron=settings.recalibration_cron)

    now = datetime.now(tz=timezone.utc)

    # ── Prediction cycle job ───────────────────────────────────────────────
    # prediction_interval_seconds is sourced from config.yaml; if absent
    # (settings field default is None), fall back to weekly.
    interval = settings.prediction_interval_seconds if settings.prediction_interval_seconds is not None else 604800
    scheduler.add_job(
        func=run_prediction_cycle,
        args=[db],
        trigger=IntervalTrigger(seconds=interval),
        next_run_time=now,
        id="prediction_cycle",
        name="Prediction cycle",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    log.info(
        "ml_scheduler_job_registered",
        job="prediction_cycle",
        interval_seconds=interval,
    )

    # ── Bootstrap check job ────────────────────────────────────────────────
    scheduler.add_job(
        func=run_bootstrap_if_needed,
        args=[db],
        trigger=IntervalTrigger(seconds=3600),
        next_run_time=now,
        id="bootstrap_check",
        name="Bootstrap check",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    log.info("ml_scheduler_job_registered", job="bootstrap_check", interval_seconds=3600)

    log.info("ml_scheduler_built")
    return scheduler
