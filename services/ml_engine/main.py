"""
ML engine service entry point.

Startup sequence (per Amendment C.5):
  1. Load and validate config
  2. Setup structured logging
  3. Probe MongoDB (blocking, with retries)
  4. Bootstrap detection (synchronous, blocking):
       - If active models found for all enabled pipelines → proceed
       - If any pipeline has no active model → run bootstrap training
  5. Start APScheduler:
       - CronTrigger for weekly recalibration
       - IntervalTrigger for prediction cycle
  6. Block until shutdown signal

Bootstrap runs BEFORE the scheduler starts to prevent bootstrap/recalibration
races (Amendment C.5).  This does NOT block the api_service or ingestion from
starting.
"""

from __future__ import annotations

import asyncio
import signal

from dss_shared.config import get_settings
from dss_shared.db import get_database, probe_mongo
from dss_shared.logging import setup_logging, get_logger

from services.ml_engine.scheduler import build_ml_scheduler
from services.ml_engine.training.bootstrap import run_bootstrap_if_needed

log = get_logger(__name__)


async def main() -> None:
    settings = get_settings()

    setup_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        service="ml_engine",
    )

    log.info("ml_engine_starting", env=settings.env)

    # 1. MongoDB readiness probe
    db = get_database()
    await probe_mongo(db)
    log.info("mongodb_connected", db=settings.mongo_db_name)

    # 2. Bootstrap detection (blocking before scheduler starts)
    await run_bootstrap_if_needed(db)

    # 3. Build and start scheduler
    scheduler = build_ml_scheduler(db)
    scheduler.start()
    log.info("ml_scheduler_started")

    # 4. Block until shutdown
    stop_event = asyncio.Event()

    def _handle_shutdown(*_):
        log.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_shutdown)

    await stop_event.wait()

    log.info("ml_engine_stopping")
    scheduler.shutdown(wait=True)
    log.info("ml_engine_stopped")


if __name__ == "__main__":
    asyncio.run(main())
