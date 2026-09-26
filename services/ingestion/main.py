"""
Ingestion service entry point.

Startup sequence (per Amendment C.4, C.5):
  1. Load and validate config
  2. Setup structured logging
  3. Probe MongoDB (blocking, with retries)
  4. Build APScheduler with one job per (source, variable) pair
  5. Start scheduler
  6. Block until shutdown signal

This service owns the complete data pipeline:
  API polling → normalization → preprocessing (6-stage) → feature engineering → persistence

No other service is a prerequisite for this service to start.
"""

from __future__ import annotations

import asyncio
import signal

from dss_shared.config import get_settings
from dss_shared.db import bootstrap_db, get_database, probe_mongo
from dss_shared.db.collections import CHECKPOINTS
from dss_shared.logging import setup_logging, get_logger

from services.ingestion.scheduler import build_ingestion_scheduler

log = get_logger(__name__)


async def main() -> None:
    settings = get_settings()

    setup_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        service="ingestion",
    )

    log.info("ingestion_service_starting", env=settings.env)

    # 1. MongoDB readiness probe
    db = get_database()
    await probe_mongo(db)
    log.info("mongodb_connected", db=settings.mongo_db_name)

    # 1b. Ensure all indexes exist (idempotent).  The unique index on
    #     preprocessed_measurements is what makes re-fetches safe; without it
    #     every overlapping fetch inserts a full duplicate copy.
    await bootstrap_db(db)

    # 1c. Guard against the most expensive misconfiguration: bootstrap mode
    #     left on after the initial backfill.  Every tick then re-fetches the
    #     entire history for every variable.
    if settings.bootstrap_mode:
        existing_checkpoints = await db[CHECKPOINTS].count_documents({})
        if existing_checkpoints:
            log.warning(
                "bootstrap_mode_with_existing_checkpoints",
                checkpoint_count=existing_checkpoints,
                hint="Set DSS_BOOTSTRAP_MODE=false unless a full re-fetch is intended",
            )

    # 2. Build and start scheduler
    scheduler = build_ingestion_scheduler(db)
    scheduler.start()
    log.info("scheduler_started", job_count=len(scheduler.get_jobs()))

    # 3. Block until shutdown
    stop_event = asyncio.Event()

    def _handle_shutdown(*_):
        log.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_shutdown)

    await stop_event.wait()

    log.info("ingestion_service_stopping")
    scheduler.shutdown(wait=True)
    log.info("ingestion_service_stopped")


if __name__ == "__main__":
    asyncio.run(main())
