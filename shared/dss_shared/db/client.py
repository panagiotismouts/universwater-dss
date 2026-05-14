"""
MongoDB client factory.

Provides a lazily-initialised Motor async client singleton.  All services
obtain their database handle through this module — never by constructing
their own AsyncIOMotorClient.

Design decisions:
  - Singleton client is intentionally not created at import time so that
    tests can set DSS_MONGO_URI before the first call.
  - probe_mongo() raises MongoUnavailableError if MongoDB is unreachable
    after all retries.  Services must not silently continue without a DB.
  - close_motor_client() should be called at service shutdown (lifespan
    teardown in FastAPI, signal handler in APScheduler services).

Usage:
    from dss_shared.db import get_database, probe_mongo

    db = get_database()
    await probe_mongo(db)   # call once at startup; raises on failure
"""

from __future__ import annotations

import asyncio
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from dss_shared.config import get_settings
from dss_shared.exceptions import MongoUnavailableError

_client: Optional[AsyncIOMotorClient] = None


def get_motor_client() -> AsyncIOMotorClient:
    """
    Return the singleton Motor client, creating it on first call.

    The client is created with the connection and server-selection timeouts
    from settings.  These control how long Motor waits before giving up on
    each individual connection attempt (not the total probe retry budget).
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            connectTimeoutMS=settings.mongo_connect_timeout_ms,
            serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
            tz_aware=True,
        )
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """Return the configured database handle."""
    settings = get_settings()
    return get_motor_client()[settings.mongo_db_name]


async def probe_mongo(
    db: Optional[AsyncIOMotorDatabase] = None,
    *,
    retry_count: Optional[int] = None,
    backoff_seconds: Optional[float] = None,
) -> None:
    """
    Blocking MongoDB readiness probe with retries and exponential back-off.

    Sends a "ping" command to MongoDB.  On failure, waits backoff_seconds
    and retries up to retry_count times.  Raises MongoUnavailableError if
    all attempts fail.

    Services must call this at startup before registering jobs or handling
    requests.  There is intentionally no silent fallback.

    Args:
        db:              Database handle (defaults to get_database()).
        retry_count:     Override for settings.mongo_retry_count.
        backoff_seconds: Override for settings.mongo_retry_backoff_seconds.

    Raises:
        MongoUnavailableError: if MongoDB is unreachable after all retries.
    """
    # Import here to avoid circular import at module level
    from dss_shared.logging import get_logger
    log = get_logger(__name__)

    settings = get_settings()
    db = get_database() if db is None else db
    max_retries = retry_count if retry_count is not None else settings.mongo_retry_count
    backoff = backoff_seconds if backoff_seconds is not None else settings.mongo_retry_backoff_seconds

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            await db.command("ping")
            log.info("mongo_probe_success", attempt=attempt)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning(
                "mongo_probe_failed",
                attempt=attempt,
                max_retries=max_retries,
                error=str(exc),
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff)

    raise MongoUnavailableError(
        f"MongoDB unreachable after {max_retries} attempts. Last error: {last_exc}"
    ) from last_exc


def close_motor_client() -> None:
    """
    Close and discard the singleton client.

    Call at service shutdown: FastAPI lifespan teardown, SIGTERM handler in
    APScheduler services.  Safe to call when no client exists.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None
