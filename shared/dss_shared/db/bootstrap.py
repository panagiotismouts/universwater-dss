"""
Database bootstrap — collection and index creation.

create_indexes(db)
    Idempotent.  Creates all DSS indexes defined in db.collections.
    Safe to call on every startup or after a schema change.
    Uses motor (async) so it integrates cleanly into service startup.

    On first run: creates all indexes.
    On subsequent runs: MongoDB detects existing indexes by name and skips them.
    If an index definition changes (different keys or options), MongoDB raises
    an error — the old index must be dropped manually before re-running.

bootstrap_db(db)
    Thin wrapper: calls create_indexes and logs the result.
    Call once at service startup after probe_mongo() succeeds.

Usage in a service main():
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.db.bootstrap import bootstrap_db

    db = get_database()
    await probe_mongo(db)
    await bootstrap_db(db)
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.collections import INDEX_DEFINITIONS
from dss_shared.logging import get_logger

log = get_logger(__name__)


async def create_indexes(db: AsyncIOMotorDatabase) -> None:
    """
    Create all DSS indexes idempotently.

    Iterates INDEX_DEFINITIONS from db.collections and calls create_index
    on each collection.  Each index is named so that MongoDB can detect
    existing definitions and skip re-creation.

    Raises:
        pymongo.errors.OperationFailure  if an existing index has the same
        name but different keys/options.  This requires manual intervention
        (drop the old index and re-run).
    """
    created: list[str] = []
    skipped: list[str] = []

    for collection_name, keys, options in INDEX_DEFINITIONS:
        col = db[collection_name]
        index_name: str = options.get("name", str(keys))
        try:
            result = await col.create_index(keys, **options)
            # Motor returns the index name; if it matched an existing index the
            # return value is still the name (not an error).
            created.append(f"{collection_name}.{index_name}")
            log.debug("index_created", collection=collection_name, index=index_name)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "index_creation_failed",
                collection=collection_name,
                index=index_name,
                error=str(exc),
            )
            raise

    log.info("indexes_created", count=len(created))


async def bootstrap_db(db: AsyncIOMotorDatabase) -> None:
    """
    Run database bootstrap: create all indexes.

    Call once per service startup, after probe_mongo() confirms MongoDB
    is reachable.  Idempotent — safe to call on every restart.
    """
    log.info("db_bootstrap_starting")
    await create_indexes(db)
    log.info("db_bootstrap_complete")
