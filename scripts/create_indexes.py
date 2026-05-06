"""
scripts/create_indexes.py

Idempotent MongoDB index creation for all 9 DSS collections.

Delegates to dss_shared.db.bootstrap which is the single source of truth
for index definitions.  Run this script:
  - On initial deployment
  - After any index definition change in dss_shared/db/collections.py

Usage:
    python scripts/create_indexes.py

Environment variables read (via dss_shared.config):
    DSS_MONGO_URI      (default: mongodb://localhost:27017)
    DSS_MONGO_DB_NAME  (default: dss)
    DSS_CONFIG_PATH    (optional: path to config.yaml)

If an existing index has the same name but different keys/options, MongoDB
raises OperationFailure.  Drop the old index manually before re-running.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure dss_shared is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parents[1] / "shared"))


async def main() -> None:
    from dss_shared.db import bootstrap_db, get_database, probe_mongo
    from dss_shared.logging import setup_logging

    setup_logging(level="INFO", fmt="human", service="create_indexes_script")

    print("Connecting to MongoDB...")
    db = get_database()
    await probe_mongo(db, retry_count=3, backoff_seconds=2.0)
    print(f"Connected to database: {db.name}")

    print("Creating indexes...")
    await bootstrap_db(db)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
