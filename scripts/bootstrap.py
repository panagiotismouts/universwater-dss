"""
scripts/bootstrap.py

Trigger initial model training for all enabled pipelines.

Run this manually on first deployment if bootstrap detection in ml_engine
startup is not sufficient, or to force a re-bootstrap:

    python scripts/bootstrap.py

This script connects directly to MongoDB, loads feature data, trains models,
and registers them in model_registry.  It is the v1 substitute for the
POST /admin/bootstrap API endpoint (Amendment C.7).

TODO: Implement full bootstrap flow, reusing ml_engine training modules.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure dss_shared is importable when running from repo root
sys.path.insert(0, str(__file__ + "/../../.."))


async def main() -> None:
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.logging import setup_logging

    setup_logging(level="INFO", fmt="human", service="bootstrap_script")

    db = get_database()
    await probe_mongo(db)

    # TODO: import and call run_bootstrap_if_needed from ml_engine
    # from services.ml_engine.training.bootstrap import run_bootstrap_if_needed
    # await run_bootstrap_if_needed(db)
    print("Bootstrap script: TODO — implement by calling ml_engine bootstrap module.")


if __name__ == "__main__":
    asyncio.run(main())
