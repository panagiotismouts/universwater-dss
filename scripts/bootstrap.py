"""
scripts/bootstrap.py

Trigger initial model training for all enabled pipelines.

Run this manually on first deployment if bootstrap detection in ml_engine
startup is not sufficient, or to force a re-bootstrap:

    python scripts/bootstrap.py

This script connects directly to MongoDB, loads feature data, trains models,
and registers them in model_registry.  It is the v1 substitute for the
POST /admin/bootstrap API endpoint (Amendment C.7).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "shared"))
sys.path.insert(0, str(_repo))

os.environ.setdefault("DSS_MONGO_URI",     "mongodb://localhost:27017")
os.environ.setdefault("DSS_MONGO_DB_NAME", "dss")
os.environ.setdefault("DSS_ENV",           "local")
os.environ.setdefault("DSS_LOG_LEVEL",     "INFO")
os.environ.setdefault("DSS_LOG_FORMAT",    "human")
# Required by ml_engine settings
os.environ.setdefault("DSS_MODEL_ARTIFACT_PATH", str(_repo / "artifacts"))
os.environ.setdefault("DSS_JWT_SECRET_KEY", "bootstrap_placeholder")
os.environ.setdefault("DSS_ADMIN_API_KEY",  "bootstrap_placeholder")


async def main() -> None:
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.logging import setup_logging
    from services.ml_engine.training.bootstrap import run_bootstrap_if_needed

    setup_logging(level="INFO", fmt="human", service="bootstrap_script")

    db = get_database()
    await probe_mongo(db)

    await run_bootstrap_if_needed(db)


if __name__ == "__main__":
    asyncio.run(main())
