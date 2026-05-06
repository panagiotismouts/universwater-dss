"""
scripts/force_recalibration.py

Force a recalibration run outside the normal weekly schedule.

Usage:
    python scripts/force_recalibration.py

TODO: Implement by calling ml_engine recalibration module directly.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, str(__file__ + "/../../.."))


async def main() -> None:
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.logging import setup_logging

    setup_logging(level="INFO", fmt="human", service="force_recalibration_script")

    db = get_database()
    await probe_mongo(db)

    # TODO: import and call run_recalibration from ml_engine
    # from services.ml_engine.training.recalibration import run_recalibration
    # await run_recalibration(db)
    print("Force recalibration script: TODO — implement by calling ml_engine recalibration module.")


if __name__ == "__main__":
    asyncio.run(main())
