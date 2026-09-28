"""
scripts/flag_placeholder_readings.py

Re-flag stored measurements that are placeholders rather than readings,
using the same rule ingestion now applies at preprocessing
(services.ingestion.preprocessing.pipeline.placeholder_flag):

  - a sensor's no-data marker (new_water_station sends whole records of -1)
    → quality_flag "no_data"
  - an exact 0 of a variable that cannot be 0 (conductivity, tds)
    → quality_flag "suspect"

Readings stored before that rule existed kept quality_flag "ok", so feature
engineering used them (e.g. a 0 conductivity pinned the CCME-WQI).  This is
the one exception to measurement immutability: only quality_flag changes.
Feature vectors are not touched — recompute them afterwards with
scripts/backfill_features.py --stored for the same sensor.

Usage (inside the ingestion container, or with the repo on PYTHONPATH):
    python scripts/flag_placeholder_readings.py --sensor new_water_station [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "shared"))
sys.path.insert(0, str(_repo))

os.environ.setdefault("DSS_MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("DSS_MONGO_DB_NAME", "dss")
os.environ.setdefault("DSS_ENV", "local")
os.environ.setdefault("DSS_LOG_LEVEL", "INFO")
os.environ.setdefault("DSS_LOG_FORMAT", "human")
os.environ.setdefault("DSS_JWT_SECRET_KEY", "bootstrap_placeholder")
os.environ.setdefault("DSS_ADMIN_API_KEY", "bootstrap_placeholder")

from services.ingestion.preprocessing.pipeline import _SENSOR_PLACEHOLDERS  # noqa: E402
from services.ingestion.preprocessing.variable_registry import (  # noqa: E402
    get_all_variable_names,
    get_variable_spec,
)


def candidate_query(sensor_id: str) -> dict:
    """Mongo filter for the stored measurements that could be placeholders."""
    clauses: list[dict] = []
    markers = sorted(_SENSOR_PLACEHOLDERS.get(sensor_id, ()))
    if markers:
        clauses.append({"value": {"$in": markers}})
    zero_vars = sorted(v for v in get_all_variable_names() if get_variable_spec(v).zero_is_missing)
    if zero_vars:
        clauses.append({"variable_name": {"$in": zero_vars}, "value": 0.0})
    return {"sensor_id": sensor_id, "superseded": False, "$or": clauses}


async def run(sensor_id: str, dry_run: bool) -> Counter:
    from pymongo import UpdateOne

    from dss_shared.db import get_database, probe_mongo
    from dss_shared.db.collections import MEASUREMENTS
    from dss_shared.logging import get_logger, setup_logging
    from services.ingestion.preprocessing.pipeline import placeholder_flag

    setup_logging(level="INFO", fmt="human", service="flag_placeholder_readings")
    log = get_logger("flag_placeholder_readings")

    db = get_database()
    await probe_mongo(db)
    col = db[MEASUREMENTS]

    counts: Counter = Counter()
    ops: list[UpdateOne] = []
    cursor = col.find(
        candidate_query(sensor_id),
        projection={"variable_name": 1, "value": 1, "quality_flag": 1},
    )
    async for raw in cursor:
        spec = get_variable_spec(raw["variable_name"])
        flag = placeholder_flag(sensor_id, spec, raw["value"]) if spec else None
        if flag is None or raw.get("quality_flag") == flag.value:
            counts["unchanged"] += 1
            continue
        counts[f"{raw['variable_name']}: {raw.get('quality_flag')} -> {flag.value}"] += 1
        ops.append(UpdateOne({"_id": raw["_id"]}, {"$set": {"quality_flag": flag.value}}))

    if ops and not dry_run:
        result = await col.bulk_write(ops, ordered=False)
        counts["modified"] = result.modified_count

    log.info("flag_placeholders_completed", sensor_id=sensor_id, dry_run=dry_run, updates=len(ops))
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", required=True, help="sensor_id, e.g. new_water_station")
    p.add_argument("--dry-run", action="store_true", help="only count what would change, no writes")
    args = p.parse_args()
    counts = asyncio.run(run(args.sensor, args.dry_run))
    for key, n in sorted(counts.items()):
        print(f"{key:48s} {n}")


if __name__ == "__main__":
    main()
