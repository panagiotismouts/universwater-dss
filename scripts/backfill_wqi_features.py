"""
scripts/backfill_wqi_features.py

Patches wqi_brown, wqi_ccme, and wqi_entropy into existing water feature
documents that were created before these targets were added.

For each existing water feature document missing wqi_brown, it re-queries the
raw measurements for that sensor/timestamp and re-runs the WQI computation,
then writes only the three new fields into the document via $set.

Usage:
    python scripts/backfill_wqi_features.py [--dry-run] [--batch 200]

    --dry-run   Print how many documents would be patched, no writes.
    --batch N   Process N documents at a time (default: 100).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "shared"))
sys.path.insert(0, str(_repo / "services" / "ingestion"))

import os
os.environ.setdefault("DSS_MONGO_URI",    "mongodb://localhost:27017")
os.environ.setdefault("DSS_MONGO_DB_NAME", "dss")


async def main(dry_run: bool, batch_size: int) -> None:
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.logging import setup_logging
    from feature_engineering.water_features import _compute_wqi_brown, _compute_wqi_ccme, _compute_wqi_entropy
    from dss_shared.db.repositories.measurements import MeasurementRepository
    from datetime import timedelta

    _168H = timedelta(hours=168)

    setup_logging(level="INFO", fmt="human", service="backfill_wqi")

    db = get_database()
    await probe_mongo(db)

    col = db["engineered_features"]

    # Count documents that need patching
    query = {
        "pipeline": "water",
        "feature_schema_version": "water_v1",
        "superseded": False,
        "features.wqi_brown": {"$exists": False},
    }
    total = await col.count_documents(query)
    print(f"\nDocuments to patch: {total}")

    if dry_run:
        print("Dry-run mode — no writes.")
        return

    repo = MeasurementRepository(db)
    patched = 0
    skipped = 0

    cursor = col.find(query, {"sensor_id": 1, "feature_timestamp": 1})
    async for doc in cursor:
        sensor_id       = doc["sensor_id"]
        feature_ts      = doc["feature_timestamp"]
        window_start    = feature_ts - _168H

        ph_docs   = await repo.find_window("water", sensor_id, "ph",               window_start, feature_ts)
        do_docs   = await repo.find_window("water", sensor_id, "dissolved_oxygen",  window_start, feature_ts)
        temp_docs = await repo.find_window("water", sensor_id, "temperature_water", window_start, feature_ts)
        cond_docs = await repo.find_window("water", sensor_id, "conductivity",      window_start, feature_ts)
        orp_docs  = await repo.find_window("water", sensor_id, "orp",              window_start, feature_ts)

        wqi_brown   = _compute_wqi_brown(ph_docs, do_docs, temp_docs, cond_docs, orp_docs, feature_ts)
        wqi_ccme    = _compute_wqi_ccme(ph_docs, do_docs, temp_docs, cond_docs, orp_docs)
        wqi_entropy = _compute_wqi_entropy(ph_docs, do_docs, temp_docs, cond_docs, orp_docs, feature_ts)

        if wqi_brown is None and wqi_ccme is None and wqi_entropy is None:
            skipped += 1
            continue

        update: dict = {}
        if wqi_brown   is not None: update["features.wqi_brown"]   = wqi_brown
        if wqi_ccme    is not None: update["features.wqi_ccme"]    = wqi_ccme
        if wqi_entropy is not None: update["features.wqi_entropy"] = wqi_entropy

        await col.update_one({"_id": doc["_id"]}, {"$set": update})
        patched += 1

        if patched % 100 == 0:
            print(f"  patched {patched}/{total} ...")

    print(f"\nDone. Patched: {patched}  |  Skipped (no raw data): {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, batch_size=args.batch))
