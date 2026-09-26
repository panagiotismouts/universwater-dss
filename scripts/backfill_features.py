"""
scripts/backfill_features.py

Targeted feature backfill for one (pipeline, sensor) over an explicit time
range.  Companion to the per-run cap in the ingestion coordinator
(settings.feature_backfill_max_hours): when ingestion logs
"feature_backfill_truncated", run this for the truncated range.

Walks hourly from --start to --end (inclusive), and for each timestamp
reuses the coordinator's compute-and-persist path, which:
  - skips timestamps that already have a vector under the current schema
    version (one index lookup, no computation)
  - computes via compute_water_features / compute_soil_features otherwise
  - upserts on the natural key (idempotent; safe to re-run or overlap)

Usage (inside the ingestion container, or with the repo on PYTHONPATH):
    python scripts/backfill_features.py --pipeline water --sensor hcmr \
        --start 2026-07-14T00:00 --end 2026-09-19T23:00 [--dry-run]

Timestamps are UTC; a trailing "Z" or "+00:00" is accepted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "shared"))
sys.path.insert(0, str(_repo))

os.environ.setdefault("DSS_MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("DSS_MONGO_DB_NAME", "dss")
os.environ.setdefault("DSS_ENV", "local")
os.environ.setdefault("DSS_LOG_LEVEL", "INFO")
os.environ.setdefault("DSS_LOG_FORMAT", "human")
os.environ.setdefault("DSS_MODEL_ARTIFACT_PATH", str(_repo / "artifacts"))
os.environ.setdefault("DSS_JWT_SECRET_KEY", "bootstrap_placeholder")
os.environ.setdefault("DSS_ADMIN_API_KEY", "bootstrap_placeholder")

_1H = timedelta(hours=1)


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are taken as UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hourly_range(start: datetime, end: datetime) -> list[datetime]:
    """Inclusive hourly timestamps from start to end."""
    if end < start:
        raise ValueError("--end must not be before --start")
    out: list[datetime] = []
    ts = start
    while ts <= end:
        out.append(ts)
        ts += _1H
    return out


async def run(pipeline: str, sensor_id: str, start: datetime, end: datetime, dry_run: bool) -> Counter:
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.db.repositories.features import FeatureRepository
    from dss_shared.logging import get_logger, setup_logging
    from services.ingestion.feature_engineering.coordinator import (
        _SCHEMA_VERSIONS,
        _compute_and_persist,
    )

    setup_logging(level="INFO", fmt="human", service="backfill_features")
    log = get_logger("backfill_features")

    if pipeline not in _SCHEMA_VERSIONS:
        raise SystemExit(f"pipeline must be one of {sorted(_SCHEMA_VERSIONS)}, got {pipeline!r}")

    db = get_database()
    await probe_mongo(db)
    repo = FeatureRepository(db)
    timestamps = hourly_range(start, end)
    schema_version = _SCHEMA_VERSIONS[pipeline]

    log.info(
        "backfill_starting",
        pipeline=pipeline,
        sensor_id=sensor_id,
        start=start.isoformat(),
        end=end.isoformat(),
        hours=len(timestamps),
        schema_version=schema_version,
        dry_run=dry_run,
    )

    counts: Counter = Counter()
    for i, ts in enumerate(timestamps, 1):
        if dry_run:
            exists = await repo.exists(pipeline, sensor_id, ts, schema_version)
            counts["skipped" if exists else "would_compute"] += 1
        else:
            counts[await _compute_and_persist(pipeline, sensor_id, ts, db, repo)] += 1
        if i % 200 == 0:
            log.info("backfill_progress", done=i, total=len(timestamps), current=ts.isoformat(), **counts)

    log.info("backfill_completed", pipeline=pipeline, sensor_id=sensor_id, hours=len(timestamps), **counts)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", required=True, help="water | soil")
    p.add_argument("--sensor", required=True, help="sensor_id, e.g. hcmr")
    p.add_argument("--start", required=True, type=parse_utc, help="first feature timestamp (UTC)")
    p.add_argument("--end", required=True, type=parse_utc, help="last feature timestamp (UTC, inclusive)")
    p.add_argument("--dry-run", action="store_true", help="only count existing vs missing, no writes")
    args = p.parse_args()

    counts = asyncio.run(run(args.pipeline, args.sensor, args.start, args.end, args.dry_run))
    print(dict(counts))


if __name__ == "__main__":
    main()
