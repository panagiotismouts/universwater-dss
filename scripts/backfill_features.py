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

Regeneration (--force, --stored):
  Vectors computed while the measurements collection held duplicates (or
  any other bad source state) are wrong even though they exist.  --force
  bypasses the skip-if-exists check and recomputes every hour in the range,
  replacing stored vectors in place (same natural key, fresh created_at).
  --created-before TS narrows that to vectors whose created_at is older
  than TS: vectors written after the source data was fixed are left alone,
  so the run is safe to repeat or to overlap with live ingestion.

  Stored vectors are not always on the hour: the coordinator also writes
  one at each ingestion batch's exact end (e.g. 19:05:13).  An hourly walk
  never lands on those, so to regenerate what is actually stored use
  --stored: the timestamps come from the vectors already in the collection
  within [--start, --end] (filtered by --created-before), and each one is
  recomputed at its own timestamp.  --stored implies --force.

Usage (inside the ingestion container, or with the repo on PYTHONPATH):
    python scripts/backfill_features.py --pipeline water --sensor hcmr \
        --start 2026-07-14T00:00 --end 2026-09-19T23:00 [--dry-run]
    python scripts/backfill_features.py --pipeline soil --sensor soil_station_1 \
        --start 2025-09-23T00:00 --end 2026-09-26T14:30 \
        --stored --created-before 2026-09-26T14:30Z [--dry-run]

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


def decide_action(
    exists: bool,
    created_at: datetime | None,
    *,
    force: bool,
    created_before: datetime | None,
) -> str:
    """
    Decide what to do with one timestamp given the stored vector's state.

    Returns "compute" (no vector stored), "recompute" (stored vector must be
    replaced) or "skip" (stored vector is kept).  Pure function so the
    --force / --created-before policy can be unit-tested without Mongo.
    """
    if not exists:
        return "compute"
    if not force:
        return "skip"
    if created_before is None:
        return "recompute"
    # Legacy vectors without created_at are treated as stale.
    if created_at is None or created_at < created_before:
        return "recompute"
    return "skip"


def stored_query(
    pipeline: str,
    sensor_id: str,
    schema_version: str,
    start: datetime,
    end: datetime,
    created_before: datetime | None,
) -> dict:
    """Mongo filter selecting the stored vectors --stored will recompute."""
    q: dict = {
        "pipeline": pipeline,
        "sensor_id": sensor_id,
        "feature_schema_version": schema_version,
        "feature_timestamp": {"$gte": start, "$lte": end},
    }
    if created_before is not None:
        # Legacy vectors without created_at count as stale (see decide_action).
        q["$or"] = [{"created_at": {"$lt": created_before}}, {"created_at": {"$exists": False}}]
    return q


async def run(
    pipeline: str,
    sensor_id: str,
    start: datetime,
    end: datetime,
    dry_run: bool,
    *,
    force: bool = False,
    created_before: datetime | None = None,
    stored: bool = False,
) -> Counter:
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
    schema_version = _SCHEMA_VERSIONS[pipeline]

    if stored:
        force = True
        # Materialise the list first so the recompute upserts below never
        # race the cursor that selected them.
        cursor = repo.col.find(
            stored_query(pipeline, sensor_id, schema_version, start, end, created_before),
            projection={"feature_timestamp": 1},
            sort=[("feature_timestamp", 1)],
        )
        timestamps = [raw["feature_timestamp"] async for raw in cursor]
        timestamps = [
            ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc) for ts in timestamps
        ]
    else:
        timestamps = hourly_range(start, end)

    log.info(
        "backfill_starting",
        pipeline=pipeline,
        sensor_id=sensor_id,
        start=start.isoformat(),
        end=end.isoformat(),
        hours=len(timestamps),
        schema_version=schema_version,
        dry_run=dry_run,
        force=force,
        stored=stored,
        created_before=created_before.isoformat() if created_before else None,
    )

    counts: Counter = Counter()
    for i, ts in enumerate(timestamps, 1):
        # One covered index lookup; only the fields the policy needs.
        raw = await repo.col.find_one(
            {
                "pipeline": pipeline,
                "sensor_id": sensor_id,
                "feature_timestamp": ts,
                "feature_schema_version": schema_version,
            },
            projection={"_id": 1, "created_at": 1},
        )
        created_at = raw.get("created_at") if raw else None
        if created_at is not None and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        action = decide_action(raw is not None, created_at, force=force, created_before=created_before)

        if action == "skip":
            counts["skipped"] += 1
        elif dry_run:
            counts["would_compute" if action == "compute" else "would_recompute"] += 1
        else:
            outcome = await _compute_and_persist(
                pipeline, sensor_id, ts, db, repo, force=(action == "recompute")
            )
            counts[outcome] += 1
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
    p.add_argument(
        "--force", action="store_true",
        help="recompute vectors that already exist (replaced in place)",
    )
    p.add_argument(
        "--stored", action="store_true",
        help="recompute the vectors already stored in the range, at their own timestamps (implies --force)",
    )
    p.add_argument(
        "--created-before", type=parse_utc, default=None, metavar="TS",
        help="with --force/--stored: only recompute vectors whose created_at is before TS (UTC)",
    )
    args = p.parse_args()
    if args.created_before is not None and not (args.force or args.stored):
        p.error("--created-before only makes sense together with --force or --stored")

    counts = asyncio.run(
        run(
            args.pipeline, args.sensor, args.start, args.end, args.dry_run,
            force=args.force or args.stored, created_before=args.created_before, stored=args.stored,
        )
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
