"""
scripts/import_history.py

One-off import of historical measurements that predate the DSS's own
polling, through the same normalise -> preprocess -> insert path the
ingestion fetch job uses, so imported documents are indistinguishable from
polled ones.

Subcommands:

  hcmr-csv   HCMR Mikri Prespa export (HCMR API; the upstream of the WINGS
             `hcmr_15045987545_*` datastreams, values verified identical).
             Columns used: temperature, conductivity, ph, doMgPerL, orp.
             Stored as pipeline=water, sensor_id=hcmr, source=wings (same
             physical sensor and values as the WINGS feed).
             --until defaults to 2026-08-08T00:00Z: from that date the ORP
             probe reads -100..-509 mV together with near-zero DO, pending
             confirmation from HCMR whether that is a fault or real anoxia.

  met-json   Adcon (UOWM weather station) archive, as fetched from the Adcon
             addUPI gateway that feeds the UOWM API:
               {"<adcon node id>": [{"timestamp": <epoch s>, "value": <float>}, ...]}
             Converted with the DSS's own normalize_uowm_met, i.e. one
             reading each for pipeline=met_water and pipeline=met_soil.

Both are idempotent: the unique index on (pipeline, sensor_id,
variable_name, measured_at) turns re-imports and overlaps with already
polled data into counted duplicates, never extra documents.

Usage (inside the ingestion image, files bind-mounted):
    python scripts/import_history.py hcmr-csv --csv /data/hcmr.csv [--from TS] [--until TS] [--dry-run]
    python scripts/import_history.py met-json --json /data/adcon.json [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
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

HCMR_SENSOR_ID = "hcmr"
HCMR_SITE_ID = "15045987545"
DEFAULT_HCMR_UNTIL = "2026-08-08T00:00:00Z"

# HCMR CSV column -> DSS canonical variable_name
HCMR_COLUMNS: dict[str, str] = {
    "temperature": "temperature_water",
    "conductivity": "conductivity",
    "ph": "ph",
    "doMgPerL": "dissolved_oxygen",
    "orp": "orp",
}

# Adcon node id -> DSS canonical variable_name (mirrors config.yaml uowm.sensors)
ADCON_NODES: dict[str, str] = {
    "142": "air_temperature",
    "144": "rainfall",
    "145": "humidity",
    "154": "wind_speed",
    "155": "wind_direction",
    "156": "solar_radiation",
    "754": "infrared_temperature",
}

_BATCH = 2000


def _plain(value) -> str:
    """Enum member or plain string -> its string value (models store enum values)."""
    return str(getattr(value, "value", value))


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are taken as UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _finite(value: str | None) -> float | None:
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def hcmr_rows(
    path: Path,
    start: datetime | None,
    until: datetime | None,
) -> tuple[list[tuple[datetime, str, float]], Counter]:
    """
    Read the HCMR CSV and return (measured_at, variable_name, value) triples
    within [start, until).  Pure function (file in, list out) for unit tests.
    """
    out: list[tuple[datetime, str, float]] = []
    stats: Counter = Counter()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            stats["rows"] += 1
            if row.get("site_id") not in (HCMR_SITE_ID, HCMR_SITE_ID + ".0"):
                stats["skipped_other_site"] += 1
                continue
            ts = parse_utc(row["date"])
            if (start and ts < start) or (until and ts >= until):
                stats["skipped_out_of_range"] += 1
                continue
            stats["rows_in_range"] += 1
            for col, var in HCMR_COLUMNS.items():
                val = _finite(row.get(col))
                if val is None:
                    stats["skipped_missing_value"] += 1
                    continue
                out.append((ts, var, val))
    return out, stats


def hcmr_readings(triples, fetched_at: datetime) -> list:
    """Build NormalizedReadings for HCMR triples, with the registry's canonical units."""
    from dss_shared.schemas.enums import DataSource, Pipeline
    from dss_shared.schemas.measurement import NormalizedReading
    from services.ingestion.preprocessing.variable_registry import get_variable_spec

    units = {var: get_variable_spec(var).expected_unit for var in HCMR_COLUMNS.values()}
    return [
        NormalizedReading(
            pipeline=Pipeline.WATER,
            source=DataSource.WINGS,
            sensor_id=HCMR_SENSOR_ID,
            variable_name=var,
            raw_value=val,
            unit=units[var],
            measured_at=ts,
            fetched_at=fetched_at,
        )
        for ts, var, val in triples
    ]


def met_readings(payload: dict, fetched_at: datetime) -> tuple[list, Counter]:
    """Convert an Adcon JSON payload with the DSS's own UOWM normaliser."""
    from services.ingestion.normalizer import normalize_uowm_met

    readings: list = []
    stats: Counter = Counter()
    for node, rows in payload.items():
        var = ADCON_NODES.get(str(node))
        if var is None:
            stats[f"skipped_unknown_node_{node}"] += len(rows)
            continue
        got = normalize_uowm_met(rows, var, str(node), fetched_at)
        stats[f"slots_{var}"] += len(rows)
        readings.extend(got)
    return readings, stats


async def _existing_count(db, readings) -> int:
    """Dry-run helper: how many readings are already stored (by unique key)."""
    from dss_shared.db.collections import MEASUREMENTS

    groups: dict[tuple[str, str, str], list[datetime]] = {}
    for r in readings:
        groups.setdefault((_plain(r.pipeline), r.sensor_id, r.variable_name), []).append(r.measured_at)
    found = 0
    for (pipeline, sensor, var), stamps in groups.items():
        found += await db[MEASUREMENTS].count_documents({
            "pipeline": pipeline, "sensor_id": sensor, "variable_name": var,
            "measured_at": {"$in": stamps},
        })
    return found


async def ingest(readings, dry_run: bool, log) -> Counter:
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.db.repositories.measurements import MeasurementRepository
    from services.ingestion.preprocessing.pipeline import run_preprocessing_pipeline

    db = get_database()
    await probe_mongo(db)
    counts: Counter = Counter(readings=len(readings))

    docs = await run_preprocessing_pipeline(readings, db)
    for d in docs:
        counts[f"status_{_plain(d.processing_status)}"] += 1
        counts[f"flag_{_plain(d.quality_flag)}"] += 1

    if dry_run:
        existing = await _existing_count(db, readings)
        counts["already_stored"] = existing
        counts["would_insert"] = len(readings) - existing
        return counts

    repo = MeasurementRepository(db)
    for i in range(0, len(docs), _BATCH):
        inserted, duplicates = await repo.insert_many(docs[i:i + _BATCH])
        counts["inserted"] += inserted
        counts["duplicates"] += duplicates
        if (i // _BATCH) % 5 == 0:
            log.info("import_progress", done=min(i + _BATCH, len(docs)), total=len(docs),
                     inserted=counts["inserted"], duplicates=counts["duplicates"])
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("hcmr-csv", help="import the HCMR Mikri Prespa CSV export")
    a.add_argument("--csv", required=True, type=Path)
    a.add_argument("--from", dest="start", type=parse_utc, default=None, help="first timestamp (UTC, inclusive)")
    a.add_argument("--until", type=parse_utc, default=parse_utc(DEFAULT_HCMR_UNTIL),
                   help=f"end timestamp (UTC, exclusive); default {DEFAULT_HCMR_UNTIL}")
    a.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("met-json", help="import an Adcon weather archive JSON")
    b.add_argument("--json", required=True, type=Path)
    b.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from dss_shared.logging import get_logger, setup_logging

    setup_logging(level="INFO", fmt="human", service="import_history")
    log = get_logger("import_history")
    fetched_at = datetime.now(tz=timezone.utc)

    if args.cmd == "hcmr-csv":
        triples, stats = hcmr_rows(args.csv, args.start, args.until)
        readings = hcmr_readings(triples, fetched_at)
    else:
        readings, stats = met_readings(json.loads(args.json.read_text()), fetched_at)

    log.info("import_parsed", cmd=args.cmd, readings=len(readings), **stats)
    counts = asyncio.run(ingest(readings, args.dry_run, log))
    log.info("import_completed", cmd=args.cmd, dry_run=args.dry_run, **counts)
    print(dict(stats))
    print(dict(counts))


if __name__ == "__main__":
    main()
