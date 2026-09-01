"""
scripts/backfill_new_sensor.py

One-off historical backfill for the new water-quality sensor
(new_water_station / WINGS thing_id 865969070680657), live since
2026-07-23. Also recomputes HCMR's full feature history under the new
water_v2 schema, so both sensors share a consistent column set for pooled
training (see the sensor-integration plan for the rationale).

Three phases, run in order:
  1. Raw data: fetch the new sensor's full history for all 16 water
     variables directly from WINGS (the normal scheduled poll cannot do
     this — see checkpoint_manager.py, checkpoints are not sensor-scoped),
     drop known sensor fault-code sentinels, normalize + preprocess +
     insert into preprocessed_measurements via the same pure functions the
     live ingestion path uses.
  2. Feature engineering + WQI backfill for new_water_station, hourly from
     go-live to now.
  3. Feature engineering + WQI re-backfill for hcmr under water_v2, hourly
     across its full history.

Idempotent throughout: insert_many/upsert are both safe to re-run.

Usage:
    python scripts/backfill_new_sensor.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "shared"))
sys.path.insert(0, str(_repo))

os.environ.setdefault("DSS_MONGO_URI",     "mongodb://localhost:27017")
os.environ.setdefault("DSS_MONGO_DB_NAME", "dss")
os.environ.setdefault("DSS_ENV",           "local")
os.environ.setdefault("DSS_LOG_LEVEL",     "INFO")
os.environ.setdefault("DSS_LOG_FORMAT",    "human")
os.environ.setdefault("DSS_MODEL_ARTIFACT_PATH", str(_repo / "artifacts"))

# Secrets: only set dev placeholders in local mode. In server mode, require
# the operator to export real values so we never silently sign with "".
def _require_secret(var_name: str) -> None:
    if os.environ.get(var_name):
        return
    if os.environ.get("DSS_ENV", "local") == "local":
        os.environ.setdefault(var_name, "local_dev_only_placeholder")
        return
    sys.exit(
        f"ERROR: {var_name} must be set when DSS_ENV=server "
        f"(generate with: openssl rand -base64 48)"
    )


_require_secret("DSS_JWT_SECRET_KEY")
_require_secret("DSS_ADMIN_API_KEY")

NEW_SENSOR_ID = "new_water_station"
DEVICE_ID = "865969070680657"
HCMR_SENSOR_ID = "hcmr"

# Same sentinel fault codes identified and handled in the prior read-only
# charting session (graphs_new_sensor/fetch_and_plot_new_sensor.py):
# exact-match, confirmed to be the ONLY negative values anywhere in this
# device's 16 channels, and confirmed to land on simultaneous fault bursts
# across all channels (not real readings).
_SENTINELS = {-1.0, -2.0}

_WATER_VARIABLES = [
    "ph", "dissolved_oxygen", "temperature_water", "turbidity", "conductivity",
    "do_saturation", "tds", "salinity", "orp", "ammonia", "nitrate",
    "chlorophyll_a", "ammonium", "blue_green_algae", "cdom", "sigma_t",
]
# WINGS datastream suffixes differ slightly from our canonical variable
# names for two of them (matches config.yaml's mapping).
_WINGS_SUFFIX = {
    "temperature_water": "temperature",
    "do_saturation": "dissolved_oxygen_saturation",
    "chlorophyll_a": "chlorophyll",
}


async def _get_wings_token() -> str:
    import httpx
    resp = httpx.post(
        os.environ["DSS_WINGS_SSO_URL"],
        data={
            "grant_type": "client_credentials",
            "client_id": os.environ["DSS_WINGS_CLIENT_ID"],
            "client_secret": os.environ["DSS_WINGS_CLIENT_SECRET"],
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def _fetch_wings_observations(token: str, datastream_id: str) -> list[dict]:
    import httpx
    base_url = os.environ["DSS_WINGS_API_BASE_URL"].rstrip("/")
    url = f"{base_url}/api/v1/collections/sensor_things:observations"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    filter_obj = {"op": "eq", "a": "datastream_id", "v": datastream_id}
    all_rows: list[dict] = []
    offset, limit = 0, 1000
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            params = {
                "version": "v2",
                "fields": "phenomenon_time_start,result_number",
                "limit": str(limit),
                "offset": str(offset),
                "filter": json.dumps(filter_obj),
            }
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            rows = resp.json()
            all_rows.extend(rows)
            if len(rows) < limit:
                break
            offset += limit
    return all_rows


async def phase1_backfill_raw_data(db) -> None:
    """Fetch the new sensor's full WINGS history and insert it, sensor-fault
    sentinels excluded, via the same normalize + preprocess pipeline the
    live ingestion path uses."""
    from dss_shared.db.repositories.measurements import MeasurementRepository
    from dss_shared.logging import get_logger
    from services.ingestion.normalizer import normalize_wings_water
    from services.ingestion.preprocessing.pipeline import run_preprocessing_pipeline

    log = get_logger("backfill_new_sensor.phase1")
    token = await _get_wings_token()
    repo = MeasurementRepository(db)
    fetched_at = datetime.now(tz=timezone.utc)

    total_inserted = 0
    total_duplicates = 0
    total_sentinels_dropped = 0

    for var_name in _WATER_VARIABLES:
        suffix = _WINGS_SUFFIX.get(var_name, var_name)
        datastream_id = f"{DEVICE_ID}_{suffix}"
        rows = await _fetch_wings_observations(token, datastream_id)

        clean_rows = []
        for row in rows:
            v = row.get("result_number")
            if v is not None and float(v) in _SENTINELS:
                total_sentinels_dropped += 1
                continue
            clean_rows.append(row)

        readings = normalize_wings_water(
            rows=clean_rows,
            sensor_id=NEW_SENSOR_ID,
            variable_name=var_name,
            fetched_at=fetched_at,
        )
        docs = await run_preprocessing_pipeline(readings, db)
        inserted, duplicates = await repo.insert_many(docs)
        total_inserted += inserted
        total_duplicates += duplicates

        log.info(
            "phase1_variable_backfilled",
            variable=var_name,
            raw_rows=len(rows),
            sentinels_dropped=len(rows) - len(clean_rows),
            inserted=inserted,
            duplicates=duplicates,
        )

    log.info(
        "phase1_completed",
        total_inserted=total_inserted,
        total_duplicates=total_duplicates,
        total_sentinels_dropped=total_sentinels_dropped,
    )


async def _sensor_measurement_range(db, sensor_id: str) -> tuple[datetime, datetime] | None:
    """Earliest/latest measured_at across all water variables for a sensor."""
    col = db["preprocessed_measurements"]
    first = await col.find_one(
        {"pipeline": "water", "sensor_id": sensor_id},
        sort=[("measured_at", 1)],
    )
    last = await col.find_one(
        {"pipeline": "water", "sensor_id": sensor_id},
        sort=[("measured_at", -1)],
    )
    if first is None or last is None:
        return None
    return first["measured_at"], last["measured_at"]


async def _backfill_water_features(db, sensor_id: str) -> None:
    """Hourly compute_water_features + upsert across a sensor's full
    measurement history, mirroring coordinator.py's large-historical-batch
    loop (6h warm-up offset, hourly step)."""
    from dss_shared.db.repositories.features import FeatureRepository
    from dss_shared.logging import get_logger
    from services.ingestion.feature_engineering.coordinator import _compute_and_persist

    log = get_logger("backfill_new_sensor.features")
    time_range = await _sensor_measurement_range(db, sensor_id)
    if time_range is None:
        log.warning("no_measurements_found", sensor_id=sensor_id)
        return
    batch_start, batch_end = time_range

    feature_repo = FeatureRepository(db)
    _1H = timedelta(hours=1)
    _6H = timedelta(hours=6)

    ts = batch_start + _6H
    n = 0
    while ts <= batch_end:
        await _compute_and_persist("water", sensor_id, ts, db, feature_repo)
        ts += _1H
        n += 1
        if n % 200 == 0:
            log.info("features_backfill_progress", sensor_id=sensor_id, hours_done=n, current_ts=ts.isoformat())
    if batch_end != ts - _1H:
        await _compute_and_persist("water", sensor_id, batch_end, db, feature_repo)
        n += 1

    log.info("features_backfill_completed", sensor_id=sensor_id, total_hours=n,
              range_start=batch_start.isoformat(), range_end=batch_end.isoformat())


async def phase2_backfill_new_sensor_features(db) -> None:
    await _backfill_water_features(db, NEW_SENSOR_ID)


async def phase3_recompute_hcmr_features_v2(db) -> None:
    await _backfill_water_features(db, HCMR_SENSOR_ID)


async def main() -> None:
    from dss_shared.db import get_database, probe_mongo
    from dss_shared.logging import setup_logging, get_logger

    setup_logging(level="INFO", fmt="human", service="backfill_new_sensor_script")
    log = get_logger("backfill_new_sensor")

    db = get_database()
    await probe_mongo(db)

    log.info("phase1_started")
    await phase1_backfill_raw_data(db)

    log.info("phase2_started")
    await phase2_backfill_new_sensor_features(db)

    log.info("phase3_started")
    await phase3_recompute_hcmr_features_v2(db)

    log.info("backfill_new_sensor_all_phases_completed")


if __name__ == "__main__":
    asyncio.run(main())
