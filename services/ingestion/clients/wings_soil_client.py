"""
WINGS soil sensor API client.

Fetches all soil variables (soil_moisture, soil_temperature, nitrogen,
phosphorus, potassium, soil_conductivity) from the WINGS SensorThings (OGC) API.
Shares token management and pagination logic with WingsWaterClient; differs only
in which sensor entries it reads from config.

Datastream IDs and sensor_ids are read from config.yaml wings.sensors entries.
Soil sensors are those that carry at least one soil variable.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import httpx

from dss_shared.config import get_raw_yaml, get_settings
from dss_shared.exceptions import SourceAPIError, SourceAPITimeoutError
from dss_shared.logging import get_logger
from dss_shared.schemas.measurement import NormalizedReading
from services.ingestion.clients.base_client import BaseAPIClient
from services.ingestion.clients.wings_auth import WingsTokenManager
from services.ingestion.normalizer import normalize_wings_soil

log = get_logger(__name__)

_RETRY_BACKOFF_SECONDS = 2.0
_OBSERVATIONS_PATH = "/api/v1/collections/sensor_things:observations"


def _load_wings_soil_datastreams() -> dict[str, dict[str, str]]:
    """Return {sensor_id: {variable_name: datastream_id}} for soil sensors."""
    _soil_variables = {"soil_moisture", "soil_temperature", "nitrogen", "phosphorus", "potassium", "soil_conductivity"}
    raw = get_raw_yaml()
    sensors = raw.get("wings", {}).get("sensors", [])
    result: dict[str, dict[str, str]] = {}
    for s in sensors:
        streams: dict[str, str] = s.get("datastreams", {})
        if any(v in streams for v in _soil_variables):
            result[s["sensor_id"]] = streams
    return result


class WingsSoilClient(BaseAPIClient):
    """WINGS SensorThings client for soil variables."""

    source_name = "wings_soil"

    async def fetch(
        self,
        variable_name: str,
        since: datetime,
    ) -> list[NormalizedReading]:
        settings = get_settings()
        raw_cfg = get_raw_yaml()
        page_size: int = raw_cfg.get("wings", {}).get("page_size", 1000)
        sensor_map = _load_wings_soil_datastreams()
        fetched_at = datetime.now(tz=timezone.utc)

        all_readings: list[NormalizedReading] = []

        for sensor_id, datastreams in sensor_map.items():
            datastream_id = datastreams.get(variable_name)
            if datastream_id is None:
                continue

            sensor_readings = await self._fetch_all_pages(
                sensor_id=sensor_id,
                datastream_id=datastream_id,
                variable_name=variable_name,
                since=since,
                fetched_at=fetched_at,
                page_size=page_size,
                settings=settings,
            )
            all_readings.extend(sensor_readings)

        return all_readings

    async def _fetch_all_pages(
        self,
        sensor_id: str,
        datastream_id: str,
        variable_name: str,
        since: datetime,
        fetched_at: datetime,
        page_size: int,
        settings: Any,
    ) -> list[NormalizedReading]:
        base = settings.wings_api_base_url.rstrip("/")
        url = f"{base}{_OBSERVATIONS_PATH}"
        all_rows: list[dict[str, Any]] = []
        offset = 0

        while True:
            filter_obj = {
                "op": "and",
                "f": [
                    {"op": "eq",  "a": "datastream_id",        "v": datastream_id},
                    {"op": "gte", "a": "phenomenon_time_start", "v": since.strftime("%Y-%m-%dT%H:%M:%SZ")},
                ],
            }
            params = {
                "version": "v2",
                "fields": "phenomenon_time_start,result_number",
                "limit": str(page_size),
                "offset": str(offset),
                "filter": json.dumps(filter_obj),
            }

            rows = await self._get_with_retry(url, params, settings, variable_name)
            all_rows.extend(rows)

            if len(rows) < page_size:
                break
            offset += page_size

        readings = normalize_wings_soil(
            rows=all_rows,
            sensor_id=sensor_id,
            variable_name=variable_name,
            fetched_at=fetched_at,
        )
        log.debug(
            "wings_soil_fetch_ok",
            sensor_id=sensor_id,
            variable=variable_name,
            rows=len(all_rows),
            readings=len(readings),
        )
        return readings

    async def _get_with_retry(
        self,
        url: str,
        params: dict[str, str],
        settings: Any,
        variable_name: str,
    ) -> list[dict[str, Any]]:
        last_exc: Exception | None = None
        for attempt in range(1, settings.wings_max_retries + 1):
            try:
                token = await WingsTokenManager.get_token()
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                async with httpx.AsyncClient(
                    timeout=settings.wings_request_timeout_seconds
                ) as client:
                    response = await client.get(url, params=params, headers=headers)

                if response.status_code == 401:
                    WingsTokenManager.invalidate()
                    if attempt < settings.wings_max_retries:
                        await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                        continue
                    raise SourceAPIError(f"WINGS returned 401 for {variable_name}")

                if response.status_code in (400, 403, 404):
                    raise SourceAPIError(
                        f"WINGS returned {response.status_code} for {variable_name}: "
                        f"{response.text[:200]}"
                    )
                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as exc:
                last_exc = exc
                log.warning("wings_soil_timeout", variable=variable_name, attempt=attempt)
                if attempt < settings.wings_max_retries:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

            except SourceAPIError:
                raise

            except Exception as exc:
                last_exc = exc
                log.warning("wings_soil_fetch_error", variable=variable_name, attempt=attempt, error=str(exc))
                if attempt < settings.wings_max_retries:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

        if isinstance(last_exc, httpx.TimeoutException):
            raise SourceAPITimeoutError(
                f"WINGS soil timeout after {settings.wings_max_retries} attempts for {variable_name}"
            ) from last_exc
        raise SourceAPIError(
            f"WINGS soil fetch failed after {settings.wings_max_retries} attempts for {variable_name}"
        ) from last_exc

    async def health_check(self) -> bool:
        settings = get_settings()
        base = settings.wings_api_base_url.rstrip("/")
        try:
            token = await WingsTokenManager.get_token()
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{base}{_OBSERVATIONS_PATH}",
                    params={"version": "v2", "limit": "1"},
                    headers={"Authorization": f"Bearer {token}"},
                )
            return response.status_code < 500
        except Exception:
            return False
