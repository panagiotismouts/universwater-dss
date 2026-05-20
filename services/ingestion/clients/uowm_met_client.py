"""
UOWM meteorological API client.

Fetches meteorological variables (rainfall, air_temperature, humidity) from
the UOWM local REST API.  No authentication required.

Endpoint:
  GET {base_url}/sensors/{sensor_id}/data
  Params: start_time={unix_epoch}, end_time={unix_epoch}, limit=10000

Response:
  [{"timestamp": 1728388800, "datetime": "2024-10-08T10:00:00", "value": 15.2}, ...]

Pagination (cursor-based):
  Advance start_time to max(timestamp) + 1 after each page.
  Stop when response is empty.

Each reading produces TWO NormalizedReading objects — one for pipeline="met_water"
and one for pipeline="met_soil" (§E.3).

Sensor IDs are read from config.yaml uowm.sensors:
  air_temperature: 142
  rainfall: 144
  humidity: 145
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from dss_shared.config import get_raw_yaml, get_settings
from dss_shared.exceptions import SourceAPIError, SourceAPITimeoutError
from dss_shared.logging import get_logger
from dss_shared.schemas.measurement import NormalizedReading
from services.ingestion.clients.base_client import BaseAPIClient
from services.ingestion.normalizer import normalize_uowm_met

log = get_logger(__name__)

_RETRY_BACKOFF_SECONDS = 2.0
_PAGE_LIMIT = 500  # UOWM server truncates responses above ~109 KB (~780 records)


def _load_uowm_sensor_id(variable_name: str) -> int | None:
    """Return the UOWM numeric sensor ID for a canonical variable name."""
    raw = get_raw_yaml()
    return raw.get("uowm", {}).get("sensors", {}).get(variable_name)


class UOWMMetClient(BaseAPIClient):
    """UOWM REST client for meteorological variables (no auth)."""

    source_name = "uowm_met"

    async def fetch(
        self,
        variable_name: str,
        since: datetime,
    ) -> list[NormalizedReading]:
        settings = get_settings()
        sensor_id = _load_uowm_sensor_id(variable_name)
        if sensor_id is None:
            log.warning("uowm_sensor_id_not_configured", variable=variable_name)
            return []

        fetched_at = datetime.now(tz=timezone.utc)
        now = fetched_at
        base = settings.uowm_api_base_url.rstrip("/")
        url = f"{base}/sensors/{sensor_id}/data"

        all_rows: list[dict[str, Any]] = []
        start_ts = int(since.timestamp())
        end_cursor = int(now.timestamp())

        # The UOWM API returns records sorted DESC (newest first).
        # Paginate backward: each page moves end_cursor to just before the
        # oldest record in the previous page.
        while True:
            params = {
                "start_time": str(start_ts),
                "end_time": str(end_cursor),
                "limit": str(_PAGE_LIMIT),
            }
            rows = await self._get_with_retry(url, params, settings, variable_name)
            if not rows:
                break

            all_rows.extend(rows)

            if len(rows) < _PAGE_LIMIT:
                break   # last page (fewer records than limit)
            # Move end_cursor backward past the oldest record on this page
            end_cursor = min(r["timestamp"] for r in rows) - 1

        readings = normalize_uowm_met(
            rows=all_rows,
            variable_name=variable_name,
            sensor_id=str(sensor_id),
            fetched_at=fetched_at,
        )
        log.debug(
            "uowm_fetch_ok",
            variable=variable_name,
            sensor_id=sensor_id,
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
        for attempt in range(1, settings.uowm_max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=settings.uowm_request_timeout_seconds
                ) as client:
                    response = await client.get(url, params=params, headers={"Accept": "application/json"})

                if response.status_code in (400, 404):
                    raise SourceAPIError(
                        f"UOWM returned {response.status_code} for {variable_name}: "
                        f"{response.text[:200]}"
                    )
                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as exc:
                last_exc = exc
                log.warning("uowm_timeout", variable=variable_name, attempt=attempt)
                if attempt < settings.uowm_max_retries:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

            except SourceAPIError:
                raise

            except Exception as exc:
                last_exc = exc
                log.warning("uowm_fetch_error", variable=variable_name, attempt=attempt, error=str(exc))
                if attempt < settings.uowm_max_retries:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

        if isinstance(last_exc, httpx.TimeoutException):
            raise SourceAPITimeoutError(
                f"UOWM timeout after {settings.uowm_max_retries} attempts for {variable_name}"
            ) from last_exc
        raise SourceAPIError(
            f"UOWM fetch failed after {settings.uowm_max_retries} attempts for {variable_name}"
        ) from last_exc

    async def health_check(self) -> bool:
        settings = get_settings()
        base = settings.uowm_api_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base}/sensors")
            return response.status_code < 500
        except Exception:
            return False
