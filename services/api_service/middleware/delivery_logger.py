"""
Delivery logger middleware.

After each HTTP response is sent, writes one DeliveryLogDocument to
api_delivery_logs recording: client_id, endpoint, response status, latency.

Implemented as a Starlette BaseHTTPMiddleware subclass.

client_id is read from request.state.client_id (set by verify_bearer_token).
Falls back to "anonymous" when auth failed or the endpoint is unauthenticated.

Failures in the logger are caught and logged — they must never break the
API response.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from dss_shared.db import get_database
from dss_shared.db.repositories.delivery_logs import DeliveryLogRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.delivery_log import DeliveryLogDocument

log = get_logger(__name__)

# Query params that must never appear in the log
_SENSITIVE_PARAMS = {"client_secret", "password", "token", "secret"}


def _sanitize_params(params: dict) -> dict:
    return {k: "***" if k.lower() in _SENSITIVE_PARAMS else v for k, v in params.items()}


class DeliveryLoggerMiddleware(BaseHTTPMiddleware):
    """Write one audit log entry per HTTP request after the response is sent."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_ts = time.monotonic()
        request_timestamp = datetime.now(tz=timezone.utc)

        response = await call_next(request)

        latency_ms = int((time.monotonic() - start_ts) * 1000)
        client_id = getattr(request.state, "client_id", "anonymous")

        # Extract safe query params
        raw_params = dict(request.query_params)
        safe_params = _sanitize_params(raw_params) if raw_params else None

        # Best-effort pipeline detection from query params
        pipeline = raw_params.get("pipeline")

        try:
            db = get_database()
            doc = DeliveryLogDocument(
                client_id=client_id,
                endpoint=request.url.path,
                http_method=request.method,
                request_timestamp=request_timestamp,
                pipeline=pipeline,
                query_params=safe_params,
                response_status=response.status_code,
                latency_ms=latency_ms,
            )
            repo = DeliveryLogRepository(db)
            await repo.insert(doc)
        except Exception as exc:
            log.warning("delivery_log_write_failed", endpoint=request.url.path, error=str(exc))

        return response
