"""
API delivery log schema.

DeliveryLogDocument
    Persistence model for the api_delivery_logs collection.
    One document per HTTP request served.  Append-only.
    Subject to TTL-based expiry (TTL index on created_at).

    query_params must be sanitized before storage — never log auth credentials.
    prediction_ids are ObjectId strings referencing prediction_results._id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from dss_shared.schemas.base import MongoDocument, PyObjectId, utc_now


class DeliveryLogDocument(MongoDocument):
    """
    MongoDB document shape for the api_delivery_logs collection.

    Append-only.  The ObjectId _id provides approximate insertion order.

    Timestamp discipline:
      request_timestamp — domain: when the HTTP request was received
      created_at        — system: when this document was written (≈ request end)
    """

    # Request identity
    client_id: str = Field(..., min_length=1)   # "anonymous" for failed auth
    endpoint: str = Field(..., min_length=1)    # e.g. "/results/latest"
    http_method: str = Field(..., pattern=r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)$")

    # Request context
    request_timestamp: datetime     # UTC — domain
    pipeline: Optional[str] = None
    sensor_id: Optional[str] = None
    query_params: Optional[dict[str, Any]] = None   # Sanitized — no credentials

    # Response
    response_status: int = Field(..., ge=100, le=599)
    result_count: Optional[int] = Field(default=None, ge=0)
    prediction_ids: list[PyObjectId] = Field(default_factory=list)
    latency_ms: int = Field(..., ge=0)
    error_message: Optional[str] = None   # Brief error; no stack traces

    # System timestamp
    created_at: datetime = Field(default_factory=utc_now)
