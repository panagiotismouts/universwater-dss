"""
Repository for the api_delivery_logs collection.

Append-only.  One document per HTTP request served.
TTL index on request_timestamp drives automatic 90-day rotation.

Retrieval patterns (per Persistence Blueprint §C.8):
  - Insert a log entry
  - Find entries by client_id + time range (per-client audit)
  - Count by response_status (error rate aggregation)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.collections import DELIVERY_LOGS
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.schemas.delivery_log import DeliveryLogDocument


class DeliveryLogRepository(BaseRepository):
    """Repository for api_delivery_logs.  Append-only."""

    COLLECTION_NAME = DELIVERY_LOGS

    async def insert(self, doc: DeliveryLogDocument) -> str:
        """
        Append a delivery log entry.

        Best-effort — the caller (delivery logger middleware) should catch
        and log any exception from this method rather than letting it bubble
        up and break the API response.

        Returns the inserted _id string.
        """
        result = await self.col.insert_one(self._to_doc(doc))
        return str(result.inserted_id)

    async def find_by_client(
        self,
        client_id: str,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[DeliveryLogDocument]:
        """
        Return log entries for a client, optionally within a time range.
        Sorted by request_timestamp descending (newest first).
        """
        query: dict = {"client_id": client_id}
        if from_time or to_time:
            time_filter: dict = {}
            if from_time:
                time_filter["$gte"] = from_time
            if to_time:
                time_filter["$lte"] = to_time
            query["request_timestamp"] = time_filter

        cursor = self.col.find(
            query,
            sort=[("request_timestamp", pymongo.DESCENDING)],
            limit=limit,
        )
        return [self._from_doc(raw, DeliveryLogDocument) async for raw in cursor]

    async def count_by_status(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> dict[int, int]:
        """
        Return {response_status: count} for entries in the time range.
        Used for error rate aggregation.
        """
        match: dict = {}
        if from_time or to_time:
            time_filter: dict = {}
            if from_time:
                time_filter["$gte"] = from_time
            if to_time:
                time_filter["$lte"] = to_time
            match["request_timestamp"] = time_filter

        pipeline = [
            *([{"$match": match}] if match else []),
            {"$group": {"_id": "$response_status", "count": {"$sum": 1}}},
            {"$sort": {"_id": pymongo.ASCENDING}},
        ]
        cursor = self.col.aggregate(pipeline)
        return {doc["_id"]: doc["count"] async for doc in cursor}
