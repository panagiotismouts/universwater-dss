"""
Repository for the prediction_results collection.

Retrieval patterns (per Persistence Blueprint §C.5 and API Contract Blueprint):
  - Insert a new prediction (idempotent on natural key)
  - Find latest prediction per sensor for a pipeline  ← GET /results/latest
  - Find paginated history for a pipeline/sensor      ← GET /results/history
  - Find by model_id (admin/audit)
  - Check whether any predictions exist for a pipeline (DATA_UNAVAILABLE gate)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from dss_shared.db.collections import PREDICTIONS
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.prediction import PredictionDocument

log = get_logger(__name__)


class PredictionRepository(BaseRepository):
    """
    Repository for prediction_results.

    Insert-only — prediction documents are never updated.
    Duplicate inserts (same pipeline, sensor_id, input_feature_timestamp)
    are silently ignored.
    """

    COLLECTION_NAME = PREDICTIONS

    async def insert(self, doc: PredictionDocument) -> Optional[str]:
        """
        Insert a prediction document.

        Returns the inserted _id string on success.
        Returns None on duplicate (idempotent).
        """
        try:
            result = await self.col.insert_one(self._to_doc(doc))
            return str(result.inserted_id)
        except DuplicateKeyError:
            log.debug(
                "prediction_duplicate_ignored",
                pipeline=doc.pipeline,
                sensor_id=doc.sensor_id,
                input_feature_timestamp=doc.input_feature_timestamp.isoformat(),
            )
            return None

    async def find_latest(
        self,
        pipeline: str,
        sensor_id: Optional[str] = None,
    ) -> list[PredictionDocument]:
        """
        Return the most recent prediction for each sensor in a pipeline.

        If sensor_id is provided, returns at most one document.
        If sensor_id is omitted, returns the latest prediction per unique
        sensor_id in the pipeline.

        Used by GET /results/latest.

        Implementation note: MongoDB does not natively support "latest per group"
        in a single query without $group.  We use a sort+group approach via
        aggregation pipeline for the multi-sensor case.
        """
        if sensor_id:
            raw = await self.col.find_one(
                {"pipeline": pipeline, "sensor_id": sensor_id, "superseded": False},
                sort=[("input_feature_timestamp", pymongo.DESCENDING)],
            )
            if raw is None:
                return []
            return [self._from_doc(raw, PredictionDocument)]

        # Multi-sensor: aggregate to get latest per sensor_id
        pipeline_stages = [
            {"$match": {"pipeline": pipeline, "superseded": False}},
            {"$sort": {"input_feature_timestamp": pymongo.DESCENDING}},
            {
                "$group": {
                    "_id": "$sensor_id",
                    "doc": {"$first": "$$ROOT"},
                }
            },
            {"$replaceRoot": {"newRoot": "$doc"}},
            {"$sort": {"sensor_id": pymongo.ASCENDING}},
        ]
        cursor = self.col.aggregate(pipeline_stages)
        return [self._from_doc(raw, PredictionDocument) async for raw in cursor]

    async def find_history(
        self,
        pipeline: str,
        sensor_id: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[PredictionDocument], int]:
        """
        Return paginated historical predictions.

        Time filters operate on prediction_generated_at (API Contract §C.3 note).
        Results are sorted by prediction_generated_at ascending (oldest first).

        Returns:
            (results_page, total_count)

        total_count is the total matching records across all pages.
        Used by GET /results/history.
        """
        query: dict = {"pipeline": pipeline, "superseded": False}
        if sensor_id:
            query["sensor_id"] = sensor_id
        if from_time or to_time:
            time_filter: dict = {}
            if from_time:
                time_filter["$gte"] = from_time
            if to_time:
                time_filter["$lte"] = to_time
            query["prediction_generated_at"] = time_filter

        total = await self.col.count_documents(query)
        skip = (page - 1) * page_size
        cursor = self.col.find(
            query,
            sort=[("prediction_generated_at", pymongo.ASCENDING)],
            skip=skip,
            limit=page_size,
        )
        results = [self._from_doc(raw, PredictionDocument) async for raw in cursor]
        return results, total

    async def has_any(self, pipeline: str) -> bool:
        """
        Return True if any non-superseded predictions exist for the pipeline.

        Used to determine whether to return DATA_UNAVAILABLE (404) from the
        API.  Uses limit=1 for efficiency.
        """
        count = await self.col.count_documents(
            {"pipeline": pipeline, "superseded": False}, limit=1
        )
        return count > 0

    async def find_by_model_id(
        self, model_id: str, limit: int = 100
    ) -> list[PredictionDocument]:
        """Return predictions produced by a specific model.  Admin/audit use."""
        cursor = self.col.find(
            {"model_id": model_id},
            sort=[("prediction_generated_at", pymongo.DESCENDING)],
            limit=limit,
        )
        return [self._from_doc(raw, PredictionDocument) async for raw in cursor]
