"""
Repository for the xai_results collection.

Retrieval patterns (per Persistence Blueprint §C.6):
  - Insert XAI result (unique on prediction_id)
  - Find by prediction_id  ← used when assembling full API response
  - Find by model_id       ← XAI audit
  - Find time-range for operational analysis
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from dss_shared.db.collections import XAI_RESULTS
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.xai import XAIResultDocument

log = get_logger(__name__)


class XAIResultRepository(BaseRepository):
    """Repository for xai_results.  Insert-only."""

    COLLECTION_NAME = XAI_RESULTS

    async def insert(self, doc: XAIResultDocument) -> Optional[str]:
        """
        Insert an XAI result document.

        Returns the inserted _id string.
        Returns None on duplicate (prediction_id already has an XAI record).
        The unique index on prediction_id enforces one XAI doc per prediction.
        """
        try:
            result = await self.col.insert_one(self._to_doc(doc))
            return str(result.inserted_id)
        except DuplicateKeyError:
            log.debug(
                "xai_duplicate_ignored",
                prediction_id=doc.prediction_id,
            )
            return None

    async def find_by_prediction_id(
        self, prediction_id: str
    ) -> Optional[XAIResultDocument]:
        """
        Return the XAI result for a given prediction.

        Called when assembling a full API response with full_shap_values.
        Returns None if not found.
        """
        raw = await self.col.find_one({"prediction_id": prediction_id})
        if raw is None:
            return None
        return self._from_doc(raw, XAIResultDocument)

    async def find_by_model_id(
        self, model_id: str, limit: int = 100
    ) -> list[XAIResultDocument]:
        """Return XAI results for a specific model.  Audit use."""
        cursor = self.col.find(
            {"model_id": model_id},
            sort=[("input_feature_timestamp", pymongo.DESCENDING)],
            limit=limit,
        )
        return [self._from_doc(raw, XAIResultDocument) async for raw in cursor]

    async def find_time_range(
        self,
        pipeline: str,
        start: datetime,
        end: datetime,
    ) -> list[XAIResultDocument]:
        """Return XAI results for a pipeline within a time range."""
        cursor = self.col.find(
            {
                "pipeline": pipeline,
                "input_feature_timestamp": {"$gte": start, "$lte": end},
            },
            sort=[("input_feature_timestamp", pymongo.ASCENDING)],
        )
        return [self._from_doc(raw, XAIResultDocument) async for raw in cursor]
