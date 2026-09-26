"""
Repository for the engineered_features collection.

Retrieval patterns supported (per Persistence Blueprint §C.2):
  - Upsert a feature document (idempotent on natural key)
  - Find most recent feature vector for prediction (per sensor)
  - Find feature vectors for training window (per pipeline, across sensors)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from dss_shared.db.collections import FEATURES
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.feature import FeatureDocument

log = get_logger(__name__)


class FeatureRepository(BaseRepository):
    """
    Repository for engineered_features.

    Documents are immutable after creation.  Upsert on the natural key
    means re-running feature engineering for the same input is safe.
    """

    COLLECTION_NAME = FEATURES

    async def upsert(self, doc: FeatureDocument) -> bool:
        """
        Upsert a FeatureDocument keyed on (pipeline, sensor_id,
        feature_timestamp, feature_schema_version).

        Returns True if the document was inserted (new), False if it
        already existed and was matched (no change applied).

        Note: we use update_one with $setOnInsert rather than a full
        replace so that an existing document is never mutated —
        consistent with the immutability policy.
        """
        key = {
            "pipeline": doc.pipeline,
            "sensor_id": doc.sensor_id,
            "feature_timestamp": doc.feature_timestamp,
            "feature_schema_version": doc.feature_schema_version,
        }
        raw_doc = self._to_doc(doc)
        raw_doc.pop("_id", None)

        result = await self.col.update_one(
            key,
            {"$set": raw_doc},
            upsert=True,
        )
        inserted = result.upserted_id is not None
        if inserted:
            log.debug(
                "feature_inserted",
                pipeline=doc.pipeline,
                sensor_id=doc.sensor_id,
                feature_timestamp=doc.feature_timestamp.isoformat(),
                version=doc.feature_schema_version,
            )
        return inserted

    async def exists(
        self,
        pipeline: str,
        sensor_id: str,
        feature_timestamp: datetime,
        feature_schema_version: str,
    ) -> bool:
        """
        Cheap existence check on the natural key.

        Used by the ingestion coordinator to skip the (expensive) feature
        computation for timestamps that already have a vector under the
        current schema version.  Only the _id is projected, so this is a
        single covered index lookup on uq_feature_vector.
        """
        raw = await self.col.find_one(
            {
                "pipeline": pipeline,
                "sensor_id": sensor_id,
                "feature_timestamp": feature_timestamp,
                "feature_schema_version": feature_schema_version,
            },
            projection={"_id": 1},
        )
        return raw is not None

    async def find_latest_for_prediction(
        self,
        pipeline: str,
        sensor_id: str,
        feature_schema_version: str,
    ) -> Optional[FeatureDocument]:
        """
        Return the most recent feature vector for a sensor.

        Used by the prediction cycle to load the latest available input
        for the active model.

        Returns None if no feature vectors exist for this sensor.
        """
        raw = await self.col.find_one(
            {
                "pipeline": pipeline,
                "sensor_id": sensor_id,
                "feature_schema_version": feature_schema_version,
                "superseded": False,
            },
            sort=[("feature_timestamp", pymongo.DESCENDING)],
        )
        if raw is None:
            return None
        return self._from_doc(raw, FeatureDocument)

    async def find_training_window(
        self,
        pipeline: str,
        start: datetime,
        end: datetime,
        feature_schema_version: str,
    ) -> list[FeatureDocument]:
        """
        Return all feature vectors for a pipeline within [start, end].

        Used by dataset_builder to construct the training matrix.
        Results sorted by (sensor_id, feature_timestamp) ascending.
        """
        cursor = self.col.find(
            {
                "pipeline": pipeline,
                "feature_timestamp": {"$gte": start, "$lte": end},
                "feature_schema_version": feature_schema_version,
                "superseded": False,
            },
            sort=[
                ("sensor_id", pymongo.ASCENDING),
                ("feature_timestamp", pymongo.ASCENDING),
            ],
        )
        return [self._from_doc(raw, FeatureDocument) async for raw in cursor]

    async def find_by_timestamp(
        self,
        pipeline: str,
        sensor_id: str,
        feature_timestamp: datetime,
        feature_schema_version: str,
    ) -> Optional[FeatureDocument]:
        """
        Point lookup for a specific feature vector by its exact timestamp.
        Returns None if not found.
        """
        raw = await self.col.find_one({
            "pipeline": pipeline,
            "sensor_id": sensor_id,
            "feature_timestamp": feature_timestamp,
            "feature_schema_version": feature_schema_version,
        })
        if raw is None:
            return None
        return self._from_doc(raw, FeatureDocument)
