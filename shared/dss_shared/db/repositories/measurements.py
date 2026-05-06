"""
Repository for the preprocessed_measurements collection.

Retrieval patterns supported (per Persistence Blueprint §C.1):
  - Insert single measurement (idempotent — duplicate key → no-op)
  - Insert batch of measurements
  - Forward-fill lookup: most recent processed reading for (sensor_id, variable_name)
  - Training data load: range scan by (pipeline, measured_at) window
  - Feature engineering window: range scan by (pipeline, sensor_id, variable_name, time range)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from dss_shared.db.collections import MEASUREMENTS
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.measurement import MeasurementDocument

log = get_logger(__name__)


class MeasurementRepository(BaseRepository):
    """
    Repository for preprocessed_measurements.

    Documents are immutable after creation.  Duplicate inserts (same natural
    key) are silently ignored — this is correct behaviour for the checkpoint-
    based ingestion model where a job may re-process readings after a partial
    failure.
    """

    COLLECTION_NAME = MEASUREMENTS

    async def insert_one(self, doc: MeasurementDocument) -> Optional[str]:
        """
        Insert a single MeasurementDocument.

        Returns the inserted _id as a string on success.
        Returns None if the document already exists (DuplicateKeyError on
        the unique compound index — treated as idempotent no-op).
        """
        try:
            result = await self.col.insert_one(self._to_doc(doc))
            return str(result.inserted_id)
        except DuplicateKeyError:
            log.debug(
                "measurement_duplicate_ignored",
                sensor_id=doc.sensor_id,
                variable=doc.variable_name,
                measured_at=doc.measured_at.isoformat(),
            )
            return None

    async def insert_many(
        self, docs: list[MeasurementDocument]
    ) -> tuple[int, int]:
        """
        Insert a batch of MeasurementDocuments.

        Uses ordered=False so that duplicate-key errors on individual
        documents do not abort the batch.

        Returns:
            (inserted_count, duplicate_count)
        """
        if not docs:
            return 0, 0

        raw_docs = [self._to_doc(d) for d in docs]
        try:
            result = await self.col.insert_many(raw_docs, ordered=False)
            return len(result.inserted_ids), 0
        except pymongo.errors.BulkWriteError as exc:
            inserted = exc.details.get("nInserted", 0)
            duplicates = sum(
                1 for err in exc.details.get("writeErrors", [])
                if err.get("code") == 11000  # DuplicateKey
            )
            non_duplicate_errors = [
                err for err in exc.details.get("writeErrors", [])
                if err.get("code") != 11000
            ]
            if non_duplicate_errors:
                log.error(
                    "measurement_batch_write_errors",
                    error_count=len(non_duplicate_errors),
                    errors=non_duplicate_errors[:3],  # log first 3 only
                )
                raise
            log.debug(
                "measurement_batch_inserted",
                inserted=inserted,
                duplicates=duplicates,
            )
            return inserted, duplicates

    async def find_last_known_value(
        self,
        sensor_id: str,
        variable_name: str,
        before: datetime,
        pipeline: str,
    ) -> Optional[MeasurementDocument]:
        """
        Return the most recent successfully processed measurement for
        (sensor_id, variable_name) with measured_at < before.

        Used by the forward-fill stage of the preprocessing pipeline.
        Returns None if no prior value exists.
        """
        raw = await self.col.find_one(
            {
                "sensor_id": sensor_id,
                "variable_name": variable_name,
                "pipeline": pipeline,
                "measured_at": {"$lt": before},
                "processing_status": "processed",
                "filled": False,     # prefer original readings as fill source
                "superseded": False,
            },
            sort=[("measured_at", pymongo.DESCENDING)],
        )
        if raw is None:
            return None
        return self._from_doc(raw, MeasurementDocument)

    async def find_window(
        self,
        pipeline: str,
        sensor_id: str,
        variable_name: str,
        start: datetime,
        end: datetime,
    ) -> list[MeasurementDocument]:
        """
        Return all processed measurements for a (pipeline, sensor, variable)
        within [start, end], sorted ascending by measured_at.

        Used by feature engineering to build rolling-window features.
        """
        cursor = self.col.find(
            {
                "pipeline": pipeline,
                "sensor_id": sensor_id,
                "variable_name": variable_name,
                "measured_at": {"$gte": start, "$lte": end},
                "processing_status": "processed",
                "superseded": False,
            },
            sort=[("measured_at", pymongo.ASCENDING)],
        )
        return [self._from_doc(raw, MeasurementDocument) async for raw in cursor]

    async def find_training_window(
        self,
        pipeline: str,
        start: datetime,
        end: datetime,
    ) -> list[MeasurementDocument]:
        """
        Return all processed measurements for a pipeline within [start, end].
        Used by the dataset builder to gather training data across all variables.

        Results are sorted by (sensor_id, variable_name, measured_at) for
        deterministic dataset construction.
        """
        cursor = self.col.find(
            {
                "pipeline": pipeline,
                "measured_at": {"$gte": start, "$lte": end},
                "processing_status": "processed",
                "superseded": False,
            },
            sort=[
                ("sensor_id", pymongo.ASCENDING),
                ("variable_name", pymongo.ASCENDING),
                ("measured_at", pymongo.ASCENDING),
            ],
        )
        return [self._from_doc(raw, MeasurementDocument) async for raw in cursor]
