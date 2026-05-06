"""
Repository for the model_metrics collection.

Retrieval patterns (per Persistence Blueprint §C.4):
  - Insert evaluation results for a model
  - Find all metrics for a specific model_id
  - Find metrics by evaluation_type for a model
  - Find most recent validation metrics for the active model
    (used by recalibration to get baseline MAE for threshold comparison)
"""

from __future__ import annotations

from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.collections import MODEL_METRICS
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.schemas.enums import EvaluationType
from dss_shared.schemas.model import ModelMetricsDocument


class ModelMetricsRepository(BaseRepository):
    """Repository for model_metrics collection.  Append-only."""

    COLLECTION_NAME = MODEL_METRICS

    async def insert(self, doc: ModelMetricsDocument) -> str:
        """
        Insert a metrics document.

        Returns the inserted _id as a string.
        Multiple evaluation documents may exist for the same model_id
        (different evaluation_type values).
        """
        result = await self.col.insert_one(self._to_doc(doc))
        return str(result.inserted_id)

    async def find_all_for_model(
        self, model_id: str
    ) -> list[ModelMetricsDocument]:
        """
        Return all metrics documents for a model_id.

        Sorted by evaluated_at ascending (oldest first).
        Used for audit and analysis.
        """
        cursor = self.col.find(
            {"model_id": model_id},
            sort=[("evaluated_at", pymongo.ASCENDING)],
        )
        return [self._from_doc(raw, ModelMetricsDocument) async for raw in cursor]

    async def find_by_evaluation_type(
        self,
        model_id: str,
        evaluation_type: EvaluationType | str,
    ) -> list[ModelMetricsDocument]:
        """
        Return metrics for a model filtered by evaluation_type.

        Example: find all "validation" runs for a model.
        """
        cursor = self.col.find(
            {"model_id": model_id, "evaluation_type": str(evaluation_type)},
            sort=[("evaluated_at", pymongo.ASCENDING)],
        )
        return [self._from_doc(raw, ModelMetricsDocument) async for raw in cursor]

    async def find_latest_validation(
        self, model_id: str
    ) -> Optional[ModelMetricsDocument]:
        """
        Return the most recent validation metrics for a model.

        Used during recalibration to get the baseline active model's MAE
        for relative threshold comparison.

        Returns None if no validation metrics exist for the model.
        """
        raw = await self.col.find_one(
            {
                "model_id": model_id,
                "evaluation_type": EvaluationType.VALIDATION,
            },
            sort=[("evaluated_at", pymongo.DESCENDING)],
        )
        if raw is None:
            return None
        return self._from_doc(raw, ModelMetricsDocument)
