"""
Repository for the model_registry collection.

This repository is the ONLY place in the codebase that reads or writes
model_registry documents.  Both the training path (bootstrap, recalibration)
and the prediction path (loading the active model) use this repository.

Status transitions managed here:
  candidate → active    (activate_model)
  active    → retired   (retire_model, called inside activate_model)
  candidate → rejected  (reject_model)

The partial unique index on {pipeline} where {status: "active"} enforces
at most one active model per pipeline at the database level.
activate_model retires the current active model atomically before promoting
the candidate to avoid violating this constraint.

Retrieval patterns (per Persistence Blueprint §C.3):
  - Insert new candidate
  - Find active model for a pipeline  ← hottest query
  - Activate a candidate model
  - Reject a candidate model
  - List models for admin view
  - Find most recent retired model (rollback lookup)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.collections import MODEL_REGISTRY
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.exceptions import ModelNotFoundError
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import ModelStatus, Pipeline
from dss_shared.schemas.model import ModelRegistryDocument

log = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class ModelRegistryRepository(BaseRepository):
    """Repository for model_registry collection."""

    COLLECTION_NAME = MODEL_REGISTRY

    async def insert_candidate(self, doc: ModelRegistryDocument) -> str:
        """
        Insert a new model candidate document.

        The document must have status=CANDIDATE.  Raises ValueError if
        status is anything else — candidates are always inserted first,
        then transitioned via activate/reject.

        Returns the inserted _id as a string.
        """
        if doc.status != ModelStatus.CANDIDATE:
            raise ValueError(
                f"insert_candidate requires status=CANDIDATE, got: {doc.status!r}"
            )
        result = await self.col.insert_one(self._to_doc(doc))
        log.info(
            "model_candidate_inserted",
            model_id=doc.model_id,
            pipeline=doc.pipeline,
        )
        return str(result.inserted_id)

    async def find_active(
        self,
        pipeline: Pipeline | str,
    ) -> Optional[ModelRegistryDocument]:
        """
        Return the currently active model for a pipeline.

        This is the hottest query in the system — runs on every prediction
        cycle.  Covered by the (pipeline, status) compound index.

        Returns None if no active model exists (bootstrap not yet complete).
        """
        raw = await self.col.find_one({
            "pipeline": str(pipeline),
            "status": ModelStatus.ACTIVE,
        })
        if raw is None:
            return None
        return self._from_doc(raw, ModelRegistryDocument)

    async def find_by_model_id(self, model_id: str) -> Optional[ModelRegistryDocument]:
        """Return a registry document by its human-readable model_id."""
        raw = await self.col.find_one({"model_id": model_id})
        if raw is None:
            return None
        return self._from_doc(raw, ModelRegistryDocument)

    async def activate_model(self, model_id: str) -> None:
        """
        Promote a candidate model to active status.

        Steps (executed sequentially — not in a transaction for simplicity):
          1. Find the current active model for the same pipeline.
          2. Retire the current active model (set status=retired, retired_at=now).
          3. Promote the candidate to active (set status=active, activated_at=now).

        The partial unique index prevents two active models per pipeline.
        If step 3 fails after step 2, the pipeline has no active model
        temporarily — the prediction cycle will log ModelNotFoundError and
        skip until manual intervention restores state.  This is an acceptable
        edge case for a single-developer system.

        Raises:
            ModelNotFoundError if model_id is not found or is not a candidate.
        """
        candidate = await self.find_by_model_id(model_id)
        if candidate is None:
            raise ModelNotFoundError(f"Model not found: {model_id!r}")
        if candidate.status != ModelStatus.CANDIDATE:
            raise ValueError(
                f"Cannot activate model {model_id!r}: "
                f"current status is {candidate.status!r}, expected 'candidate'."
            )

        now = _utc_now()
        pipeline = candidate.pipeline

        # Step 1+2: retire current active model if one exists
        await self.col.update_one(
            {"pipeline": pipeline, "status": ModelStatus.ACTIVE},
            {"$set": {"status": ModelStatus.RETIRED, "retired_at": now}},
        )

        # Step 3: promote candidate to active
        result = await self.col.update_one(
            {"model_id": model_id, "status": ModelStatus.CANDIDATE},
            {"$set": {"status": ModelStatus.ACTIVE, "activated_at": now}},
        )
        if result.modified_count == 0:
            raise ModelNotFoundError(
                f"Failed to activate model {model_id!r}: document not modified."
            )
        log.info("model_activated", model_id=model_id, pipeline=pipeline)

    async def reject_model(self, model_id: str, reason: str) -> None:
        """
        Mark a candidate model as rejected.

        Raises:
            ModelNotFoundError if model_id is not found.
        """
        now = _utc_now()
        result = await self.col.update_one(
            {"model_id": model_id, "status": ModelStatus.CANDIDATE},
            {"$set": {
                "status": ModelStatus.REJECTED,
                "rejection_reason": reason,
                "retired_at": now,   # reuse retired_at as the rejection timestamp
            }},
        )
        if result.matched_count == 0:
            raise ModelNotFoundError(
                f"Model not found or not a candidate: {model_id!r}"
            )
        log.info("model_rejected", model_id=model_id, reason=reason)

    async def find_latest_retired(
        self, pipeline: Pipeline | str
    ) -> Optional[ModelRegistryDocument]:
        """
        Return the most recently retired model for a pipeline.

        Used for rollback inspection — not for prediction.
        """
        raw = await self.col.find_one(
            {"pipeline": str(pipeline), "status": ModelStatus.RETIRED},
            sort=[("retired_at", pymongo.DESCENDING)],
        )
        if raw is None:
            return None
        return self._from_doc(raw, ModelRegistryDocument)

    async def list_all(
        self,
        pipeline: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[ModelRegistryDocument]:
        """
        List model registry entries for admin view.

        Optional filters: pipeline and/or status.
        Sorted by trained_at descending (newest first).
        """
        query: dict = {}
        if pipeline:
            query["pipeline"] = pipeline
        if status:
            query["status"] = status
        cursor = self.col.find(
            query,
            sort=[("trained_at", pymongo.DESCENDING)],
            limit=limit,
        )
        return [self._from_doc(raw, ModelRegistryDocument) async for raw in cursor]

    async def has_active_model(self, pipeline: Pipeline | str) -> bool:
        """Return True if an active model exists for the pipeline."""
        count = await self.col.count_documents({
            "pipeline": str(pipeline),
            "status": ModelStatus.ACTIVE,
        }, limit=1)
        return count > 0
