"""
BaseRepository — shared infrastructure for all DSS repository classes.

Provides:
  - database handle storage
  - typed collection accessor
  - shared serialization shortcuts

Every concrete repository inherits from this class and declares its
COLLECTION_NAME class variable.  No business logic lives here.
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pydantic import BaseModel

from dss_shared.db.serialization import coerce_object_ids, doc_to_model, to_document

M = TypeVar("M", bound=BaseModel)


class BaseRepository:
    """
    Base class for all DSS MongoDB repository classes.

    Subclasses must define:
        COLLECTION_NAME: str   — the MongoDB collection name constant from db.collections

    Usage:
        class MeasurementRepository(BaseRepository):
            COLLECTION_NAME = MEASUREMENTS

            async def insert(self, doc: MeasurementDocument) -> str:
                ...
    """

    COLLECTION_NAME: str  # Must be defined on each subclass

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    @property
    def col(self) -> AsyncIOMotorCollection:
        """Return the Motor collection handle for this repository."""
        return self._db[self.COLLECTION_NAME]

    # ------------------------------------------------------------------
    # Serialization helpers — thin wrappers so repositories don't import
    # from db.serialization directly in every method.
    # ------------------------------------------------------------------

    def _to_doc(self, model: BaseModel) -> dict[str, Any]:
        return to_document(model)

    def _from_doc(self, raw: dict[str, Any], model_cls: Type[M]) -> M:
        return doc_to_model(raw, model_cls)
