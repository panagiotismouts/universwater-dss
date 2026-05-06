"""
Model registry manager.

Thin facade over ModelRegistryRepository.  All reads and writes to the
model_registry collection go through this module.  Both the training path
(bootstrap, recalibration) and the prediction path use this interface.
"""

from __future__ import annotations

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.repositories.model_registry import ModelRegistryRepository
from dss_shared.logging import get_logger
from dss_shared.schemas.model import ModelRegistryDocument

log = get_logger(__name__)


async def find_active_model(
    db: AsyncIOMotorDatabase,
    pipeline: str,
    model_type: Optional[str] = None,
) -> Optional[ModelRegistryDocument]:
    """
    Return the active model for the given pipeline.

    If model_type is provided, verifies that the active model matches it;
    returns None if there is a type mismatch (should not happen in practice).
    """
    repo = ModelRegistryRepository(db)
    doc = await repo.find_active(pipeline)
    if doc is None:
        return None
    if model_type is not None and doc.model_type != model_type:
        return None
    return doc


async def has_active_model(db: AsyncIOMotorDatabase, pipeline: str) -> bool:
    """Return True if an active model exists for the pipeline."""
    repo = ModelRegistryRepository(db)
    return await repo.has_active_model(pipeline)


async def insert_candidate(
    db: AsyncIOMotorDatabase,
    doc: ModelRegistryDocument,
) -> str:
    """Insert a candidate model document. Returns the inserted _id."""
    repo = ModelRegistryRepository(db)
    return await repo.insert_candidate(doc)


async def activate_model(db: AsyncIOMotorDatabase, model_id: str) -> None:
    """Promote candidate → active, retire the current active model."""
    repo = ModelRegistryRepository(db)
    await repo.activate_model(model_id)


async def reject_model(
    db: AsyncIOMotorDatabase,
    model_id: str,
    reason: str,
) -> None:
    """Mark a candidate model as rejected."""
    repo = ModelRegistryRepository(db)
    await repo.reject_model(model_id, reason)
