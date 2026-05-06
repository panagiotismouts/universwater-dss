"""FastAPI dependency: inject Motor database handle."""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db import get_database


async def get_db() -> AsyncIOMotorDatabase:
    return get_database()
