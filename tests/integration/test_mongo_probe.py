"""
Integration tests for MongoDB probe.

Requires a running MongoDB instance reachable at DSS_MONGO_URI.
Run with:  pytest tests/integration/ -m integration

TODO: Add tests for probe_mongo success and failure paths.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.anyio
async def test_probe_mongo_success():
    """probe_mongo completes without raising when MongoDB is available."""
    from dss_shared.db import get_database, probe_mongo
    db = get_database()
    await probe_mongo(db, retry_count=1)
