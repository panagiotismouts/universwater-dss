"""
Pytest configuration and shared fixtures.

Fixtures provided here are available to all tests in unit/ and integration/.

TODO: Add fixtures for:
  - in-memory / mongomock database for unit tests
  - real MongoDB connection for integration tests (uses DSS_MONGO_URI from env)
  - settings override (monkeypatch DSS_* env vars before calling get_settings)
  - sample NormalizedReading, MeasurementDocument, FeatureDocument factories
"""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


# TODO: add shared fixtures
