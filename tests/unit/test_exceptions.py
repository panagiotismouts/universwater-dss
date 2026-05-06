"""
Unit tests for dss_shared.exceptions.

Tests that exception hierarchy is correctly defined.
"""

from __future__ import annotations

from dss_shared.exceptions import (
    DSSException,
    MongoUnavailableError,
    SourceAPIError,
    SourceAPITimeoutError,
    TokenExpiredError,
    AuthenticationError,
)


def test_exception_hierarchy():
    assert issubclass(MongoUnavailableError, DSSException)
    assert issubclass(SourceAPITimeoutError, SourceAPIError)
    assert issubclass(SourceAPIError, DSSException)
    assert issubclass(TokenExpiredError, AuthenticationError)
    assert issubclass(AuthenticationError, DSSException)
