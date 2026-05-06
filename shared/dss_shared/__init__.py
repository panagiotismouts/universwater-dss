"""
dss_shared — Shared library for the DSS.

Provides:
- config:      Two-layer config loader (config.yaml + env vars) with Pydantic validation
- db:          MongoDB async client factory and collection-level helpers
- schemas:     Pydantic document schemas for all 9 MongoDB collections
- logging:     Structured JSON logging setup (structlog)
- exceptions:  DSS exception hierarchy
"""

__version__ = "0.1.0"
