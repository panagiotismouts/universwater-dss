"""
DSS exception hierarchy.

All custom exceptions raised within the DSS extend DSSException.
This makes it straightforward to catch any DSS-level error at a service
boundary without accidentally swallowing unrelated exceptions.

Exception taxonomy:
  DSSException                    — base for all DSS exceptions
    ConfigurationError            — invalid or missing config at startup
    MongoUnavailableError         — MongoDB probe failed at startup or runtime
    MongoWriteError               — unexpected write failure
    ModelNotFoundError            — no active model found for a pipeline
    ModelRejectedError            — trained candidate did not pass metric gate
    RecalibrationRejectedError    — recalibration run completed but was rejected
    SourceAPIError                — external API returned an unexpected response
      SourceAPITimeoutError       — external API request timed out
    IngestionValidationError      — raw API payload failed validation
    PreprocessingError            — preprocessing pipeline stage failure
    FeatureEngineeringError       — feature computation failure
    AuthenticationError           — invalid credentials or token
      TokenExpiredError           — valid token but past expiry
      ClientNotFoundError         — client_id not found in client_tokens
"""

from __future__ import annotations


class DSSException(Exception):
    """Base class for all DSS exceptions."""


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(DSSException):
    """Raised when required configuration is missing or invalid at startup."""


# ── Persistence ───────────────────────────────────────────────────────────────

class MongoUnavailableError(DSSException):
    """Raised when the MongoDB readiness probe fails after all retries."""


class MongoWriteError(DSSException):
    """Raised when a MongoDB write operation fails unexpectedly."""


# ── Model lifecycle ───────────────────────────────────────────────────────────

class ModelNotFoundError(DSSException):
    """Raised when no active model exists for a requested pipeline."""


class ModelRejectedError(DSSException):
    """Raised when a trained model candidate does not pass the metric gate."""


class RecalibrationRejectedError(DSSException):
    """Raised when a recalibration run completes but the candidate is rejected."""


# ── Data ingestion ────────────────────────────────────────────────────────────

class SourceAPIError(DSSException):
    """Raised when an external source API returns an error or unexpected payload."""


class SourceAPITimeoutError(SourceAPIError):
    """Raised when an external source API request times out."""


class IngestionValidationError(DSSException):
    """Raised when a raw API payload fails payload validation checks."""


# ── Preprocessing & feature engineering ──────────────────────────────────────

class PreprocessingError(DSSException):
    """Raised when a preprocessing pipeline stage fails for a reading."""


class FeatureEngineeringError(DSSException):
    """Raised when feature computation fails for a preprocessed measurement."""


# ── Authentication ────────────────────────────────────────────────────────────

class AuthenticationError(DSSException):
    """Raised when a client presents invalid credentials or a bad token."""


class TokenExpiredError(AuthenticationError):
    """Raised when a token is structurally valid but has passed its expiry."""


class ClientNotFoundError(AuthenticationError):
    """Raised when the client_id in a token request is not in client_tokens."""
