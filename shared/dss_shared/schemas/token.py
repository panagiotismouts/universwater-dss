"""
Client token and authentication schemas.

ClientTokenDocument
    Persistence model for the client_tokens collection.
    One document per registered API client.
    Never stores plaintext secrets or tokens.

TokenRequest
    API contract: form-encoded POST /auth/token request body.
    Uses OAuth2 Client Credentials convention.

TokenResponse
    API contract: POST /auth/token success response.

JWTClaims
    Internal model: the decoded payload of a validated JWT.
    Used by the auth dependency — never serialized to an API response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from dss_shared.schemas.base import DSSBaseModel, MongoDocument, utc_now


# ---------------------------------------------------------------------------
# ClientTokenDocument — persistence model for client_tokens
# ---------------------------------------------------------------------------

class ClientTokenDocument(MongoDocument):
    """
    MongoDB document shape for the client_tokens collection.

    One document per API client.  Never stores plaintext secrets or tokens.

    Immutability:
      client_id, client_name, created_at are immutable after registration.
      token_hash, token_expiry, token_issued_at, last_used_at, active are
      updated during normal operation.
      hashed_secret is updated only on explicit secret rotation.

    token_hash stores SHA-256(jti) where jti is the UUID claim in the JWT.
    This allows explicit revocation: setting token_hash=None invalidates the
    current token even if the JWT signature is still valid.

    Timestamp discipline:
      token_expiry       — domain: when the current token expires
      token_issued_at    — domain: when the current token was issued
      last_used_at       — domain: when the client last made a successful call
      created_at         — system: when this client was registered
    """

    # Stable human-readable identifier — immutable after registration
    client_id: str = Field(..., min_length=1)
    client_name: str = Field(..., min_length=1)

    # Credential storage (bcrypt hash of client_secret)
    hashed_secret: str = Field(..., min_length=1)

    # Current token state (null = no active token)
    token_hash: Optional[str] = None       # SHA-256(jti) of issued JWT
    token_expiry: Optional[datetime] = None
    token_issued_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    # Lifecycle
    active: bool = True                    # False = soft revocation

    # Audit
    created_at: datetime = Field(default_factory=utc_now)
    created_by: Optional[str] = None      # Operator note
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# TokenRequest — API request model for POST /auth/token
# ---------------------------------------------------------------------------

class TokenRequest(DSSBaseModel):
    """
    Request body for POST /auth/token.

    Uses OAuth2 Client Credentials convention.  Fields are form-encoded.
    The raw client_secret must never appear in logs.

    FastAPI usage:
        from fastapi import Form
        async def issue_token(
            client_id: str = Form(...),
            client_secret: str = Form(...),
        ):
    This model is used for validation in the service layer, not directly
    as a FastAPI form model.
    """

    client_id: str = Field(..., min_length=1)
    client_secret: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# TokenResponse — API response model for POST /auth/token
# ---------------------------------------------------------------------------

class TokenResponse(DSSBaseModel):
    """
    Response body for POST /auth/token (200 OK).

    access_token — signed JWT string
    token_type   — always "bearer"
    expires_in   — token lifetime in seconds (for client-side expiry tracking)
    """

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., ge=1)


# ---------------------------------------------------------------------------
# JWTClaims — internal model for decoded JWT payload
# ---------------------------------------------------------------------------

class JWTClaims(DSSBaseModel):
    """
    Decoded JWT payload used internally by the auth dependency.

    sub — client_id (the "subject" of the token)
    iat — issued-at (Unix epoch seconds)
    exp — expiry (Unix epoch seconds)
    jti — unique token ID (UUID); SHA-256 hashed for storage as token_hash

    Never serialized to an API response.
    """

    sub: str        # client_id
    iat: int        # issued-at (Unix epoch)
    exp: int        # expiry (Unix epoch)
    jti: str        # unique token identifier (UUID)
