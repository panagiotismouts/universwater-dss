"""
Token service.

Handles credential verification, JWT creation, and token hash persistence.

Auth flow:
  1. Look up client by client_id in client_tokens
  2. bcrypt-verify the submitted client_secret against hashed_secret
  3. Create a signed JWT: sub=client_id, jti=uuid4(), iat=now, exp=now+TTL
  4. SHA-256(jti) → token_hash stored in DB (revocation support)
  5. Return access_token string and expires_in seconds

JWT signing uses settings.jwt_secret_key + settings.jwt_algorithm (HS256).
jose library handles signing/verification.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import bcrypt
from jose import jwt
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_settings
from dss_shared.db.repositories.client_tokens import ClientTokenRepository
from dss_shared.exceptions import AuthenticationError, ClientNotFoundError
from dss_shared.logging import get_logger
from dss_shared.schemas.token import TokenRequest, TokenResponse

log = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def hash_secret(plain: str) -> str:
    """Return bcrypt hash of a plaintext secret.  Used by register_client script."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify_secret(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


async def issue_token(
    client_id: str,
    client_secret: str,
    db: AsyncIOMotorDatabase,
) -> TokenResponse:
    """
    Verify credentials and issue a new JWT bearer token.

    Raises AuthenticationError on bad credentials or inactive client.
    """
    settings = get_settings()
    repo = ClientTokenRepository(db)

    client = await repo.find_by_client_id(client_id)
    if client is None or not client.active:
        log.warning("token_issue_client_not_found", client_id=client_id)
        raise AuthenticationError("Invalid client credentials.")

    if not _verify_secret(client_secret, client.hashed_secret):
        log.warning("token_issue_bad_secret", client_id=client_id)
        raise AuthenticationError("Invalid client credentials.")

    now = _utc_now()
    now_ts = int(now.timestamp())
    jti = str(uuid.uuid4())
    expiry_ts = now_ts + settings.token_expiry_seconds

    claims = {
        "sub": client_id,
        "iat": now_ts,
        "exp": expiry_ts,
        "jti": jti,
    }
    token = jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    token_hash = _sha256(jti)
    expiry_dt = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)

    await repo.update_issued_token(
        client_id=client_id,
        token_hash=token_hash,
        token_expiry=expiry_dt,
        token_issued_at=now,
    )
    log.info("token_issued", client_id=client_id, expires_in=settings.token_expiry_seconds)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.token_expiry_seconds,
    )


async def verify_token(raw_token: str, db: AsyncIOMotorDatabase) -> str:
    """
    Validate a JWT Bearer token.

    Steps:
      1. Decode and verify JWT signature + expiry (jose raises on failure)
      2. Extract jti → SHA-256(jti) → look up token_hash in DB
      3. Update last_used_at (best-effort)

    Returns client_id on success.
    Raises AuthenticationError or TokenExpiredError on failure.
    """
    from dss_shared.exceptions import TokenExpiredError
    from jose import ExpiredSignatureError, JWTError

    settings = get_settings()
    repo = ClientTokenRepository(db)

    try:
        claims = jwt.decode(
            raw_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except ExpiredSignatureError:
        raise TokenExpiredError("Token has expired.")
    except JWTError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc

    jti = claims.get("jti")
    client_id = claims.get("sub")
    if not jti or not client_id:
        raise AuthenticationError("Token missing required claims.")

    token_hash = _sha256(jti)
    client = await repo.find_by_token_hash(token_hash)
    if client is None:
        raise AuthenticationError("Token not recognized or has been revoked.")

    # Best-effort update of last_used_at
    await repo.update_last_used(client_id)

    return client_id
