"""
FastAPI auth dependency.

Extracts and validates the Bearer token from the Authorization header.
Sets request.state.client_id for use by the delivery logger middleware.

Usage:
    @router.get("/results/latest")
    async def get_latest(client_id: str = Depends(verify_bearer_token)):
        ...
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from dss_shared.exceptions import AuthenticationError, TokenExpiredError
from services.api_service.dependencies.db import get_db
from services.api_service.services.token_service import verify_token

_bearer_scheme = HTTPBearer(auto_error=True)


async def verify_bearer_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db=Depends(get_db),
) -> str:
    """
    Validate the Bearer token.

    Sets request.state.client_id on success (used by delivery logger).
    Returns client_id.
    Raises HTTP 401 on invalid, expired, or revoked token.
    """
    try:
        client_id = await verify_token(credentials.credentials, db)
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "TOKEN_EXPIRED", "message": "Token has expired."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "INVALID_TOKEN", "message": str(exc)},
            headers={"WWW-Authenticate": "Bearer"},
        )

    request.state.client_id = client_id
    return client_id


async def verify_admin_key(request: Request) -> None:
    """
    Validate the X-Admin-Key header against settings.admin_api_key.

    Used by /admin/* endpoints.
    Raises HTTP 401 if key is missing or incorrect.
    """
    from dss_shared.config import get_settings
    settings = get_settings()
    key = request.headers.get("X-Admin-Key", "")
    if not settings.admin_api_key or key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "INVALID_CREDENTIALS", "message": "Invalid admin key."},
        )
