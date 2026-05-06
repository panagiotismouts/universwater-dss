"""
Auth router — POST /auth/token

Issues JWT bearer tokens to registered clients.
Uses OAuth2 Client Credentials convention with form-encoded body.

Flow:
  1. Receive client_id + client_secret as form fields
  2. Delegate to token_service.issue_token (bcrypt verify + JWT create)
  3. Return TokenResponse on success, 401 on bad credentials
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, status

from dss_shared.exceptions import AuthenticationError
from dss_shared.schemas.token import TokenResponse
from services.api_service.dependencies.db import get_db
from services.api_service.services.token_service import issue_token

router = APIRouter()


@router.post("/token", response_model=TokenResponse)
async def post_token(
    client_id: str = Form(..., description="Registered API client ID"),
    client_secret: str = Form(..., description="Client secret (never logged)"),
    db=Depends(get_db),
) -> TokenResponse:
    """Issue a JWT bearer token for a registered client."""
    try:
        return await issue_token(client_id=client_id, client_secret=client_secret, db=db)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "INVALID_CREDENTIALS", "message": str(exc)},
            headers={"WWW-Authenticate": "Bearer"},
        )
