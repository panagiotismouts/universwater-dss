"""
WINGS OAuth2 token manager.

Handles client_credentials flow against the WINGS SSO endpoint.
Tokens are cached in-process and refreshed automatically when they
expire (or within a 60-second safety margin before expiry).

Usage:
    token = await WingsTokenManager.get_token()
    headers = {"Authorization": f"Bearer {token}"}
"""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar

import httpx

from dss_shared.config import get_settings
from dss_shared.exceptions import SourceAPIError
from dss_shared.logging import get_logger

log = get_logger(__name__)

_EXPIRY_MARGIN_SECONDS = 60  # refresh this many seconds before actual expiry


class WingsTokenManager:
    """
    Singleton-style async OAuth2 token manager for the WINGS SSO.

    All callers share the same cached token; a lock prevents thundering-herd
    on simultaneous refresh attempts.
    """

    _token: ClassVar[str | None] = None
    _expires_at: ClassVar[float] = 0.0          # epoch seconds
    _lock: ClassVar[asyncio.Lock | None] = None  # created lazily per event loop

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def get_token(cls) -> str:
        """Return a valid Bearer token, fetching a new one if needed."""
        if cls._token and time.time() < cls._expires_at - _EXPIRY_MARGIN_SECONDS:
            return cls._token

        async with cls._get_lock():
            # Re-check inside lock to avoid duplicate refreshes
            if cls._token and time.time() < cls._expires_at - _EXPIRY_MARGIN_SECONDS:
                return cls._token
            await cls._refresh()
            return cls._token  # type: ignore[return-value]

    @classmethod
    async def _refresh(cls) -> None:
        settings = get_settings()
        log.debug("wings_auth_token_refresh_start")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    settings.wings_sso_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": settings.wings_client_id,
                        "client_secret": settings.wings_client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            if response.status_code != 200:
                raise SourceAPIError(
                    f"WINGS SSO returned {response.status_code}: {response.text[:200]}"
                )
            body = response.json()
            cls._token = body["access_token"]
            expires_in = int(body.get("expires_in", 300))
            cls._expires_at = time.time() + expires_in
            log.debug("wings_auth_token_refreshed", expires_in=expires_in)
        except SourceAPIError:
            raise
        except Exception as exc:
            raise SourceAPIError(f"WINGS SSO token fetch failed: {exc}") from exc

    @classmethod
    def invalidate(cls) -> None:
        """Force the next call to get_token() to fetch a fresh token."""
        cls._token = None
        cls._expires_at = 0.0
