"""
Repository for the client_tokens collection.

Retrieval patterns (per Persistence Blueprint §C.7 and API Contract §B.4):
  - Insert a new client registration
  - Find client by client_id  ← credential lookup during token issuance
  - Find client by token_hash ← token validation on every authenticated request
  - Update token fields after issuance (hash, expiry, issued_at)
  - Update last_used_at after successful API call
  - Soft-revoke a client (set active=False)
  - Hard-revoke a token (set token_hash=None, token_expiry=past)
  - List all clients (admin)

Security notes:
  - hashed_secret stores bcrypt hash only — plaintext never persisted
  - token_hash stores SHA-256(jti) — raw token never persisted
  - find_by_token_hash is the hot path on every API request; covered by sparse index
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from dss_shared.db.collections import CLIENT_TOKENS
from dss_shared.db.repositories.base import BaseRepository
from dss_shared.exceptions import ClientNotFoundError
from dss_shared.logging import get_logger
from dss_shared.schemas.token import ClientTokenDocument

log = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class ClientTokenRepository(BaseRepository):
    """Repository for client_tokens collection."""

    COLLECTION_NAME = CLIENT_TOKENS

    async def insert_client(self, doc: ClientTokenDocument) -> str:
        """
        Register a new API client.

        Returns the inserted _id string.
        Raises DuplicateKeyError if client_id already exists.
        """
        try:
            result = await self.col.insert_one(self._to_doc(doc))
            log.info("client_registered", client_id=doc.client_id)
            return str(result.inserted_id)
        except DuplicateKeyError:
            raise DuplicateKeyError(
                f"Client already exists: {doc.client_id!r}"
            ) from None

    async def find_by_client_id(
        self, client_id: str
    ) -> Optional[ClientTokenDocument]:
        """
        Find a client by client_id.

        Used during token issuance to validate credentials.
        Returns None if not found.
        """
        raw = await self.col.find_one({"client_id": client_id})
        if raw is None:
            return None
        return self._from_doc(raw, ClientTokenDocument)

    async def find_by_token_hash(
        self, token_hash: str
    ) -> Optional[ClientTokenDocument]:
        """
        Find an active client by the SHA-256 hash of their current JWT jti.

        This is called on every authenticated API request.  The sparse index
        on token_hash makes this fast.  Returns None if hash not found.
        """
        raw = await self.col.find_one({
            "token_hash": token_hash,
            "active": True,
        })
        if raw is None:
            return None
        return self._from_doc(raw, ClientTokenDocument)

    async def update_issued_token(
        self,
        client_id: str,
        token_hash: str,
        token_expiry: datetime,
        token_issued_at: datetime,
    ) -> None:
        """
        Persist a newly issued token's hash and expiry.

        Called by token_service after generating a JWT.  Overwrites any
        previously issued token for this client (single active token per client).

        Raises ClientNotFoundError if client_id does not exist.
        """
        result = await self.col.update_one(
            {"client_id": client_id},
            {"$set": {
                "token_hash": token_hash,
                "token_expiry": token_expiry,
                "token_issued_at": token_issued_at,
            }},
        )
        if result.matched_count == 0:
            raise ClientNotFoundError(f"Client not found: {client_id!r}")

    async def update_last_used(self, client_id: str) -> None:
        """
        Update last_used_at to now.  Called after a successful authenticated call.
        Best-effort — failure is logged but not re-raised.
        """
        try:
            await self.col.update_one(
                {"client_id": client_id},
                {"$set": {"last_used_at": _utc_now()}},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("last_used_update_failed", client_id=client_id, error=str(exc))

    async def revoke_token(self, client_id: str) -> None:
        """
        Hard-revoke the current token for a client.

        Sets token_hash=None and token_expiry to now.  The JWT may still be
        structurally valid but the missing hash means token validation fails.

        Raises ClientNotFoundError if client_id does not exist.
        """
        now = _utc_now()
        result = await self.col.update_one(
            {"client_id": client_id},
            {"$set": {
                "token_hash": None,
                "token_expiry": now,
            }},
        )
        if result.matched_count == 0:
            raise ClientNotFoundError(f"Client not found: {client_id!r}")
        log.info("token_revoked", client_id=client_id)

    async def deactivate_client(self, client_id: str) -> None:
        """
        Soft-revoke a client (active=False).  Prevents new token issuance.

        Raises ClientNotFoundError if client_id does not exist.
        """
        result = await self.col.update_one(
            {"client_id": client_id},
            {"$set": {"active": False, "token_hash": None}},
        )
        if result.matched_count == 0:
            raise ClientNotFoundError(f"Client not found: {client_id!r}")
        log.info("client_deactivated", client_id=client_id)

    async def list_all(self) -> list[ClientTokenDocument]:
        """Return all registered clients sorted by created_at ascending."""
        cursor = self.col.find(
            {},
            sort=[("created_at", pymongo.ASCENDING)],
        )
        return [self._from_doc(raw, ClientTokenDocument) async for raw in cursor]
