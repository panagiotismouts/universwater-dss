"""
scripts/register_client.py

Register a new API client in the client_tokens collection.
Generates a random client_secret, bcrypt-hashes it, and persists the document.
The plaintext secret is printed once — capture it immediately.

Usage:
    python scripts/register_client.py --client-id my_client [--name "My Client"]

The script reads .env from the repo root for DSS_MONGO_URI / DSS_MONGO_DB_NAME.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from pathlib import Path

# Allow running from repo root without installing the packages
_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "shared"))


async def main(client_id: str, client_name: str) -> None:
    import bcrypt

    from dss_shared.db import get_database, probe_mongo
    from dss_shared.db.repositories.client_tokens import ClientTokenRepository
    from dss_shared.logging import setup_logging
    from dss_shared.schemas.token import ClientTokenDocument

    def hash_secret(plain: str) -> str:
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

    setup_logging(level="INFO", fmt="human", service="register_client")

    db = get_database()
    await probe_mongo(db)

    client_secret = secrets.token_urlsafe(32)
    hashed = hash_secret(client_secret)

    doc = ClientTokenDocument(
        client_id=client_id,
        client_name=client_name,
        hashed_secret=hashed,
    )

    repo = ClientTokenRepository(db)
    await repo.insert_client(doc)

    print(f"\n{'='*60}")
    print(f"Client registered successfully.")
    print(f"  client_id:     {client_id}")
    print(f"  client_secret: {client_secret}")
    print(f"  (secret shown once — store it securely)")
    print(f"{'='*60}\n")
    print("Use POST /auth/token with these credentials to get a Bearer token.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a DSS API client.")
    parser.add_argument("--client-id", required=True, help="Unique client identifier")
    parser.add_argument("--name", default=None, help="Human-readable client name")
    args = parser.parse_args()
    client_name = args.name or args.client_id
    asyncio.run(main(args.client_id, client_name))
