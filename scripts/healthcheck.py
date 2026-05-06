"""
scripts/healthcheck.py

Check liveness of all DSS services.

Usage:
    python scripts/healthcheck.py

TODO: Call GET /health on the api_service and probe MongoDB directly.
"""

from __future__ import annotations

import asyncio
import sys

import httpx

DSS_API_URL = "http://localhost:8000"


async def main() -> None:
    print("Checking api_service health...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DSS_API_URL}/health")
            print(f"  api_service: HTTP {resp.status_code} — {resp.json()}")
    except Exception as exc:
        print(f"  api_service: UNREACHABLE — {exc}")

    print("Checking MongoDB...")
    try:
        from dss_shared.db import get_database, probe_mongo
        db = get_database()
        await probe_mongo(db, retry_count=1)
        print("  mongodb: REACHABLE")
    except Exception as exc:
        print(f"  mongodb: UNREACHABLE — {exc}")


if __name__ == "__main__":
    sys.path.insert(0, str(__file__ + "/../../.."))
    asyncio.run(main())
