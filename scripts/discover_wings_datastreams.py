#!/usr/bin/env python3
"""
WINGS datastream discovery utility.

Queries the WINGS SensorThings API and prints all available datastreams
with their IDs, names, and associated sensor information.

Usage:
    # From the dss/ directory with .env loaded:
    python scripts/discover_wings_datastreams.py

    # Or with explicit credentials:
    DSS_WINGS_CLIENT_ID=universWater \
    DSS_WINGS_CLIENT_SECRET=<secret> \
    python scripts/discover_wings_datastreams.py

Output:
    Prints a table of datastream_id, name, and unit_of_measurement for
    all datastreams accessible to the configured client.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the packages
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "shared"))
sys.path.insert(0, str(_repo_root / "services" / "ingestion"))

import httpx


async def get_token(sso_url: str, client_id: str, client_secret: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            sso_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def list_datastreams(base_url: str, token: str) -> list[dict]:
    url = f"{base_url.rstrip('/')}/api/v1/collections/sensor_things:datastreams"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    all_items: list[dict] = []
    offset = 0
    limit = 200

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            resp = await client.get(
                url,
                params={"version": "v2", "limit": str(limit), "offset": str(offset)},
                headers=headers,
            )
            resp.raise_for_status()
            items: list[dict] = resp.json()
            all_items.extend(items)
            if len(items) < limit:
                break
            offset += limit

    return all_items


async def main() -> None:
    sso_url    = os.environ.get("DSS_WINGS_SSO_URL",
        "https://serv-sso.staging.wi-sense-water.eu/auth/realms/wi-sense-water/protocol/openid-connect/token")
    base_url   = os.environ.get("DSS_WINGS_API_BASE_URL",
        "https://serv-api.staging.wi-sense-water.eu")
    client_id  = os.environ.get("DSS_WINGS_CLIENT_ID", "")
    client_secret = os.environ.get("DSS_WINGS_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        print("ERROR: DSS_WINGS_CLIENT_ID and DSS_WINGS_CLIENT_SECRET must be set.", file=sys.stderr)
        sys.exit(1)

    print(f"Authenticating to {sso_url} ...")
    token = await get_token(sso_url, client_id, client_secret)
    print("Token acquired.\n")

    print(f"Listing datastreams from {base_url} ...")
    datastreams = await list_datastreams(base_url, token)
    print(f"Found {len(datastreams)} datastreams.\n")

    # Print header
    col_id   = 55
    col_name = 45
    col_unit = 20
    print(f"{'datastream_id':<{col_id}}  {'name':<{col_name}}  {'unit':<{col_unit}}")
    print("-" * (col_id + col_name + col_unit + 4))

    for ds in sorted(datastreams, key=lambda d: str(d.get("id", ""))):
        ds_id   = str(ds.get("id", ""))
        name    = str(ds.get("name", ""))
        unit    = str(ds.get("unit_of_measurement", {}).get("symbol", "") if isinstance(ds.get("unit_of_measurement"), dict) else ds.get("unit_of_measurement", ""))
        print(f"{ds_id:<{col_id}}  {name:<{col_name}}  {unit:<{col_unit}}")

    print(f"\nTotal: {len(datastreams)} datastreams")
    print("\nRaw first entry (for field inspection):")
    if datastreams:
        print(json.dumps(datastreams[0], indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
