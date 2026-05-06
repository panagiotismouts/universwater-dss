"""
Client registry.

Maps source name strings from config to concrete client classes.
This is the single point of change when a new source API is added.

Usage:
    from services.ingestion.clients.registry import get_client
    client = get_client("wings_water")
"""

from __future__ import annotations

from typing import Type

from services.ingestion.clients.base_client import BaseAPIClient
from services.ingestion.clients.wings_water_client import WingsWaterClient
from services.ingestion.clients.wings_soil_client import WingsSoilClient
from services.ingestion.clients.uowm_met_client import UOWMMetClient

_REGISTRY: dict[str, Type[BaseAPIClient]] = {
    "wings_water": WingsWaterClient,
    "wings_soil": WingsSoilClient,
    "uowm_met": UOWMMetClient,
    # Add future clients here: "satellite": SatelliteClient
}


def get_client(source_key: str) -> BaseAPIClient:
    """
    Return an instantiated client for the given source_key.

    Args:
        source_key: Key from config.yaml sources definition, e.g. "wings_water".

    Raises:
        KeyError: If source_key is not registered.
    """
    cls = _REGISTRY[source_key]
    return cls()
