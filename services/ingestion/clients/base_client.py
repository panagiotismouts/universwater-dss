"""
BaseAPIClient abstract interface.

All source API clients (WINGS, UOWM) must implement this interface.
The scheduler and job functions interact with clients only through
this contract, making it safe to swap or add sources.

TODO: Implement concrete clients in wings_water_client.py, wings_soil_client.py,
      uowm_met_client.py.
"""

from __future__ import annotations

import abc
from datetime import datetime

from dss_shared.schemas.measurement import NormalizedReading


class BaseAPIClient(abc.ABC):
    """
    Abstract base class for all external source API clients.

    Each concrete client is responsible for one source system (e.g. WINGS,
    UOWM).  A client may serve multiple variables — the `variable_name`
    parameter on `fetch` selects which data stream to retrieve.
    """

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        """Canonical source identifier string, e.g. "wings" or "uowm"."""

    @abc.abstractmethod
    async def fetch(
        self,
        variable_name: str,
        since: datetime,
    ) -> list[NormalizedReading]:
        """
        Fetch readings for `variable_name` with domain timestamp > `since`.

        Returns:
            A list of NormalizedReading objects.  Empty list if no new data.

        Raises:
            SourceAPIError: on HTTP error or unexpected response shape.
            SourceAPITimeoutError: on request timeout.
        """

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Lightweight connectivity check.

        Returns True if the source API responds to a ping-style request.
        Used by health check scripts, not in the normal job path.
        """
