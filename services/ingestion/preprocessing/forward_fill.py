"""
Forward-fill logic (§G, Amendment B.6).

Queries preprocessed_measurements for the most recent "ok" non-filled
reading for a given (sensor_id, variable_name, pipeline) combination and
determines whether forward-fill should be applied to a new reading.

Fill policy (§G.3, §G.4):
  - Only fills from readings with quality_flag="ok" AND filled=False
    (prevents fill-chains and prevents propagating suspect values)
  - If the gap from the fill source to the current reading exceeds
    max_fill_duration_seconds for this variable, no fill is applied
  - If no eligible fill source exists, no fill is applied

Returns:
  (fill_value, fill_source_measured_at, fill_gap_seconds) on success
  None when no fill is applicable
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.repositories.measurements import MeasurementRepository
from dss_shared.logging import get_logger

log = get_logger(__name__)


async def find_fill_source(
    db: AsyncIOMotorDatabase,
    sensor_id: str,
    variable_name: str,
    pipeline: str,
    before: datetime,
    max_fill_duration_seconds: int,
) -> Optional[tuple[float, datetime, int]]:
    """
    Find the most recent eligible fill source for a missing measurement.

    Queries for the most recent processed, non-filled, quality_flag="ok"
    reading for (sensor_id, variable_name, pipeline) with measured_at < before.

    Returns:
        (fill_value, fill_source_measured_at, fill_gap_seconds) if an eligible
        fill source exists within the max gap window.
        None if no eligible source exists or the gap is too large.
    """
    repo = MeasurementRepository(db)
    prior = await repo.find_last_known_value(
        sensor_id=sensor_id,
        variable_name=variable_name,
        before=before,
        pipeline=pipeline,
    )
    if prior is None:
        log.debug(
            "forward_fill_no_prior",
            sensor_id=sensor_id,
            variable=variable_name,
            pipeline=pipeline,
        )
        return None

    # Fill chain prevention (§G.4): find_last_known_value already filters
    # filled=False, but guard defensively.
    if prior.filled:
        log.debug(
            "forward_fill_chain_prevented",
            sensor_id=sensor_id,
            variable=variable_name,
        )
        return None

    gap_seconds = int((before - prior.measured_at).total_seconds())
    if gap_seconds > max_fill_duration_seconds:
        log.debug(
            "forward_fill_gap_too_large",
            sensor_id=sensor_id,
            variable=variable_name,
            gap_seconds=gap_seconds,
            max_seconds=max_fill_duration_seconds,
        )
        return None

    return (prior.value, prior.measured_at, gap_seconds)
