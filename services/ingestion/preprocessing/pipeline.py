"""
Inline 6-stage preprocessing pipeline (Amendment B.6).

Input:  list[NormalizedReading]  (de-duplicated, new-only)
Output: list[MeasurementDocument] (ready for MongoDB insert)

Stages:
  1. Variable name validation  — reject unknown variables
  2. Range validation          — set quality_flag (ok / suspect / no_data
                                 for placeholder values)
  3. Unit validation           — warn on unit mismatch (non-fatal)
  4. Forward-fill              — no-op in v1 (raw_value always float)
  5. Build MeasurementDocument
  6. Domain completeness check

Per-reading failure handling (Amendment B.6):
  If Stage 1 raises PreprocessingError, the reading is written with
  processing_status="failed" and failure_reason set.
  Processing continues for all remaining readings.

Note on forward-fill (Stage 4):
  NormalizedReading.raw_value is float (non-optional), so null source values
  are filtered by the client before reaching this pipeline.
  The forward-fill infrastructure (forward_fill.py) is ready for gap detection
  in a future version.  Stage 4 is a no-op in v1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.exceptions import PreprocessingError
from dss_shared.logging import get_logger
from dss_shared.schemas.enums import ProcessingStatus, QualityFlag
from dss_shared.schemas.measurement import MeasurementDocument, NormalizedReading
from services.ingestion.preprocessing.variable_registry import (
    VariableSpec,
    get_variable_spec,
)

log = get_logger(__name__)


async def run_preprocessing_pipeline(
    readings: list[NormalizedReading],
    db: AsyncIOMotorDatabase,
) -> list[MeasurementDocument]:
    """
    Apply all preprocessing stages to the input readings.

    Returns one MeasurementDocument per input reading.
    Failed readings are included with processing_status="failed".
    """
    results: list[MeasurementDocument] = []
    processed_at = datetime.now(tz=timezone.utc)

    for reading in readings:
        try:
            doc = _process_reading(reading, processed_at)
            results.append(doc)
        except PreprocessingError as exc:
            log.warning(
                "preprocessing_failed",
                sensor_id=reading.sensor_id,
                variable=reading.variable_name,
                measured_at=reading.measured_at.isoformat(),
                reason=str(exc),
            )
            results.append(_make_failure_document(reading, reason=str(exc)))

    return results


def _process_reading(
    reading: NormalizedReading,
    processed_at: datetime,
) -> MeasurementDocument:
    """Apply stages 1–6 to a single reading. Raises PreprocessingError on failure."""
    spec = _stage1_variable_validation(reading)
    quality_flag = _stage2_range_validation(reading, spec)
    _stage3_unit_validation(reading, spec)
    # Stage 4: forward-fill — no-op in v1
    doc = _stage5_build_document(reading, quality_flag, processed_at)
    _stage6_completeness_check(doc)
    return doc


# ── Stage implementations ──────────────────────────────────────────────────────

def _stage1_variable_validation(reading: NormalizedReading) -> VariableSpec:
    """Validate variable_name is a known canonical variable."""
    spec = get_variable_spec(reading.variable_name)
    if spec is None:
        raise PreprocessingError(
            f"Unknown variable '{reading.variable_name}' — not in variable registry."
        )
    return spec


# Values a sensor sends in place of a reading.  new_water_station reports
# whole records of exactly -1 (every variable at once) when it has no data;
# -1 is inside the physical range of ORP, water temperature and sigma_t, so
# the range check alone stores them as valid readings.
_SENSOR_PLACEHOLDERS: dict[str, frozenset[float]] = {
    "new_water_station": frozenset({-1.0}),
}


def placeholder_flag(sensor_id: str, spec: VariableSpec, value: float) -> Optional[QualityFlag]:
    """
    Return the quality flag for a placeholder (non-)reading, or None.

    NO_DATA for the sensor's no-data marker, SUSPECT for an exact 0 of a
    variable that cannot be 0 (spec.zero_is_missing).  Also used by
    scripts/flag_placeholder_readings.py to re-flag stored measurements.
    """
    if value in _SENSOR_PLACEHOLDERS.get(sensor_id, ()):
        return QualityFlag.NO_DATA
    if spec.zero_is_missing and value == 0.0:
        return QualityFlag.SUSPECT
    return None


def _stage2_range_validation(
    reading: NormalizedReading,
    spec: VariableSpec,
) -> QualityFlag:
    """
    Check value against plausible physical range.

    Returns QualityFlag.SUSPECT for out-of-range values, and the
    placeholder_flag for values that are not readings at all.
    Suspect values are still written to MongoDB per blueprint §F.2 Stage 4.
    """
    value = reading.raw_value
    flag = placeholder_flag(reading.sensor_id, spec, value)
    if flag is not None:
        log.warning(
            "value_placeholder",
            sensor_id=reading.sensor_id,
            variable=reading.variable_name,
            value=value,
            flag=flag.value,
        )
        return flag
    out_of_range = (
        (spec.min_value is not None and value < spec.min_value)
        or (spec.max_value is not None and value > spec.max_value)
    )
    if out_of_range:
        log.warning(
            "value_out_of_range",
            sensor_id=reading.sensor_id,
            variable=reading.variable_name,
            value=value,
            min=spec.min_value,
            max=spec.max_value,
        )
        return QualityFlag.SUSPECT
    return QualityFlag.OK


def _stage3_unit_validation(reading: NormalizedReading, spec: VariableSpec) -> None:
    """Warn if unit doesn't match expected canonical unit (non-fatal)."""
    if reading.unit != spec.expected_unit:
        log.warning(
            "unit_mismatch",
            sensor_id=reading.sensor_id,
            variable=reading.variable_name,
            actual_unit=reading.unit,
            expected_unit=spec.expected_unit,
        )


def _stage5_build_document(
    reading: NormalizedReading,
    quality_flag: QualityFlag,
    processed_at: datetime,
) -> MeasurementDocument:
    """Assemble MeasurementDocument from processed fields."""
    return MeasurementDocument(
        pipeline=reading.pipeline,
        source=reading.source,
        sensor_id=reading.sensor_id,
        variable_name=reading.variable_name,
        value=reading.raw_value,
        unit=reading.unit,
        measured_at=reading.measured_at,
        processed_at=processed_at,
        processing_status=ProcessingStatus.PROCESSED,
        quality_flag=quality_flag,
        filled=False,
        fill_metadata=None,
    )


def _stage6_completeness_check(doc: MeasurementDocument) -> None:
    """Final invariant check before persistence (Pydantic catches most issues)."""
    if not doc.sensor_id:
        raise PreprocessingError("sensor_id is empty after processing.")
    if not doc.variable_name:
        raise PreprocessingError("variable_name is empty after processing.")


def _make_failure_document(
    reading: NormalizedReading,
    reason: str,
) -> MeasurementDocument:
    """Build a failed MeasurementDocument when a stage raises PreprocessingError."""
    return MeasurementDocument(
        pipeline=reading.pipeline,
        source=reading.source,
        sensor_id=reading.sensor_id,
        variable_name=reading.variable_name,
        value=reading.raw_value,
        unit=reading.unit,
        measured_at=reading.measured_at,
        processed_at=datetime.now(tz=timezone.utc),
        processing_status=ProcessingStatus.FAILED,
        failure_reason=reason,
        quality_flag=QualityFlag.OK,
        filled=False,
        fill_metadata=None,
    )
