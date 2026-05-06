"""
Variable registry.

Per-variable validation rules: expected type, plausible min/max ranges,
max forward-fill duration (§G.2), and pipeline assignment.

Used by:
  - preprocessing.pipeline — Stage 1 (range validation) and Stage 4 (fill limit)
  - normalizer — unit mapping validation

Variable ownership (§I.4):
  WINGS owns:  water variables (Aquaread ×2 + HCMR: ph, do, temperature_water,
               turbidity, conductivity, orp, do_saturation, tds, salinity,
               ammonia, nitrate, chlorophyll_a, ammonium, blue_green_algae,
               cdom, sigma_t)
               soil variables (soil_moisture, soil_temperature, nitrogen,
               phosphorus, potassium, soil_conductivity)
  UOWM owns:   meteorological variables (air_temperature, rainfall, humidity,
               wind_speed, wind_direction, solar_radiation, infrared_temperature)
  NOTE: Universwater meteo station on WINGS is EXCLUDED (stale data source).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VariableSpec:
    """Validation and preprocessing specification for a single variable."""

    name: str
    pipeline: str          # "water" | "soil" | "met_water" | "met_soil"
    source: str            # "wings" | "uowm"
    expected_unit: str

    # Plausible physical range — values outside this window get quality_flag="suspect"
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Forward-fill limit (§G.2).  A gap > this value is NOT filled.
    max_fill_duration_seconds: int = 3600   # 1 hour default

    # Whether the variable is required for the pipeline's feature vector
    is_required: bool = True


# ── Complete variable registry ─────────────────────────────────────────────────
# Fill durations per blueprint §G.2.
# Range bounds are conservative physical limits, not statistical outlier thresholds.

_VARIABLE_SPECS: dict[str, VariableSpec] = {
    # ── Water quality (WINGS → "water" pipeline) ──────────────────────────────
    "ph": VariableSpec(
        name="ph",
        pipeline="water",
        source="wings",
        expected_unit="pH units",
        min_value=0.0,
        max_value=14.0,
        max_fill_duration_seconds=7200,   # §G.2: 2 hours
    ),
    "dissolved_oxygen": VariableSpec(
        name="dissolved_oxygen",
        pipeline="water",
        source="wings",
        expected_unit="mg/L",
        min_value=0.0,
        max_value=20.0,
        max_fill_duration_seconds=14400,  # §G.2: 4 hours
    ),
    "temperature_water": VariableSpec(
        name="temperature_water",
        pipeline="water",
        source="wings",
        expected_unit="°C",
        min_value=-5.0,
        max_value=40.0,
        max_fill_duration_seconds=7200,
    ),
    "turbidity": VariableSpec(
        name="turbidity",
        pipeline="water",
        source="wings",
        expected_unit="NTU",
        min_value=0.0,
        max_value=1000.0,
        max_fill_duration_seconds=3600,   # §G.2: 1 hour (rapid dynamics)
    ),
    "conductivity": VariableSpec(
        name="conductivity",
        pipeline="water",
        source="wings",
        expected_unit="μS/cm",
        min_value=0.0,
        max_value=10000.0,
        max_fill_duration_seconds=7200,
    ),
    "orp": VariableSpec(
        name="orp",
        pipeline="water",
        source="wings",
        expected_unit="mV",
        min_value=-1000.0,
        max_value=1000.0,
        max_fill_duration_seconds=3600,
    ),
    "do_saturation": VariableSpec(
        name="do_saturation",
        pipeline="water",
        source="wings",
        expected_unit="%",
        min_value=0.0,
        max_value=200.0,
        max_fill_duration_seconds=14400,
    ),
    "tds": VariableSpec(
        name="tds",
        pipeline="water",
        source="wings",
        expected_unit="mg/L",
        min_value=0.0,
        max_value=50000.0,
        max_fill_duration_seconds=7200,
    ),
    "salinity": VariableSpec(
        name="salinity",
        pipeline="water",
        source="wings",
        expected_unit="ppt",
        min_value=0.0,
        max_value=50.0,
        max_fill_duration_seconds=7200,
    ),
    "ammonia": VariableSpec(
        name="ammonia",
        pipeline="water",
        source="wings",
        expected_unit="mg/L",
        min_value=0.0,
        max_value=100.0,
        max_fill_duration_seconds=3600,
        is_required=False,
    ),
    "nitrate": VariableSpec(
        name="nitrate",
        pipeline="water",
        source="wings",
        expected_unit="mg/L",
        min_value=0.0,
        max_value=100.0,
        max_fill_duration_seconds=3600,
        is_required=False,
    ),
    "chlorophyll_a": VariableSpec(
        name="chlorophyll_a",
        pipeline="water",
        source="wings",
        expected_unit="μg/L",
        min_value=0.0,
        max_value=500.0,
        max_fill_duration_seconds=3600,
        is_required=False,
    ),
    "ammonium": VariableSpec(
        name="ammonium",
        pipeline="water",
        source="wings",
        expected_unit="mg/L",
        min_value=0.0,
        max_value=100.0,
        max_fill_duration_seconds=3600,
        is_required=False,
    ),
    "blue_green_algae": VariableSpec(
        name="blue_green_algae",
        pipeline="water",
        source="wings",
        expected_unit="cells/mL",
        min_value=0.0,
        max_value=1_000_000.0,
        max_fill_duration_seconds=3600,
        is_required=False,
    ),
    "cdom": VariableSpec(
        name="cdom",
        pipeline="water",
        source="wings",
        expected_unit="ppb",
        min_value=0.0,
        max_value=1000.0,
        max_fill_duration_seconds=3600,
        is_required=False,
    ),
    "sigma_t": VariableSpec(
        name="sigma_t",
        pipeline="water",
        source="wings",
        expected_unit="kg/m³",
        min_value=-5.0,
        max_value=50.0,
        max_fill_duration_seconds=7200,
        is_required=False,
    ),

    # ── Soil (WINGS → "soil" pipeline) ────────────────────────────────────────
    "soil_moisture": VariableSpec(
        name="soil_moisture",
        pipeline="soil",
        source="wings",
        expected_unit="%",
        min_value=0.0,
        max_value=100.0,
        max_fill_duration_seconds=14400,  # §G.2: 4 hours (slow dynamics)
    ),
    "soil_temperature": VariableSpec(
        name="soil_temperature",
        pipeline="soil",
        source="wings",
        expected_unit="°C",
        min_value=-20.0,
        max_value=60.0,
        max_fill_duration_seconds=14400,
    ),
    "nitrogen": VariableSpec(
        name="nitrogen",
        pipeline="soil",
        source="wings",
        expected_unit="mg/kg",
        min_value=0.0,
        max_value=500.0,
        max_fill_duration_seconds=14400,
        is_required=False,
    ),
    "phosphorus": VariableSpec(
        name="phosphorus",
        pipeline="soil",
        source="wings",
        expected_unit="mg/kg",
        min_value=0.0,
        max_value=500.0,
        max_fill_duration_seconds=14400,
        is_required=False,
    ),
    "potassium": VariableSpec(
        name="potassium",
        pipeline="soil",
        source="wings",
        expected_unit="mg/kg",
        min_value=0.0,
        max_value=2000.0,
        max_fill_duration_seconds=14400,
        is_required=False,
    ),
    "soil_conductivity": VariableSpec(
        name="soil_conductivity",
        pipeline="soil",
        source="wings",
        expected_unit="μS/cm",
        min_value=0.0,
        max_value=10000.0,
        max_fill_duration_seconds=14400,
        is_required=False,
    ),

    # ── Meteorological (UOWM → "met_water" AND "met_soil" pipelines) ──────────
    # Met variables are duplicated across pipelines by the normalizer (§E.3).
    # The registry stores the "met_water" variant as primary; the normalizer
    # also creates a "met_soil" variant for each reading.
    "rainfall": VariableSpec(
        name="rainfall",
        pipeline="met_water",
        source="uowm",
        expected_unit="mm",
        min_value=0.0,
        max_value=200.0,
        max_fill_duration_seconds=1800,   # §G.2: 30 minutes
    ),
    "air_temperature": VariableSpec(
        name="air_temperature",
        pipeline="met_water",
        source="uowm",
        expected_unit="°C",
        min_value=-30.0,
        max_value=55.0,
        max_fill_duration_seconds=7200,   # §G.2: 2 hours
    ),
    "humidity": VariableSpec(
        name="humidity",
        pipeline="met_water",
        source="uowm",
        expected_unit="%",
        min_value=0.0,
        max_value=100.0,
        max_fill_duration_seconds=7200,
    ),
    "wind_speed": VariableSpec(
        name="wind_speed",
        pipeline="met_water",
        source="uowm",
        expected_unit="m/s",
        min_value=0.0,
        max_value=60.0,
        max_fill_duration_seconds=7200,
        is_required=False,
    ),
    "wind_direction": VariableSpec(
        name="wind_direction",
        pipeline="met_water",
        source="uowm",
        expected_unit="°",
        min_value=0.0,
        max_value=360.0,
        max_fill_duration_seconds=7200,
        is_required=False,
    ),
    "solar_radiation": VariableSpec(
        name="solar_radiation",
        pipeline="met_water",
        source="uowm",
        expected_unit="W/m²",
        min_value=0.0,
        max_value=1500.0,
        max_fill_duration_seconds=3600,
        is_required=False,
    ),
    "infrared_temperature": VariableSpec(
        name="infrared_temperature",
        pipeline="met_water",
        source="uowm",
        expected_unit="°C",
        min_value=-30.0,
        max_value=80.0,
        max_fill_duration_seconds=7200,
        is_required=False,
    ),
}


def get_variable_spec(variable_name: str) -> Optional[VariableSpec]:
    """Return the VariableSpec for variable_name, or None if not registered."""
    return _VARIABLE_SPECS.get(variable_name)


def get_all_variable_names() -> list[str]:
    """Return all registered canonical variable names."""
    return list(_VARIABLE_SPECS.keys())


def is_registered(variable_name: str) -> bool:
    """Return True if variable_name is a known canonical variable."""
    return variable_name in _VARIABLE_SPECS
