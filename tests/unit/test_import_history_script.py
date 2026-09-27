"""Unit tests for scripts/import_history.py (parsing and reading construction; no Mongo)."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "import_history", Path(__file__).resolve().parents[2] / "scripts" / "import_history.py"
)
ih = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ih)

_HEADER = "id,date,temperature,conductivity,ph,doMgPerL,orp,site_id,site_name\n"


def _csv(tmp_path, lines):
    p = tmp_path / "hcmr.csv"
    p.write_text(_HEADER + "".join(lines))
    return p


def test_hcmr_rows_maps_columns_and_filters_range(tmp_path):
    p = _csv(tmp_path, [
        "1,2025-06-02T15:00:00Z,21.4,281.2,8.85,10.8,125.9,15045987545,Mikri_Prespa\n",
        "2,2026-08-08T00:00:00Z,25.0,300.0,7.70,0.1,-400.0,15045987545,Mikri_Prespa\n",  # at --until: excluded
        "3,2025-06-02T16:00:00Z,21.0,,8.80,10.5,120.0,15045987545,Mikri_Prespa\n",       # missing conductivity
        "4,2025-06-02T17:00:00Z,21.0,280.0,8.80,10.5,120.0,99999,Other\n",               # other site
    ])
    until = datetime(2026, 8, 8, tzinfo=timezone.utc)
    triples, stats = ih.hcmr_rows(p, None, until)
    assert stats["skipped_out_of_range"] == 1
    assert stats["skipped_other_site"] == 1
    assert stats["skipped_missing_value"] == 1
    t0 = datetime(2025, 6, 2, 15, tzinfo=timezone.utc)
    first = {var: val for ts, var, val in triples if ts == t0}
    assert first == {
        "temperature_water": 21.4, "conductivity": 281.2, "ph": 8.85,
        "dissolved_oxygen": 10.8, "orp": 125.9,
    }
    assert len(triples) == 5 + 4


def test_hcmr_readings_use_registry_units_and_wings_source():
    ts = datetime(2025, 6, 2, 15, tzinfo=timezone.utc)
    rs = ih.hcmr_readings([(ts, "ph", 8.85), (ts, "dissolved_oxygen", 10.8)], ts)
    assert {r.unit for r in rs} == {"pH units", "mg/L"}
    assert all(r.sensor_id == "hcmr" and ih._plain(r.pipeline) == "water" and ih._plain(r.source) == "wings" for r in rs)


def test_met_readings_writes_both_met_pipelines():
    payload = {"142": [{"timestamp": 1748867400, "value": 25.0}], "999": [{"timestamp": 1, "value": 0.0}]}
    rs, stats = ih.met_readings(payload, datetime.now(tz=timezone.utc))
    assert sorted(ih._plain(r.pipeline) for r in rs) == ["met_soil", "met_water"]
    assert all(r.variable_name == "air_temperature" and r.sensor_id == "142" for r in rs)
    assert rs[0].measured_at == datetime(2025, 6, 2, 12, 30, tzinfo=timezone.utc)
    assert stats["skipped_unknown_node_999"] == 1
