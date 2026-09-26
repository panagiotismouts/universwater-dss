"""
Unit tests for scripts/force_recalibration.py helpers (pure functions only).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "force_recalibration", Path(__file__).resolve().parents[2] / "scripts" / "force_recalibration.py"
)
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)

_CONFIGS = [SimpleNamespace(pipeline_name=n) for n in ("water_wqi_brown_7d", "water_wqi_brown_14d", "soil")]


def test_select_pipelines_defaults_to_all():
    assert [c.pipeline_name for c in fr.select_pipelines(_CONFIGS, None)] == [
        "water_wqi_brown_7d", "water_wqi_brown_14d", "soil",
    ]
    assert fr.select_pipelines(_CONFIGS, []) == _CONFIGS


def test_select_pipelines_keeps_requested_order():
    chosen = fr.select_pipelines(_CONFIGS, ["soil", "water_wqi_brown_7d"])
    assert [c.pipeline_name for c in chosen] == ["soil", "water_wqi_brown_7d"]


def test_select_pipelines_rejects_unknown_names():
    with pytest.raises(SystemExit, match="unknown pipeline"):
        fr.select_pipelines(_CONFIGS, ["soil", "nope"])


def test_format_candidate_table_marks_gate_and_reason():
    rows = [
        {"pipeline": "soil", "model_type": "xgboost", "status": "active", "r2": 0.91, "mae": 0.5,
         "baseline_mae": None, "n_train": 1200, "passed": True, "reason": None},
        {"pipeline": "soil", "model_type": "svr", "status": "rejected", "r2": -0.2, "mae": 3.0,
         "baseline_mae": None, "n_train": 1200, "passed": False, "reason": "R²=-0.2000 < 0.75"},
    ]
    out = fr.format_candidate_table(rows)
    lines = out.splitlines()
    assert lines[0].startswith("pipeline")
    assert "xgboost" in lines[2] and "pass" in lines[2] and "0.9100" in lines[2]
    assert "svr" in lines[3] and "FAIL" in lines[3] and "R²=-0.2000 < 0.75" in lines[3]


def test_plain_unwraps_enum_values():
    from dss_shared.schemas.enums import ModelStatus, ModelType

    assert fr._plain(ModelType.XGBOOST) == "xgboost"
    assert fr._plain(ModelStatus.ACTIVE) == "active"
    assert fr._plain("soil") == "soil"
