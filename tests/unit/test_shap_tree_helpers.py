"""Unit tests for services.ml_engine.xai.shap_tree.scalar_expected_value."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load the module file directly: the xai package __init__ pulls in shap,
# which the unit-test environment does not install.
_spec = importlib.util.spec_from_file_location(
    "shap_tree", Path(__file__).resolve().parents[2] / "services/ml_engine/xai/shap_tree.py"
)


def _load():
    import sys, types
    if "shap" not in sys.modules:
        sys.modules["shap"] = types.ModuleType("shap")
    if "services.ml_engine.xai.base" not in sys.modules:
        base = types.ModuleType("services.ml_engine.xai.base")
        class BaseExplainer:  # minimal stand-in
            pass
        base.BaseExplainer = BaseExplainer
        sys.modules["services.ml_engine.xai.base"] = base
    mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mod)
    return mod


def test_scalar_from_python_float():
    assert _load().scalar_expected_value(1.5) == 1.5


def test_scalar_from_numpy_scalar():
    assert _load().scalar_expected_value(np.float64(-0.91)) == pytest.approx(-0.91)


def test_scalar_from_one_element_array_sklearn_forest_style():
    assert _load().scalar_expected_value(np.array([2.25])) == 2.25


def test_scalar_from_list_takes_first_output():
    assert _load().scalar_expected_value([3.0, 4.0]) == 3.0


def test_empty_is_an_error():
    with pytest.raises(ValueError):
        _load().scalar_expected_value(np.array([]))
