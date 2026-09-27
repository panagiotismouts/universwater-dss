"""
Input standardisation inside the scale-sensitive wrappers, backward-compatible
artifacts, and exact linear SHAP.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from services.ml_engine.models.black_box.svr_model import SVRModel
from services.ml_engine.models.white_box.elastic_net import ElasticNetModel
from services.ml_engine.models.white_box.linear_regression import LinearRegressionModel
from services.ml_engine.xai.shap_linear import LinearExplainerWrapper


@pytest.fixture
def data():
    """Features on wildly different scales, like pH (~9), conductivity (~300), ORP (~400), sin/cos (±1)."""
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([
        rng.normal(9.0, 0.3, n),
        rng.normal(300.0, 40.0, n),
        rng.normal(400.0, 60.0, n),
        np.sin(np.linspace(0, 12, n)),
    ])
    y = 2.0 * X[:, 0] + 0.01 * X[:, 1] - 0.005 * X[:, 2] + 3.0 * X[:, 3] + rng.normal(0, 0.1, n)
    return X, y


class _ScaledSVR(SVRModel):
    _scale_inputs = True


class _ScaledElasticNet(ElasticNetModel):
    _scale_inputs = True


def test_default_wrappers_are_unscaled_like_before(data):
    X, y = data
    for w, ref in [(SVRModel(), SVR()), (ElasticNetModel(), ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000))]:
        w.fit(X, y)
        assert w._scaler is None
        np.testing.assert_allclose(w.predict(X[:20]), ref.fit(X, y).predict(X[:20]), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(w.feature_mean_, X.mean(axis=0))


def test_scaling_flag_matches_scaler_plus_estimator_pipeline(data):
    X, y = data
    for w, ref in [(_ScaledSVR(), make_pipeline(StandardScaler(), SVR())),
                   (_ScaledElasticNet(), make_pipeline(StandardScaler(), ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)))]:
        w.fit(X, y)
        np.testing.assert_allclose(w.predict(X[:20]), ref.fit(X, y).predict(X[:20]), rtol=1e-9, atol=1e-9)


def test_artifact_round_trip_keeps_scaler_and_means(data, tmp_path):
    X, y = data
    w = _ScaledSVR(); w.fit(X, y)
    path = tmp_path / "m.joblib"
    w.save(path)
    w2 = _ScaledSVR(); w2.load(path)
    assert w2._scaler is not None
    np.testing.assert_allclose(w2.predict(X[:10]), w.predict(X[:10]))
    np.testing.assert_allclose(w2.feature_mean_, X.mean(axis=0))


def test_legacy_bare_estimator_artifact_still_loads_unscaled(data, tmp_path):
    X, y = data
    legacy = SVR().fit(X, y)                     # how artifacts were written before
    path = tmp_path / "legacy.joblib"
    joblib.dump(legacy, path)
    w = SVRModel(); w.load(path)
    assert w._scaler is None and w.feature_mean_ is None
    np.testing.assert_allclose(w.predict(X[:10]), legacy.predict(X[:10]))


@pytest.mark.parametrize("cls", [ElasticNetModel, _ScaledElasticNet])
def test_linear_terms_are_in_raw_units(data, cls):
    X, y = data
    w = cls(); w.fit(X, y)
    coef_raw, intercept_raw = w.linear_terms()
    np.testing.assert_allclose(intercept_raw + X[:10] @ coef_raw, w.predict(X[:10]), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("cls", [ElasticNetModel, _ScaledElasticNet, LinearRegressionModel])
def test_linear_shap_is_exact_and_mean_referenced(data, cls):
    X, y = data
    w = cls(); w.fit(X, y)
    names = ["ph", "cond", "orp", "sin"]
    ex = LinearExplainerWrapper()
    out = ex.explain(w, X[5], names)
    assert out["explainer_type"] == "linear_shap"
    assert out["base_value"] + sum(out["shap_values"].values()) == pytest.approx(float(w.predict(X[5:6])[0]), abs=1e-9)
    at_mean = ex.explain(w, X.mean(axis=0), names)
    assert all(abs(v) < 1e-9 for v in at_mean["shap_values"].values())


def test_linear_shap_refuses_legacy_artifacts(data, tmp_path):
    X, y = data
    path = tmp_path / "legacy.joblib"
    joblib.dump(ElasticNet().fit(X, y), path)
    w = ElasticNetModel(); w.load(path)
    with pytest.raises(ValueError, match="training feature means"):
        LinearExplainerWrapper().explain(w, X[0], ["a", "b", "c", "d"])
