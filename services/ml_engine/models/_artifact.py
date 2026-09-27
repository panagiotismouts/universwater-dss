"""
Shared fit/predict/persist helpers for the scikit-learn wrappers that need
input scaling or training statistics.

Why:
  - SVR (RBF kernel) and penalised linear models (ElasticNet, Ridge) are
    scale-sensitive.  Features here range from sin/cos encodings (±1) and pH
    (~9) to conductivity (~300 µS/cm) and ORP (~400 mV); unscaled, the kernel
    distance and the L1/L2 penalty are dominated by the large-unit features.
    These wrappers standardise inputs inside the wrapper, so every caller
    (trainers, predictor, explainers) keeps passing raw feature rows.
  - Linear SHAP needs the training feature means as its reference point.
    Every wrapper using this helper records them at fit time.

Artifact format:
  format 2  {"format": 2, "model": est, "scaler": StandardScaler|None,
             "feature_mean": ndarray}
  legacy    the bare estimator (artifacts written before this change);
            loads unscaled with no stored means, so existing active models
            keep predicting exactly as before until they are retrained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

ARTIFACT_FORMAT = 2


class ScaledEstimatorMixin:
    """Mixin for wrappers holding a scikit-learn estimator in self._model."""

    _scale_inputs: bool = False
    _scaler: Any = None
    feature_mean_: np.ndarray | None = None

    def _prepare_fit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self.feature_mean_ = X.mean(axis=0)
        if self._scale_inputs:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler().fit(X)
            return self._scaler.transform(X)
        self._scaler = None
        return X

    def _prepare_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._scaler.transform(X) if self._scaler is not None else X

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._prepare_fit(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._prepare_predict(X))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": ARTIFACT_FORMAT,
                "model": self._model,
                "scaler": self._scaler,
                "feature_mean": self.feature_mean_,
            },
            path,
        )

    def load(self, path: Path) -> None:
        obj = joblib.load(path)
        if isinstance(obj, dict) and obj.get("format") == ARTIFACT_FORMAT:
            self._model = obj["model"]
            self._scaler = obj["scaler"]
            self.feature_mean_ = obj["feature_mean"]
        else:  # legacy: bare estimator, trained on raw inputs
            self._model = obj
            self._scaler = None
            self.feature_mean_ = None

    def linear_terms(self) -> tuple[np.ndarray, float]:
        """
        Coefficients and intercept of a linear estimator expressed in RAW
        feature units (undoing the scaler), so that
            prediction = intercept_raw + coef_raw · x_raw.
        """
        coef = np.ravel(np.asarray(self._model.coef_, dtype=np.float64))
        intercept = float(np.ravel(np.asarray(self._model.intercept_, dtype=np.float64))[0])
        if self._scaler is None:
            return coef, intercept
        coef_raw = coef / self._scaler.scale_
        intercept_raw = intercept - float(np.dot(coef_raw, self._scaler.mean_))
        return coef_raw, intercept_raw
