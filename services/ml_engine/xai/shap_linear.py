"""
Exact SHAP for linear models (LinearRegression, ElasticNet, Ridge).

For a linear model  f(x) = b + w·x  with features treated as independent,
the SHAP value of feature i relative to the training distribution is
    phi_i = w_i * (x_i - mean_i)
and the base value is f(mean).  Therefore  base + sum(phi) = f(x)  exactly.

w and b are taken in RAW feature units from the wrapper (linear_terms(),
which undoes any internal standardisation), and mean is the training
feature mean recorded at fit time.  Legacy artifacts (no stored mean)
cannot be explained this way and raise ValueError; callers treat
explanation failure as non-fatal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from services.ml_engine.xai.base import BaseExplainer


class LinearExplainerWrapper(BaseExplainer):
    """Closed-form linear SHAP; no sampling, no shap library call."""

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        row = np.asarray(X, dtype=np.float64)
        row = row.reshape(1, -1) if row.ndim == 1 else row
        mean = getattr(model, "feature_mean_", None)
        terms = getattr(model, "linear_terms", None)
        if mean is None or terms is None:
            raise ValueError(
                "linear SHAP needs the training feature means; this artifact predates them "
                "(retrain the model to enable explanations)"
            )
        coef_raw, intercept_raw = terms()
        mean = np.asarray(mean, dtype=np.float64)
        x = row[0]
        phi = coef_raw * (x - mean)
        base_value = float(intercept_raw + np.dot(coef_raw, mean))
        return {
            "shap_values":    {name: float(v) for name, v in zip(feature_names, phi)},
            "base_value":     base_value,
            "feature_values": {name: float(v) for name, v in zip(feature_names, x)},
            "explainer_type": "linear_shap",
        }
