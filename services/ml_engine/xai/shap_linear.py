"""SHAP LinearExplainer wrapper for white-box linear models (LinearRegression, Ridge)."""

from __future__ import annotations

from typing import Any

import numpy as np

from services.ml_engine.xai.base import BaseExplainer


class LinearExplainerWrapper(BaseExplainer):
    """SHAP LinearExplainer for sklearn linear models."""

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Compute SHAP values for a single input row X using LinearExplainer.

        X must be shape (1, n_features) or (n_features,).
        Returns a dict with shap_values, base_value, feature_values, explainer_type.
        """
        import shap

        row = X.reshape(1, -1) if X.ndim == 1 else X

        inner = getattr(model, "_model", model)

        # Use zero background so SHAP values represent each feature's contribution
        # relative to a baseline of all-zero inputs.  This avoids the degenerate case
        # where using the input itself as the masker makes all SHAP values zero.
        background = np.zeros((1, row.shape[1]))
        explainer = shap.LinearExplainer(
            inner,
            masker=shap.maskers.Independent(background),
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(row)   # shape (1, n_features)

        shap_row = shap_values[0]
        base_value = float(explainer.expected_value)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[0])

        return {
            "shap_values":     {name: float(val) for name, val in zip(feature_names, shap_row)},
            "base_value":      base_value,
            "feature_values":  {name: float(val) for name, val in zip(feature_names, row[0])},
            "explainer_type":  "linear_shap",
        }
