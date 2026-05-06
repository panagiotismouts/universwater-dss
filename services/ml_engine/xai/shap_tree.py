"""SHAP TreeExplainer wrapper for black-box tree models (XGBoost, RandomForest)."""

from __future__ import annotations

from typing import Any

import numpy as np

from services.ml_engine.xai.base import BaseExplainer


class TreeExplainerWrapper(BaseExplainer):
    """SHAP TreeExplainer for XGBoost and RandomForest models."""

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Compute SHAP values for a single input row X using TreeExplainer.

        X must be shape (1, n_features) or (n_features,).
        Returns a dict with shap_values, base_value, feature_values, explainer_type.
        """
        import shap

        row = X.reshape(1, -1) if X.ndim == 1 else X

        # Extract the underlying sklearn/xgboost model from the wrapper
        inner = getattr(model, "_model", model)
        explainer = shap.TreeExplainer(inner)
        shap_values = explainer.shap_values(row)   # shape (1, n_features)

        # shap_values may be a list (multi-output) — take first output for regression
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_row = shap_values[0]   # 1D array of length n_features
        base_value = float(explainer.expected_value)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[0])

        return {
            "shap_values":     {name: float(val) for name, val in zip(feature_names, shap_row)},
            "base_value":      base_value,
            "feature_values":  {name: float(val) for name, val in zip(feature_names, row[0])},
            "explainer_type":  "tree_shap",
        }
