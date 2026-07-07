"""SHAP KernelExplainer wrapper for SVR and other non-tree, non-linear models."""

from __future__ import annotations

from typing import Any

import numpy as np

from services.ml_engine.xai.base import BaseExplainer


class KernelExplainerWrapper(BaseExplainer):
    """SHAP KernelExplainer for models not supported by Tree or Linear explainers (e.g. SVR).

    Uses a zero-vector background and nsamples=100 to bound runtime.
    At weekly prediction cadence (~30 s/call) this is acceptable.
    """

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        import shap

        row = X.reshape(1, -1) if X.ndim == 1 else X
        inner = getattr(model, "_model", model)

        background = np.zeros((1, row.shape[1]))
        explainer = shap.KernelExplainer(inner.predict, background)
        shap_values = explainer.shap_values(row, nsamples=100)

        shap_row = shap_values[0]
        base_value = float(explainer.expected_value)

        return {
            "shap_values":    {name: float(val) for name, val in zip(feature_names, shap_row)},
            "base_value":     base_value,
            "feature_values": {name: float(val) for name, val in zip(feature_names, row[0])},
            "explainer_type": "kernel_shap",
        }
