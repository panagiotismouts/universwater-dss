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

        from services.ml_engine.xai.shap_tree import scalar_expected_value

        row = X.reshape(1, -1) if X.ndim == 1 else X

        # Call the WRAPPER's predict: it applies the input scaler the estimator
        # was trained with.  Calling the inner estimator on raw rows would feed
        # unscaled inputs to a model fitted on standardised ones.
        # Background: the training feature mean when the artifact records it
        # (contributions relative to a typical input); zeros for legacy
        # artifacts, as before.
        mean = getattr(model, "feature_mean_", None)
        background = (
            np.asarray(mean, dtype=np.float64).reshape(1, -1)
            if mean is not None else np.zeros((1, row.shape[1]))
        )
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(row, nsamples=100)

        shap_row = np.asarray(shap_values).reshape(-1)[: row.shape[1]]
        base_value = scalar_expected_value(explainer.expected_value)

        return {
            "shap_values":    {name: float(val) for name, val in zip(feature_names, shap_row)},
            "base_value":     base_value,
            "feature_values": {name: float(val) for name, val in zip(feature_names, row[0])},
            "explainer_type": "kernel_shap",
        }
