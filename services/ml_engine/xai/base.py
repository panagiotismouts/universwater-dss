"""BaseExplainer abstract class for SHAP explainer wrappers."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class BaseExplainer(abc.ABC):
    """
    Abstract base class for SHAP explainer wrappers.

    Each concrete subclass wraps one SHAP explainer type and adapts it to
    this interface.  The training and prediction layers interact with explainers
    only through this contract.
    """

    @abc.abstractmethod
    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Compute SHAP values for input X using the given model.

        Returns:
            A dict with keys:
                "shap_values":    dict[feature_name, shap_value] for the input row
                "base_value":     SHAP base/expected value
                "feature_values": dict[feature_name, input_value]
                "explainer_type": str identifier ("tree" | "linear" | "kernel")
        """
