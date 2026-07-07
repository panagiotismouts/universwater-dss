"""
XAI explainer registry.

Maps model_type to the correct SHAP explainer class.
Dispatch is by model_type (not model_family) because Decision Tree is white_box
but needs TreeExplainer, and SVR is black_box but needs KernelExplainer.
"""

from __future__ import annotations

from services.ml_engine.xai.base import BaseExplainer
from services.ml_engine.xai.shap_tree import TreeExplainerWrapper
from services.ml_engine.xai.shap_linear import LinearExplainerWrapper
from services.ml_engine.xai.shap_kernel import KernelExplainerWrapper

_TREE_MODELS = {"xgboost", "random_forest", "lightgbm", "catboost", "decision_tree"}
_LINEAR_MODELS = {"linear_regression", "elastic_net", "ridge_regression"}


def get_explainer(model_family: str, model_type: str = "") -> BaseExplainer:
    """Return an instantiated explainer for the given model.

    model_type takes precedence over model_family for routing.
    Falls back to KernelExplainer for unknown types.
    """
    if model_type in _TREE_MODELS:
        return TreeExplainerWrapper()
    if model_type in _LINEAR_MODELS:
        return LinearExplainerWrapper()
    # SVR and any future model types not covered above
    return KernelExplainerWrapper()
