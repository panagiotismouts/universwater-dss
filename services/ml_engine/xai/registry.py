"""
XAI explainer registry.

Maps (model_family, model_type) to the correct SHAP explainer class.
"""

from __future__ import annotations

from services.ml_engine.xai.base import BaseExplainer
from services.ml_engine.xai.shap_tree import TreeExplainerWrapper
from services.ml_engine.xai.shap_linear import LinearExplainerWrapper

_REGISTRY: dict[str, type[BaseExplainer]] = {
    "black_box": TreeExplainerWrapper,
    "white_box": LinearExplainerWrapper,
}


def get_explainer(model_family: str) -> BaseExplainer:
    """Return an instantiated explainer for the given model_family."""
    if model_family not in _REGISTRY:
        raise KeyError(f"No explainer registered for model_family: {model_family!r}")
    return _REGISTRY[model_family]()
