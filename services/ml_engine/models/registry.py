"""
Model class registry.

Maps model_type strings from config/MongoDB to concrete model wrapper classes.
This is the single point of change when a new model type is added.

TODO: Register all model types.
"""

from __future__ import annotations

from typing import Type

from services.ml_engine.models.base import BaseModel
from services.ml_engine.models.white_box.linear_regression import LinearRegressionModel
from services.ml_engine.models.white_box.ridge_regression import RidgeRegressionModel
from services.ml_engine.models.black_box.xgboost_model import XGBoostModel
from services.ml_engine.models.black_box.random_forest import RandomForestModel

_REGISTRY: dict[str, Type[BaseModel]] = {
    "linear_regression": LinearRegressionModel,
    "ridge_regression":  RidgeRegressionModel,
    "xgboost":           XGBoostModel,
    "random_forest":     RandomForestModel,
}


def get_model_class(model_type: str) -> Type[BaseModel]:
    """Return the model class for the given model_type string."""
    if model_type not in _REGISTRY:
        raise KeyError(f"Unknown model_type: {model_type!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[model_type]


def instantiate_model(model_type: str) -> BaseModel:
    """Instantiate and return a fresh model of the given type."""
    return get_model_class(model_type)()
