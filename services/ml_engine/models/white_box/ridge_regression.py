"""sklearn Ridge regression wrapper implementing BaseModel."""

from __future__ import annotations

from services.ml_engine.models._artifact import ScaledEstimatorMixin
from services.ml_engine.models.base import BaseModel


class RidgeRegressionModel(ScaledEstimatorMixin, BaseModel):
    """Wrapper for sklearn Ridge."""

    # Standardise inputs inside the wrapper (scale-sensitive estimator).
    _scale_inputs = True

    def __init__(self, alpha: float = 1.0) -> None:
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=alpha)

    @property
    def model_type(self) -> str:
        return "ridge_regression"

    @property
    def model_family(self) -> str:
        return "white_box"
