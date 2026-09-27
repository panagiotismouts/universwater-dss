"""sklearn LinearRegression wrapper implementing BaseModel."""

from __future__ import annotations

from services.ml_engine.models._artifact import ScaledEstimatorMixin
from services.ml_engine.models.base import BaseModel


class LinearRegressionModel(ScaledEstimatorMixin, BaseModel):
    """Wrapper for sklearn LinearRegression."""

    # OLS predictions are scale-invariant; the mixin is used for the stored
    # training means (linear SHAP reference) and the artifact format.
    _scale_inputs = False

    def __init__(self) -> None:
        from sklearn.linear_model import LinearRegression
        self._model = LinearRegression()

    @property
    def model_type(self) -> str:
        return "linear_regression"

    @property
    def model_family(self) -> str:
        return "white_box"
