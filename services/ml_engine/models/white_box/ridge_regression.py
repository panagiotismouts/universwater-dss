"""sklearn Ridge regression wrapper implementing BaseModel."""

from __future__ import annotations

from services.ml_engine.models._artifact import ScaledEstimatorMixin
from services.ml_engine.models.base import BaseModel


class RidgeRegressionModel(ScaledEstimatorMixin, BaseModel):
    """Wrapper for sklearn Ridge."""

    # Input standardisation is available (set True) but OFF: on the WQI
    # pipelines (2026-09-27 A/B, same split and features) it made SVR worse
    # on all six and ElasticNet worse on the best one (brown_7d). The
    # mixin still records training means for exact linear/kernel SHAP.
    _scale_inputs = False

    def __init__(self, alpha: float = 1.0) -> None:
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=alpha)

    @property
    def model_type(self) -> str:
        return "ridge_regression"

    @property
    def model_family(self) -> str:
        return "white_box"
