"""sklearn SVR (Support Vector Regression) wrapper implementing BaseModel."""

from __future__ import annotations

from services.ml_engine.models._artifact import ScaledEstimatorMixin
from services.ml_engine.models.base import BaseModel


class SVRModel(ScaledEstimatorMixin, BaseModel):
    """Wrapper for sklearn SVR with RBF kernel."""

    # Input standardisation is available (set True) but OFF: on the WQI
    # pipelines (2026-09-27 A/B, same split and features) it made SVR worse
    # on all six and ElasticNet worse on the best one (brown_7d). The
    # mixin still records training means for exact linear/kernel SHAP.
    _scale_inputs = False

    def __init__(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> None:
        from sklearn.svm import SVR
        self._model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    @property
    def model_type(self) -> str:
        return "svr"

    @property
    def model_family(self) -> str:
        return "black_box"
