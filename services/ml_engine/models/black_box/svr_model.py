"""sklearn SVR (Support Vector Regression) wrapper implementing BaseModel."""

from __future__ import annotations

from services.ml_engine.models._artifact import ScaledEstimatorMixin
from services.ml_engine.models.base import BaseModel


class SVRModel(ScaledEstimatorMixin, BaseModel):
    """Wrapper for sklearn SVR with RBF kernel."""

    # Standardise inputs inside the wrapper (scale-sensitive estimator).
    _scale_inputs = True

    def __init__(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> None:
        from sklearn.svm import SVR
        self._model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    @property
    def model_type(self) -> str:
        return "svr"

    @property
    def model_family(self) -> str:
        return "black_box"
