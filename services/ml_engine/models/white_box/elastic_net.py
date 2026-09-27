"""sklearn ElasticNet regression wrapper implementing BaseModel."""

from __future__ import annotations

from services.ml_engine.models._artifact import ScaledEstimatorMixin
from services.ml_engine.models.base import BaseModel


class ElasticNetModel(ScaledEstimatorMixin, BaseModel):
    """Wrapper for sklearn ElasticNet."""

    # Standardise inputs inside the wrapper (scale-sensitive estimator).
    _scale_inputs = True

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        from sklearn.linear_model import ElasticNet
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    @property
    def model_type(self) -> str:
        return "elastic_net"

    @property
    def model_family(self) -> str:
        return "white_box"
