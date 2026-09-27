"""sklearn ElasticNet regression wrapper implementing BaseModel."""

from __future__ import annotations

from services.ml_engine.models._artifact import ScaledEstimatorMixin
from services.ml_engine.models.base import BaseModel


class ElasticNetModel(ScaledEstimatorMixin, BaseModel):
    """Wrapper for sklearn ElasticNet."""

    # Input standardisation is available (set True) but OFF: on the WQI
    # pipelines (2026-09-27 A/B, same split and features) it made SVR worse
    # on all six and ElasticNet worse on the best one (brown_7d). The
    # mixin still records training means for exact linear/kernel SHAP.
    _scale_inputs = False

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        from sklearn.linear_model import ElasticNet
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    @property
    def model_type(self) -> str:
        return "elastic_net"

    @property
    def model_family(self) -> str:
        return "white_box"
