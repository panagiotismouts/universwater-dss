"""sklearn ElasticNet regression wrapper implementing BaseModel."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from services.ml_engine.models.base import BaseModel


class ElasticNetModel(BaseModel):
    """Wrapper for sklearn ElasticNet."""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        from sklearn.linear_model import ElasticNet
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    @property
    def model_type(self) -> str:
        return "elastic_net"

    @property
    def model_family(self) -> str:
        return "white_box"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        self._model = joblib.load(path)
