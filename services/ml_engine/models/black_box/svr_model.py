"""sklearn SVR (Support Vector Regression) wrapper implementing BaseModel."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from services.ml_engine.models.base import BaseModel


class SVRModel(BaseModel):
    """Wrapper for sklearn SVR with RBF kernel."""

    def __init__(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> None:
        from sklearn.svm import SVR
        self._model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    @property
    def model_type(self) -> str:
        return "svr"

    @property
    def model_family(self) -> str:
        return "black_box"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        self._model = joblib.load(path)
