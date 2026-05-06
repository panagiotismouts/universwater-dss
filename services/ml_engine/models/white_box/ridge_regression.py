"""sklearn Ridge regression wrapper implementing BaseModel."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from services.ml_engine.models.base import BaseModel


class RidgeRegressionModel(BaseModel):
    """Wrapper for sklearn Ridge."""

    def __init__(self, alpha: float = 1.0) -> None:
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=alpha)

    @property
    def model_type(self) -> str:
        return "ridge_regression"

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
