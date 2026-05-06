"""XGBoost regression wrapper implementing BaseModel."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from services.ml_engine.models.base import BaseModel


class XGBoostModel(BaseModel):
    """Wrapper for XGBRegressor."""

    def __init__(self, **kwargs) -> None:
        from xgboost import XGBRegressor
        self._model = XGBRegressor(**kwargs)

    @property
    def model_type(self) -> str:
        return "xgboost"

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
