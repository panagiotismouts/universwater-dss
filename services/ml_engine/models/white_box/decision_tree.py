"""sklearn DecisionTreeRegressor wrapper implementing BaseModel."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from services.ml_engine.models.base import BaseModel


class DecisionTreeModel(BaseModel):
    """Wrapper for sklearn DecisionTreeRegressor.

    Classified as white_box because a shallow decision tree's logic can be
    fully inspected and visualised.  XAI routing uses TreeExplainer (not
    LinearExplainer) regardless of the white_box family label.
    """

    def __init__(self, max_depth: int = 10) -> None:
        from sklearn.tree import DecisionTreeRegressor
        self._model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

    @property
    def model_type(self) -> str:
        return "decision_tree"

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
