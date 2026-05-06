"""
BaseModel abstract class.

All model wrappers (white-box and black-box) implement this interface.
The training, recalibration, and prediction layers interact with models
only through this contract.

TODO: Implement concrete wrappers in white_box/ and black_box/ sub-packages.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(abc.ABC):
    """
    Abstract base class for all DSS model wrappers.

    Each concrete subclass wraps one sklearn/XGBoost model family and
    adapts it to this interface.
    """

    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        """Canonical model type string, e.g. "xgboost", "linear_regression"."""

    @property
    @abc.abstractmethod
    def model_family(self) -> str:
        """Model family: "white_box" or "black_box"."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on (X, y)."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X."""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the trained model to disk at path."""

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Load a serialized model from disk at path."""
