"""
Model artifact store.

Wraps joblib.dump/load behind a clean interface.  All artifact paths are
constructed relative to settings.model_artifact_path.

Usage:
    from services.ml_engine.artifact_store import ArtifactStore
    store = ArtifactStore()
    store.save(model, "water/xgboost/20260401T0200.joblib")
    model = store.load("water/xgboost/20260401T0200.joblib")

TODO: Verify path construction is Docker-volume-aware (uses absolute path
      from settings.model_artifact_path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from dss_shared.config import get_settings
from dss_shared.logging import get_logger

log = get_logger(__name__)


class ArtifactStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._root = Path(settings.model_artifact_path)

    def _resolve(self, relative_path: str) -> Path:
        return self._root / relative_path

    def save(self, obj: Any, relative_path: str) -> Path:
        """Serialize obj to disk.  Creates parent directories as needed."""
        full_path = self._resolve(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, full_path)
        log.info("artifact_saved", path=str(full_path))
        return full_path

    def load(self, relative_path: str) -> Any:
        """Load and return a serialized object from disk."""
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {full_path}")
        obj = joblib.load(full_path)
        log.info("artifact_loaded", path=str(full_path))
        return obj

    def exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).exists()
