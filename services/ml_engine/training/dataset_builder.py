"""
Dataset builder.

Loads the training-ready feature matrix from the engineered_features collection
for a given (pipeline, time_range, feature_schema_version).

Contract:
  - y = features[target_variable] — the target column must be present in every
    FeatureDocument for this pipeline.
  - X = all other feature columns, in a consistent sorted order.
  - Rows with a missing target column are skipped and logged.
  - Raises ValueError if fewer than 10 rows remain after filtering.

The dataset is sorted by feature_timestamp ascending.  No shuffling is done here;
callers are responsible for train/validation splitting.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.db.repositories.features import FeatureRepository
from dss_shared.logging import get_logger

log = get_logger(__name__)

_MIN_TRAINING_ROWS = 10

# Max distance between (t + horizon) and the matched future document for
# horizon-target construction.  Feature docs are ~hourly with gaps.
_HORIZON_MATCH_TOLERANCE = timedelta(hours=6)


async def build_dataset(
    db: AsyncIOMotorDatabase,
    pipeline: str,
    start: datetime,
    end: datetime,
    feature_schema_version: str,
    target_variable: str,
    feature_pipeline: str | None = None,
    excluded_features: list[str] | None = None,
    horizon_days: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load and assemble a feature matrix for training.

    Args:
        db:                     Motor database handle.
        pipeline:               Pipeline name ("water" | "soil").
        start:                  Training window start (inclusive).
        end:                    Training window end (inclusive).
        feature_schema_version: Must match FeatureDocument.feature_schema_version.
        target_variable:        Key inside FeatureDocument.features to use as y.
        horizon_days:           0 = nowcast (y from the same document).
                                > 0 = forecast: y is taken from the same-sensor
                                document nearest (t + horizon_days), within
                                _HORIZON_MATCH_TOLERANCE; unmatched rows are
                                dropped.  The current value of target_variable
                                remains in X (it is past information relative
                                to the forecast target).

    Returns:
        (X, y, feature_names)
          X:             np.ndarray of shape (n_samples, n_features), dtype float64.
          y:             np.ndarray of shape (n_samples,), dtype float64.
          feature_names: list[str] of length n_features, aligned with X columns.

    Raises:
        ValueError: If fewer than _MIN_TRAINING_ROWS valid rows are available.
    """
    repo = FeatureRepository(db)
    docs = await repo.find_training_window(feature_pipeline or pipeline, start, end, feature_schema_version)
    log.debug(
        "dataset_builder_docs_loaded",
        pipeline=pipeline,
        n_docs=len(docs),
        start=start.isoformat(),
        end=end.isoformat(),
    )

    if not docs:
        raise ValueError(
            f"No feature documents found for pipeline={pipeline!r} "
            f"schema={feature_schema_version!r} in [{start.date()}, {end.date()}]."
        )

    # Determine the full, sorted list of non-target feature names from the first doc
    # then verify consistency across all docs.
    first_features = docs[0].features
    _excluded = set(excluded_features or [])
    if horizon_days == 0:
        # Nowcast: the target column IS the current value — exclude it from X.
        # Forecast: the current value is legitimate past information — keep it.
        _excluded |= {target_variable}
    all_feature_keys = sorted(k for k in first_features if k not in _excluded)

    if not all_feature_keys:
        raise ValueError(
            f"Feature document for pipeline={pipeline!r} has no columns besides "
            f"the target variable {target_variable!r}."
        )

    # Per-sensor timestamp index for horizon-target matching (docs are already
    # time-sorted, so per-sensor lists remain ascending for bisect).
    by_sensor: dict[str, tuple[list[datetime], list]] = {}
    if horizon_days > 0:
        for d in docs:
            ts_list, doc_list = by_sensor.setdefault(d.sensor_id, ([], []))
            ts_list.append(d.feature_timestamp)
            doc_list.append(d)

    horizon = timedelta(days=horizon_days)
    rows_X: list[list[float]] = []
    rows_y: list[float] = []
    skipped = 0
    unmatched = 0

    for doc in docs:
        if horizon_days > 0:
            ts_list, doc_list = by_sensor[doc.sensor_id]
            wanted = doc.feature_timestamp + horizon
            idx = bisect_left(ts_list, wanted)
            future_doc = None
            best_dt: Optional[timedelta] = None
            for j in (idx - 1, idx):
                if 0 <= j < len(ts_list):
                    dt = abs(ts_list[j] - wanted)
                    if best_dt is None or dt < best_dt:
                        future_doc, best_dt = doc_list[j], dt
            if future_doc is None or best_dt > _HORIZON_MATCH_TOLERANCE:
                unmatched += 1
                continue
            target_val = future_doc.features.get(target_variable)
        else:
            target_val = doc.features.get(target_variable)

        if target_val is None:
            skipped += 1
            continue

        row = []
        for key in all_feature_keys:
            val = doc.features.get(key)
            if val is None:
                # Fill missing non-target features with 0.0 rather than dropping
                # the whole row.  The caller may decide to use imputation instead.
                val = 0.0
            row.append(val)

        rows_X.append(row)
        rows_y.append(float(target_val))

    if skipped:
        log.warning(
            "dataset_builder_rows_skipped",
            pipeline=pipeline,
            skipped=skipped,
            reason=f"missing target column {target_variable!r}",
        )
    if unmatched:
        log.info(
            "dataset_builder_horizon_unmatched",
            pipeline=pipeline,
            unmatched=unmatched,
            horizon_days=horizon_days,
        )

    if len(rows_X) < _MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient training data for pipeline={pipeline!r}: "
            f"{len(rows_X)} valid rows (minimum {_MIN_TRAINING_ROWS})."
        )

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)

    log.info(
        "dataset_built",
        pipeline=pipeline,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        target=target_variable,
    )
    return X, y, all_feature_keys
