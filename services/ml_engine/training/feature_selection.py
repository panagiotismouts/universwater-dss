"""
Automated feature selection: cross-validated RFE, cross-checked by SHAP
importance on the same fitted estimator.

No manual/human-chosen thresholds anywhere: the number of features kept is
decided by cross-validated score (RFECV), and the returned importance
ranking comes from SHAP attribution on the resulting fit — not from a
hand-picked cutoff. Runs once per pipeline per training cycle (bootstrap or
weekly recalibration), on the pooled feature matrix from every sensor
feeding that pipeline; the resulting column subset is then shared by all 8
candidates in the pipeline's model pool, keeping the pool comparison fair.

Time-series aware: RFECV's cross-validation uses TimeSeriesSplit (not a
shuffled K-fold), consistent with the chronological, no-shuffling discipline
used everywhere else in this training pipeline (80/20 chronological splits,
horizon-target construction, etc.) — a shuffled CV split would leak future
information across folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit

from dss_shared.logging import get_logger

log = get_logger(__name__)

# Below this many rows or features, RFECV is unreliable / not worth the
# compute — fall back to "keep everything" (a safe no-op, not a manual
# override of which features to keep).
_MIN_ROWS_FOR_SELECTION = 40
_MIN_FEATURES_FOR_SELECTION = 12

_MAX_SHAP_SAMPLE_ROWS = 500  # cap for SHAP compute cost; sampling is uniform


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cv_folds: int = 3,
    min_features_to_select: int = 8,
    random_state: int = 42,
) -> tuple[list[str], dict[str, float]]:
    """
    Select the best-scoring feature subset via cross-validated RFE, and
    return a SHAP-based importance ranking over the selected subset for
    auditability.

    Returns:
        (selected_feature_names, shap_importance_by_feature)
        On any degenerate-input fallback, selected_feature_names is the
        full input list and shap_importance_by_feature is empty (still
        automatic — not a manual choice, just "too little data to select").
    """
    n_rows, n_features = X.shape

    if n_rows < _MIN_ROWS_FOR_SELECTION or n_features <= _MIN_FEATURES_FOR_SELECTION:
        log.info(
            "feature_selection_skipped_insufficient_data",
            n_rows=n_rows,
            n_features=n_features,
        )
        return list(feature_names), {}

    estimator = RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    cv = TimeSeriesSplit(n_splits=cv_folds)

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="r2",
        min_features_to_select=min(min_features_to_select, n_features),
        n_jobs=-1,
    )

    try:
        rfecv.fit(X, y)
    except Exception as exc:
        log.warning("feature_selection_rfecv_failed", error=str(exc))
        return list(feature_names), {}

    selected_mask = rfecv.support_
    selected_names = [name for name, keep in zip(feature_names, selected_mask) if keep]
    X_selected = X[:, selected_mask]

    # rfecv.estimator_ is the RandomForest already refit on the selected
    # subset (sklearn refits automatically once the CV-optimal count is
    # found) — reuse it directly for the SHAP cross-check rather than
    # fitting a second model.
    importance: dict[str, float] = {}
    try:
        import shap

        n_sample = min(_MAX_SHAP_SAMPLE_ROWS, X_selected.shape[0])
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_selected.shape[0], size=n_sample, replace=False)
        sample = X_selected[idx]

        explainer = shap.TreeExplainer(rfecv.estimator_)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = {
            name: float(val) for name, val in zip(selected_names, mean_abs_shap)
        }
    except Exception as exc:
        # SHAP cross-check is an auditability add-on; RFECV's own selection
        # is already complete and valid without it.
        log.warning("feature_selection_shap_crosscheck_failed", error=str(exc))

    log.info(
        "feature_selection_completed",
        n_features_in=n_features,
        n_features_selected=len(selected_names),
        cv_folds=cv_folds,
    )
    return selected_names, importance
