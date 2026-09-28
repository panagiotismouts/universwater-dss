"""
Unit tests for the CCME-WQI in services.ingestion.feature_engineering.water_features.

Covers the excursion ("how far out of objective") term, in particular ORP,
which is a signed potential: a negative reading used to hit the 1e-9 clamp in
objective/value - 1, giving an excursion of ~2e11 that pinned F3 at 100 and
froze the index for the whole 168-h window.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.ingestion.feature_engineering.water_features import _ccme_shortfall, _compute_wqi_ccme


def _docs(values):
    return [SimpleNamespace(value=v, quality_flag="ok") for v in values]


def _ccme(n=24, ph=7.6, do=6.0, temp=20.0, cond=300.0, orp=250.0, orp_values=None):
    return _compute_wqi_ccme(
        _docs([ph] * n), _docs([do] * n), _docs([temp] * n), _docs([cond] * n),
        _docs(orp_values if orp_values is not None else [orp] * n),
    )


# ── _ccme_shortfall ────────────────────────────────────────────────────────────

def test_signed_shortfall_is_relative_difference():
    assert _ccme_shortfall(-84.0, 200.0, signed=True) == pytest.approx(1.42)
    assert _ccme_shortfall(180.0, 200.0, signed=True) == pytest.approx(0.1)


def test_signed_shortfall_is_bounded_for_zero_and_negative():
    assert _ccme_shortfall(0.0, 200.0, signed=True) == pytest.approx(1.0)
    assert _ccme_shortfall(-490.0, 200.0, signed=True) == pytest.approx(3.45)


def test_unsigned_shortfall_keeps_ccme_ratio():
    # Dissolved oxygen, pH, conductivity: standard CCME objective/value - 1.
    assert _ccme_shortfall(2.0, 4.0, signed=False) == pytest.approx(1.0)


# ── _compute_wqi_ccme ──────────────────────────────────────────────────────────

def test_all_within_objectives_scores_100():
    assert _ccme() == pytest.approx(100.0)


def test_too_few_observations_returns_none():
    assert _ccme(n=5) is None


def test_negative_orp_does_not_pin_the_index():
    # 23 passing ORP readings and one at -84 mV: one variable fails (F1 = 20),
    # 1 of 120 tests fails, excursion 1.42 / 120 → index stays in the 80s.
    value = _ccme(orp_values=[250.0] * 23 + [-84.0])
    assert 80.0 < value < 90.0


def test_index_responds_to_orp_magnitude():
    # The frozen-index symptom: with the old clamp both windows scored the same.
    mild = _ccme(orp_values=[250.0] * 12 + [150.0] * 12)
    severe = _ccme(orp_values=[250.0] * 12 + [-400.0] * 12)
    assert severe < mild
    assert mild - severe > 1.0


def test_near_objective_orp_matches_old_ratio_closely():
    # Just below the objective the difference form ≈ the old ratio form.
    assert _ccme_shortfall(195.0, 200.0, signed=True) == pytest.approx(200.0 / 195.0 - 1.0, rel=0.03)
