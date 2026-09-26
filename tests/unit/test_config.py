"""
Unit tests for dss_shared.config.settings and dss_shared.config.loader.

Covers:
  - Settings field validation (log_level, env)
  - _flatten_yaml: basic flattening, nested dicts, leaf types
  - get_settings: YAML values used as defaults, env vars override YAML
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from dss_shared.config.settings import Settings
from dss_shared.config.loader import _flatten_yaml, get_settings


# ── Settings validators ────────────────────────────────────────────────────────

def test_settings_defaults():
    """Settings instantiates successfully with all defaults."""
    s = Settings()
    assert s.env == "local"
    assert s.mongo_db_name == "dss"
    assert s.log_level == "INFO"


def test_invalid_log_level():
    """Settings rejects an invalid log_level."""
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


def test_invalid_env():
    """Settings rejects an unknown env label."""
    with pytest.raises(ValidationError):
        Settings(env="production")


def test_feature_backfill_cap_default_and_bounds():
    """The per-run feature backfill cap defaults to one week and must be >= 1."""
    assert Settings().feature_backfill_max_hours == 168
    assert Settings(feature_backfill_max_hours=24).feature_backfill_max_hours == 24
    with pytest.raises(ValidationError):
        Settings(feature_backfill_max_hours=0)


# ── _flatten_yaml ──────────────────────────────────────────────────────────────

def test_flatten_yaml_flat_dict():
    flat = _flatten_yaml({"retry_count": 5, "enabled": True})
    assert flat == {"retry_count": 5, "enabled": True}


def test_flatten_yaml_one_level_nesting():
    nested = {"mongo": {"uri": "mongodb://localhost:27017", "db_name": "dss"}}
    flat = _flatten_yaml(nested)
    assert flat == {
        "mongo_uri": "mongodb://localhost:27017",
        "mongo_db_name": "dss",
    }


def test_flatten_yaml_two_level_nesting():
    data = {"training": {"metric_thresholds": {"min_r2": 0.75, "max_mae_relative": 0.15}}}
    flat = _flatten_yaml(data)
    assert flat["training_metric_thresholds_min_r2"] == 0.75
    assert flat["training_metric_thresholds_max_mae_relative"] == 0.15


def test_flatten_yaml_list_value_preserved():
    """List values (e.g. ingestion sources) are kept as-is, not recursed into."""
    data = {"ingestion": {"sources": [{"name": "wings_water"}]}}
    flat = _flatten_yaml(data)
    assert flat["ingestion_sources"] == [{"name": "wings_water"}]


def test_flatten_yaml_empty_dict():
    assert _flatten_yaml({}) == {}


# ── get_settings with config.yaml ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear the lru_cache before and after each test so tests are isolated."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_reads_yaml(tmp_path, monkeypatch):
    """Values in config.yaml are used when no env var overrides are present."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "api:\n  default_page_size: 42\nfeatures:\n  enable_xai: false\n"
    )
    monkeypatch.setenv("DSS_CONFIG_PATH", str(cfg))
    monkeypatch.delenv("DSS_API_DEFAULT_PAGE_SIZE", raising=False)
    monkeypatch.delenv("DSS_ENABLE_XAI", raising=False)

    s = get_settings()
    assert s.api_default_page_size == 42
    assert s.enable_xai is False


def test_get_settings_env_overrides_yaml(tmp_path, monkeypatch):
    """Environment variables take priority over config.yaml values."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("api:\n  default_page_size: 42\n")
    monkeypatch.setenv("DSS_CONFIG_PATH", str(cfg))
    monkeypatch.setenv("DSS_API_DEFAULT_PAGE_SIZE", "99")

    s = get_settings()
    assert s.api_default_page_size == 99


def test_get_settings_missing_yaml_uses_defaults(tmp_path, monkeypatch):
    """When config.yaml does not exist, field defaults are still applied."""
    monkeypatch.setenv("DSS_CONFIG_PATH", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.delenv("DSS_API_DEFAULT_PAGE_SIZE", raising=False)

    s = get_settings()
    assert s.api_default_page_size == 24  # pydantic default
