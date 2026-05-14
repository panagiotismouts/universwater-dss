"""
Config loader.

Merges config.yaml values with environment variable overrides and returns a
validated Settings instance.  Calling order of precedence (highest wins):
  1. Environment variables (DSS_* prefix)
  2. config.yaml values
  3. Pydantic field defaults (lowest priority)

Usage:
    from dss_shared.config import get_settings
    settings = get_settings()          # cached after first call
    print(settings.mongo_uri)
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from dss_shared.config.settings import Settings

# Default config.yaml search path: repo root/config/config.yaml
# Computed lazily to avoid IndexError when the package is installed via pip
# (in that case DSS_CONFIG_PATH must be set explicitly, e.g. via Dockerfile ENV).
def _default_config_path() -> Path:
    parents = Path(__file__).parents
    if len(parents) > 3:
        return parents[3] / "config" / "config.yaml"
    # Fallback for installed package: rely on DSS_CONFIG_PATH env var
    return Path("/app/config/config.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _flatten_yaml(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Recursively flatten a nested YAML dict into a flat dict keyed by
    underscore-joined path segments.

    The config.yaml top-level sections map to Settings field name prefixes:
        mongo.connect_timeout_ms  → mongo_connect_timeout_ms
        scheduling.recalibration_cron → recalibration_cron
        api.default_page_size → api_default_page_size
        features.enable_xai → enable_xai
        training.metric_thresholds.min_r2 → training_metric_thresholds_min_r2

    Leaf values (non-dict) are kept as-is; pydantic-settings coerces them.
    Non-scalar branches (lists) are kept as-is under their flattened key.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_yaml(value, flat_key))
        else:
            result[flat_key] = value
    return result


# ── Custom pydantic-settings source for YAML ──────────────────────────────────

class _YamlSettingsSource(PydanticBaseSettingsSource):
    """
    pydantic-settings source that reads from a pre-loaded flat YAML dict.

    Placed after env_settings in the priority chain so environment variables
    always win.  Placed before default_settings so YAML values beat field
    defaults.
    """

    def __init__(self, settings_cls: type[BaseSettings], flat: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._flat = {k.lower(): v for k, v in flat.items()}

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        value = self._flat.get(field_name)
        return value, field_name, False

    def __call__(self) -> dict[str, Any]:
        known_fields = set(self.settings_cls.model_fields.keys())
        d: dict[str, Any] = {}
        for flat_key, value in self._flat.items():
            if value is None:
                continue
            if flat_key in known_fields:
                d[flat_key] = value
            else:
                # Try stripping leading section prefixes (e.g. "features_enable_xai"
                # → "enable_xai", "scheduling_recalibration_cron" → "recalibration_cron")
                parts = flat_key.split("_", 1)
                if len(parts) == 2 and parts[1] in known_fields:
                    d[parts[1]] = value
        return d


# ── Patched Settings subclass ─────────────────────────────────────────────────

def _make_settings_with_yaml(flat_yaml: dict[str, Any]) -> Settings:
    """
    Return a Settings instance where YAML values act as defaults that env vars
    can still override.

    We subclass Settings locally to inject the YAML source at the correct
    priority slot: env_settings > yaml_settings > default_settings.
    """

    yaml_source = None  # captured via closure

    class _YamlSettings(Settings):
        @classmethod
        def settings_customise_sources(  # type: ignore[override]
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                _YamlSettingsSource(settings_cls, flat_yaml),
                file_secret_settings,
            )

    return _YamlSettings()  # type: ignore[return-value]


@functools.lru_cache(maxsize=1)
def get_raw_yaml() -> dict[str, Any]:
    """
    Return the raw (un-flattened) config.yaml dict.

    Use this to access structured sub-sections (e.g. ingestion.sources) that
    are not mapped to individual Settings fields.  Result is cached alongside
    get_settings().
    """
    config_path = Path(os.environ.get("DSS_CONFIG_PATH", str(_default_config_path())))
    return _load_yaml(config_path)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the validated Settings singleton.

    Reads config.yaml from the path in DSS_CONFIG_PATH env var (falls back to
    the default path), then constructs a Settings object where:
      - env vars (DSS_* prefix) override everything
      - config.yaml values override field defaults
    """
    config_path = Path(os.environ.get("DSS_CONFIG_PATH", str(_default_config_path())))
    raw_yaml = _load_yaml(config_path)
    flat_yaml = _flatten_yaml(raw_yaml)
    return _make_settings_with_yaml(flat_yaml)
