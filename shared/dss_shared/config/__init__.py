"""Config sub-package. Entry points: get_settings(), get_raw_yaml()."""
from dss_shared.config.loader import get_raw_yaml, get_settings

__all__ = ["get_settings", "get_raw_yaml"]
