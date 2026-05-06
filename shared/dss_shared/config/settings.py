"""
Pydantic Settings model.

All runtime configuration for the DSS is typed and validated here.
No service may read os.environ directly — all config access goes through
get_settings() which returns an instance of this class.

Environment variables use the DSS_ prefix (env_prefix = "DSS_").
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Validated runtime settings for the DSS.

    Fields are grouped by concern.  Service-specific fields (e.g. JWT secret)
    are present on this model but are only *used* by the owning service.
    Other services will simply never read those fields.
    """

    model_config = SettingsConfigDict(
        env_prefix="DSS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service identity ──────────────────────────────────────────────────────
    env: str = Field(default="local", description="Runtime environment: local | docker | server")
    service_name: str = Field(default="dss", description="Owning service name, set per container")

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI.  Include credentials in the URI string.",
    )
    mongo_db_name: str = Field(default="dss", description="MongoDB database name")
    mongo_connect_timeout_ms: int = Field(default=5000)
    mongo_server_selection_timeout_ms: int = Field(default=10000)
    mongo_retry_count: int = Field(default=5)
    mongo_retry_backoff_seconds: float = Field(default=2.0)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Log level: DEBUG | INFO | WARNING | ERROR")
    log_format: str = Field(default="json", description="Log output format: json | human")

    # ── External APIs — WINGS ─────────────────────────────────────────────────
    wings_api_base_url: str = Field(default="https://serv-api.staging.wi-sense-water.eu", description="WINGS SensorThings API base URL")
    wings_sso_url: str = Field(
        default="https://serv-sso.staging.wi-sense-water.eu/auth/realms/wi-sense-water/protocol/openid-connect/token",
        description="WINGS SSO token endpoint (OAuth2 client_credentials)",
    )
    wings_client_id: str = Field(default="", description="WINGS OAuth2 client_id (secret)")
    wings_client_secret: str = Field(default="", description="WINGS OAuth2 client_secret (secret)")
    wings_request_timeout_seconds: float = Field(default=30.0)
    wings_max_retries: int = Field(default=3)

    # ── External APIs — UOWM ─────────────────────────────────────────────────
    uowm_api_base_url: str = Field(default="http://universwater.ece.uowm.gr", description="UOWM meteorological API base URL (no auth)")
    uowm_request_timeout_seconds: float = Field(default=30.0)
    uowm_max_retries: int = Field(default=3)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    bootstrap_mode: bool = Field(
        default=False,
        description="When True, ingestion fetches full historical data instead of incremental.",
    )

    # ── ML Engine ─────────────────────────────────────────────────────────────
    model_artifact_path: str = Field(
        default="./artifacts",
        description="Root directory for model artifact files (deployment-specific path).",
    )
    recalibration_cron: str = Field(
        default="0 2 * * 1",
        description="Cron expression for weekly recalibration (default: Monday 02:00 UTC).",
    )
    prediction_interval_seconds: int = Field(default=3600, description="Prediction cycle interval.")

    # ── API Service ───────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    jwt_secret_key: str = Field(default="", description="JWT signing secret (secret — env var only)")
    jwt_algorithm: str = Field(default="HS256")
    token_expiry_seconds: int = Field(default=86400, description="Token TTL in seconds")
    admin_api_key: str = Field(default="", description="Admin API key (secret — env var only)")
    api_default_page_size: int = Field(default=24)
    api_max_page_size: int = Field(default=200)
    api_shap_top_n: int = Field(default=5, description="Number of top SHAP features in API responses")

    # ── Feature flags ─────────────────────────────────────────────────────────
    enable_water_pipeline: bool = Field(default=True)
    enable_soil_pipeline: bool = Field(default=True)
    enable_satellite_pipeline: bool = Field(default=False, description="NDVI/satellite — disabled in v1")
    enable_xai: bool = Field(default=True)
    enable_white_box_models: bool = Field(default=True)
    enable_black_box_models: bool = Field(default=True)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got: {v!r}")
        return v.upper()

    @field_validator("env")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        allowed = {"local", "docker", "server"}
        if v.lower() not in allowed:
            raise ValueError(f"env must be one of {allowed}, got: {v!r}")
        return v.lower()
