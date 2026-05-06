"""
MongoDB collection name constants and index definitions.

This is the single source of truth for:
  - Every collection name string used in the DSS
  - Every index that exists on each collection

Collection name constants prevent the string "preprocessed_measurements"
from being scattered across 15 files.  All repositories and the bootstrap
module import from here.

Index definitions are grouped per collection.  Each entry is a tuple of
(keys, kwargs) matching the pymongo/motor create_index signature:
  keys   — list of (field, direction) tuples
  kwargs — dict of index options (unique, sparse, name, expireAfterSeconds, etc.)

Index directions:
  pymongo.ASCENDING  = 1
  pymongo.DESCENDING = -1

All index names are explicit so that create_index is idempotent: if the
index already exists with the same name and spec, MongoDB is a no-op.
"""

from __future__ import annotations

from typing import Any

import pymongo

# ---------------------------------------------------------------------------
# Collection name constants
# ---------------------------------------------------------------------------

MEASUREMENTS = "preprocessed_measurements"
FEATURES = "engineered_features"
MODEL_REGISTRY = "model_registry"
MODEL_METRICS = "model_metrics"
PREDICTIONS = "prediction_results"
XAI_RESULTS = "xai_results"
CLIENT_TOKENS = "client_tokens"
DELIVERY_LOGS = "api_delivery_logs"
CHECKPOINTS = "ingestion_checkpoints"

# Ordered list used by bootstrap to create all collections
ALL_COLLECTIONS: list[str] = [
    MEASUREMENTS,
    FEATURES,
    MODEL_REGISTRY,
    MODEL_METRICS,
    PREDICTIONS,
    XAI_RESULTS,
    CLIENT_TOKENS,
    DELIVERY_LOGS,
    CHECKPOINTS,
]

# ---------------------------------------------------------------------------
# TTL for api_delivery_logs (90 days, configurable)
# ---------------------------------------------------------------------------

DELIVERY_LOG_TTL_SECONDS: int = 90 * 24 * 3600  # 7_776_000

# ---------------------------------------------------------------------------
# Index definitions
#
# Each entry: (collection_name, keys_list, options_dict)
# ---------------------------------------------------------------------------

IndexSpec = tuple[str, list[tuple[str, int]], dict[str, Any]]

INDEX_DEFINITIONS: list[IndexSpec] = [

    # ── preprocessed_measurements ─────────────────────────────────────────────
    # Unique: one document per physical observation
    (
        MEASUREMENTS,
        [("pipeline", pymongo.ASCENDING), ("sensor_id", pymongo.ASCENDING),
         ("variable_name", pymongo.ASCENDING), ("measured_at", pymongo.ASCENDING)],
        {"unique": True, "name": "uq_measurement_observation"},
    ),
    # Forward-fill lookup + feature engineering window scans
    (
        MEASUREMENTS,
        [("pipeline", pymongo.ASCENDING), ("sensor_id", pymongo.ASCENDING),
         ("variable_name", pymongo.ASCENDING), ("measured_at", pymongo.DESCENDING)],
        {"name": "ix_measurement_sensor_variable_time"},
    ),
    # Training data loads across all variables for a pipeline
    (
        MEASUREMENTS,
        [("pipeline", pymongo.ASCENDING), ("measured_at", pymongo.DESCENDING)],
        {"name": "ix_measurement_pipeline_time"},
    ),
    # processing_status scan (retained for future use per Amendment C.3)
    (
        MEASUREMENTS,
        [("processing_status", pymongo.ASCENDING), ("measured_at", pymongo.ASCENDING)],
        {"name": "ix_measurement_status_time"},
    ),

    # ── engineered_features ───────────────────────────────────────────────────
    # Unique: one feature vector per computation cycle
    (
        FEATURES,
        [("pipeline", pymongo.ASCENDING), ("sensor_id", pymongo.ASCENDING),
         ("feature_timestamp", pymongo.ASCENDING),
         ("feature_schema_version", pymongo.ASCENDING)],
        {"unique": True, "name": "uq_feature_vector"},
    ),
    # Prediction engine: find most recent feature vector for a sensor
    (
        FEATURES,
        [("pipeline", pymongo.ASCENDING), ("sensor_id", pymongo.ASCENDING),
         ("feature_timestamp", pymongo.DESCENDING),
         ("feature_schema_version", pymongo.ASCENDING)],
        {"name": "ix_feature_sensor_time"},
    ),
    # Training dataset construction across all sensors
    (
        FEATURES,
        [("pipeline", pymongo.ASCENDING), ("feature_timestamp", pymongo.DESCENDING),
         ("feature_schema_version", pymongo.ASCENDING)],
        {"name": "ix_feature_pipeline_time"},
    ),

    # ── model_registry ────────────────────────────────────────────────────────
    # Stable human-readable key lookup
    (
        MODEL_REGISTRY,
        [("model_id", pymongo.ASCENDING)],
        {"unique": True, "name": "uq_model_id"},
    ),
    # Active model lookup — hottest query in the system
    (
        MODEL_REGISTRY,
        [("pipeline", pymongo.ASCENDING), ("status", pymongo.ASCENDING)],
        {"name": "ix_model_pipeline_status"},
    ),
    # Partial unique: at most one active model per pipeline (database-level enforcement)
    (
        MODEL_REGISTRY,
        [("pipeline", pymongo.ASCENDING)],
        {
            "unique": True,
            "partialFilterExpression": {"status": "active"},
            "name": "uq_one_active_per_pipeline",
        },
    ),
    # Admin listing of recent models
    (
        MODEL_REGISTRY,
        [("trained_at", pymongo.DESCENDING)],
        {"name": "ix_model_trained_at"},
    ),

    # ── model_metrics ─────────────────────────────────────────────────────────
    (
        MODEL_METRICS,
        [("model_id", pymongo.ASCENDING)],
        {"name": "ix_metrics_model_id"},
    ),
    (
        MODEL_METRICS,
        [("model_id", pymongo.ASCENDING), ("evaluation_type", pymongo.ASCENDING)],
        {"name": "ix_metrics_model_eval_type"},
    ),

    # ── prediction_results ────────────────────────────────────────────────────
    # Unique: one prediction per (pipeline, sensor, input feature window)
    (
        PREDICTIONS,
        [("pipeline", pymongo.ASCENDING), ("sensor_id", pymongo.ASCENDING),
         ("input_feature_timestamp", pymongo.ASCENDING)],
        {"unique": True, "name": "uq_prediction_input"},
    ),
    # Latest + historical results API
    (
        PREDICTIONS,
        [("pipeline", pymongo.ASCENDING), ("sensor_id", pymongo.ASCENDING),
         ("input_feature_timestamp", pymongo.DESCENDING)],
        {"name": "ix_prediction_sensor_time"},
    ),
    # Audit: all predictions from a model
    (
        PREDICTIONS,
        [("model_id", pymongo.ASCENDING)],
        {"name": "ix_prediction_model_id"},
    ),

    # ── xai_results ───────────────────────────────────────────────────────────
    # Unique: one XAI document per prediction
    (
        XAI_RESULTS,
        [("prediction_id", pymongo.ASCENDING)],
        {"unique": True, "name": "uq_xai_prediction_id"},
    ),
    # XAI audit per model
    (
        XAI_RESULTS,
        [("model_id", pymongo.ASCENDING)],
        {"name": "ix_xai_model_id"},
    ),
    # Time-range scans for operational analysis
    (
        XAI_RESULTS,
        [("pipeline", pymongo.ASCENDING),
         ("input_feature_timestamp", pymongo.DESCENDING)],
        {"name": "ix_xai_pipeline_time"},
    ),

    # ── client_tokens ─────────────────────────────────────────────────────────
    # Credential lookup by client_id
    (
        CLIENT_TOKENS,
        [("client_id", pymongo.ASCENDING)],
        {"unique": True, "name": "uq_client_id"},
    ),
    # Token validation on every authenticated request — must be fast
    (
        CLIENT_TOKENS,
        [("token_hash", pymongo.ASCENDING)],
        {"sparse": True, "name": "ix_token_hash"},
    ),

    # ── api_delivery_logs ─────────────────────────────────────────────────────
    # Per-client audit
    (
        DELIVERY_LOGS,
        [("client_id", pymongo.ASCENDING),
         ("request_timestamp", pymongo.DESCENDING)],
        {"name": "ix_delivery_client_time"},
    ),
    # TTL — automatic 90-day log rotation
    (
        DELIVERY_LOGS,
        [("request_timestamp", pymongo.ASCENDING)],
        {"expireAfterSeconds": DELIVERY_LOG_TTL_SECONDS, "name": "ttl_delivery_logs"},
    ),
    # Error rate aggregation
    (
        DELIVERY_LOGS,
        [("response_status", pymongo.ASCENDING)],
        {"name": "ix_delivery_status"},
    ),

    # ── ingestion_checkpoints ─────────────────────────────────────────────────
    (
        CHECKPOINTS,
        [("source", pymongo.ASCENDING), ("pipeline", pymongo.ASCENDING),
         ("variable_name", pymongo.ASCENDING)],
        {"unique": True, "name": "uq_checkpoint_source_pipeline_variable"},
    ),
]
