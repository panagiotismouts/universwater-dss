"""
dss_shared.db — MongoDB persistence layer.

Primary entry points:
    get_database()        — return the Motor database handle (singleton)
    probe_mongo(db)       — blocking readiness probe with retries
    close_motor_client()  — close the Motor client at shutdown
    bootstrap_db(db)      — idempotent index creation (call once at startup)

Repository classes (one per collection):
    from dss_shared.db.repositories import (
        MeasurementRepository,
        FeatureRepository,
        ModelRegistryRepository,
        ModelMetricsRepository,
        PredictionRepository,
        XAIResultRepository,
        ClientTokenRepository,
        DeliveryLogRepository,
        CheckpointRepository,
    )

Collection name constants:
    from dss_shared.db.collections import MEASUREMENTS, FEATURES, ...

Usage pattern in service startup:
    db = get_database()
    await probe_mongo(db)
    await bootstrap_db(db)
    predictions = PredictionRepository(db)
    registry = ModelRegistryRepository(db)
"""

from dss_shared.db.client import (
    close_motor_client,
    get_database,
    get_motor_client,
    probe_mongo,
)
from dss_shared.db.bootstrap import bootstrap_db, create_indexes

__all__ = [
    "get_motor_client",
    "get_database",
    "probe_mongo",
    "close_motor_client",
    "bootstrap_db",
    "create_indexes",
]
