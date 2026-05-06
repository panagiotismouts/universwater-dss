"""
Structured logging setup.

Configures structlog to emit either JSON (for production/Docker) or
pretty-printed human-readable output (for local development).

Usage:
    from dss_shared.logging import setup_logging, get_logger

    setup_logging(level="INFO", fmt="json", service="ingestion")
    log = get_logger(__name__)
    log.info("started", source="wings")
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def setup_logging(
    level: str = "INFO",
    fmt: Literal["json", "human"] = "json",
    service: str = "dss",
) -> None:
    """
    Configure structlog and stdlib logging.

    Must be called once at service startup before any log statements are
    emitted.  Subsequent calls are idempotent.

    Args:
        level:   Log level string (DEBUG | INFO | WARNING | ERROR).
        fmt:     Output format.  "json" for structured production logs;
                 "human" for coloured terminal output during development.
        service: Service name injected into every log record.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libraries (motor, uvicorn)
    # emit at the same level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Inject service name into every log record via context vars.
    structlog.contextvars.bind_contextvars(service=service)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given name."""
    return structlog.get_logger(name)
