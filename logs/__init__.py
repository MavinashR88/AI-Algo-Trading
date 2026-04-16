"""Structured JSON logging setup using structlog."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Configure structlog for JSON structured logging.

    All modules use structlog.get_logger() to get a bound logger.
    Output goes to both stdout (for Railway) and a rotating file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Standard library logging config (structlog wraps this)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # File handler for persistent logs
    file_handler = logging.FileHandler(log_path / "trading.log")
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logging.getLogger().addHandler(file_handler)

    # Structlog processors pipeline
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
