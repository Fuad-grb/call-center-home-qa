"""
Structured JSON logging.
Use get_logger(__name__) in every module instead of print().
"""

import logging
import sys

from pythonjsonlogger import json as json_logger
from config.settings import settings

_CONFIGURED = False


def setup_logging() -> None:
    """Configure root logger. Call once at startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = json_logger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    setup_logging()
    return logging.getLogger(name)
