"""
Logging configuration for Vajra BM25.

As a library, Vajra only attaches a NullHandler to its "vajra_bm25" logger: records
propagate to the application's logging configuration and the application chooses the
level and destination. Scripts that want Vajra's own console output can opt in with
configure_logging(), which can be controlled via environment variables:
- VAJRA_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- VAJRA_LOG_FORMAT: Set log format (simple, detailed, json)
"""

import logging
import os
import sys
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"


def get_log_level() -> int:
    """Get log level from environment or default to INFO."""
    level_name = os.getenv("VAJRA_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_log_format() -> str:
    """Get log format from environment or default to simple format."""
    format_type = os.getenv("VAJRA_LOG_FORMAT", "simple").lower()

    formats = {
        "simple": SIMPLE_FORMAT,
        "default": DEFAULT_FORMAT,
        "detailed": DETAILED_FORMAT,
    }

    return formats.get(format_type, SIMPLE_FORMAT)


def configure_logging(
    level: Optional[int] = None,
    format_str: Optional[str] = None,
    logger_name: str = "vajra_bm25"
) -> logging.Logger:
    """
    Opt in to Vajra's own console logging (stdout handler, no propagation).

    Not called on import. Libraries should not install handlers, so by default Vajra
    records flow to the application's logging setup instead.

    Args:
        level: Logging level (defaults to environment or INFO)
        format_str: Log format string (defaults to environment or simple)
        logger_name: Name of the logger to configure

    Returns:
        Configured logger instance
    """
    if level is None:
        level = get_log_level()

    if format_str is None:
        format_str = get_log_format()

    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter and add to handler
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name under the vajra_bm25 namespace.

    Args:
        name: Component name (e.g., "search", "scorer", "index")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"vajra_bm25.{name}")


# Library default: a NullHandler and nothing else. No level is set and propagation stays on,
# so the application's root logger decides what is shown and where it goes.
_root_logger = logging.getLogger("vajra_bm25")
if not any(isinstance(h, logging.NullHandler) for h in _root_logger.handlers):
    _root_logger.addHandler(logging.NullHandler())
