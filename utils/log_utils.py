"""Logging helper functions for reusable pipeline logging."""

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        ValueError: If logger name is empty.
    """
    if not name:
        raise ValueError("Logger name cannot be empty.")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_step(logger: logging.Logger, step_name: str, details: Optional[str] = None) -> None:
    """Write a standard step log message.

    Args:
        logger: Logger to write with.
        step_name: Name of the running step.
        details: Optional extra details.

    Returns:
        None

    Raises:
        ValueError: If step name is empty.
    """
    if not step_name:
        raise ValueError("Step name cannot be empty.")
    if details:
        logger.info("%s | %s", step_name, details)
    else:
        logger.info("%s", step_name)
