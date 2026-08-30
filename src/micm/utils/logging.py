"""Logging setup.

One logger per module, stdlib `logging`, English messages. INFO marks
milestones (a subject fitted, a run written), DEBUG carries per-episode detail,
WARNING is reserved for genuine anomalies so that a warning in a run log always
means something. Progress bars belong in scripts, never in library code.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%H:%M:%S"
_ROOT_LOGGER_NAME: Final[str] = "micm"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a single stderr handler to the `micm` logger.

    Idempotent: calling it twice does not duplicate handlers, which would
    otherwise double every line when a script imports a module that also
    configures logging. Only the `micm` logger is touched, so importing this
    package never changes logging for the host application.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    else:
        for existing in logger.handlers:
            existing.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for `name`, rooted under `micm`.

    Pass `__name__`. Modules inside the package already start with `micm.`, so
    they are returned unchanged; anything else is nested under `micm` to keep a
    single configuration point.
    """
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
