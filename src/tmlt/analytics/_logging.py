"""Logging helpers and contributor guidance for Tumult Analytics.

Tumult Analytics uses the standard-library :mod:`logging` module. Library code
should follow this pattern:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)
    logger.info("...")

The library does **not** configure the root logger, handlers, formatters, or
levels. Applications and notebooks that want to see Analytics log messages
should configure logging themselves (for example with
``logging.basicConfig(level=logging.INFO)`` or a logger-specific handler on
``tmlt.analytics``).

Output channels
---------------

Use the right channel for the job:

* ``print`` — intentional interactive UX (for example ``Session.describe`` and
  ``check_installation``).
* ``warnings.warn`` — user-facing behavioral or setup advisories that must be
  visible without configuring logging.
* ``logging`` — diagnostic / lifecycle messages for operators who configure
  logging.

Keep log messages coarse: prefer query class names and privacy-budget
*types* over full query ASTs, row data, or numeric remaining budgets.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import logging
import warnings
from typing import Optional

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

_NOISY_SPARK_LOG_LEVELS = frozenset({"ALL", "DEBUG", "INFO"})
# Mutable container avoids a ``global`` statement for the once-per-process flag.
_spark_noise_warned = [False]

_SPARK_NOISE_MESSAGE = (
    "The active SparkSession has SparkContext log level {level!r}. "
    "Spark's own logging is separate from Python's logging module, so "
    "Analytics log levels will not quiet Spark output. Consider calling "
    "spark.sparkContext.setLogLevel('ERROR') after creating the session. "
    "py4j and log4j can also produce noise independently; see the Spark "
    "deployment guide for details."
)


def reset_spark_logging_warning_for_tests() -> None:
    """Reset the once-per-process Spark noise warning (for unit tests only)."""
    _spark_noise_warned[0] = False


def warn_if_spark_logging_noisy(spark: Optional[SparkSession] = None) -> None:
    """Warn once if Spark's built-in logging looks noisy.

    Uses only an existing Spark session: the optional ``spark`` argument, or
    :meth:`SparkSession.getActiveSession`. Never calls ``getOrCreate()``.

    Emits both a :class:`UserWarning` (visible by default) and a
    ``logger.warning`` (for configured log pipelines). This dual emit is
    intentional and scoped to this setup advisory only.

    Args:
        spark: Session to inspect. If omitted, the active session is used when
            one exists.
    """
    if _spark_noise_warned[0]:
        return

    if spark is None:
        spark = SparkSession.getActiveSession()
    if spark is None:
        return

    try:
        level = spark.sparkContext.getLogLevel()
    except Exception:
        # Never fail Session creation because Spark logging could not be inspected.
        return

    if level is None:
        return

    level_name = str(level).upper()
    if level_name not in _NOISY_SPARK_LOG_LEVELS:
        return

    message = _SPARK_NOISE_MESSAGE.format(level=level_name)
    warnings.warn(message, UserWarning, stacklevel=2)
    logger.warning(message)
    _spark_noise_warned[0] = True
