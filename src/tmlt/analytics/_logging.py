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

Contributor guidelines for levels, channels, and log-once behavior live in
``CONTRIBUTING.md`` (Logging section). This module documents the same channel
rules briefly for discoverability next to the helpers.

Output channels
---------------

Use the right channel for the job:

* ``print`` — intentional interactive UX (for example ``Session.describe`` and
  ``check_installation``).
* ``warnings.warn`` — advisories that must be visible without configuring
  logging (the package ``NullHandler`` prevents stdlib lastResort from
  emitting library WARNING to stderr).
* ``logging`` — diagnostic / lifecycle messages (DEBUG / INFO / WARNING).

Never log row data, query ASTs, or PII. Coarseness is operator hygiene, not a
privacy control. Budget *values* may appear at DEBUG; Session INFO stays
budget *type* only.

Lint vs review
--------------

Ruff enforces *mechanical* conventions in CI (``LOG``, ``G``, and ``TID251``
banned-api for ``logging.basicConfig`` / ``dictConfig`` / ``fileConfig``, plus
``loguru`` / ``structlog``). A unit test bans ``logger.error`` /
``logger.exception`` under ``src/tmlt/analytics`` so failure stays raise-only.
Other judgment calls — which of DEBUG/INFO/WARNING to use, channel choice
(``print`` / ``warnings`` / ``logging``), and message coarseness — are
documented in ``CONTRIBUTING.md`` and reviewed in PRs.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import logging
import warnings

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

_NOISY_SPARK_LOG_LEVELS = frozenset({"ALL", "DEBUG", "INFO"})
_spark_noise_warned = False

_SPARK_DOCS_URL = (
    "https://docs.tmlt.dev/analytics/latest/deployment/spark.html#spark-logging"
)

_SPARK_NOISE_MESSAGE = (
    "The active SparkSession has SparkContext log level {level!r}. "
    "Spark's own logging is separate from Python's logging module, so "
    "Analytics log levels will not quiet Spark output. Consider calling "
    "spark.sparkContext.setLogLevel('ERROR') after creating the session. "
    "py4j and log4j can also produce noise independently; see the Spark "
    f"deployment guide for details: {_SPARK_DOCS_URL}"
)


def _reset_spark_logging_warning_for_tests() -> None:
    """Reset the once-per-process Spark noise warning (for unit tests only)."""
    global _spark_noise_warned  # noqa: PLW0603
    _spark_noise_warned = False


def warn_if_spark_logging_noisy() -> None:
    """Warn once if Spark's built-in logging looks noisy.

    Uses :meth:`SparkSession.getActiveSession` only. Never calls
    ``getOrCreate()``.

    Emits a :class:`UserWarning` (visible without logging config) and a
    matching ``logger.warning`` (for configured log pipelines). Dual-emit is
    scoped to this setup advisory: a ``NullHandler`` on ``tmlt.analytics``
    means lastResort does not print library WARNING to stderr.
    """
    global _spark_noise_warned  # noqa: PLW0603
    if _spark_noise_warned:
        return

    spark = SparkSession.getActiveSession()
    if spark is None:
        return

    try:
        # PySpark exposes getLogLevel at runtime; stubs often omit it.
        get_log_level = getattr(spark.sparkContext, "getLogLevel", None)
        if not callable(get_log_level):
            return
        level = get_log_level()
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
    _spark_noise_warned = True
