"""Unit tests for :mod:`tmlt.analytics._logging`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import logging
import warnings
from unittest.mock import MagicMock

import pytest

from tmlt.analytics._logging import (
    _reset_spark_logging_warning_for_tests,
    warn_if_spark_logging_noisy,
)


@pytest.fixture(autouse=True)
def _reset_spark_warning():
    _reset_spark_logging_warning_for_tests()
    yield
    _reset_spark_logging_warning_for_tests()


@pytest.fixture
def mock_spark(request: pytest.FixtureRequest) -> MagicMock:
    """Spark session mock with a configurable SparkContext log level.

    Parametrize with ``@pytest.mark.parametrize("mock_spark", [...], indirect=True)``.
    Defaults to ``"INFO"`` when not parametrized.
    """
    log_level = getattr(request, "param", "INFO")
    spark = MagicMock()
    spark.sparkContext.getLogLevel.return_value = log_level
    return spark


@pytest.mark.parametrize("mock_spark", ["ALL", "DEBUG", "INFO", "info"], indirect=True)
def test_warn_if_spark_logging_noisy_warns_once(
    mock_spark: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Noisy Spark levels trigger UserWarning and logger.warning once."""
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with pytest.warns(UserWarning, match="setLogLevel"):
            warn_if_spark_logging_noisy(mock_spark)
        assert any("setLogLevel" in r.message for r in caplog.records)
        assert any("docs.tmlt.dev" in r.message for r in caplog.records)

        # Second call should be a no-op.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy(mock_spark)
            assert caught == []
        assert sum(1 for r in caplog.records if "setLogLevel" in r.message) == 1


@pytest.mark.parametrize(
    "mock_spark",
    ["ERROR", "FATAL", "OFF", "WARN", "WARNING"],
    indirect=True,
)
def test_warn_if_spark_logging_noisy_quiet_levels(
    mock_spark: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-noisy Spark log levels do not warn."""
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy(mock_spark)
            assert caught == []
        assert caplog.records == []


def test_warn_if_spark_logging_noisy_noop_without_session(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No active session and no argument means no warning."""
    monkeypatch.setattr(
        "tmlt.analytics._logging.SparkSession.getActiveSession", lambda: None
    )
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy()
            assert caught == []
        assert caplog.records == []


def test_warn_if_spark_logging_noisy_swallows_get_log_level_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failures inspecting Spark log level must not propagate."""
    spark = MagicMock()
    spark.sparkContext.getLogLevel.side_effect = RuntimeError("unavailable")
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy(spark)
            assert caught == []
        assert caplog.records == []
