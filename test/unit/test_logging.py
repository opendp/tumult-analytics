"""Unit tests for :mod:`tmlt.analytics._logging`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tmlt.analytics._logging import (
    _reset_spark_logging_warning_for_tests,
    warn_if_spark_logging_noisy,
)

_ANALYTICS_SRC = Path(__file__).resolve().parents[2] / "src" / "tmlt" / "analytics"
_BANNED_LOG_METHODS = frozenset({"error", "exception"})


@pytest.fixture(autouse=True)
def _reset_spark_warning():
    _reset_spark_logging_warning_for_tests()
    yield
    _reset_spark_logging_warning_for_tests()


@pytest.fixture
def mock_spark(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Spark session mock installed as the active session.

    Parametrize with ``@pytest.mark.parametrize("mock_spark", [...], indirect=True)``.
    Defaults to ``"INFO"`` when not parametrized.
    """
    log_level = getattr(request, "param", "INFO")
    spark = MagicMock()
    spark.sparkContext.getLogLevel.return_value = log_level
    monkeypatch.setattr(
        "tmlt.analytics._logging.SparkSession.getActiveSession", lambda: spark
    )
    return spark


@pytest.mark.parametrize("mock_spark", ["ALL", "DEBUG", "INFO", "info"], indirect=True)
def test_warn_if_spark_logging_noisy_warns_once(
    mock_spark: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Noisy Spark levels trigger UserWarning and logger.warning once."""
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with pytest.warns(UserWarning, match="setLogLevel"):
            warn_if_spark_logging_noisy()
        assert any("setLogLevel" in r.message for r in caplog.records)
        assert any("docs.tmlt.dev" in r.message for r in caplog.records)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy()
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
            warn_if_spark_logging_noisy()
            assert caught == []
        assert caplog.records == []


def test_warn_if_spark_logging_noisy_noop_without_session(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No active session means no warning."""
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
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failures inspecting Spark log level must not propagate."""
    spark = MagicMock()
    spark.sparkContext.getLogLevel.side_effect = RuntimeError("unavailable")
    monkeypatch.setattr(
        "tmlt.analytics._logging.SparkSession.getActiveSession", lambda: spark
    )
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy()
            assert caught == []
        assert caplog.records == []


def test_no_library_error_level_logging() -> None:
    """Library call sites must not use logger.error / logger.exception.

    Failures are signaled by raising (see CONTRIBUTING.md Logging). Applications
    may log at ERROR when they catch at their boundary.
    """
    violations: list[str] = []
    for path in sorted(_ANALYTICS_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _BANNED_LOG_METHODS:
                continue
            # Match logger.error(...) / logging.error(...) style call sites.
            if isinstance(func.value, ast.Name) and func.value.id in {
                "logger",
                "logging",
            }:
                rel = path.relative_to(_ANALYTICS_SRC.parent.parent.parent)
                violations.append(f"{rel}:{node.lineno}: {func.value.id}.{func.attr}")
    assert violations == [], (
        "ERROR-level logging is banned in tmlt.analytics; raise instead:\n"
        + "\n".join(violations)
    )
