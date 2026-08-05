"""Unit tests for :mod:`tmlt.analytics._logging`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import logging
import warnings
from unittest.mock import MagicMock

import pytest

from tmlt.analytics import AddOneRow, PureDPBudget, Session
from tmlt.analytics._base_builder import PrivateDataFrame
from tmlt.analytics._logging import (
    reset_spark_logging_warning_for_tests,
    warn_if_spark_logging_noisy,
)
from tmlt.analytics._neighboring_relation import AddRemoveRows, Conjunction


@pytest.fixture(autouse=True)
def _reset_spark_warning():
    reset_spark_logging_warning_for_tests()
    yield
    reset_spark_logging_warning_for_tests()


def _mock_spark(log_level: str) -> MagicMock:
    spark = MagicMock()
    spark.sparkContext.getLogLevel.return_value = log_level
    return spark


def test_warn_if_spark_logging_noisy_warns_on_info(caplog) -> None:
    """INFO Spark log level triggers UserWarning and logger.warning once."""
    spark = _mock_spark("INFO")
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with pytest.warns(UserWarning, match="setLogLevel"):
            warn_if_spark_logging_noisy(spark)
        assert any("setLogLevel" in r.message for r in caplog.records)

        # Second call should be a no-op.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy(spark)
            assert caught == []
        assert sum(1 for r in caplog.records if "setLogLevel" in r.message) == 1


@pytest.mark.parametrize("level", ["ERROR", "FATAL", "OFF", "WARN", "WARNING"])
def test_warn_if_spark_logging_noisy_quiet_levels(level: str, caplog) -> None:
    """Non-noisy Spark log levels do not warn."""
    spark = _mock_spark(level)
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy(spark)
            assert caught == []
        assert caplog.records == []


@pytest.mark.parametrize("level", ["ALL", "DEBUG", "info"])
def test_warn_if_spark_logging_noisy_noisy_levels(level: str) -> None:
    """ALL/DEBUG/INFO (case-insensitive) trigger the advisory."""
    spark = _mock_spark(level)
    with pytest.warns(UserWarning, match="SparkContext log level"):
        warn_if_spark_logging_noisy(spark)


def test_warn_if_spark_logging_noisy_noop_without_session(monkeypatch, caplog) -> None:
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


def test_warn_if_spark_logging_noisy_swallows_get_log_level_errors(caplog) -> None:
    """Failures inspecting Spark log level must not propagate."""
    spark = MagicMock()
    spark.sparkContext.getLogLevel.side_effect = RuntimeError("unavailable")
    with caplog.at_level(logging.WARNING, logger="tmlt.analytics._logging"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_spark_logging_noisy(spark)
            assert caught == []
        assert caplog.records == []


def test_session_build_logs_info(caplog, monkeypatch) -> None:
    """Session.Builder.build logs an INFO message with the budget type."""
    mock_sess = MagicMock()
    monkeypatch.setattr(
        Session,
        "_from_neighboring_relation",
        classmethod(lambda cls, *args, **kwargs: mock_sess),
    )
    monkeypatch.setattr(
        "tmlt.analytics.session._generate_neighboring_relation",
        lambda sources: Conjunction(AddRemoveRows("data", 1)),
    )
    monkeypatch.setattr(
        "tmlt.analytics.session.warn_if_spark_logging_noisy", lambda: None
    )

    builder = Session.Builder().with_privacy_budget(PureDPBudget(1))
    builder._DataFrameMixin__private_dataframes["data"] = PrivateDataFrame(
        dataframe=MagicMock(), protected_change=AddOneRow()
    )

    with caplog.at_level(logging.INFO, logger="tmlt.analytics.session"):
        result = builder.build()

    assert result is mock_sess
    assert any(
        "Created Session with privacy budget type PureDPBudget" in r.message
        for r in caplog.records
    )


def test_evaluate_logs_debug(caplog, monkeypatch) -> None:
    """Session.evaluate logs DEBUG start/finish messages."""
    sess = Session.__new__(Session)
    sess._accountant = MagicMock()
    sess._accountant.privacy_budget = object()
    sess._accountant.d_in = object()
    sess._accountant.measure.return_value = "answer"
    sess._activate_accountant = MagicMock()  # type: ignore[method-assign]

    query = type("GroupByCount", (), {})()
    query_wrapper = MagicMock()
    query_wrapper._query_expr = query

    measurement = MagicMock()
    measurement.privacy_relation.return_value = True
    adjusted_budget = MagicMock()
    adjusted_budget.value = object()

    monkeypatch.setattr(
        sess,
        "_compile_and_get_info",
        lambda *args, **kwargs: (measurement, adjusted_budget, None),
    )
    monkeypatch.setattr(
        "tmlt.analytics.session.check_type", lambda *args, **kwargs: None
    )

    with caplog.at_level(logging.DEBUG, logger="tmlt.analytics.session"):
        result = sess.evaluate(query_wrapper, PureDPBudget(0.5))

    assert result == "answer"
    messages = [r.message for r in caplog.records]
    assert any(m.startswith("Evaluating query of type GroupByCount") for m in messages)
    assert any(
        m.startswith("Finished evaluating query of type GroupByCount") for m in messages
    )
