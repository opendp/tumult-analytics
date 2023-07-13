"""Creates a Spark Context to use for each testing session."""


# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# TODO(#2206): Import these fixtures from core once it is rewritten

import logging
from typing import Any, Optional, Sequence
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from tmlt.core.domains.base import Domain
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import Measure, PureDP
from tmlt.core.metrics import AbsoluteDifference, Metric
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber


def quiet_py4j():
    """Remove noise in the logs irrelevant to testing."""
    print("Calling PySparkTest:suppress_py4j_logging")
    logger = logging.getLogger("py4j")
    # This is to silence py4j.java_gateway: DEBUG logs.
    logger.setLevel(logging.ERROR)


# this initializes one shared spark session for the duration of the test session.
# another option may be to set the scope to "module", which changes the duration to
# one session per module
@pytest.fixture(scope="session", name="spark")
def pyspark():
    """Setup a context to execute pyspark tests."""
    quiet_py4j()
    print("Setting up spark session.")
    spark = (
        SparkSession.builder.appName(__name__)
        .master("local[4]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.eventLog.enabled", "false")
        .config("spark.driver.allowMultipleContexts", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.default.parallelism", "5")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "16g")
        .getOrCreate()
    )
    # This is to silence pyspark logs.
    spark.sparkContext.setLogLevel("OFF")
    return spark


def assert_frame_equal_with_sort(
    first_df: pd.DataFrame,
    second_df: pd.DataFrame,
    sort_columns: Optional[Sequence[str]] = None,
    **kwargs: Any,
):
    """Asserts that the two data frames are equal.

    Wrapper around pandas test function. Both dataframes are sorted
    since the ordering in Spark is not guaranteed.

    Args:
        first_df: First dataframe to compare.
        second_df: Second dataframe to compare.
        sort_columns: Names of column to sort on. By default sorts by all columns.
        **kwargs: Keyword arguments that will be passed to assert_frame_equal().
    """
    if sorted(first_df.columns) != sorted(second_df.columns):
        raise ValueError(
            "Dataframes must have matching columns. "
            f"first_df: {sorted(first_df.columns)}. "
            f"second_df: {sorted(second_df.columns)}."
        )
    if first_df.empty and second_df.empty:
        return
    if sort_columns is None:
        sort_columns = list(first_df.columns)
    if sort_columns:
        first_df = first_df.set_index(sort_columns).sort_index().reset_index()
        second_df = second_df.set_index(sort_columns).sort_index().reset_index()
    pd.testing.assert_frame_equal(first_df, second_df, **kwargs)


def create_mock_measurement(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_measure: Measure = PureDP(),
    is_interactive: bool = False,
    return_value: Any = np.int64(0),
    privacy_function_implemented: bool = False,
    privacy_function_return_value: Any = ExactNumber(1),
    privacy_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Measurement with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_measure: Output measure for the mock.
        is_interactive: Whether the mock should be interactive.
        return_value: Return value for the Measurement's __call__.
        privacy_function_implemented: If True, raises a :class:`NotImplementedError`
            with the message "TEST" when the privacy function is called.
        privacy_function_return_value: Return value for the Measurement's privacy
            function.
        privacy_relation_return_value: Return value for the Measurement's privacy
            relation.
    """
    measurement = create_autospec(spec=Measurement, instance=True)
    measurement.input_domain = input_domain
    measurement.input_metric = input_metric
    measurement.output_measure = output_measure
    measurement.is_interactive = is_interactive
    measurement.return_value = return_value
    measurement.privacy_function.return_value = privacy_function_return_value
    measurement.privacy_relation.return_value = privacy_relation_return_value
    if not privacy_function_implemented:
        measurement.privacy_function.side_effect = NotImplementedError("TEST")
    return measurement


def create_mock_transformation(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_domain: Domain = NumpyIntegerDomain(),
    output_metric: Metric = AbsoluteDifference(),
    return_value: Any = 0,
    stability_function_implemented: bool = False,
    stability_function_return_value: Any = ExactNumber(1),
    stability_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Transformation with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_domain: Output domain for the mock.
        output_metric: Output metric for the mock.
        return_value: Return value for the Transformation's __call__.
        stability_function_implemented: If False, raises a :class:`NotImplementedError`
            with the message "TEST" when the stability function is called.
        stability_function_return_value: Return value for the Transformation's stability
            function.
        stability_relation_return_value: Return value for the Transformation's stability
            relation.
    """
    transformation = create_autospec(spec=Transformation, instance=True)
    transformation.input_domain = input_domain
    transformation.input_metric = input_metric
    transformation.output_domain = output_domain
    transformation.output_metric = output_metric
    transformation.return_value = return_value
    transformation.stability_function.return_value = stability_function_return_value
    transformation.stability_relation.return_value = stability_relation_return_value
    transformation.__or__ = Transformation.__or__
    if not stability_function_implemented:
        transformation.stability_function.side_effect = NotImplementedError("TEST")
    return transformation


def params(d):
    """Allows parameterizing tests with dictionaries.

    Examples:
    @params(
        {
            "test_case_1": {
                "arg1": value1,
                "arg2": value2,
            },
        }
    )
    test_func(...)
    """
    argnames = sorted({k for v in d.values() for k in v.keys()})
    return pytest.mark.parametrize(
        argnames=argnames,
        argvalues=[[v.get(k) for k in argnames] for v in d.values()],
        ids=d.keys(),
    )
