"""Creates a Spark Context to use for each testing session."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

# TODO(#2206): Import these fixtures from core once it is rewritten

import logging
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union, cast, overload
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import Measure, PureDP
from tmlt.core.metrics import AbsoluteDifference, Metric
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics import (
    ApproxDPBudget,
    BinningSpec,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PrivacyBudget,
    PureDPBudget,
    QueryBuilder,
    RhoZCDPBudget,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.truncation_strategy import TruncationStrategy

SIMPLE_TRANSFORMATION_QUERIES = [
    QueryBuilder("private_data").rename({"D": "new"}),
    QueryBuilder("private_data").filter("C>1"),
    QueryBuilder("private_data").select(["A", "B", "C"]),
    QueryBuilder("private_data").map(
        f=lambda row: {"F": 1},
        new_column_types=Schema({"F": "INTEGER"}),
        augment=True,
    ),
    QueryBuilder("private_data").flat_map(
        f=lambda row: [{"F": 1}],
        new_column_types=Schema({"F": "INTEGER"}),
        augment=True,
        max_rows=2,
    ),
    QueryBuilder("private_data").join_private(
        "join_private_data",
        truncation_strategy_left=TruncationStrategy.DropExcess(1),
        truncation_strategy_right=TruncationStrategy.DropNonUnique(),
    ),
    QueryBuilder("private_data").join_public("join_public_data"),
    QueryBuilder("private_data").replace_null_and_nan(),
    QueryBuilder("private_data").replace_infinity({"C": (-100, 100)}),
    QueryBuilder("private_data").drop_null_and_nan(["C"]),
    QueryBuilder("private_data").drop_infinity(["C"]),
    QueryBuilder("private_data").bin_column(
        column="A", spec=BinningSpec(bin_edges=[1000 * i for i in range(0, 10)])
    ),
]

KEY_SET = KeySet.from_dict(
    {
        "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True).tolist(),
        "B": np.random.choice(np.arange(0, 100, 1), 100, replace=True).tolist(),
    }
)
KEY_SET.cache()


GROUPBY_AGGREGATION_QUERIES = [
    lambda x: x.groupby(KEY_SET).count("measure_col"),
    lambda x: x.groupby(KEY_SET).sum("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).variance("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).stdev("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).min("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).max("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).median("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).count_distinct(name="measure_col"),
    lambda x: x.groupby(KEY_SET).quantile(
        "C", 0.5, low=0, high=1000, name="measure_col"
    ),
    lambda x: x.groupby(KEY_SET).count(name="measure_col").suppress(1),
    # TODO(#3342): Enable after get_bounds core slowness is fixed
    # lambda x: x.groupby(KEY_SET).get_bounds("C", lower_bound_column="measure_col"),
]

NON_GROUPBY_AGGREGATION_QUERIES = [
    QueryBuilder("private_data").count(name="measure_col"),
    QueryBuilder("private_data").count_distinct(columns=["A", "B"], name="measure_col"),
    QueryBuilder("private_data").quantile("A", 0.5, 0, 1000, name="measure_col"),
    QueryBuilder("private_data").min("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").max("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").median("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").sum("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").average("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").variance("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").stdev("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").get_groups(["A"]),
    QueryBuilder("private_data").histogram(
        column="A",
        bin_edges=BinningSpec(bin_edges=[1000 * i for i in range(0, 10)]),
        name="measure_col",
    ),
    # TODO(#3342): Enable after get_bounds core slowness is fixed
    # QueryBuilder("private_data").get_bounds("C", lower_bound_column="measure_col"),
]

ID_QUERIES = [
    QueryBuilder("id_a_private_data").enforce(MaxRowsPerID(1)).count(),
    QueryBuilder("id_a_private_data")
    .enforce(MaxRowsPerID(100))
    .filter("id >= 2")
    .groupby(KEY_SET)
    .count("measure_col"),
    QueryBuilder("id_a_private_data")
    .enforce(MaxGroupsPerID("X", 1))
    .enforce(MaxRowsPerGroupPerID("X", 1))
    .count(),
    QueryBuilder("id_a_private_data")
    .flat_map_by_id(
        lambda rows: [{"per_id_sum": sum(r["A"] for r in rows)}],
        new_column_types={
            "per_id_sum": ColumnDescriptor(
                ColumnType.INTEGER,
                allow_null=False,
            )
        },
    )
    .enforce(MaxRowsPerID(1))
    .sum("per_id_sum", low=0, high=5, name="sum"),
]


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
        SparkSession.builder.appName("analytics-test")
        .master("local[4]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.eventLog.enabled", "false")
        .config("spark.driver.allowMultipleContexts", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.default.parallelism", "5")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "16g")
        .config("spark.port.maxRetries", "30")
        .config("spark.sql.shuffle.partitions", "1")
        # Disable Spark UI / Console display
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.dagGraph.retainedRootRDDs", "1")
        .config("spark.ui.retainedJobs", "1")
        .config("spark.ui.retainedStages", "1")
        .config("spark.ui.retainedTasks", "1")
        .config("spark.sql.ui.retainedExecutions", "1")
        .config("spark.worker.ui.retainedExecutors", "1")
        .config("spark.worker.ui.retainedDrivers", "1")
        .getOrCreate()
    )
    # This is to silence pyspark logs.
    spark.sparkContext.setLogLevel("OFF")
    return spark


@pytest.fixture(scope="function", name="spark_with_progress")
def pyspark_with_progress():
    """A context to execute pyspark tests, with spark.ui.showConsoleProgress enabled."""
    quiet_py4j()
    print("Setting up spark session.")
    spark = (
        SparkSession.builder.appName("analytics-test-with-progress")
        .master("local[4]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.eventLog.enabled", "false")
        .config("spark.driver.allowMultipleContexts", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.default.parallelism", "5")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "16g")
        .config("spark.port.maxRetries", "30")
        .config("spark.sql.shuffle.partitions", "1")
        # Disable Spark UI / Console display
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.dagGraph.retainedRootRDDs", "1")
        .config("spark.ui.retainedJobs", "1")
        .config("spark.ui.retainedStages", "1")
        .config("spark.ui.retainedTasks", "1")
        .config("spark.sql.ui.retainedExecutions", "1")
        .config("spark.worker.ui.retainedExecutors", "1")
        .config("spark.worker.ui.retainedDrivers", "1")
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
    """Asserts that the two Pandas DataFrames are equal.

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
            "DataFrames must have matching columns. "
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
    # We explicitly pass check_dtype=False the equality check, so that identical
    # DataFrames which differ only in dtypes (like one with an int64 column and
    # the other with an Int64 column) are considered equal.
    pd.testing.assert_frame_equal(first_df, second_df, check_dtype=False, **kwargs)


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


T = TypeVar("T", bound=PrivacyBudget)


def assert_approx_equal_budgets(
    budget1: T, budget2: T, atol: float = 1e-8, rtol: float = 1e-5
):
    """Asserts that two budgets are approximately equal.

    Args:
        budget1: The first budget.
        budget2: The second budget.
        atol: The absolute tolerance for the comparison.
        rtol: The relative tolerance for the comparison.
    """
    if not isinstance(budget1, type(budget2)) or not isinstance(budget2, type(budget1)):
        raise AssertionError(
            f"Budgets are not of the same type: {type(budget1)} and {type(budget2)}"
        )
    if isinstance(budget1, PureDPBudget) and isinstance(budget2, PureDPBudget):
        if not np.allclose(budget1.epsilon, budget2.epsilon, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Epsilon values are not approximately equal: {budget1} and {budget2}"
            )
        return
    if isinstance(budget1, ApproxDPBudget) and isinstance(budget2, ApproxDPBudget):
        if not np.allclose(budget1.epsilon, budget2.epsilon, atol=atol, rtol=rtol):
            raise AssertionError(
                "Epsilon values are not approximately equal: "
                f"{budget1.epsilon} and {budget2.epsilon}"
            )
        if not np.allclose(budget1.delta, budget2.delta, atol=atol, rtol=rtol):
            raise AssertionError(
                "Delta values are not approximately equal: "
                f"{budget1.delta} and {budget2.delta}"
            )
        return
    if isinstance(budget1, RhoZCDPBudget) and isinstance(budget2, RhoZCDPBudget):
        if not np.allclose(budget1.rho, budget2.rho, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Rho values are not approximately equal: "
                f"{budget1.rho} and {budget2.rho}"
            )
        return
    raise AssertionError(f"Budget type not recognized: {type(budget1)}")


@overload
def create_empty_input(domain: DictDomain) -> Dict:
    ...


@overload
def create_empty_input(domain: SparkDataFrameDomain) -> DataFrame:
    ...


def create_empty_input(domain):  # pylint: disable=missing-type-doc
    """Returns an empty input for a given domain.

    Args:
        domain: The domain for which to create an empty input.
    """
    spark = SparkSession.builder.getOrCreate()
    if isinstance(domain, DictDomain):
        return {
            k: create_empty_input(cast(Union[DictDomain, SparkDataFrameDomain], v))
            for k, v in domain.key_to_domain.items()
        }
    if isinstance(domain, SparkDataFrameDomain):
        # TODO(#3092): the row is only necessary b/c of a bug in core for empty dfs
        row: List[Any] = []
        for field in domain.spark_schema.fields:
            if field.dataType.simpleString() == "string":
                row.append("")
            elif field.dataType.simpleString() == "integer":
                row.append(0)
            elif field.dataType.simpleString() == "double":
                row.append(0.0)
            elif field.dataType.simpleString() == "boolean":
                row.append(False)
            elif field.dataType.simpleString() == "bigint":
                row.append(0)
            else:
                raise ValueError(
                    f"Unsupported field type: {field.dataType.simpleString()}"
                )
        return spark.createDataFrame([row], domain.spark_schema)
    raise ValueError(f"Unsupported domain type: {type(domain)}")


def pyspark_schema_from_pandas(df: pd.DataFrame) -> StructType:
    """Create a pyspark schema corresponding to a pandas dataframe."""

    def convert_type(dtype):
        if dtype == np.int64:
            return LongType()
        elif dtype == float:
            return FloatType()
        elif dtype == str:
            return StringType()
        raise NotImplementedError("Type not implemented yet.")

    return StructType(
        [
            StructField(colname, convert_type(dtype))
            for colname, dtype in df.dtypes.items()
        ]
    )


@pytest.fixture(scope="module")
def _session_data(spark):
    base_private_data = pd.DataFrame(
        {
            "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "B": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "C": np.random.choice(np.arange(0, 100, 0.5), 100, replace=True),
            "D": np.random.choice(np.arange(0, 100, 0.5), 100, replace=True),
        }
    )
    private_id_data = pd.DataFrame(
        [
            [1, 4, 100, "X"],
            [1, 5, 100, "Y"],
            [1, 6, 100, "X"],
            [2, 7, 100, "Y"],
            [3, 8, 100, "X"],
            [3, 9, 100, "Y"],
        ],
        columns=["id", "A", "B", "X"],
    )
    join_private_data = pd.DataFrame(
        {
            "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "Y": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
        }
    )
    join_public_data = pd.DataFrame(
        {
            "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "Z": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
        }
    )
    private_sdf = spark.createDataFrame(base_private_data)
    private_id_sdf = spark.createDataFrame(private_id_data)
    join_private_sdf = spark.createDataFrame(join_private_data)
    join_public_sdf = spark.createDataFrame(join_public_data)
    return {
        "private_data": private_sdf,
        "private_id_data": private_id_sdf,
        "join_private_data": join_private_sdf,
        "join_public_data": join_public_sdf,
    }
