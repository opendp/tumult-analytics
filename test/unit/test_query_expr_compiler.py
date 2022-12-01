# type: ignore[attr-defined]
"""Tests for QueryExprCompiler."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
# pylint: disable= no-member, protected-access, no-self-use

import datetime
from typing import List, Union
from unittest.mock import patch

import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import SparkSession
from pyspark.sql.functions import col  # pylint: disable=no-name-in-module
from pyspark.sql.types import (
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler import QueryExprCompiler
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_expr import (
    AverageMechanism,
    CountMechanism,
    Filter,
    FlatMap,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    JoinPrivate,
    JoinPublic,
    Map,
    PrivateSource,
    QueryExpr,
    Rename,
    ReplaceNullAndNan,
    Select,
    StdevMechanism,
    SumMechanism,
    VarianceMechanism,
)
from tmlt.analytics.truncation_strategy import TruncationStrategy
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric, SymmetricDifference

from ..conftest import (  # pylint: disable=no-name-in-module
    assert_frame_equal_with_sort,
    create_mock_measurement,
)

GROUPBY_TWO_COLUMNS = pd.DataFrame([["0", 0], ["0", 1], ["1", 1]], columns=["A", "B"])
GROUPBY_TWO_SCHEMA = StructType(
    [StructField("A", StringType(), False), StructField("B", LongType(), False)]
)
GET_GROUPBY_TWO = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    GROUPBY_TWO_COLUMNS, schema=GROUPBY_TWO_SCHEMA
)
GROUPBY_ONE_COLUMN = pd.DataFrame([["0"], ["1"], ["2"]], columns=["A"])
GROUPBY_ONE_SCHEMA = StructType([StructField("A", StringType(), False)])
GET_GROUPBY_ONE = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    GROUPBY_ONE_COLUMN, schema=GROUPBY_ONE_SCHEMA
)


QUERY_EXPR_COMPILER_TESTS = [
    (  # Total
        [
            GroupByCount(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({}),
                output_column="total",
            )
        ],
        [pd.DataFrame({"total": [4]})],
    ),
    (
        [
            GroupByCountDistinct(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({}),
                output_column="total",
            )
        ],
        [pd.DataFrame({"total": [4]})],
    ),
    (  # Full marginal from domain description
        [
            GroupByCount(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
            )
        ],
        [
            pd.DataFrame(
                {"A": ["0", "0", "1", "1"], "B": [0, 1, 0, 1], "count": [2, 1, 1, 0]}
            )
        ],
    ),
    (  # Incomplete two-column marginal with a dataframe
        [
            GroupByCount(
                child=PrivateSource("private"),
                # pylint: disable=protected-access
                groupby_keys=KeySet(dataframe=GET_GROUPBY_TWO),
            )
        ],
        [pd.DataFrame({"A": ["0", "0", "1"], "B": [0, 1, 1], "count": [2, 1, 0]})],
    ),
    (  # One-column marginal with additional value
        [
            GroupByCount(
                child=PrivateSource("private"),
                # pylint: disable=protected-access
                groupby_keys=KeySet(dataframe=GET_GROUPBY_ONE),
            )
        ],
        [pd.DataFrame({"A": ["0", "1", "2"], "count": [3, 1, 0]})],
    ),
    (  # BoundedAverage
        [
            GroupByBoundedAverage(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                measure_column="X",
                low=0.0,
                high=1.0,
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "average": [0.666667, 1.0]})],
    ),
    (  # BoundedSTDEV
        [
            GroupByBoundedSTDEV(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                measure_column="X",
                low=0.0,
                high=1.0,
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "stdev": [0.471405, 0.5]})],
    ),
    (  # BoundedVariance
        [
            GroupByBoundedVariance(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                measure_column="X",
                low=0.0,
                high=1.0,
                output_column="var",
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "var": [0.22222, 0.25]})],
    ),
    (  # BoundedSum
        [
            GroupByBoundedSum(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                measure_column="X",
                low=0.0,
                high=1.0,
                output_column="sum",
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "sum": [2.0, 1.0]})],
    ),
    (  # Marginal over A, Marginal over B
        [
            GroupByCount(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
            ),
            GroupByCount(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"B": [0, 1]}),
            ),
        ],
        [
            pd.DataFrame({"A": ["0", "1"], "count": [3, 1]}),
            pd.DataFrame({"B": [0, 1], "count": [3, 1]}),
        ],
    ),
    (  # FlatMap
        [
            GroupByBoundedSum(
                child=ReplaceNullAndNan(
                    replace_with={},
                    child=FlatMap(
                        child=PrivateSource("private"),
                        f=lambda row: [{}, {}],
                        max_num_rows=2,
                        schema_new_columns=Schema({}),
                        augment=True,
                    ),
                ),
                groupby_keys=KeySet.from_dict({}),
                measure_column="X",
                low=0.0,
                high=3.0,
            )
        ],
        [pd.DataFrame({"sum": [12.0]})],
    ),
    (  # Multiple flat maps
        [
            GroupByBoundedSum(
                child=ReplaceNullAndNan(
                    replace_with={},
                    child=FlatMap(
                        child=FlatMap(
                            child=PrivateSource("private"),
                            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
                            max_num_rows=1,
                            schema_new_columns=Schema({"Repeat": "INTEGER"}),
                            augment=True,
                        ),
                        f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
                        max_num_rows=2,
                        schema_new_columns=Schema({"i": "DECIMAL"}),
                        augment=False,
                    ),
                ),
                groupby_keys=KeySet.from_dict({}),
                measure_column="i",
                low=0.0,
                high=3.0,
            )
        ],
        [pd.DataFrame({"sum": [9.0]})],
    ),
    (  # Grouping flat map
        [
            GroupByBoundedSum(
                child=ReplaceNullAndNan(
                    replace_with={},
                    child=FlatMap(
                        child=FlatMap(
                            child=PrivateSource("private"),
                            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
                            max_num_rows=1,
                            schema_new_columns=Schema(
                                {"Repeat": "INTEGER"}, grouping_column="Repeat"
                            ),
                            augment=True,
                        ),
                        f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
                        max_num_rows=2,
                        schema_new_columns=Schema({"i": "DECIMAL"}),
                        augment=True,
                    ),
                ),
                groupby_keys=KeySet.from_dict({"Repeat": [1, 2]}),
                measure_column="i",
                low=0.0,
                high=3.0,
            )
        ],
        [pd.DataFrame({"Repeat": [1, 2], "sum": [3.0, 6.0]})],
    ),
    (  # Filter
        [
            GroupByCount(
                child=Filter(child=PrivateSource("private"), predicate="A == '0'"),
                groupby_keys=KeySet.from_dict({}),
            )
        ],
        [pd.DataFrame({"count": [3]})],
    ),
    (  # Rename
        [
            GroupByCount(
                child=Rename(child=PrivateSource("private"), column_mapper={"A": "Z"}),
                groupby_keys=KeySet.from_dict({"Z": ["0", "1"]}),
            )
        ],
        [pd.DataFrame({"Z": ["0", "1"], "count": [3, 1]})],
    ),
    (  # Select
        [
            GroupByCount(
                child=Select(child=PrivateSource("private"), columns=["A"]),
                groupby_keys=KeySet.from_dict({}),
            )
        ],
        [pd.DataFrame({"count": [4]})],
    ),
    (  # Map
        [
            GroupByCount(
                child=ReplaceNullAndNan(
                    replace_with={},
                    child=Map(
                        child=PrivateSource("private"),
                        f=lambda row: {"C": 2 * str(row["B"])},
                        schema_new_columns=Schema({"C": "VARCHAR"}),
                        augment=True,
                    ),
                ),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"], "C": ["00", "11"]}),
            )
        ],
        [
            pd.DataFrame(
                [["0", "00", 2], ["0", "11", 1], ["1", "00", 1], ["1", "11", 0]],
                columns=["A", "C", "count"],
            )
        ],
    ),
    (  # JoinPublic
        [
            GroupByCount(
                child=JoinPublic(child=PrivateSource("private"), public_table="public"),
                groupby_keys=KeySet.from_dict({"A+B": [0, 1, 2]}),
            )
        ],
        [pd.DataFrame({"A+B": [0, 1, 2], "count": [2, 2, 0]})],
    ),
    (  # JoinPublic with One Join Column
        [
            GroupByCount(
                child=JoinPublic(
                    child=PrivateSource("private"),
                    public_table="public",
                    join_columns=["A"],
                ),
                groupby_keys=KeySet.from_dict({"A+B": [0, 1, 2]}),
            )
        ],
        [pd.DataFrame({"A+B": [0, 1, 2], "count": [3, 4, 1]})],
    ),
    # Tests on less-common data types
    (
        [
            GroupByCount(
                JoinPublic(PrivateSource("private"), "dtypes"),
                KeySet.from_dict({"A": ["0", "1"]}),
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "count": [3, 1]})],
    ),
    (
        [
            GroupByCount(
                Filter(
                    JoinPublic(PrivateSource("private"), "dtypes"),
                    "date < '2022-01-02'",
                ),
                KeySet.from_dict({"A": ["0", "1"]}),
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "count": [3, 0]})],
    ),
    (
        [
            GroupByCount(
                Filter(
                    JoinPublic(PrivateSource("private"), "dtypes"),
                    "date = '2022-01-02'",
                ),
                KeySet.from_dict({"A": ["0", "1"]}),
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "count": [0, 1]})],
    ),
    (
        [
            GroupByCount(
                Filter(
                    JoinPublic(PrivateSource("private"), "dtypes"),
                    "timestamp < '2022-01-01T12:40:00'",
                ),
                KeySet.from_dict({"A": ["0", "1"]}),
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "count": [3, 0]})],
    ),
    (
        [
            GroupByCount(
                Filter(
                    JoinPublic(PrivateSource("private"), "dtypes"),
                    "timestamp >= '2022-01-01T12:45:00'",
                ),
                KeySet.from_dict({"A": ["0", "1"]}),
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "count": [0, 1]})],
    ),
    (
        [
            GroupByBoundedSum(
                ReplaceNullAndNan(
                    replace_with={},
                    child=Map(
                        JoinPublic(PrivateSource("private"), "dtypes"),
                        lambda row: {"day": row["date"].day},
                        Schema({"day": ColumnDescriptor(ColumnType.INTEGER)}),
                        augment=True,
                    ),
                ),
                KeySet.from_dict({"A": ["0", "1"]}),
                "day",
                0,
                2,
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "sum": [3, 2]})],
    ),
    (
        [
            GroupByBoundedSum(
                ReplaceNullAndNan(
                    replace_with={},
                    child=Map(
                        JoinPublic(PrivateSource("private"), "dtypes"),
                        lambda row: {"minute": row["timestamp"].minute},
                        Schema({"minute": ColumnDescriptor(ColumnType.INTEGER)}),
                        augment=True,
                    ),
                ),
                KeySet.from_dict({"A": ["0", "1"]}),
                "minute",
                0,
                59,
            )
        ],
        [pd.DataFrame({"A": ["0", "1"], "sum": [90, 45]})],
    ),
]

# Shorthands for some values used in tests
_DATE1 = datetime.date.fromisoformat("2022-01-01")
_DATE2 = datetime.date.fromisoformat("2022-01-02")
_TIMESTAMP1 = datetime.datetime.fromisoformat("2022-01-01T12:30:00")
_TIMESTAMP2 = datetime.datetime.fromisoformat("2022-01-01T12:45:00")


@pytest.fixture(name="test_data", scope="class")
def setup(spark, request) -> None:
    "Set up test data."
    sdf = spark.createDataFrame(
        pd.DataFrame(
            [["0", 0, 0.0], ["0", 0, 1.0], ["0", 1, 2.0], ["1", 0, 3.0]],
            columns=["A", "B", "X"],
        ),
        schema=StructType(
            [
                StructField("A", StringType(), False),
                StructField("B", LongType(), False),
                StructField("X", DoubleType(), False),
            ]
        ),
    )
    request.cls.sdf = sdf

    join_df = spark.createDataFrame(
        pd.DataFrame(
            [["0", 0, 0], ["0", 1, 1], ["1", 0, 1], ["1", 1, 2]],
            columns=["A", "B", "A+B"],
        ),
        schema=StructType(
            [
                StructField("A", StringType(), False),
                StructField("B", LongType(), False),
                StructField("A+B", LongType(), False),
            ]
        ),
    )
    request.cls.join_df = join_df

    dtypes_df = spark.createDataFrame(
        pd.DataFrame(
            [["0", 0, 0.1, _DATE1, _TIMESTAMP1], ["1", 1, 0.2, _DATE2, _TIMESTAMP2]]
        ),
        schema=StructType(
            [
                StructField("A", StringType(), False),
                StructField("int", LongType(), False),
                StructField("float", DoubleType(), False),
                StructField("date", DateType(), False),
                StructField("timestamp", TimestampType(), False),
            ]
        ),
    )
    request.cls.dtypes_df = dtypes_df

    groupby_two_columns_df = spark.createDataFrame(
        pd.DataFrame([["0", 0], ["0", 1], ["1", 1]], columns=["A", "B"]),
        schema=StructType(
            [StructField("A", StringType(), False), StructField("B", LongType(), False)]
        ),
    )
    request.cls.groupby_two_columns_df = groupby_two_columns_df

    groupby_one_column_df = spark.createDataFrame(
        pd.DataFrame([["0"], ["1"], ["2"]], columns=["A"]),
        schema=StructType([StructField("A", StringType(), False)]),
    )
    request.cls.groupby_one_column_df = groupby_one_column_df

    stability = {"private": sp.Integer(3), "private_2": sp.Integer(3)}

    request.cls.stability = stability

    input_domain = DictDomain(
        {
            "private": SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkIntegerColumnDescriptor(),
                    "X": SparkFloatColumnDescriptor(),
                }
            ),
            "private_2": SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "C": SparkIntegerColumnDescriptor(),
                }
            ),
        }
    )

    request.cls.input_domain = input_domain

    catalog = Catalog()
    catalog.add_private_source(
        "private",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "X": ColumnDescriptor(ColumnType.DECIMAL),
        },
        stability=3,
    )
    catalog.add_private_view(
        "private_2",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "C": ColumnDescriptor(ColumnType.INTEGER),
        },
        stability=3,
    )
    catalog.add_public_source(
        "public",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "A+B": ColumnDescriptor(ColumnType.INTEGER),
        },
    )
    catalog.add_public_source(
        "dtypes",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "int": ColumnDescriptor(ColumnType.INTEGER),
            "float": ColumnDescriptor(
                ColumnType.DECIMAL, allow_nan=True, allow_inf=True
            ),
            "date": ColumnDescriptor(ColumnType.DATE),
            "timestamp": ColumnDescriptor(ColumnType.TIMESTAMP),
        },
    )
    catalog.add_public_source(
        "groupby_two_columns",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    )
    catalog.add_public_source(
        "groupby_one_column", {"A": ColumnDescriptor(ColumnType.VARCHAR)}
    )

    request.cls.catalog = catalog

    input_metric = DictMetric(
        {"private": SymmetricDifference(), "private_2": SymmetricDifference()}
    )
    request.cls.input_metric = input_metric

    request.cls.compiler = QueryExprCompiler()


@pytest.mark.usefixtures("test_data")
class TestQueryExprCompiler:
    """Unit tests for class QueryExprCompiler.

    Tests :class:`~tmlt.analytics._query_expr_compiler.QueryExprCompiler`.
    """

    @pytest.mark.parametrize(
        "query_expr,expected",
        [
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    output_column="total",
                ),
                pd.DataFrame({"total": [4]}),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    output_column="distinct",
                    columns_to_count=["B"],
                ),
                pd.DataFrame({"distinct": [2]}),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                ),
                pd.DataFrame([["0", 3], ["1", 1]], columns=["A", "count_distinct"]),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    columns_to_count=["B"],
                ),
                pd.DataFrame([["0", 2], ["1", 1]], columns=["A", "count_distinct"]),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    # pylint: disable=protected-access
                    groupby_keys=KeySet(dataframe=GET_GROUPBY_ONE),
                ),
                pd.DataFrame({"A": ["0", "1", "2"], "count_distinct": [3, 1, 0]}),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    # pylint: disable=protected-access
                    groupby_keys=KeySet(dataframe=GET_GROUPBY_ONE),
                    columns_to_count=["B"],
                ),
                pd.DataFrame({"A": ["0", "1", "2"], "count_distinct": [2, 1, 0]}),
            ),
        ],
    )
    def test_count_distinct(self, spark, query_expr: QueryExpr, expected: pd.DataFrame):
        """Test that count_distinct works correctly."""
        count_distinct_df = spark.createDataFrame(
            pd.DataFrame(
                [
                    ["0", 0, 0.0],
                    ["0", 0, 0.0],
                    ["0", 0, 1.0],
                    ["0", 1, 2.0],
                    ["1", 0, 3.0],
                ],
                columns=["A", "B", "X"],
            )
        )
        measurement = self.compiler(
            [query_expr],
            privacy_budget=sp.oo,
            stability=self.stability,
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={
                "public": self.join_df,
                "groupby_two_columns": self.groupby_two_columns_df,
                "groupby_one_column": self.groupby_one_column_df,
            },
            catalog=self.catalog,
        )
        actual = measurement({"private": count_distinct_df})
        assert len(actual) == 1
        assert_frame_equal_with_sort(actual[0].toPandas(), expected)

    @pytest.mark.parametrize("query_exprs,expected", QUERY_EXPR_COMPILER_TESTS)
    def test_queries(self, query_exprs: List[QueryExpr], expected: List[pd.DataFrame]):
        """Tests that compiled measurement produces correct results.

        Args:
            query_exprs: The queries to evaluate.
            expected: The expected answers.
        """
        measurement = self.compiler(
            query_exprs,
            privacy_budget=sp.oo,
            stability=self.stability,
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={
                "public": self.join_df,
                "dtypes": self.dtypes_df,
                "groupby_two_columns": self.groupby_two_columns_df,
                "groupby_one_column": self.groupby_one_column_df,
            },
            catalog=self.catalog,
        )
        actual = measurement({"private": self.sdf})
        assert len(actual) == len(expected)
        for actual_sdf, expected_df in zip(actual, expected):
            assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @pytest.mark.parametrize(
        "query,output_measure,expected",
        [
            (  # Total with LAPLACE (Geometric noise gets applied)
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    output_column="total",
                    mechanism=CountMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"total": [4]})],
            ),
            (  # Total with GAUSSIAN
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    output_column="total",
                    mechanism=CountMechanism.GAUSSIAN,
                ),
                RhoZCDP(),
                [pd.DataFrame({"total": [4]})],
            ),
            (  # BoundedAverage on floating-point valued measure column with LAPLACE
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="X",
                    low=0.0,
                    high=1.0,
                    mechanism=AverageMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "average": [0.666667, 1.0]})],
            ),
            (  # BoundedAverage with integer valued measure column with LAPLACE
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    mechanism=AverageMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "average": [0.33333, 0.0]})],
            ),
            (  # BoundedAverage with integer valued measure column with GAUSSIAN
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    mechanism=AverageMechanism.GAUSSIAN,
                ),
                RhoZCDP(),
                [pd.DataFrame({"A": ["0", "1"], "average": [0.33333, 0.0]})],
            ),
            (  # BoundedSTDEV on floating-point valued measure column with LAPLACE
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="X",
                    low=0.0,
                    high=1.0,
                    mechanism=StdevMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "stdev": [0.471405, 0.5]})],
            ),
            (  # BoundedSTDEV on integer valued measure column with LAPLACE
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    mechanism=StdevMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "stdev": [0.471405, 0.0]})],
            ),
            (  # BoundedSTDEV on integer valued measure column with GAUSSIAN
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    mechanism=StdevMechanism.GAUSSIAN,
                ),
                RhoZCDP(),
                [pd.DataFrame({"A": ["0", "1"], "stdev": [0.471405, 0.0]})],
            ),
            (  # BoundedVariance on floating-point valued measure column with LAPLACE
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="X",
                    low=0.0,
                    high=1.0,
                    output_column="var",
                    mechanism=VarianceMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "var": [0.22222, 0.25]})],
            ),
            (  # BoundedVariance on integer valued measure column with LAPLACE
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    output_column="var",
                    mechanism=VarianceMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "var": [0.22222, 0.0]})],
            ),
            (  # BoundedVariance on integer valued measure column with GAUSSIAN
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    output_column="var",
                    mechanism=VarianceMechanism.GAUSSIAN,
                ),
                RhoZCDP(),
                [pd.DataFrame({"A": ["0", "1"], "var": [0.22222, 0.0]})],
            ),
            (  # BoundedSum on floating-point valued measure column with LAPLACE
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="X",
                    low=0.0,
                    high=1.0,
                    output_column="sum",
                    mechanism=SumMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "sum": [2.0, 1.0]})],
            ),
            (  # BoundedSum on integer valued measure column with LAPLACE
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    output_column="sum",
                    mechanism=SumMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"A": ["0", "1"], "sum": [1, 0]})],
            ),
            (  # BoundedSum on integer valued measure column with GAUSSIAN
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    output_column="sum",
                    mechanism=SumMechanism.GAUSSIAN,
                ),
                RhoZCDP(),
                [pd.DataFrame({"A": ["0", "1"], "sum": [1, 0]})],
            ),
            (  # Grouping flat map with LAPLACE
                GroupByBoundedSum(
                    child=ReplaceNullAndNan(
                        replace_with={},
                        child=FlatMap(
                            child=FlatMap(
                                child=PrivateSource("private"),
                                f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
                                max_num_rows=1,
                                schema_new_columns=Schema(
                                    {"Repeat": "INTEGER"}, grouping_column="Repeat"
                                ),
                                augment=True,
                            ),
                            f=lambda row: [
                                {"i": row["X"]} for i in range(row["Repeat"])
                            ],
                            max_num_rows=2,
                            schema_new_columns=Schema({"i": "DECIMAL"}),
                            augment=True,
                        ),
                    ),
                    groupby_keys=KeySet.from_dict({"Repeat": [1, 2]}),
                    measure_column="i",
                    low=0.0,
                    high=3.0,
                    mechanism=SumMechanism.LAPLACE,
                ),
                PureDP(),
                [pd.DataFrame({"Repeat": [1, 2], "sum": [3.0, 6.0]})],
            ),
        ],
    )
    def test_noise_param_combinations(
        self,
        query: QueryExpr,
        output_measure: Union[PureDP, RhoZCDP],
        expected: List[pd.DataFrame],
    ):
        """Tests aggregation with various privacy definition and mechanism."""
        compiler = QueryExprCompiler(output_measure=output_measure)
        measurement = compiler(
            [query],
            privacy_budget=sp.oo,
            stability=self.stability,
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={"public": self.join_df},
            catalog=self.catalog,
        )
        actual = measurement({"private": self.sdf})
        assert len(actual) == len(expected)
        for actual_sdf, expected_df in zip(actual, expected):
            assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @pytest.mark.parametrize(
        "query_exprs",
        [
            (  # BoundedAverage with floating-point valued measure column with GAUSSIAN
                [
                    GroupByBoundedAverage(
                        child=PrivateSource("private"),
                        groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                        measure_column="X",
                        low=0.0,
                        high=1.0,
                        mechanism=AverageMechanism.GAUSSIAN,
                    )
                ]
            ),
            (  # BoundedSTDEV on floating-point valued measure column with GAUSSIAN
                [
                    GroupByBoundedSTDEV(
                        child=PrivateSource("private"),
                        groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                        measure_column="X",
                        low=0.0,
                        high=1.0,
                        mechanism=StdevMechanism.GAUSSIAN,
                    )
                ]
            ),
            (  # BoundedVariance on floating-point valued measure column with GAUSSIAN
                [
                    GroupByBoundedVariance(
                        child=PrivateSource("private"),
                        groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                        measure_column="X",
                        low=0.0,
                        high=1.0,
                        output_column="var",
                        mechanism=VarianceMechanism.GAUSSIAN,
                    )
                ]
            ),
            (  # BoundedSum on floating-point valued measure column with GAUSSIAN
                [
                    GroupByBoundedSum(
                        child=PrivateSource("private"),
                        groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                        measure_column="X",
                        low=0.0,
                        high=1.0,
                        output_column="sum",
                        mechanism=SumMechanism.GAUSSIAN,
                    )
                ]
            ),
            (  # Grouping flat map with GAUSSIAN
                [
                    GroupByBoundedSum(
                        child=ReplaceNullAndNan(
                            replace_with={},
                            child=FlatMap(
                                child=FlatMap(
                                    child=PrivateSource("private"),
                                    f=lambda row: [
                                        {"Repeat": 1 if row["A"] == "0" else 2}
                                    ],
                                    max_num_rows=1,
                                    schema_new_columns=Schema(
                                        {"Repeat": "INTEGER"}, grouping_column="Repeat"
                                    ),
                                    augment=True,
                                ),
                                f=lambda row: [
                                    {"i": row["X"]} for i in range(row["Repeat"])
                                ],
                                max_num_rows=2,
                                schema_new_columns=Schema({"i": "DECIMAL"}),
                                augment=True,
                            ),
                        ),
                        groupby_keys=KeySet.from_dict({"Repeat": [1, 2]}),
                        measure_column="i",
                        low=0.0,
                        high=3.0,
                        mechanism=SumMechanism.GAUSSIAN,
                    )
                ]
            ),
        ],
    )
    def test_gaussian_noise_param_on_float_errors(self, query_exprs: List[QueryExpr]):
        """Tests that Gaussian noise with floating-point values errors."""
        compiler = QueryExprCompiler(output_measure=RhoZCDP())
        with pytest.raises(
            NotImplementedError,
            match="GAUSSIAN noise is not yet compatible with floating-point values.",
        ):
            compiler(
                query_exprs,
                privacy_budget=sp.oo,
                stability=self.stability,
                input_domain=self.input_domain,
                input_metric=self.input_metric,
                public_sources={"public": self.join_df},
                catalog=self.catalog,
            )

    def test_join_public_dataframe(self, spark):
        """Public join works with public tables given as Spark dataframes."""
        # This sets up a DF with a column that Spark thinks could contain NaNs,
        # but which doesn't actually contain any. This is allowed by Analytics,
        # but it has caused some bugs in the past which this test should
        # detect.
        public_sdf = spark.createDataFrame(
            pd.DataFrame({"A": ["0", "1"], "Y": [0.1, float("nan")]})
        ).fillna(0)

        output_sdf = self.compiler.build_transformation(
            JoinPublic(PrivateSource("private"), public_sdf),
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={},
            catalog=self.catalog,
        )({"private": self.sdf})

        assert_frame_equal_with_sort(
            output_sdf.toPandas(),
            pd.DataFrame(
                [
                    ("0", 0, 0.0, 0.1),
                    ("0", 0, 1.0, 0.1),
                    ("0", 1, 2.0, 0.1),
                    ("1", 0, 3.0, 0),
                ],
                columns=["A", "B", "X", "Y"],
            ),
        )

    def test_join_private(self, spark):
        """Tests that join private works."""
        sdf_2 = spark.createDataFrame(  # pylint: disable=no-member
            pd.DataFrame(
                [["0", 0], ["0", 2], ["1", 2], ["0", 0], ["1", 4]], columns=["A", "C"]
            )
        )
        output_sdf = self.compiler.build_transformation(
            JoinPrivate(
                child=PrivateSource("private"),
                right_operand_expr=PrivateSource("private_2"),
                truncation_strategy_left=TruncationStrategy.DropExcess(3),
                truncation_strategy_right=TruncationStrategy.DropExcess(3),
            ),
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={},
            catalog=self.catalog,
        )({"private": self.sdf, "private_2": sdf_2})

        assert_frame_equal_with_sort(
            output_sdf.toPandas(),
            pd.DataFrame(
                [
                    ["0", 0, 0.0, 0],
                    ["0", 0, 0.0, 0],
                    ["0", 0, 0.0, 2],
                    ["0", 0, 1.0, 0],
                    ["0", 0, 1.0, 0],
                    ["0", 0, 1.0, 2],
                    ["0", 1, 2.0, 0],
                    ["0", 1, 2.0, 0],
                    ["0", 1, 2.0, 2],
                    ["1", 0, 3.0, 2],
                    ["1", 0, 3.0, 4],
                ],
                columns=["A", "B", "X", "C"],
            ),
        )

    @pytest.mark.parametrize(
        "join_query,expected_output_stability",
        [
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropExcess(3),
                ),
                36,  # 3 * (2 * 3) + 3 * (2 * 3)
            ),
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropExcess(1),
                ),
                24,  # 3 * (2 * 3) + 1 * (2 * 3)
            ),
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(1),
                    truncation_strategy_right=TruncationStrategy.DropExcess(1),
                ),
                12,  # 1 * (2 * 3) + 1 * (2 * 3)
            ),
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropNonUnique(),
                ),
                15,  # 3 * (1 * 3) + 1 * (2 * 3)
            ),
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropNonUnique(),
                    truncation_strategy_right=TruncationStrategy.DropNonUnique(),
                ),
                6,  # 1 * (1 * 3) + 1 * (1 * 3)
            ),
        ],
    )
    def test_join_private_output_stability(self, join_query, expected_output_stability):
        """Tests that join private gives correct output stability."""
        transformation = self.compiler.build_transformation(
            join_query,
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={},
            catalog=self.catalog,
        )
        assert (
            transformation.stability_function(self.stability)
            == expected_output_stability
        )

    def test_join_private_invalid_truncation_strategy(self):
        """Tests that join_private raises error if truncation strategy is invalid."""

        class Strategy(TruncationStrategy.Type):
            """An invalid truncation strategy."""

        query = JoinPrivate(
            child=PrivateSource("private"),
            right_operand_expr=PrivateSource("private_2"),
            truncation_strategy_left=Strategy(),
            truncation_strategy_right=Strategy(),
        )
        expected_error_msg = (
            f"Truncation strategy type {Strategy.__qualname__} is not supported."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.compiler.build_transformation(
                query,
                input_domain=self.input_domain,
                input_metric=self.input_metric,
                public_sources={},
                catalog=self.catalog,
            )

    @pytest.mark.parametrize(
        "flatmap_query,measure,expected_output_stability",
        [
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda _: [{"G": "a"}, {"G": "b"}],
                    max_num_rows=2,
                    schema_new_columns=Schema({"G": "VARCHAR"}, grouping_column="G"),
                    augment=True,
                ),
                RhoZCDP(),
                3 * sp.sqrt(2),
            ),
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda _: [{"G": "a"}, {"G": "b"}],
                    max_num_rows=2,
                    schema_new_columns=Schema({"G": "VARCHAR"}),
                    augment=True,
                ),
                PureDP(),
                6,
            ),
        ],
    )
    def test_flatmap_output_stability(
        self, flatmap_query, measure, expected_output_stability
    ):
        """Tests that flatmap gives correct output stability."""
        compiler = QueryExprCompiler(output_measure=measure)
        transformation = compiler.build_transformation(
            flatmap_query,
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={},
            catalog=self.catalog,
        )
        assert (
            transformation.stability_function(self.stability)
            == expected_output_stability
        )

    def test_float_groupby_sum(self, spark):
        """Tests that groupby sum on floating-point-valued column uses laplace."""
        sdf_float = spark.createDataFrame(
            pd.DataFrame(
                [["0", 0, 0.0], ["0", 0, 1.0], ["0", 1, 2.0], ["1", 0, 3.0]],
                columns=["A", "B", "X"],
            )
        )
        query_exprs = [
            GroupByBoundedSum(
                child=PrivateSource("private"),
                groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                measure_column="X",
                low=0.0,
                high=1.0,
                output_column="sum",
                mechanism=SumMechanism.LAPLACE,
            )
        ]
        measurement = QueryExprCompiler()(
            query_exprs,
            privacy_budget=sp.oo,
            stability=self.stability,
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            public_sources={},
            catalog=self.catalog,
        )
        actual = measurement({"private": sdf_float})
        expected = [pd.DataFrame({"A": ["0", "1"], "sum": [2.0, 1.0]})]
        for actual_sdf, expected_df in zip(actual, expected):
            assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @pytest.mark.parametrize(
        "query_exprs",
        [
            (
                # Top-level query needs to be instance of measurement QueryExpr
                [
                    FlatMap(
                        child=PrivateSource("private"),
                        f=lambda row: [{}, {}],
                        max_num_rows=2,
                        schema_new_columns=Schema({}),
                        augment=True,
                    )
                ]
            ),
            (  # Query's child has to be transformation QueryExpr
                [
                    GroupByBoundedSum(
                        child=GroupByCount(
                            child=PrivateSource("private"),
                            groupby_keys=KeySet.from_dict(
                                {"A": ["0", "1"], "B": [0, 1]}
                            ),
                        ),
                        groupby_keys=KeySet.from_dict({}),
                        measure_column="B",
                        low=0,
                        high=3,
                    )
                ]
            ),
        ],
    )
    def test_invalid_queries(self, query_exprs: List[QueryExpr]):
        """QueryExprCompiler raises error on unsupported queries."""
        with pytest.raises(NotImplementedError):
            self.compiler(
                query_exprs,
                privacy_budget=sp.oo,
                stability=self.stability,
                input_domain=self.input_domain,
                input_metric=self.input_metric,
                public_sources={"public": self.join_df},
                catalog=self.catalog,
            )

    def test_different_source_id(self):
        """Tests that different source ids are allowed."""
        self.catalog.add_private_view(
            source_id="doubled",
            col_types={
                "A": ColumnType.VARCHAR,
                "B": ColumnType.INTEGER,
                "X": ColumnType.DECIMAL,
            },
            stability=self.catalog.private_table.stability * 2,
        )
        query_exprs = [
            GroupByCount(
                child=PrivateSource("private"), groupby_keys=KeySet.from_dict({})
            ),
            GroupByCount(
                child=PrivateSource("doubled"), groupby_keys=KeySet.from_dict({})
            ),
        ]
        stability = {"doubled": self.stability["private"] * 2, **self.stability}
        input_domain = DictDomain(
            {"doubled": self.input_domain["private"], **self.input_domain.key_to_domain}
        )
        input_metric = DictMetric(
            {"doubled": self.input_metric["private"], **self.input_metric.key_to_metric}
        )

        measurement = self.compiler(
            query_exprs,
            privacy_budget=sp.Integer(10),
            stability=stability,
            input_domain=input_domain,
            input_metric=input_metric,
            public_sources={"public": self.join_df},
            catalog=self.catalog,
        )
        assert measurement.privacy_relation(stability, sp.Integer(10))

    def test_call_no_queries(self):
        """``__call__`` raises error if the sequence of queries has length 0."""
        with pytest.raises(
            ValueError, match=r"At least one query needs to be provided"
        ):
            self.compiler(
                queries=[],
                privacy_budget=sp.Integer(10),
                stability=self.stability,
                input_domain=self.input_domain,
                input_metric=self.input_metric,
                public_sources={"public": self.join_df},
                catalog=self.catalog,
            )


class TestCompileGroupByQuantile:
    """Test compiling GroupByQuantile.

    This is separate from other tests because some noise is added even when
    privacy_budget is infinity. Therefore, we must fix a seed and make sure
    the output is within a range, rather than asserting the exact output.
    """

    @pytest.mark.parametrize("output_measure", [(PureDP()), (RhoZCDP())])
    def test_compile_groupby_quantile(
        self, spark, output_measure: Union[PureDP, RhoZCDP]
    ):
        """GroupByQuantile is compiles correctly."""
        sdf = spark.createDataFrame(
            pd.DataFrame(
                [
                    ["F", 28],
                    ["F", 26],
                    ["F", 27],
                    ["M", 23],
                    ["F", 29],
                    ["M", 22],
                    ["M", 24],
                    ["M", 25],
                ],
                columns=["Gender", "Age"],
            )
        )
        catalog = Catalog()
        catalog.add_private_source(
            "private",
            {
                "Gender": ColumnDescriptor(ColumnType.VARCHAR),
                "Age": ColumnDescriptor(ColumnType.INTEGER),
            },
            stability=1,
        )
        stability = {"private": sp.Integer(1)}
        input_domain = DictDomain(
            {
                "private": SparkDataFrameDomain(
                    {
                        "Gender": SparkStringColumnDescriptor(),
                        "Age": SparkIntegerColumnDescriptor(),
                    }
                )
            }
        )
        input_metric = DictMetric({"private": SymmetricDifference()})
        compiler = QueryExprCompiler(output_measure=output_measure)

        query_expr = GroupByQuantile(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"Gender": ["M", "F"]}),
            measure_column="Age",
            quantile=0.5,
            low=22,
            high=29,
            output_column="out",
        )
        measurement = compiler(
            [query_expr],
            privacy_budget=sp.Integer(1000),
            stability=stability,
            input_domain=input_domain,
            input_metric=input_metric,
            public_sources={},
            catalog=catalog,
        )
        assert measurement.input_domain == input_domain
        assert measurement.input_metric == input_metric
        assert measurement.output_measure == output_measure
        assert measurement.privacy_function(stability) == sp.Integer(1000)

        [actual] = measurement({"private": sdf})

        assert 26 < actual.filter(col("Gender") == "F").collect()[0]["out"] <= 28
        assert 22 < actual.filter(col("Gender") == "M").collect()[0]["out"] <= 24


@pytest.fixture(name="test_component_data", scope="class")
def setup_components(request):
    """Set up test."""
    request.cls._stability = {"test": sp.Integer(3)}
    request.cls._privacy_budget = sp.Integer(5)
    request.cls._input_domain = DictDomain(
        {
            "test": SparkDataFrameDomain(
                analytics_to_spark_columns_descriptor(
                    Schema({"A": "VARCHAR", "X": "INTEGER"})
                )
            )
        }
    )
    request.cls._input_metric = DictMetric({"test": SymmetricDifference()})
    request.cls._catalog = Catalog()
    request.cls._catalog.add_private_source(
        source_id="test",
        col_types={"A": ColumnType.VARCHAR, "X": ColumnType.INTEGER},
        stability=3,
    )


@pytest.mark.usefixtures("test_component_data")
class TestComponentIsUsed:
    """Tests that specific components are used inside of compiled measurements."""

    @pytest.mark.parametrize(
        "output_measure,query_expr,column,preprocessing_stability",
        [
            (output_measure, query_expr, column, preprocessing_stability)
            for output_measure in [PureDP(), RhoZCDP()]
            for preprocessing_expr, preprocessing_stability in [
                (
                    ReplaceNullAndNan(
                        replace_with={},
                        child=FlatMap(
                            child=PrivateSource(source_id="test"),
                            f=lambda row: [{}, {}],
                            max_num_rows=2,
                            schema_new_columns=Schema({}),
                            augment=True,
                        ),
                    ),
                    2,
                ),
                (PrivateSource(source_id="test"), 1),
            ]
            for query_expr, column in [
                (
                    GroupByCount(
                        child=preprocessing_expr,
                        groupby_keys=KeySet.from_dict({"A": ["a1", "a2"]}),
                    ),
                    "count",
                ),
                (
                    GroupByBoundedSum(
                        child=preprocessing_expr,
                        groupby_keys=KeySet.from_dict({"A": ["a1", "a2"]}),
                        measure_column="X",
                        low=-1,
                        high=5,
                        output_column="sum",
                    ),
                    "sum",
                ),
                (
                    GroupByBoundedAverage(
                        child=preprocessing_expr,
                        groupby_keys=KeySet.from_dict({"A": ["a1", "a2"]}),
                        measure_column="X",
                        low=-1,
                        high=5,
                        output_column="average",
                    ),
                    "average",
                ),
                (
                    GroupByBoundedSTDEV(
                        child=preprocessing_expr,
                        groupby_keys=KeySet.from_dict({"A": ["a1", "a2"]}),
                        measure_column="X",
                        low=-1,
                        high=5,
                        output_column="standard deviation",
                    ),
                    "standard deviation",
                ),
                (
                    GroupByBoundedVariance(
                        child=preprocessing_expr,
                        groupby_keys=KeySet.from_dict({"A": ["a1", "a2"]}),
                        measure_column="X",
                        low=-1,
                        high=5,
                        output_column="variance",
                    ),
                    "variance",
                ),
                (
                    GroupByQuantile(
                        child=preprocessing_expr,
                        groupby_keys=KeySet.from_dict({"A": ["a1", "a2"]}),
                        measure_column="X",
                        quantile=0.6,
                        low=-1,
                        high=5,
                        output_column="quantile",
                    ),
                    "quantile",
                ),
            ]
        ],
    )
    def test_used_create_measurement(
        self,
        output_measure: Union[PureDP, RhoZCDP],
        query_expr: QueryExpr,
        column: str,
        preprocessing_stability: int,
    ):
        """Compiled measurements contain aggregations with the expected noise scale."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor."
            "create_quantile_measurement"
        ) as mock_create_quantile_measurement, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor."
            "create_standard_deviation_measurement"
        ) as mock_create_standard_deviation_measurement, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor."
            "create_variance_measurement"
        ) as mock_create_variance_measurement, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor."
            "create_average_measurement"
        ) as mock_create_average_measurement, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor."
            "create_sum_measurement"
        ) as mock_create_sum_measurement, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor."
            "create_count_measurement"
        ) as mock_create_count_measurement:

            d_in = sp.Integer(3)
            d_out = sp.Integer(5)
            compiler = QueryExprCompiler(output_measure=output_measure)
            mock_create_measurement_dict = {
                "count": mock_create_count_measurement,
                "sum": mock_create_sum_measurement,
                "average": mock_create_average_measurement,
                "variance": mock_create_variance_measurement,
                "standard deviation": mock_create_standard_deviation_measurement,
                "quantile": mock_create_quantile_measurement,
            }
            mock_create_measurement = mock_create_measurement_dict[column]
            mock_create_measurement.return_value = create_mock_measurement(
                input_domain=self._input_domain["test"],
                input_metric=self._input_metric["test"],
                output_measure=output_measure,
                privacy_function_return_value=d_out,
                privacy_function_implemented=True,
            )
            _ = compiler(
                [query_expr],
                privacy_budget=self._privacy_budget,
                stability=self._stability,
                input_domain=self._input_domain,
                input_metric=self._input_metric,
                public_sources={},
                catalog=self._catalog,
            )
            expected_arguments = {  # Other arguments are not checked
                "input_domain": self._input_domain["test"],
                "input_metric": self._input_metric["test"],
                "d_in": d_in * preprocessing_stability,
                "d_out": d_out,
            }
            if column != "quantile":
                expected_arguments["noise_mechanism"] = (
                    NoiseMechanism.GEOMETRIC
                    if isinstance(output_measure, PureDP)
                    else NoiseMechanism.DISCRETE_GAUSSIAN
                )
            _, kwargs = mock_create_measurement.call_args_list[-1]
            for kwarg, value in expected_arguments.items():
                assert kwargs[kwarg] == value
