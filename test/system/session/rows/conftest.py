"""Common fixtures for non-IDs Session tests."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Any, List

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType
from tmlt.core.domains.spark_domains import SparkDataFrameDomain

from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
    analytics_to_spark_schema,
)
from tmlt.analytics.binning_spec import BinningSpec
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import (
    CountDistinctMechanism,
    CountMechanism,
    Filter,
    FlatMap,
    GroupByBoundedSum,
    GroupByCount,
    GroupByCountDistinct,
    JoinPublic,
    Map,
    PrivateSource,
    ReplaceNullAndNan,
    Select,
    SumMechanism,
)

# Shorthands for some values used in tests
_DATE1 = datetime.date.fromisoformat("2022-01-01")
_DATE2 = datetime.date.fromisoformat("2022-01-02")

# Dataframes for public data,
# placed here so that test case KeySets can use them
GROUPBY_TWO_COLUMNS = pd.DataFrame([["0", 0], ["0", 1], ["1", 1]], columns=["A", "B"])
GET_GROUPBY_TWO_COLUMNS = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    GROUPBY_TWO_COLUMNS
)
GROUPBY_ONE_COLUMN = pd.DataFrame([["0"], ["1"], ["2"]], columns=["A"])
GET_GROUPBY_ONE_COLUMN = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    GROUPBY_ONE_COLUMN
)
GROUPBY_WITH_DUPLICATES = pd.DataFrame(
    [["0"], ["0"], ["1"], ["1"], ["2"], ["2"]], columns=["A"]
)
GET_GROUPBY_WITH_DUPLICATES = (
    lambda: SparkSession.builder.getOrCreate().createDataFrame(GROUPBY_WITH_DUPLICATES)
)
GROUPBY_EMPTY: List[Any] = []
GROUPBY_EMPTY_SCHEMA = StructType()
GET_GROUPBY_EMPTY = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    GROUPBY_EMPTY, schema=GROUPBY_EMPTY_SCHEMA
)


EVALUATE_TESTS = [
    (  # Total with DEFAULT mechanism
        # (Geometric noise gets applied if PureDP; Gaussian noise gets applied if ZCDP)
        QueryBuilder("private").count(name="total"),
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Total with DEFAULT mechanism
        # (Geometric noise gets applied if PureDP; Gaussian noise gets applied if ZCDP)
        QueryBuilder("private").count_distinct(name="total"),
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Total with LAPLACE (Geometric noise gets applied)
        QueryBuilder("private").count(name="total", mechanism=CountMechanism.LAPLACE),
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=CountMechanism.LAPLACE,
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Total with LAPLACE (Geometric noise gets applied)
        QueryBuilder("private").count_distinct(
            name="total", mechanism=CountDistinctMechanism.LAPLACE
        ),
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=CountDistinctMechanism.LAPLACE,
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Full marginal from domain description (Geometric noise gets applied)
        QueryBuilder("private")
        .groupby(KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}))
        .count(),
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
        ),
        pd.DataFrame(
            {"A": ["0", "0", "1", "1"], "B": [0, 1, 0, 1], "count": [2, 1, 1, 0]}
        ),
    ),
    (  # Full marginal from domain description (Geometric noise gets applied)
        QueryBuilder("private")
        .groupby(KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}))
        .count_distinct(),
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
        ),
        pd.DataFrame(
            {
                "A": ["0", "0", "1", "1"],
                "B": [0, 1, 0, 1],
                "count_distinct": [2, 1, 1, 0],
            }
        ),
    ),
    (  # Incomplete two-column marginal with a dataframe
        QueryBuilder("private")
        .groupby(KeySet(dataframe=GET_GROUPBY_TWO_COLUMNS))
        .count(),
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_TWO_COLUMNS),
        ),
        pd.DataFrame({"A": ["0", "0", "1"], "B": [0, 1, 1], "count": [2, 1, 0]}),
    ),
    (  # Incomplete two-column marginal with a dataframe
        QueryBuilder("private")
        .groupby(KeySet(dataframe=GET_GROUPBY_TWO_COLUMNS))
        .count_distinct(),
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_TWO_COLUMNS),
        ),
        pd.DataFrame(
            {"A": ["0", "0", "1"], "B": [0, 1, 1], "count_distinct": [2, 1, 0]}
        ),
    ),
    (  # One-column marginal with additional value
        QueryBuilder("private")
        .groupby(KeySet(dataframe=GET_GROUPBY_ONE_COLUMN))
        .count(),
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_ONE_COLUMN),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count": [3, 1, 0]}),
    ),
    (  # One-column marginal with additional value
        QueryBuilder("private")
        .groupby(KeySet(dataframe=GET_GROUPBY_ONE_COLUMN))
        .count_distinct(),
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_ONE_COLUMN),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count_distinct": [3, 1, 0]}),
    ),
    (  # One-column marginal with duplicate rows
        QueryBuilder("private")
        .groupby(KeySet(dataframe=GET_GROUPBY_WITH_DUPLICATES))
        .count(),
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_WITH_DUPLICATES),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count": [3, 1, 0]}),
    ),
    (  # One-column marginal with duplicate rows
        QueryBuilder("private")
        .groupby(KeySet(dataframe=GET_GROUPBY_WITH_DUPLICATES))
        .count_distinct(),
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_WITH_DUPLICATES),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count_distinct": [3, 1, 0]}),
    ),
    (  # empty public source
        QueryBuilder("private").groupby(KeySet(dataframe=GET_GROUPBY_EMPTY)).count(),
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_EMPTY),
        ),
        pd.DataFrame({"count": [4]}),
    ),
    (  # empty public source
        QueryBuilder("private")
        .groupby(KeySet(dataframe=GET_GROUPBY_EMPTY))
        .count_distinct(),
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_EMPTY),
        ),
        pd.DataFrame({"count_distinct": [4]}),
    ),
    (  # BoundedSum
        QueryBuilder("private")
        .groupby(KeySet.from_dict({"A": ["0", "1"]}))
        .sum(column="X", low=0, high=1, name="sum"),
        GroupByBoundedSum(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
            measure_column="X",
            low=0,
            high=1,
            output_column="sum",
        ),
        pd.DataFrame({"A": ["0", "1"], "sum": [2, 1]}),
    ),
    (  # FlatMap
        QueryBuilder("private")
        .flat_map(
            f=lambda _: [{}, {}], max_num_rows=2, new_column_types={}, augment=True
        )
        .replace_null_and_nan()
        .sum(column="X", low=0, high=3),
        GroupByBoundedSum(
            child=ReplaceNullAndNan(
                replace_with={},
                child=FlatMap(
                    child=PrivateSource("private"),
                    f=lambda _: [{}, {}],
                    max_num_rows=2,
                    schema_new_columns=Schema({}),
                    augment=True,
                ),
            ),
            groupby_keys=KeySet.from_dict({}),
            measure_column="X",
            output_column="X_sum",
            low=0,
            high=3,
        ),
        pd.DataFrame({"X_sum": [12]}),
    ),
    (  # Multiple flat maps on integer-valued measure_column
        # (Geometric noise gets applied if PureDP; Gaussian noise gets applied if ZCDP
        QueryBuilder("private")
        .flat_map(
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            max_num_rows=1,
            new_column_types={"Repeat": ColumnDescriptor(ColumnType.INTEGER)},
            augment=True,
        )
        .flat_map(
            f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            max_num_rows=2,
            new_column_types={"i": ColumnDescriptor(ColumnType.INTEGER)},
            augment=False,
        )
        .replace_null_and_nan()
        .sum(column="i", low=0, high=3),
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
                    schema_new_columns=Schema({"i": "INTEGER"}),
                    augment=False,
                ),
            ),
            groupby_keys=KeySet.from_dict({}),
            measure_column="i",
            output_column="i_sum",
            low=0,
            high=3,
            mechanism=SumMechanism.DEFAULT,
        ),
        pd.DataFrame({"i_sum": [9]}),
    ),
    (  # Grouping flat map with DEFAULT mechanism and integer-valued measure column
        # (Geometric noise gets applied)
        QueryBuilder("private")
        .flat_map(
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            max_num_rows=1,
            new_column_types={"Repeat": ColumnDescriptor(ColumnType.INTEGER)},
            augment=True,
            grouping=True,
        )
        .flat_map(
            f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            max_num_rows=2,
            new_column_types={"i": ColumnDescriptor(ColumnType.INTEGER)},
            augment=True,
        )
        .replace_null_and_nan()
        .groupby(KeySet.from_dict({"Repeat": [1, 2]}))
        .sum(column="i", low=0, high=3),
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
                    schema_new_columns=Schema({"i": "INTEGER"}),
                    augment=True,
                ),
            ),
            groupby_keys=KeySet.from_dict({"Repeat": [1, 2]}),
            measure_column="i",
            output_column="i_sum",
            low=0,
            high=3,
        ),
        pd.DataFrame({"Repeat": [1, 2], "i_sum": [3, 6]}),
    ),
    (  # Grouping flat map with LAPLACE mechanism (Geometric noise gets applied)
        QueryBuilder("private")
        .flat_map(
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            max_num_rows=1,
            new_column_types={"Repeat": ColumnDescriptor(ColumnType.INTEGER)},
            grouping=True,
            augment=True,
        )
        .flat_map(
            f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            max_num_rows=2,
            new_column_types={"i": ColumnDescriptor(ColumnType.INTEGER)},
            augment=True,
        )
        .replace_null_and_nan()
        .groupby(KeySet.from_dict({"Repeat": [1, 2]}))
        .sum(column="i", low=0, high=3, mechanism=SumMechanism.LAPLACE),
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
                    schema_new_columns=Schema({"i": "INTEGER"}),
                    augment=True,
                ),
            ),
            groupby_keys=KeySet.from_dict({"Repeat": [1, 2]}),
            measure_column="i",
            output_column="i_sum",
            low=0,
            high=3,
            mechanism=SumMechanism.LAPLACE,
        ),
        pd.DataFrame({"Repeat": [1, 2], "i_sum": [3, 6]}),
    ),
    (  # Binning
        QueryBuilder("private")
        .bin_column("X", BinningSpec([0, 2, 4], names=["0,1", "2,3"], right=False))
        .groupby(KeySet.from_dict({"X_binned": ["0,1", "2,3"]}))
        .count(),
        None,
        pd.DataFrame({"X_binned": ["0,1", "2,3"], "count": [2, 2]}),
    ),
    (  # Histogram Syntax
        QueryBuilder("private").histogram(
            "X", BinningSpec([0, 2, 4], names=["0,1", "2,3"], right=False)
        ),
        None,
        pd.DataFrame({"X_binned": ["0,1", "2,3"], "count": [2, 2]}),
    ),
    (  # Binning Nulls
        QueryBuilder("private")
        .map(
            lambda row: {"X": row["X"] if row["X"] != 3 else None},
            new_column_types={"X": ColumnDescriptor(ColumnType.INTEGER)},
        )
        .bin_column(
            "X", BinningSpec([10, 12, 14], names=["10,12", "12,14"], right=False)
        )
        .groupby(KeySet.from_dict({"X_binned": ["10,12", "12,14", None]}))
        .count(),
        None,
        pd.DataFrame({"X_binned": ["10,12", "12,14", None], "count": [0, 0, 4]}),
    ),
    (  # Binning NaN bin names
        QueryBuilder("private")
        .bin_column(
            "X",
            BinningSpec(
                [0, 2, 4], names=[0.1, float("nan")], nan_bin=float("nan"), right=False
            ),
        )
        .map(
            f=lambda row: {"X_binned": 0 if row["X_binned"] == 0.1 else 1},
            new_column_types={"X_binned": ColumnType.INTEGER},
        )
        .groupby(KeySet.from_dict({"X_binned": [0, 1]}))
        .count(),
        None,
        pd.DataFrame({"X_binned": [0, 1], "count": [2, 2]}),
    ),
    (  # GroupByCount Filter
        QueryBuilder("private").filter("A == '0'").count(),
        GroupByCount(
            child=Filter(child=PrivateSource("private"), condition="A == '0'"),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count": [3]}),
    ),
    (  # GroupByCountDistinct Filter
        QueryBuilder("private").filter("A == '0'").count_distinct(),
        GroupByCountDistinct(
            child=Filter(child=PrivateSource("private"), condition="A == '0'"),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count_distinct": [3]}),
    ),
    (  # GroupByCount Select
        QueryBuilder("private").select(["A"]).count(),
        GroupByCount(
            child=Select(child=PrivateSource("private"), columns=["A"]),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count": [4]}),
    ),
    (  # GroupByCountDistinct Select
        QueryBuilder("private").select(["A"]).count_distinct(),
        GroupByCountDistinct(
            child=Select(child=PrivateSource("private"), columns=["A"]),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count_distinct": [2]}),
    ),
    (  # GroupByCount Map
        QueryBuilder("private")
        .map(
            f=lambda row: {"C": 2 * str(row["B"])},
            new_column_types={"C": ColumnDescriptor(ColumnType.VARCHAR)},
            augment=True,
        )
        .replace_null_and_nan()
        .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["00", "11"]}))
        .count(),
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
        ),
        pd.DataFrame(
            [["0", "00", 2], ["0", "11", 1], ["1", "00", 1], ["1", "11", 0]],
            columns=["A", "C", "count"],
        ),
    ),
    (  # GroupByCountDistinct Map
        QueryBuilder("private")
        .map(
            f=lambda row: {"C": 2 * str(row["B"])},
            new_column_types={"C": ColumnDescriptor(ColumnType.VARCHAR)},
            augment=True,
        )
        .replace_null_and_nan()
        .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["00", "11"]}))
        .count_distinct(),
        GroupByCountDistinct(
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
        ),
        pd.DataFrame(
            [["0", "00", 2], ["0", "11", 1], ["1", "00", 1], ["1", "11", 0]],
            columns=["A", "C", "count_distinct"],
        ),
    ),
    (  # GroupByCount JoinPublic
        QueryBuilder("private")
        .join_public("public")
        .groupby(KeySet.from_dict({"A+B": [0, 1, 2]}))
        .count(),
        GroupByCount(
            child=JoinPublic(child=PrivateSource("private"), public_table="public"),
            groupby_keys=KeySet.from_dict({"A+B": [0, 1, 2]}),
        ),
        pd.DataFrame({"A+B": [0, 1, 2], "count": [3, 4, 1]}),
    ),
    (  # GroupByCountDistinct JoinPublic
        QueryBuilder("private")
        .join_public("public")
        .groupby(KeySet.from_dict({"A+B": [0, 1, 2]}))
        .count_distinct(),
        GroupByCountDistinct(
            child=JoinPublic(child=PrivateSource("private"), public_table="public"),
            groupby_keys=KeySet.from_dict({"A+B": [0, 1, 2]}),
        ),
        pd.DataFrame({"A+B": [0, 1, 2], "count_distinct": [3, 4, 1]}),
    ),
    (  # GroupByCount with dates as groupby keys
        QueryBuilder("private")
        .join_public("join_dtypes")
        .groupby(KeySet.from_dict({"DATE": [_DATE1, _DATE2]}))
        .count(),
        GroupByCount(
            child=JoinPublic(
                child=PrivateSource("private"), public_table="join_dtypes"
            ),
            groupby_keys=KeySet.from_dict({"DATE": [_DATE1, _DATE2]}),
        ),
        pd.DataFrame({"DATE": [_DATE1, _DATE2], "count": [3, 1]}),
    ),
    (  # GroupByCountDistinct checking distinctness of dates
        QueryBuilder("private")
        .join_public("join_dtypes")
        .count_distinct(columns=["DATE"]),
        GroupByCountDistinct(
            child=JoinPublic(
                child=PrivateSource("private"), public_table="join_dtypes"
            ),
            columns_to_count=["DATE"],
            output_column="count_distinct(DATE)",
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count_distinct(DATE)": [2]}),
    ),
    pytest.param(
        QueryBuilder("private")
        .join_public("public")
        .join_public("public", ["A"])
        .join_public("public", ["A"])
        .groupby(
            KeySet.from_dict(
                {"A+B": [0, 1, 2], "A+B_left": [0, 1, 2], "A+B_right": [0, 1, 2]}
            )
        )
        .count(),
        None,
        pd.DataFrame(
            [
                (0, 0, 0, 3),
                (0, 0, 1, 3),
                (0, 1, 0, 3),
                (0, 1, 1, 3),
                (1, 0, 0, 3),
                (1, 0, 1, 3),
                (1, 1, 0, 3),
                (1, 1, 1, 4),
                (1, 1, 2, 1),
                (1, 2, 1, 1),
                (1, 2, 2, 1),
                (2, 1, 1, 1),
                (2, 1, 2, 1),
                (2, 2, 1, 1),
                (2, 2, 2, 1),
                (0, 0, 2, 0),
                (0, 1, 2, 0),
                (0, 2, 0, 0),
                (0, 2, 1, 0),
                (0, 2, 2, 0),
                (1, 0, 2, 0),
                (1, 2, 0, 0),
                (2, 0, 0, 0),
                (2, 0, 1, 0),
                (2, 0, 2, 0),
                (2, 1, 0, 0),
                (2, 2, 0, 0),
            ],
            columns=["A+B", "A+B_left", "A+B_right", "count"],
        ),
        id="public_join_disambiguation",
        marks=pytest.mark.slow,
    ),
]


### DATA FOR GENERAL SESSIONS ###
@pytest.fixture(name="session_data", scope="class")
def sess_data(spark, request):
    """Set up test data for basic session tests."""
    sdf = spark.createDataFrame(
        [["0", 0, 0], ["0", 0, 1], ["0", 1, 2], ["1", 0, 3]],
        schema=StructType(
            [
                StructField("A", StringType(), nullable=False),
                StructField("B", LongType(), nullable=False),
                StructField("X", LongType(), nullable=False),
            ]
        ),
    )
    request.cls.sdf = sdf
    join_df = spark.createDataFrame(
        [["0", 0], ["0", 1], ["1", 1], ["1", 2]],
        schema=StructType(
            [
                StructField("A", StringType(), nullable=False),
                StructField("A+B", LongType(), nullable=False),
            ]
        ),
    )
    request.cls.join_df = join_df

    join_dtypes_df = spark.createDataFrame(
        pd.DataFrame(
            [[0, _DATE1], [1, _DATE1], [2, _DATE1], [3, _DATE2]], columns=["X", "DATE"]
        )
    )
    request.cls.join_dtypes_df = join_dtypes_df

    groupby_two_columns_df = spark.createDataFrame(
        pd.DataFrame([["0", 0], ["0", 1], ["1", 1]], columns=["A", "B"])
    )
    request.cls.groupby_two_columns_df = groupby_two_columns_df

    groupby_one_column_df = spark.createDataFrame(
        pd.DataFrame([["0"], ["1"], ["2"]], columns=["A"])
    )
    request.cls.groupby_one_column_df = groupby_one_column_df

    groupby_with_duplicates_df = spark.createDataFrame(
        pd.DataFrame([["0"], ["0"], ["1"], ["1"], ["2"], ["2"]], columns=["A"])
    )
    request.cls.groupby_with_duplicates_df = groupby_with_duplicates_df

    groupby_empty_df = spark.createDataFrame([], schema=StructType())
    request.cls.groupby_empty_df = groupby_empty_df

    sdf_col_types = {"A": "VARCHAR", "B": "INTEGER", "X": "DECIMAL"}

    request.cls.sdf_col_types = sdf_col_types

    sdf_input_domain = SparkDataFrameDomain(
        analytics_to_spark_columns_descriptor(Schema(sdf_col_types))
    )
    request.cls.sdf_input_domain = sdf_input_domain


###DATA FOR SESSIONS WITH NULLS###
@pytest.fixture(name="null_session_data", scope="class")
def null_setup(spark, request):
    """Set up test data for sessions with nulls."""
    pdf = pd.DataFrame(
        [
            ["a0", 0, 0.0, datetime.date(2000, 1, 1), datetime.datetime(2020, 1, 1)],
            [None, 1, 1.0, datetime.date(2001, 1, 1), datetime.datetime(2021, 1, 1)],
            ["a2", None, 2.0, datetime.date(2002, 1, 1), datetime.datetime(2022, 1, 1)],
            ["a3", 3, None, datetime.date(2003, 1, 1), datetime.datetime(2023, 1, 1)],
            ["a4", 4, 4.0, None, datetime.datetime(2024, 1, 1)],
            ["a5", 5, 5.0, datetime.date(2005, 1, 1), None],
        ],
        columns=["A", "I", "X", "D", "T"],
    )

    request.cls.pdf = pdf

    sdf_col_types = {
        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        "I": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        "X": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
    }

    sdf = spark.createDataFrame(
        pdf, schema=analytics_to_spark_schema(Schema(sdf_col_types))
    )
    request.cls.sdf = sdf


###DATA FOR SESSIONS WITH INF VALUES###
@pytest.fixture(name="infs_test_data", scope="class")
def infs_setup(spark, request):
    """Set up tests."""
    pdf = pd.DataFrame(
        {"A": ["a0", "a0", "a1", "a1"], "B": [float("-inf"), 2.0, 5.0, float("inf")]}
    )
    request.cls.pdf = pdf

    sdf_col_types = {
        "A": ColumnDescriptor(ColumnType.VARCHAR),
        "B": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
    }
    sdf = spark.createDataFrame(
        pdf, schema=analytics_to_spark_schema(Schema(sdf_col_types))
    )
    request.cls.sdf = sdf
