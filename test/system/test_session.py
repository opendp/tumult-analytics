"""System tests for Session."""

# <placeholder: boilerplate>

import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
from unittest.mock import patch

import pandas as pd
from parameterized import parameterized
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
    analytics_to_spark_schema,
)
from tmlt.analytics.binning_spec import BinningSpec
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import (
    AnalyticsDefault,
    CountDistinctMechanism,
    CountMechanism,
    Filter,
    FlatMap,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByCount,
    GroupByCountDistinct,
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
)
from tmlt.analytics.session import Session
from tmlt.analytics.truncation_strategy import TruncationStrategy
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountantState,
    SequentialQueryable,
)
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import PySparkTest

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
            # pylint: disable=protected-access
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
            # pylint: disable=protected-access
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
            # pylint: disable=protected-access
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
    (  # GroupByCount Filter
        QueryBuilder("private").filter("A == '0'").count(),
        GroupByCount(
            child=Filter(child=PrivateSource("private"), predicate="A == '0'"),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count": [3]}),
    ),
    (  # GroupByCountDistinct Filter
        QueryBuilder("private").filter("A == '0'").count_distinct(),
        GroupByCountDistinct(
            child=Filter(child=PrivateSource("private"), predicate="A == '0'"),
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
        .count_distinct(cols=["DATE"]),
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
]


class TestEvaluate(PySparkTest):
    """Tests for evaluate."""

    def setUp(self) -> None:
        """Set up test data."""
        self.sdf = self.spark.createDataFrame(
            pd.DataFrame(
                [["0", 0, 0], ["0", 0, 1], ["0", 1, 2], ["1", 0, 3]],
                columns=["A", "B", "X"],
            )
        )
        self.join_df = self.spark.createDataFrame(
            pd.DataFrame([["0", 0], ["0", 1], ["1", 1], ["1", 2]], columns=["A", "A+B"])
        )
        self.join_dtypes_df = self.spark.createDataFrame(
            pd.DataFrame(
                [[0, _DATE1], [1, _DATE1], [2, _DATE1], [3, _DATE2]],
                columns=["X", "DATE"],
            )
        )
        self.groupby_two_columns_df = self.spark.createDataFrame(
            pd.DataFrame([["0", 0], ["0", 1], ["1", 1]], columns=["A", "B"])
        )
        self.groupby_one_column_df = self.spark.createDataFrame(
            pd.DataFrame([["0"], ["1"], ["2"]], columns=["A"])
        )
        self.groupby_with_duplicates_df = self.spark.createDataFrame(
            pd.DataFrame([["0"], ["0"], ["1"], ["1"], ["2"], ["2"]], columns=["A"])
        )
        self.groupby_empty_df = self.spark.createDataFrame([], schema=StructType())
        self.sdf_col_types = {"A": "VARCHAR", "B": "INTEGER", "X": "DECIMAL"}
        self.sdf_input_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(Schema(self.sdf_col_types))
        )

    @parameterized.expand(EVALUATE_TESTS)
    def test_queries_privacy_budget_infinity_puredp(
        self,
        query_expr: QueryExpr,
        expected_expr: Optional[QueryExpr],
        expected_df: pd.DataFrame,
    ):
        """Session :func:`evaluate` returns the correct results for eps=inf and PureDP.

        Args:
            query_expr: The query to evaluate.
            expected_expr: Expected value for query_expr.
            expected_df: The expected answer.
        """
        if expected_expr is not None:
            self.assertEqual(query_expr, expected_expr)
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
        )
        session.add_public_dataframe(source_id="public", dataframe=self.join_df)
        session.add_public_dataframe(
            source_id="join_dtypes", dataframe=self.join_dtypes_df
        )
        session.add_public_dataframe(
            source_id="groupby_two_columns", dataframe=self.groupby_two_columns_df
        )
        session.add_public_dataframe(
            source_id="groupby_one_column", dataframe=self.groupby_one_column_df
        )
        session.add_public_dataframe(
            source_id="groupby_with_duplicates",
            dataframe=self.groupby_with_duplicates_df,
        )
        session.add_public_dataframe(
            source_id="groupby_empty", dataframe=self.groupby_empty_df
        )
        actual_sdf = session.evaluate(
            query_expr, privacy_budget=PureDPBudget(float("inf"))
        )
        self.assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @parameterized.expand(
        EVALUATE_TESTS
        + [
            (  # Total with GAUSSIAN
                QueryBuilder("private").count(
                    name="total", mechanism=CountMechanism.GAUSSIAN
                ),
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    output_column="total",
                    mechanism=CountMechanism.GAUSSIAN,
                ),
                pd.DataFrame({"total": [4]}),
            ),
            (  # BoundedSTDEV on integer valued measure column with GAUSSIAN
                QueryBuilder("private")
                .groupby(KeySet.from_dict({"A": ["0", "1"]}))
                .stdev(column="B", low=0, high=1, mechanism=StdevMechanism.GAUSSIAN),
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="B",
                    low=0,
                    high=1,
                    mechanism=StdevMechanism.GAUSSIAN,
                    output_column="B_stdev",
                ),
                pd.DataFrame({"A": ["0", "1"], "B_stdev": [0.471405, 0.0]}),
            ),
        ]
    )
    def test_queries_privacy_budget_infinity_rhozcdp(
        self,
        query_expr: QueryExpr,
        expected_expr: Optional[QueryExpr],
        expected_df: pd.DataFrame,
    ):
        """Session :func:`evaluate` returns the correct results for eps=inf and RhoZCDP.

        Args:
            query_expr: The query to evaluate.
            expected_expr: What to expect query_expr to be.
            expected_df: The expected answer.
        """
        if expected_expr is not None:
            self.assertEqual(query_expr, expected_expr)
        session = Session.from_dataframe(
            privacy_budget=RhoZCDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
        )
        session.add_public_dataframe(source_id="public", dataframe=self.join_df)
        session.add_public_dataframe(
            source_id="join_dtypes", dataframe=self.join_dtypes_df
        )
        session.add_public_dataframe(
            source_id="groupby_two_columns", dataframe=self.groupby_two_columns_df
        )
        session.add_public_dataframe(
            source_id="groupby_one_column", dataframe=self.groupby_one_column_df
        )
        session.add_public_dataframe(
            source_id="groupby_with_duplicates",
            dataframe=self.groupby_with_duplicates_df,
        )
        session.add_public_dataframe(
            source_id="groupby_empty", dataframe=self.groupby_empty_df
        )
        actual_sdf = session.evaluate(
            query_expr, privacy_budget=RhoZCDPBudget(float("inf"))
        )
        self.assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @parameterized.expand(
        [(PureDPBudget(float("inf")),), (RhoZCDPBudget(float("inf")),)]
    )
    def test_private_join_privacy_budget_infinity(self, privacy_budget: PrivacyBudget):
        """Session :func:`evaluate` returns correct result for private join, eps=inf."""
        query_expr = GroupByCount(
            child=ReplaceNullAndNan(
                replace_with={},
                child=JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropExcess(3),
                ),
            ),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
        )

        expected_df = pd.DataFrame({"A": ["0", "1"], "count": [3, 1]})
        session = Session.from_dataframe(
            privacy_budget=privacy_budget, source_id="private", dataframe=self.sdf
        )
        session.create_view(
            query_expr=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"C": 1 if row["A"] == "0" else 2}],
                max_num_rows=1,
                schema_new_columns=Schema({"C": "INTEGER"}),
                augment=True,
            ),
            source_id="private_2",
            cache=False,
        )
        actual_sdf = session.evaluate(query_expr, privacy_budget=privacy_budget)
        self.assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @parameterized.expand([(CountMechanism.DEFAULT,), (CountMechanism.LAPLACE,)])
    def test_interactivity_puredp(self, mechanism: CountMechanism):
        """Test that interactivity works with PureDP."""
        query_expr = GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=mechanism,
        )

        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(10), source_id="private", dataframe=self.sdf
        )
        session.evaluate(query_expr, privacy_budget=PureDPBudget(5))
        self.assertEqual(session.remaining_privacy_budget, PureDPBudget(5))
        session.evaluate(query_expr, privacy_budget=PureDPBudget(5))
        self.assertEqual(session.remaining_privacy_budget, PureDPBudget(0))

    @parameterized.expand(
        [
            (CountMechanism.DEFAULT,),
            (CountMechanism.LAPLACE,),
            (CountMechanism.GAUSSIAN,),
        ]
    )
    def test_interactivity_zcdp(self, mechanism: CountMechanism):
        """Test that interactivity works with RhoZCDP."""
        query_expr = GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=mechanism,
        )

        session = Session.from_dataframe(
            privacy_budget=RhoZCDPBudget(10), source_id="private", dataframe=self.sdf
        )
        session.evaluate(query_expr, privacy_budget=RhoZCDPBudget(5))
        self.assertEqual(session.remaining_privacy_budget, RhoZCDPBudget(5))
        session.evaluate(query_expr, privacy_budget=RhoZCDPBudget(5))
        self.assertEqual(session.remaining_privacy_budget, RhoZCDPBudget(0))

    @parameterized.expand([(PureDPBudget(1),), (RhoZCDPBudget(1),)])
    def test_zero_budget(self, budget: PrivacyBudget):
        """Test that a call to `evaluate` raises a ValueError if budget is 0."""
        query_expr = GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=CountMechanism.DEFAULT,
        )
        session = Session.from_dataframe(
            privacy_budget=budget, source_id="private", dataframe=self.sdf
        )
        zero_budget: PrivacyBudget
        if isinstance(budget, PureDPBudget):
            zero_budget = PureDPBudget(0)
        else:
            zero_budget = RhoZCDPBudget(0)
        with self.assertRaisesRegex(
            ValueError, "You need a non-zero privacy budget to evaluate a query."
        ):
            session.evaluate(query_expr, privacy_budget=zero_budget)

    @parameterized.expand(
        [
            (  # GEOMETRIC noise since integer measure_column and PureDP
                PureDPBudget(10000),
                pd.DataFrame({"sum": [12]}),
            ),
            (  # GAUSSIAN noise since RhoZCDP
                RhoZCDPBudget(10000),
                pd.DataFrame({"sum": [12]}),
            ),
        ]
    )
    def test_create_view_with_stability(
        self, privacy_budget: PrivacyBudget, expected: pd.DataFrame
    ):
        """Smoke test for querying on views with stability changes"""
        session = Session.from_dataframe(
            privacy_budget=privacy_budget, source_id="private", dataframe=self.sdf
        )

        transformation_query = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{}, {}],
            max_num_rows=2,
            schema_new_columns=Schema({}),
            augment=True,
        )
        session.create_view(transformation_query, "flatmap_transformation", cache=False)

        sum_query = GroupByBoundedSum(
            child=ReplaceNullAndNan(
                replace_with={}, child=PrivateSource("flatmap_transformation")
            ),
            groupby_keys=KeySet.from_dict({}),
            measure_column="X",
            low=0,
            high=3,
        )
        actual = session.evaluate(sum_query, privacy_budget)
        self.assert_frame_equal_with_sort(actual.toPandas(), expected, rtol=1)

    @parameterized.expand(
        [(PureDPBudget(20), PureDPBudget(10)), (RhoZCDPBudget(20), RhoZCDPBudget(10))]
    )
    def test_partition_and_create(
        self, starting_budget: PrivacyBudget, partition_budget: PrivacyBudget
    ):
        """Tests using :func:`partition_and_create` to create a new session."""
        session1 = Session.from_dataframe(
            privacy_budget=starting_budget, source_id="private", dataframe=self.sdf
        )

        sessions = session1.partition_and_create(
            source_id="private",
            privacy_budget=partition_budget,
            attr_name="A",
            splits={"private0": "0", "private1": "1"},
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]
        self.assertEqual(session1.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.private_sources, ["private0"])
        self.assertEqual(
            session2.get_schema("private0"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )
        self.assertEqual(session3.remaining_privacy_budget, partition_budget)
        self.assertEqual(session3.private_sources, ["private1"])
        self.assertEqual(
            session3.get_schema("private1"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )

    @parameterized.expand(
        [(PureDPBudget(20), PureDPBudget(10)), (RhoZCDPBudget(20), RhoZCDPBudget(10))]
    )
    def test_partition_and_create_query(
        self, starting_budget: PrivacyBudget, partition_budget: PrivacyBudget
    ):
        """Querying on a partitioned session with stability>1 works."""
        session1 = Session.from_dataframe(
            privacy_budget=starting_budget, source_id="private", dataframe=self.sdf
        )

        transformation_query = ReplaceNullAndNan(
            replace_with={},
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda _: [{}, {}],
                max_num_rows=2,
                schema_new_columns=Schema({}),
                augment=True,
            ),
        )
        session1.create_view(transformation_query, "flatmap", True)

        sessions = session1.partition_and_create(
            "flatmap", partition_budget, "A", splits={"private0": "0", "private1": "1"}
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]
        self.assertEqual(session1.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.private_sources, ["private0"])
        self.assertEqual(
            session2.get_schema("private0"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )
        self.assertEqual(session3.remaining_privacy_budget, partition_budget)
        self.assertEqual(session3.private_sources, ["private1"])
        self.assertEqual(
            session3.get_schema("private1"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )
        query = GroupByCount(
            child=PrivateSource("private0"), groupby_keys=KeySet.from_dict({})
        )
        session2.evaluate(query, partition_budget)

    @parameterized.expand(
        [
            (PureDPBudget(float("inf")), CountMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), CountMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), CountMechanism.GAUSSIAN),
        ]
    )
    def test_partition_and_create_correct_answer(
        self, inf_budget: PrivacyBudget, mechanism: CountMechanism
    ):
        """Using :func:`partition_and_create` gives the correct answer if budget=inf."""
        session1 = Session.from_dataframe(
            privacy_budget=inf_budget, source_id="private", dataframe=self.sdf
        )

        sessions = session1.partition_and_create(
            "private", inf_budget, "A", splits={"private0": "0", "private1": "1"}
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]

        answer_session2 = session2.evaluate(
            GroupByCount(
                child=PrivateSource("private0"),
                groupby_keys=KeySet.from_dict({}),
                mechanism=mechanism,
            ),
            inf_budget,
        )
        self.assert_frame_equal_with_sort(
            answer_session2.toPandas(), pd.DataFrame({"count": [3]})
        )
        answer_session3 = session3.evaluate(
            GroupByCount(
                child=PrivateSource("private1"), groupby_keys=KeySet.from_dict({})
            ),
            inf_budget,
        )
        self.assert_frame_equal_with_sort(
            answer_session3.toPandas(), pd.DataFrame({"count": [1]})
        )

    @parameterized.expand([(PureDP(),), (RhoZCDP(),)])
    def test_partitions_composed(self, output_measure: Union[PureDP, RhoZCDP]):
        """Smoke test for composing :func:`partition_and_create`."""
        starting_budget: Union[PureDPBudget, RhoZCDPBudget]
        partition_budget: Union[PureDPBudget, RhoZCDPBudget]
        second_partition_budget: Union[PureDPBudget, RhoZCDPBudget]
        final_evaluate_budget: Union[PureDPBudget, RhoZCDPBudget]
        if output_measure == PureDP():
            starting_budget = PureDPBudget(20)
            partition_budget = PureDPBudget(10)
            second_partition_budget = PureDPBudget(5)
            final_evaluate_budget = PureDPBudget(2)
        elif output_measure == RhoZCDP():
            starting_budget = RhoZCDPBudget(20)
            partition_budget = RhoZCDPBudget(10)
            second_partition_budget = RhoZCDPBudget(5)
            final_evaluate_budget = RhoZCDPBudget(2)
        else:
            self.fail(f"must use PureDP or RhoZCDP, found {output_measure}")

        session1 = Session.from_dataframe(
            privacy_budget=starting_budget, source_id="private", dataframe=self.sdf
        )

        transformation_query1 = ReplaceNullAndNan(
            replace_with={},
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{}, {}],
                max_num_rows=2,
                schema_new_columns=Schema({}),
                augment=True,
            ),
        )
        session1.create_view(transformation_query1, "transform1", cache=False)

        sessions = session1.partition_and_create(
            "transform1",
            partition_budget,
            "A",
            splits={"private0": "0", "private1": "1"},
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]
        self.assertEqual(session1.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.private_sources, ["private0"])
        self.assertEqual(
            session2.get_schema("private0"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )
        self.assertEqual(session3.remaining_privacy_budget, partition_budget)
        self.assertEqual(session3.private_sources, ["private1"])
        self.assertEqual(
            session3.get_schema("private1"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )

        transformation_query2 = ReplaceNullAndNan(
            replace_with={},
            child=FlatMap(
                child=PrivateSource("private0"),
                f=lambda row: [{}, {}, {}],
                max_num_rows=3,
                schema_new_columns=Schema({}),
                augment=True,
            ),
        )
        session2.create_view(transformation_query2, "transform2", cache=False)

        sessions = session2.partition_and_create(
            "transform2",
            second_partition_budget,
            "A",
            splits={"private0": "0", "private1": "1"},
        )
        session4 = sessions["private0"]
        session5 = sessions["private1"]
        self.assertEqual(session2.remaining_privacy_budget, second_partition_budget)
        self.assertEqual(session4.remaining_privacy_budget, second_partition_budget)
        self.assertEqual(session4.private_sources, ["private0"])
        self.assertEqual(
            session4.get_schema("private0"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )
        self.assertEqual(session5.remaining_privacy_budget, second_partition_budget)
        self.assertEqual(session5.private_sources, ["private1"])
        self.assertEqual(
            session5.get_schema("private1"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )

        query = GroupByCount(
            child=PrivateSource("private0"), groupby_keys=KeySet.from_dict({})
        )
        session4.evaluate(query_expr=query, privacy_budget=final_evaluate_budget)

    @parameterized.expand(
        [(PureDPBudget(20), PureDPBudget(10)), (RhoZCDPBudget(20), RhoZCDPBudget(10))]
    )
    def test_partition_execution_order(
        self, starting_budget: PrivacyBudget, partition_budget: PrivacyBudget
    ):
        """Tests behavior using :func:`partition_and_create` sessions out of order."""
        session1 = Session.from_dataframe(
            privacy_budget=starting_budget, source_id="private", dataframe=self.sdf
        )

        sessions = session1.partition_and_create(
            source_id="private",
            privacy_budget=partition_budget,
            attr_name="A",
            splits={"private0": "0", "private1": "1"},
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]
        self.assertEqual(session1.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.remaining_privacy_budget, partition_budget)
        self.assertEqual(session2.private_sources, ["private0"])
        self.assertEqual(
            session2.get_schema("private0"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )
        self.assertEqual(session3.remaining_privacy_budget, partition_budget)
        self.assertEqual(session3.private_sources, ["private1"])
        self.assertEqual(
            session3.get_schema("private1"),
            Schema(
                {
                    "A": ColumnType.VARCHAR,
                    "B": ColumnType.INTEGER,
                    "X": ColumnType.INTEGER,
                }
            ),
        )

        # pylint: disable=protected-access
        self.assertEqual(
            session1._accountant.state, PrivacyAccountantState.WAITING_FOR_CHILDREN
        )
        self.assertEqual(session2._accountant.state, PrivacyAccountantState.ACTIVE)
        self.assertEqual(
            session3._accountant.state, PrivacyAccountantState.WAITING_FOR_SIBLING
        )

        # This should work, but it should also retire session2
        select_query3 = Select(columns=["A"], child=PrivateSource(source_id="private1"))
        session3.create_view(select_query3, "select_view", cache=False)
        self.assertEqual(session2._accountant.state, PrivacyAccountantState.RETIRED)

        # Now trying to do operations on session2 should raise an error
        select_query2 = Select(columns=["A"], child=PrivateSource(source_id="private0"))
        with self.assertRaisesRegex(
            RuntimeError,
            "This session is no longer active, and no new queries can be performed",
        ):
            session2.create_view(select_query2, "select_view", cache=False)

        # This should work, but it should also retire session3
        select_query1 = Select(columns=["A"], child=PrivateSource(source_id="private"))
        session1.create_view(select_query1, "select_view", cache=False)
        self.assertEqual(session3._accountant.state, PrivacyAccountantState.RETIRED)

        # Now trying to do operations on session3 should raise an error
        with self.assertRaisesRegex(
            RuntimeError,
            "This session is no longer active, and no new queries can be performed",
        ):
            session3.create_view(select_query3, "select_view_again", cache=False)

        # pylint: enable=protected-access

    @parameterized.expand([(PureDPBudget(20),), (RhoZCDPBudget(20),)])
    def test_partition_on_grouping_column(self, budget: PrivacyBudget):
        """Tests that you can partition on columns created by grouping flat maps."""
        session = Session.from_dataframe(
            privacy_budget=budget, source_id="private", dataframe=self.sdf
        )
        grouping_flat_map = QueryBuilder("private").flat_map(
            f=lambda row: [{"new": 1}, {"new": 2}],
            max_num_rows=2,
            new_column_types={"new": ColumnType.INTEGER},
            augment=True,
            grouping=True,
        )
        session.create_view(grouping_flat_map, "duplicated", cache=False)
        new_sessions = session.partition_and_create(
            source_id="duplicated",
            privacy_budget=budget,
            attr_name="new",
            splits={"new1": 1, "new2": 2},
        )
        new_sessions["new1"].evaluate(QueryBuilder("new1").count(), budget)
        new_sessions["new2"].evaluate(QueryBuilder("new2").count(), budget)

    @parameterized.expand([(PureDPBudget(20),), (RhoZCDPBudget(20),)])
    def test_partition_on_nongrouping_column(self, budget: PrivacyBudget):
        """Tests that you can partition on other columns after grouping flat maps."""
        session = Session.from_dataframe(
            privacy_budget=budget, source_id="private", dataframe=self.sdf
        )
        grouping_flat_map = QueryBuilder("private").flat_map(
            f=lambda row: [{"new": 1}, {"new": 2}],
            max_num_rows=2,
            new_column_types={"new": ColumnType.INTEGER},
            augment=True,
            grouping=True,
        )
        session.create_view(grouping_flat_map, "duplicated", cache=False)
        new_sessions = session.partition_and_create(
            source_id="duplicated",
            privacy_budget=budget,
            attr_name="A",
            splits={"zero": "0", "one": "1"},
        )
        keys = KeySet.from_dict({"new": [1, 2]})
        new_sessions["zero"].evaluate(
            QueryBuilder("zero").groupby(keys).count(), budget
        )
        new_sessions["one"].evaluate(QueryBuilder("one").groupby(keys).count(), budget)

    @parameterized.expand([(PureDPBudget(20),), (RhoZCDPBudget(20),)])
    def test_create_view_composed(self, budget: PrivacyBudget):
        """Composing views with :func:`create_view` works."""

        session = Session.from_dataframe(
            privacy_budget=budget, source_id="private", dataframe=self.sdf
        )
        transformation_query1 = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{}, {}],
            max_num_rows=2,
            schema_new_columns=Schema({}),
            augment=True,
        )
        session.create_view(transformation_query1, "flatmap1", cache=False)
        self.assertEqual(
            session._stability["flatmap1"], 2  # pylint: disable=protected-access
        )

        transformation_query2 = FlatMap(
            child=PrivateSource("flatmap1"),
            f=lambda row: [{}, {}],
            max_num_rows=3,
            schema_new_columns=Schema({}),
            augment=True,
        )
        session.create_view(transformation_query2, "flatmap2", cache=False)
        self.assertEqual(
            session._stability["flatmap2"], 6  # pylint: disable=protected-access
        )

    @parameterized.expand([(PureDPBudget(10),), (RhoZCDPBudget(10),)])
    def test_create_view_composed_query(self, budget: PrivacyBudget):
        """Smoke test for composing views and querying."""
        session = Session.from_dataframe(
            privacy_budget=budget, source_id="private", dataframe=self.sdf
        )
        transformation_query1 = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{}, {}],
            max_num_rows=2,
            schema_new_columns=Schema({}),
            augment=True,
        )
        session.create_view(transformation_query1, "flatmap1", cache=False)

        transformation_query2 = FlatMap(
            child=PrivateSource("flatmap1"),
            f=lambda row: [{}, {}],
            max_num_rows=3,
            schema_new_columns=Schema({}),
            augment=True,
        )
        session.create_view(transformation_query2, "flatmap2", cache=False)

        # Check that we can query on the view.
        sum_query = GroupByBoundedSum(
            child=ReplaceNullAndNan(replace_with={}, child=PrivateSource("flatmap2")),
            groupby_keys=KeySet.from_dict({}),
            measure_column="X",
            low=0,
            high=3,
        )
        session.evaluate(query_expr=sum_query, privacy_budget=budget)

    @parameterized.expand(
        [
            (PureDPBudget(float("inf")), SumMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), SumMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), SumMechanism.GAUSSIAN),
        ]
    )
    def test_create_view_composed_correct_answer(
        self, inf_budget: PrivacyBudget, mechanism: SumMechanism
    ):
        """Composing :func:`create_view` gives the correct answer if budget=inf."""
        session = Session.from_dataframe(
            privacy_budget=inf_budget, source_id="private", dataframe=self.sdf
        )

        transformation_query1 = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            max_num_rows=1,
            schema_new_columns=Schema({"Repeat": "INTEGER"}),
            augment=True,
        )
        session.create_view(transformation_query1, "flatmap1", cache=False)
        transformation_query2 = FlatMap(
            child=PrivateSource("flatmap1"),
            f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            max_num_rows=2,
            schema_new_columns=Schema({"i": "INTEGER"}),
            augment=False,
        )
        session.create_view(transformation_query2, "flatmap2", cache=False)

        # Check that we can query on the view.
        sum_query = GroupByBoundedSum(
            child=ReplaceNullAndNan(replace_with={}, child=PrivateSource("flatmap2")),
            groupby_keys=KeySet.from_dict({}),
            measure_column="i",
            low=0,
            high=3,
            mechanism=mechanism,
        )
        answer = session.evaluate(sum_query, inf_budget).toPandas()
        expected = pd.DataFrame({"sum": [9]})
        self.assert_frame_equal_with_sort(answer, expected)

    def test_caching(self):
        """Tests that caching works as expected."""
        # pylint: disable=protected-access
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
        )
        view1_query = QueryBuilder("private").filter("B = 0")
        view2_query = QueryBuilder("private").join_public(self.join_df)
        session.create_view(view1_query, "view1", cache=True)
        session.create_view(view2_query, "view2", cache=True)
        # Views have been created, but are lazy - nothing in cache yet
        self.assertEqual(
            len(list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo())), 0
        )
        # Evaluate a query on view1
        session.evaluate(QueryBuilder("view1").count(), privacy_budget=PureDPBudget(1))
        self.assertEqual(
            len(list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo())), 1
        )
        # Evaluate another query on view1
        session.evaluate(QueryBuilder("view1").count(), privacy_budget=PureDPBudget(1))
        self.assertEqual(
            len(list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo())), 1
        )
        # Evaluate a query on view2
        session.evaluate(QueryBuilder("view2").count(), privacy_budget=PureDPBudget(1))
        self.assertEqual(
            len(list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo())), 2
        )
        # Delete views
        session.delete_view("view1")
        self.assertEqual(
            len(list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo())), 1
        )
        session.delete_view("view2")
        self.assertEqual(
            len(list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo())), 0
        )


class TestInvalidSession(PySparkTest):
    """Tests for invalid session."""

    def setUp(self) -> None:
        """Set up test data."""
        # pylint: disable=no-member
        self.sdf = self.spark.createDataFrame(
            pd.DataFrame(
                [["0", 0, 0.0], ["0", 0, 1.0], ["0", 1, 2.0], ["1", 0, 3.0]],
                columns=["A", "B", "X"],
            )
        )
        self.sdf_col_types = {"A": "VARCHAR", "B": "INTEGER", "X": "DECIMAL"}
        self.sdf_input_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(Schema(self.sdf_col_types))
        )

    def test_invalid_data(self):
        """Tests that domain validation produces an appropriate error."""
        sdf = self.spark.createDataFrame(
            [["0", 0, 0.0], ["0", 0, float("nan")], ["0", 1, 2.0], ["1", 0, 3.0]],
            ["A", "B", "X"],
        )
        with self.assertRaisesRegex(
            ValueError, "This DataFrame contains a null or nan value"
        ):
            Session.from_dataframe(
                source_id="private_nan",
                dataframe=sdf,
                privacy_budget=PureDPBudget(float("inf")),
            )

    @parameterized.expand(
        [
            (
                GroupByCount(
                    child=PrivateSource("private_source_not_in_catalog"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
                ),
                ValueError,
                "Query references invalid source 'private_source_not_in_catalog'.",
            )
        ]
    )
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_queries_evaluate(
        self,
        query_expr: QueryExpr,
        error_type: Type[Exception],
        expected_error_msg: str,
        mock_accountant,
    ):
        """evaluate raises error on invalid queries."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": ExactNumber(1)}
        mock_accountant.privacy_budget = ExactNumber(float("inf"))

        session = Session(accountant=mock_accountant, public_sources=dict())
        session.create_view(PrivateSource("private"), "view", cache=False)
        with self.assertRaisesRegex(error_type, expected_error_msg):
            session.evaluate(query_expr, privacy_budget=PureDPBudget(float("inf")))

    @parameterized.expand([(PureDP(),), (RhoZCDP(),)])
    def test_invalid_privacy_budget_evaluate_and_create(
        self, output_measure: Union[PureDP, RhoZCDP]
    ):
        """evaluate and create functions raise error on invalid privacy_budget."""
        one_budget: Union[PureDPBudget, RhoZCDPBudget]
        two_budget: Union[PureDPBudget, RhoZCDPBudget]
        if output_measure == PureDP():
            one_budget = PureDPBudget(1)
            two_budget = PureDPBudget(2)
        elif output_measure == RhoZCDP():
            one_budget = RhoZCDPBudget(1)
            two_budget = RhoZCDPBudget(2)
        else:
            self.fail(f"must use PureDP or RhoZCDP, found {output_measure}")

        query_expr = GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
        )
        session = Session.from_dataframe(
            privacy_budget=one_budget, source_id="private", dataframe=self.sdf
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot answer query without exceeding privacy budget: "
            "it needs approximately 2.000, but the remaining budget is "
            r"approximately 1.000 \(difference: 1.000e\+00\)",
        ):
            session.evaluate(query_expr, privacy_budget=two_budget)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot perform this partition without exceeding privacy budget: "
            "it needs approximately 2.000, but the remaining budget is approximately "
            r"1.000 \(difference: 1.000e\+00\)",
        ):
            session.partition_and_create(
                "private",
                privacy_budget=two_budget,
                attr_name="A",
                splits={"part_0": "0", "part_1": "1"},
            )

    def test_invalid_grouping_with_view(self):
        """Tests that grouping flatmap + rename fails if not used in a later groupby."""
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
        )

        grouping_flatmap = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            max_num_rows=1,
            schema_new_columns=Schema({"Repeat": "INTEGER"}, grouping_column="Repeat"),
            augment=True,
        )
        session.create_view(
            Rename(child=grouping_flatmap, column_mapper={"Repeat": "repeated"}),
            "grouping_flatmap_renamed",
            cache=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Column produced by grouping transformation 'repeated' is not in "
            "groupby columns",
        ):
            session.evaluate(
                query_expr=GroupByBoundedSum(
                    child=ReplaceNullAndNan(
                        replace_with={}, child=PrivateSource("grouping_flatmap_renamed")
                    ),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"]}),
                    measure_column="X",
                    low=0,
                    high=3,
                ),
                privacy_budget=PureDPBudget(10),
            )

    def test_invalid_double_grouping_with_view(self):
        """Tests that multiple grouping transformations aren't allowed."""
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
        )

        grouping_flatmap = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            max_num_rows=1,
            schema_new_columns=Schema({"Repeat": "INTEGER"}, grouping_column="Repeat"),
            augment=True,
        )
        session.create_view(grouping_flatmap, "grouping_flatmap", cache=False)

        grouping_flatmap_2 = FlatMap(
            child=PrivateSource("grouping_flatmap"),
            f=lambda row: [{"i": row["X"]} for _ in range(row["Repeat"])],
            max_num_rows=2,
            schema_new_columns=Schema({"i": "INTEGER"}, grouping_column="i"),
            augment=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Multiple grouping transformations are used in this query. "
            "Only one grouping transformation is allowed.",
        ):
            session.create_view(grouping_flatmap_2, "grouping_flatmap_2", cache=False)


class TestSessionWithNull(PySparkTest):
    """Test Sessions that allow null values."""

    def setUp(self) -> None:
        """Set up tests"""
        self.pdf = pd.DataFrame(
            [
                [
                    "a0",
                    0,
                    0.0,
                    datetime.date(2000, 1, 1),
                    datetime.datetime(2020, 1, 1),
                ],
                [
                    None,
                    1,
                    1.0,
                    datetime.date(2001, 1, 1),
                    datetime.datetime(2021, 1, 1),
                ],
                [
                    "a2",
                    None,
                    2.0,
                    datetime.date(2002, 1, 1),
                    datetime.datetime(2022, 1, 1),
                ],
                [
                    "a3",
                    3,
                    None,
                    datetime.date(2003, 1, 1),
                    datetime.datetime(2023, 1, 1),
                ],
                ["a4", 4, 4.0, None, datetime.datetime(2024, 1, 1)],
                ["a5", 5, 5.0, datetime.date(2005, 1, 1), None],
            ],
            columns=["A", "I", "X", "D", "T"],
        )

        self.sdf_col_types = {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "I": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "X": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
            "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
            "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
        }
        self.sdf = self.spark.createDataFrame(
            self.pdf, schema=analytics_to_spark_schema(Schema(self.sdf_col_types))
        )
        self.sdf_input_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(Schema(self.sdf_col_types))
        )

    def _expected_replace(self, d: Mapping[str, Any]) -> pd.DataFrame:
        """The expected value if you replace None with default values in d."""
        new_cols: List[pd.DataFrame] = []
        for col in list(self.pdf.columns):
            if col in dict(d):
                # make sure I becomes an integer here
                if col == "I":
                    new_cols.append(self.pdf[col].fillna(dict(d)[col]).astype(int))
                else:
                    new_cols.append(self.pdf[col].fillna(dict(d)[col]))
            else:
                new_cols.append(self.pdf[col])
        # `axis=1` means that you want to "concatenate" by columns
        # i.e., you want your new table to look like this:
        # df1 | df2 | df3 | ...
        # df1 | df2 | df3 | ...
        return pd.concat(new_cols, axis=1)

    def test_expected_replace(self) -> None:
        """Test the test method _expected_replace."""
        d = {
            "A": "a999",
            "I": -999,
            "X": 99.9,
            "D": datetime.date(1999, 1, 1),
            "T": datetime.datetime(2019, 1, 1),
        }
        expected = pd.DataFrame(
            [
                [
                    "a0",
                    0,
                    0.0,
                    datetime.date(2000, 1, 1),
                    datetime.datetime(2020, 1, 1),
                ],
                [
                    "a999",
                    1,
                    1.0,
                    datetime.date(2001, 1, 1),
                    datetime.datetime(2021, 1, 1),
                ],
                [
                    "a2",
                    -999,
                    2.0,
                    datetime.date(2002, 1, 1),
                    datetime.datetime(2022, 1, 1),
                ],
                [
                    "a3",
                    3,
                    99.9,
                    datetime.date(2003, 1, 1),
                    datetime.datetime(2023, 1, 1),
                ],
                [
                    "a4",
                    4,
                    4.0,
                    datetime.date(1999, 1, 1),
                    datetime.datetime(2024, 1, 1),
                ],
                [
                    "a5",
                    5,
                    5.0,
                    datetime.date(2005, 1, 1),
                    datetime.datetime(2019, 1, 1),
                ],
            ],
            columns=["A", "I", "X", "D", "T"],
        )
        self.assert_frame_equal_with_sort(self.pdf, self._expected_replace({}))
        self.assert_frame_equal_with_sort(expected, self._expected_replace(d))

    @parameterized.expand(
        [
            ({"A": "aaaaaaa"},),
            ({"I": 999},),
            (
                {
                    "A": "aaa",
                    "I": 999,
                    "X": -99.9,
                    "D": datetime.date.fromtimestamp(0),
                    "T": datetime.datetime.fromtimestamp(0),
                },
            ),
        ]
    )
    def test_replace_null_and_nan(
        self,
        cols_to_defaults: Mapping[
            str, Union[int, float, str, datetime.date, datetime.datetime]
        ],
    ) -> None:
        """Test Session.replace_null_and_nan."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")), "private", self.sdf, validate=False
        )
        session.create_view(
            QueryBuilder("private").replace_null_and_nan(cols_to_defaults),
            "replaced",
            cache=False,
        )
        # pylint: disable=protected-access
        queryable = session._accountant._queryable
        self.assertIsInstance(queryable, SequentialQueryable)
        assert isinstance(queryable, SequentialQueryable)
        data = queryable._data
        self.assertIsInstance(data, dict)
        assert isinstance(data, dict)
        self.assertIsInstance(data["replaced"], DataFrame)
        assert isinstance(data["replaced"], DataFrame)
        # pylint: enable=protected-access
        self.assert_frame_equal_with_sort(
            data["replaced"].toPandas(), self._expected_replace(cols_to_defaults)
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    [[None, 0], [None, 1], ["a2", 1], ["a2", 2]],
                    columns=["A", "new_column"],
                ),
                KeySet.from_dict({"new_column": [0, 1, 2]}),
                pd.DataFrame([[0, 1], [1, 2], [2, 1]], columns=["new_column", "count"]),
            ),
            (
                pd.DataFrame(
                    [["a0", 0, 0], [None, 1, 17], ["a5", 5, 17], ["a5", 5, 400]],
                    columns=["A", "I", "new_column"],
                ),
                KeySet.from_dict({"new_column": [0, 17, 400]}),
                pd.DataFrame(
                    [[0, 1], [17, 2], [400, 1]], columns=["new_column", "count"]
                ),
            ),
            (
                pd.DataFrame(
                    [
                        [datetime.date(2000, 1, 1), "2000"],
                        [datetime.date(2001, 1, 1), "2001"],
                        [None, "none"],
                        [None, "also none"],
                    ],
                    columns=["D", "year"],
                ),
                KeySet.from_dict(
                    {"D": [datetime.date(2000, 1, 1), datetime.date(2001, 1, 1), None]}
                ),
                pd.DataFrame(
                    [
                        [datetime.date(2000, 1, 1), 1],
                        [datetime.date(2001, 1, 1), 1],
                        [None, 2],
                    ],
                    columns=["D", "count"],
                ),
            ),
        ]
    )
    def test_join_public(
        self, public_df: pd.DataFrame, keyset: KeySet, expected: pd.DataFrame
    ) -> None:
        """Test that join_public creates the correct results.

        The query used to evaluate this is a GroupByCount on the new dataframe,
        using the keyset provided.
        """
        session = Session.from_dataframe(
            PureDPBudget(float("inf")), "private", self.sdf, validate=False
        )
        session.add_public_dataframe(
            "public", self.spark.createDataFrame(public_df), validate=False
        )
        result = session.evaluate(
            QueryBuilder("private").join_public("public").groupby(keyset).count(),
            privacy_budget=PureDPBudget(float("inf")),
        )
        self.assert_frame_equal_with_sort(result.toPandas(), expected)

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    [[None, 0], [None, 1], ["a2", 1], ["a2", 2]],
                    columns=["A", "new_column"],
                ),
                KeySet.from_dict({"new_column": [0, 1, 2]}),
                pd.DataFrame([[0, 1], [1, 2], [2, 1]], columns=["new_column", "count"]),
            ),
            (
                pd.DataFrame(
                    [["a0", 0, 0], [None, 1, 17], ["a5", 5, 17], ["a5", 5, 400]],
                    columns=["A", "I", "new_column"],
                ),
                KeySet.from_dict({"new_column": [0, 17, 400]}),
                pd.DataFrame(
                    [[0, 1], [17, 2], [400, 1]], columns=["new_column", "count"]
                ),
            ),
            (
                pd.DataFrame(
                    [
                        [datetime.date(2000, 1, 1), "2000"],
                        [datetime.date(2001, 1, 1), "2001"],
                        [None, "none"],
                        [None, "also none"],
                    ],
                    columns=["D", "year"],
                ),
                KeySet.from_dict(
                    {"D": [datetime.date(2000, 1, 1), datetime.date(2001, 1, 1), None]}
                ),
                pd.DataFrame(
                    [
                        [datetime.date(2000, 1, 1), 1],
                        [datetime.date(2001, 1, 1), 1],
                        [None, 2],
                    ],
                    columns=["D", "count"],
                ),
            ),
        ]
    )
    def test_join_private(
        self, private_df: pd.DataFrame, keyset: KeySet, expected: pd.DataFrame
    ) -> None:
        """Test that join_private creates the correct results.

        The query used to evaluate this is a GroupByCount on the joined dataframe,
        using the keyset provided.
        """
        session = (
            Session.Builder()
            .with_privacy_budget(PureDPBudget(float("inf")))
            .with_private_dataframe("private", self.sdf, validate=False)
            .with_private_dataframe(
                "private2", self.spark.createDataFrame(private_df), validate=False
            )
            .build()
        )
        result = session.evaluate(
            QueryBuilder("private")
            .join_private(
                QueryBuilder("private2"),
                TruncationStrategy.DropExcess(100),
                TruncationStrategy.DropExcess(100),
            )
            .groupby(keyset)
            .count(),
            PureDPBudget(float("inf")),
        )
        self.assert_frame_equal_with_sort(result.toPandas(), expected)


class TestSessionWithInf(PySparkTest):
    """Test Sessions that allow infinite values."""

    def setUp(self) -> None:
        """Set up tests."""
        self.pdf = pd.DataFrame(
            {
                "A": ["a0", "a0", "a1", "a1"],
                "B": [float("-inf"), 2.0, 5.0, float("inf")],
            }
        )
        self.sdf_col_types = {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
        }
        self.sdf = self.spark.createDataFrame(
            self.pdf, schema=analytics_to_spark_schema(Schema(self.sdf_col_types))
        )
        self.sdf_input_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(Schema(self.sdf_col_types))
        )

    @parameterized.expand(
        [
            ({},),
            ({"B": (-100.0, 100.0)},),
            ({"B": (123.45, 678.90)},),
            ({"B": (999.9, 111.1)},),
        ]
    )
    def test_replace_infinity(
        self, replace_with: Dict[str, Tuple[float, float]]
    ) -> None:
        """Test replace_infinity query."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")), "private", self.sdf, validate=False
        )
        session.create_view(
            QueryBuilder("private").replace_infinity(replace_with),
            "replaced",
            cache=False,
        )
        # pylint: disable=protected-access
        queryable = session._accountant._queryable
        self.assertIsInstance(queryable, SequentialQueryable)
        assert isinstance(queryable, SequentialQueryable)
        data = queryable._data
        self.assertIsInstance(data, dict)
        assert isinstance(data, dict)
        self.assertIsInstance(data["replaced"], DataFrame)
        assert isinstance(data["replaced"], DataFrame)
        # pylint: enable=protected-access
        (replace_negative, replace_positive) = replace_with.get(
            "B", (AnalyticsDefault.DECIMAL, AnalyticsDefault.DECIMAL)
        )
        expected = self.pdf.replace(float("-inf"), replace_negative).replace(
            float("inf"), replace_positive
        )
        self.assert_frame_equal_with_sort(data["replaced"].toPandas(), expected)

    @parameterized.expand(
        [
            ({}, pd.DataFrame([["a0", 2.0], ["a1", 5.0]], columns=["A", "sum"])),
            (
                {"B": (-100.0, 100.0)},
                pd.DataFrame([["a0", -98.0], ["a1", 105.0]], columns=["A", "sum"]),
            ),
            (
                {"B": (500.0, 100.0)},
                pd.DataFrame([["a0", 502.0], ["a1", 105.0]], columns=["A", "sum"]),
            ),
        ]
    )
    def test_sum(
        self, replace_with: Dict[str, Tuple[float, float]], expected: pd.DataFrame
    ) -> None:
        """Test GroupByBoundedSum after replacing infinite values."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")), "private", self.sdf, validate=False
        )
        result = session.evaluate(
            QueryBuilder("private")
            .replace_infinity(replace_with)
            .groupby(KeySet.from_dict({"A": ["a0", "a1"]}))
            .sum("B", low=-1000, high=1000, name="sum"),
            PureDPBudget(float("inf")),
        )
        self.assert_frame_equal_with_sort(result.toPandas(), expected)

    def test_drop_invalid(self):
        """Test GroupByBoundedSum after dropping infinite values."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")), "private", self.sdf, validate=False
        )
        result = session.evaluate(
            QueryBuilder("private")
            .drop_invalid(columns=["B"])
            .groupby(KeySet.from_dict({"A": ["a0", "a1"]}))
            .sum("B", low=-1000, high=1000, name="sum"),
            PureDPBudget(float("inf")),
        )
        expected = pd.DataFrame([["a0", 2.0], ["a1", 5.0]], columns=["A", "sum"])
        self.assert_frame_equal_with_sort(result.toPandas(), expected)
