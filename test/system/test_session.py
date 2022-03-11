"""System tests for Session."""

# <placeholder: boilerplate>

import os
import shutil
import tempfile
from typing import Type, Union
from unittest.mock import patch

import pandas as pd
from parameterized import parameterized
from pyspark.sql.types import StructType

from tmlt.analytics._schema import (
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.query_expr import (
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
    Select,
    StdevMechanism,
    SumMechanism,
)
from tmlt.analytics.session import Session
from tmlt.analytics.truncation_strategy import TruncationStrategy
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import PySparkTest

EVALUATE_TESTS = [
    (  # Total with DEFAULT mechanism
        # (Geometric noise gets applied if PureDP; Gaussian noise gets applied if ZCDP)
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Total with DEFAULT mechanism
        # (Geometric noise gets applied if PureDP; Gaussian noise gets applied if ZCDP)
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Total with LAPLACE (Geometric noise gets applied)
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=CountMechanism.LAPLACE,
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Total with LAPLACE (Geometric noise gets applied)
        GroupByCountDistinct(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=CountDistinctMechanism.LAPLACE,
        ),
        pd.DataFrame({"total": [4]}),
    ),
    (  # Full marginal from domain description (Geometric noise gets applied)
        GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
        ),
        pd.DataFrame(
            {"A": ["0", "0", "1", "1"], "B": [0, 1, 0, 1], "count": [2, 1, 1, 0]}
        ),
    ),
    (  # Full marginal from domain description (Geometric noise gets applied)
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
        GroupByCount(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_two_columns"),
        ),
        pd.DataFrame({"A": ["0", "0", "1"], "B": [0, 1, 1], "count": [2, 1, 0]}),
    ),
    (  # Incomplete two-column marginal with a dataframe
        GroupByCountDistinct(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_two_columns"),
        ),
        pd.DataFrame(
            {"A": ["0", "0", "1"], "B": [0, 1, 1], "count_distinct": [2, 1, 0]}
        ),
    ),
    (  # One-column marginal with additional value
        GroupByCount(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_one_column"),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count": [3, 1, 0]}),
    ),
    (  # One-column marginal with additional value
        GroupByCountDistinct(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_one_column"),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count_distinct": [3, 1, 0]}),
    ),
    (  # One-column marginal with duplicate rows
        GroupByCount(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_with_duplicates"),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count": [3, 1, 0]}),
    ),
    (  # One-column marginal with duplicate rows
        GroupByCountDistinct(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_with_duplicates"),
        ),
        pd.DataFrame({"A": ["0", "1", "2"], "count_distinct": [3, 1, 0]}),
    ),
    (  # empty public source
        GroupByCount(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_empty"),
        ),
        pd.DataFrame({"count": [4]}),
    ),
    (  # empty public source
        GroupByCountDistinct(
            child=PrivateSource("private"),
            # pylint: disable=protected-access
            groupby_keys=KeySet._from_public_source("groupby_empty"),
        ),
        pd.DataFrame({"count_distinct": [4]}),
    ),
    (  # BoundedSum
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
        GroupByBoundedSum(
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda _: [{}, {}],
                max_num_rows=2,
                schema_new_columns=Schema({}),
                augment=True,
            ),
            groupby_keys=KeySet.from_dict({}),
            measure_column="X",
            low=0,
            high=3,
        ),
        pd.DataFrame({"sum": [12]}),
    ),
    (  # Multiple flat maps on integer-valued measure_column
        # (Geometric noise gets applied if PureDP; Gaussian noise gets applied if ZCDP)
        GroupByBoundedSum(
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
            groupby_keys=KeySet.from_dict({}),
            measure_column="i",
            low=0,
            high=3,
        ),
        pd.DataFrame({"sum": [9]}),
    ),
    (  # Grouping flat map with DEFAULT mechanism and integer-valued measure column
        # (Geometric noise gets applied)
        GroupByBoundedSum(
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
            groupby_keys=KeySet.from_dict({"Repeat": [1, 2]}),
            measure_column="i",
            low=0,
            high=3,
        ),
        pd.DataFrame({"Repeat": [1, 2], "sum": [3, 6]}),
    ),
    (  # Grouping flat map with LAPLACE mechanism (Geometric noise gets applied)
        GroupByBoundedSum(
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
            groupby_keys=KeySet.from_dict({"Repeat": [1, 2]}),
            measure_column="i",
            low=0,
            high=3,
            mechanism=SumMechanism.LAPLACE,
        ),
        pd.DataFrame({"Repeat": [1, 2], "sum": [3, 6]}),
    ),
    (  # GroupByCount Filter
        GroupByCount(
            child=Filter(child=PrivateSource("private"), predicate="A == '0'"),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count": [3]}),
    ),
    (  # GroupByCountDistinct Filter
        GroupByCountDistinct(
            child=Filter(child=PrivateSource("private"), predicate="A == '0'"),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count_distinct": [3]}),
    ),
    (  # GroupByCount Select
        GroupByCount(
            child=Select(child=PrivateSource("private"), columns=["A"]),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count": [4]}),
    ),
    (  # GroupByCountDistinct Select
        GroupByCountDistinct(
            child=Select(child=PrivateSource("private"), columns=["A"]),
            groupby_keys=KeySet.from_dict({}),
        ),
        pd.DataFrame({"count_distinct": [2]}),
    ),
    (  # GroupByCount Map
        GroupByCount(
            child=Map(
                child=PrivateSource("private"),
                f=lambda row: {"C": 2 * str(row["B"])},
                schema_new_columns=Schema({"C": "VARCHAR"}),
                augment=True,
            ),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"], "C": ["00", "11"]}),
        ),
        pd.DataFrame(
            [["0", "00", 2], ["0", "11", 1], ["1", "00", 1], ["1", "11", 0]],
            columns=["A", "C", "count"],
        ),
    ),
    (  # GroupByCountDistinct Map
        GroupByCountDistinct(
            child=Map(
                child=PrivateSource("private"),
                f=lambda row: {"C": 2 * str(row["B"])},
                schema_new_columns=Schema({"C": "VARCHAR"}),
                augment=True,
            ),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"], "C": ["00", "11"]}),
        ),
        pd.DataFrame(
            [["0", "00", 2], ["0", "11", 1], ["1", "00", 1], ["1", "11", 0]],
            columns=["A", "C", "count_distinct"],
        ),
    ),
    (  # GroupByCount JoinPublic
        GroupByCount(
            child=JoinPublic(child=PrivateSource("private"), public_table="public"),
            groupby_keys=KeySet.from_dict({"A+B": [0, 1, 2]}),
        ),
        pd.DataFrame({"A+B": [0, 1, 2], "count": [3, 4, 1]}),
    ),
    (  # GroupByCountDistinct JoinPublic
        GroupByCountDistinct(
            child=JoinPublic(child=PrivateSource("private"), public_table="public"),
            groupby_keys=KeySet.from_dict({"A+B": [0, 1, 2]}),
        ),
        pd.DataFrame({"A+B": [0, 1, 2], "count_distinct": [3, 4, 1]}),
    ),
]


class TestEvaluate(PySparkTest):
    """Unit tests for evaluate."""

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

        self.data_dir = tempfile.mkdtemp()
        self.private_csv_path = os.path.join(self.data_dir, "private.csv")
        self.public_csv_path = os.path.join(self.data_dir, "public.csv")
        private_csv = """A,B,X
0,0,0
0,0,1
0,1,2
1,0,3"""
        public_csv = """A,A+B,EXTRA
0,0,A
0,1,B
1,1,C
1,2,D"""
        with open(self.private_csv_path, "w") as f:
            f.write(private_csv)
            f.flush()
        with open(self.public_csv_path, "w") as f:
            f.write(public_csv)
            f.flush()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.data_dir)

    @parameterized.expand(EVALUATE_TESTS)
    def test_queries_privacy_budget_infinity_puredp(
        self, query_expr: QueryExpr, expected_df: pd.DataFrame
    ):
        """Session :func:`evaluate` returns the correct results for eps=inf and PureDP.

        Args:
            query_expr: The query to evaluate.
            expected_df: The expected answer.
        """
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
        )
        session.add_public_dataframe(source_id="public", dataframe=self.join_df)
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
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    output_column="total",
                    mechanism=CountMechanism.GAUSSIAN,
                ),
                pd.DataFrame({"total": [4]}),
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
                pd.DataFrame({"A": ["0", "1"], "stdev": [0.471405, 0.0]}),
            ),
        ]
    )
    def test_queries_privacy_budget_infinity_rhozcdp(
        self, query_expr: QueryExpr, expected_df: pd.DataFrame
    ):
        """Session :func:`evaluate` returns the correct results for eps=inf and RhoZCDP.

        Args:
            query_expr: The query to evaluate.
            expected_df: The expected answer.
        """
        session = Session.from_dataframe(
            privacy_budget=RhoZCDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
        )
        session.add_public_dataframe(source_id="public", dataframe=self.join_df)
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
            child=JoinPrivate(
                child=PrivateSource("private"),
                right_operand_expr=PrivateSource("private_2"),
                truncation_strategy_left=TruncationStrategy.DropExcess(3),
                truncation_strategy_right=TruncationStrategy.DropExcess(3),
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
            child=PrivateSource("flatmap_transformation"),
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
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
        )
        self.assertEqual(session3.remaining_privacy_budget, partition_budget)
        self.assertEqual(session3.private_sources, ["private1"])
        self.assertEqual(
            session3.get_schema("private1"),
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
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

        transformation_query = FlatMap(
            child=PrivateSource("private"),
            f=lambda _: [{}, {}],
            max_num_rows=2,
            schema_new_columns=Schema({}),
            augment=True,
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
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
        )
        self.assertEqual(session3.remaining_privacy_budget, partition_budget)
        self.assertEqual(session3.private_sources, ["private1"])
        self.assertEqual(
            session3.get_schema("private1"),
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
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

        transformation_query1 = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{}, {}],
            max_num_rows=2,
            schema_new_columns=Schema({}),
            augment=True,
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
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
        )
        self.assertEqual(session3.remaining_privacy_budget, partition_budget)
        self.assertEqual(session3.private_sources, ["private1"])
        self.assertEqual(
            session3.get_schema("private1"),
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
        )

        transformation_query2 = FlatMap(
            child=PrivateSource("private0"),
            f=lambda row: [{}, {}, {}],
            max_num_rows=3,
            schema_new_columns=Schema({}),
            augment=True,
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
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
        )
        self.assertEqual(session5.remaining_privacy_budget, second_partition_budget)
        self.assertEqual(session5.private_sources, ["private1"])
        self.assertEqual(
            session5.get_schema("private1"),
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER},
        )

        query = GroupByCount(
            child=PrivateSource("private0"), groupby_keys=KeySet.from_dict({})
        )
        session4.evaluate(query_expr=query, privacy_budget=final_evaluate_budget)

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
            child=PrivateSource("flatmap2"),
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
            child=PrivateSource("flatmap2"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="i",
            low=0,
            high=3,
            mechanism=mechanism,
        )
        answer = session.evaluate(sum_query, inf_budget).toPandas()
        expected = pd.DataFrame({"sum": [9]})
        self.assert_frame_equal_with_sort(answer, expected)


class TestInvalidSession(PySparkTest):
    """Unit tests for invalid session."""

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
            ValueError,
            "Tumult Analytics does not yet handle DataFrames containing null or nan"
            " values",
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
    @patch("tmlt.analytics.session.AdaptiveCompositionQueryable")
    def test_invalid_queries_evaluate(
        self,
        query_expr: QueryExpr,
        error_type: Type[Exception],
        expected_error_msg: str,
        mock_queryable,
    ):
        """evaluate raises error on invalid queries."""
        mock_queryable.output_measure = PureDP()
        mock_queryable.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_queryable.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_queryable.d_in = {"private": ExactNumber(1)}
        mock_queryable.remaining_budget = ExactNumber(float("inf"))

        session = Session(queryable=mock_queryable, public_sources=dict())
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
            "Cannot answer measurement without exceeding maximum privacy loss: "
            "it needs 2, but the remaining budget is 1",
        ):
            session.evaluate(query_expr, privacy_budget=two_budget)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot answer measurement without exceeding maximum privacy loss: "
            "it needs 2, but the remaining budget is 1",
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
                    child=PrivateSource("grouping_flatmap_renamed"),
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
