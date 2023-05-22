"""Tests for Session with the AddMaxRows ProtectedChange."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=no-self-use

import math
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from tmlt.core.measurements.interactive_measurements import PrivacyAccountantState
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.parameters import calculate_noise_scale

from tmlt.analytics._noise_info import _NoiseMechanism
from tmlt.analytics._schema import ColumnType, Schema
from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)
from tmlt.analytics.protected_change import AddOneRow
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import (
    AverageMechanism,
    CountMechanism,
    FlatMap,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByCount,
    JoinPrivate,
    PrivateSource,
    QueryExpr,
    ReplaceNullAndNan,
    Select,
    StdevMechanism,
    SumMechanism,
)
from tmlt.analytics.session import Session
from tmlt.analytics.truncation_strategy import TruncationStrategy

from ....conftest import assert_frame_equal_with_sort
from .conftest import EVALUATE_TESTS


@pytest.mark.usefixtures("session_data")
class TestSession:
    """Tests for Valid Sessions."""

    sdf: DataFrame
    join_df: DataFrame
    join_dtypes_df: DataFrame
    groupby_two_columns_df: DataFrame
    groupby_one_column_df: DataFrame
    groupby_with_duplicates_df: DataFrame
    groupby_empty_df: DataFrame

    @pytest.mark.parametrize("query_expr,expected_expr,expected_df", EVALUATE_TESTS)
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
            assert query_expr == expected_expr
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
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
        assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @pytest.mark.parametrize(
        "query_expr,expected_expr,expected_df",
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
        ],
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
            assert query_expr == expected_expr
        session = Session.from_dataframe(
            privacy_budget=RhoZCDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
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
        assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @pytest.mark.parametrize(
        "query_expr,session_budget,query_budget,expected",
        [
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.LAPLACE,
                ),
                PureDPBudget(11),
                PureDPBudget(7),
                [
                    {
                        "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                        "noise_parameter": (1.0 / 7.0),
                    }
                ],
            ),
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    low=-111,
                    high=234,
                    mechanism=AverageMechanism.GAUSSIAN,
                    measure_column="X",
                ),
                RhoZCDPBudget(31),
                RhoZCDPBudget(11),
                [
                    # Noise for the sum query (which uses half the budget)
                    {
                        "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                        # the upper and lower bounds of the sum aggregation
                        # are -173 and 172;
                        # this is (lower - midpoint) and (upper-midpoint) respectively
                        "noise_parameter": (
                            calculate_noise_scale(
                                173, ExactNumber(11) / ExactNumber(2), RhoZCDP()
                            )
                            ** 2
                        ).to_float(round_up=False),
                    },
                    # Noise for the count query (which uses half the budget)
                    {
                        "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                        "noise_parameter": (
                            calculate_noise_scale(
                                1, ExactNumber(11) / ExactNumber(2), RhoZCDP()
                            )
                            ** 2
                        ).to_float(round_up=False),
                    },
                ],
            ),
        ],
    )
    def test_noise_info(
        self,
        query_expr: QueryExpr,
        session_budget: PrivacyBudget,
        query_budget: PrivacyBudget,
        expected: List[Dict[str, Any]],
    ):
        """Test _noise_info."""
        session = Session.from_dataframe(
            privacy_budget=session_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        # pylint: disable=protected-access
        info = session._noise_info(query_expr, query_budget)
        # pylint: enable=protected-access
        assert info == expected

    @pytest.mark.parametrize(
        "privacy_budget", [(PureDPBudget(float("inf"))), (RhoZCDPBudget(float("inf")))]
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
            privacy_budget=privacy_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        session.create_view(
            query_expr=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"C": 1 if row["A"] == "0" else 2}],
                schema_new_columns=Schema({"C": "INTEGER"}),
                augment=True,
                max_num_rows=1,
            ),
            source_id="private_2",
            cache=False,
        )
        actual_sdf = session.evaluate(query_expr, privacy_budget=privacy_budget)
        assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @pytest.mark.parametrize(
        "mechanism", [(CountMechanism.DEFAULT), (CountMechanism.LAPLACE)]
    )
    def test_interactivity_puredp(self, mechanism: CountMechanism):
        """Test that interactivity works with PureDP."""
        query_expr = GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=mechanism,
        )

        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(10),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        session.evaluate(query_expr, privacy_budget=PureDPBudget(5))
        assert session.remaining_privacy_budget == PureDPBudget(5)
        session.evaluate(query_expr, privacy_budget=PureDPBudget(5))
        assert session.remaining_privacy_budget == PureDPBudget(0)

    @pytest.mark.parametrize(
        "mechanism",
        [(CountMechanism.DEFAULT), (CountMechanism.LAPLACE), (CountMechanism.GAUSSIAN)],
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
            privacy_budget=RhoZCDPBudget(10),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        session.evaluate(query_expr, privacy_budget=RhoZCDPBudget(5))
        assert session.remaining_privacy_budget == RhoZCDPBudget(5)
        session.evaluate(query_expr, privacy_budget=RhoZCDPBudget(5))
        assert session.remaining_privacy_budget == RhoZCDPBudget(0)

    @pytest.mark.parametrize(
        "budget",
        [
            (PureDPBudget(1)),
            (ApproxDPBudget(0, 0.5)),
            (ApproxDPBudget(1, 0)),
            (RhoZCDPBudget(1)),
        ],
    )
    def test_zero_budget(self, budget: PrivacyBudget):
        """Test that a call to ``evaluate`` raises a ValueError if budget is 0."""
        query_expr = GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            output_column="total",
            mechanism=CountMechanism.DEFAULT,
        )
        session = Session.from_dataframe(
            privacy_budget=budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        zero_budget: PrivacyBudget
        if isinstance(budget, PureDPBudget):
            zero_budget = PureDPBudget(0)
        elif isinstance(budget, ApproxDPBudget):
            zero_budget = ApproxDPBudget(0, 0)
        else:
            zero_budget = RhoZCDPBudget(0)
        with pytest.raises(
            ValueError, match="You need a non-zero privacy budget to evaluate a query."
        ):
            session.evaluate(query_expr, privacy_budget=zero_budget)

    @pytest.mark.parametrize(
        "privacy_budget,expected",
        [
            (  # GEOMETRIC noise since integer measure_column and PureDP
                PureDPBudget(10000),
                pd.DataFrame({"sum": [12]}),
            ),
            (  # GAUSSIAN noise since RhoZCDP
                RhoZCDPBudget(10000),
                pd.DataFrame({"sum": [12]}),
            ),
        ],
    )
    def test_create_view_with_stability(
        self, privacy_budget: PrivacyBudget, expected: pd.DataFrame
    ):
        """Smoke test for querying on views with stability changes"""
        session = Session.from_dataframe(
            privacy_budget=privacy_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        transformation_query = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{}, {}],
            schema_new_columns=Schema({}),
            augment=True,
            max_num_rows=1,
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
        assert_frame_equal_with_sort(actual.toPandas(), expected, rtol=1)

    @pytest.mark.parametrize(
        "starting_budget,partition_budget",
        [
            (PureDPBudget(20), PureDPBudget(10)),
            (ApproxDPBudget(20, 0.5), ApproxDPBudget(10, 0)),
            (RhoZCDPBudget(20), RhoZCDPBudget(10)),
        ],
    )
    def test_partition_and_create(
        self, starting_budget: PrivacyBudget, partition_budget: PrivacyBudget
    ):
        """Tests using :func:`partition_and_create` to create a new session."""
        session1 = Session.from_dataframe(
            privacy_budget=starting_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        sessions = session1.partition_and_create(
            source_id="private",
            privacy_budget=partition_budget,
            column="A",
            splits={"private0": "0", "private1": "1"},
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]
        assert session1.remaining_privacy_budget == partition_budget
        assert session2.remaining_privacy_budget == partition_budget
        assert session2.private_sources == ["private0"]
        assert session2.get_schema("private0") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )
        assert session3.remaining_privacy_budget == partition_budget
        assert session3.private_sources == ["private1"]
        assert session3.get_schema("private1") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )

    @pytest.mark.parametrize(
        "starting_budget,partition_budget",
        [
            (PureDPBudget(20), PureDPBudget(10)),
            (ApproxDPBudget(20, 0.5), ApproxDPBudget(10, 0)),
            (RhoZCDPBudget(20), RhoZCDPBudget(10)),
        ],
    )
    def test_partition_and_create_query(
        self, starting_budget: PrivacyBudget, partition_budget: PrivacyBudget
    ):
        """Querying on a partitioned session with stability>1 works."""
        session1 = Session.from_dataframe(
            privacy_budget=starting_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        transformation_query = ReplaceNullAndNan(
            replace_with={},
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda _: [{}, {}],
                schema_new_columns=Schema({}),
                augment=True,
                max_num_rows=2,
            ),
        )
        session1.create_view(transformation_query, "flatmap", True)

        sessions = session1.partition_and_create(
            "flatmap", partition_budget, "A", splits={"private0": "0", "private1": "1"}
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]
        assert session1.remaining_privacy_budget == partition_budget
        assert session2.remaining_privacy_budget == partition_budget
        assert session2.private_sources == ["private0"]
        assert session2.get_schema("private0") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )
        assert session3.remaining_privacy_budget == partition_budget
        assert session3.private_sources == ["private1"]
        assert session3.get_schema("private1") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )
        query = GroupByCount(
            child=PrivateSource("private0"), groupby_keys=KeySet.from_dict({})
        )
        session2.evaluate(query, partition_budget)

    @pytest.mark.parametrize(
        "starting_budget,partition_budget",
        [(ApproxDPBudget(20, 0.5), PureDPBudget(10))],
    )
    def test_partition_and_create_approxDP_session_pureDP_partition(
        self, starting_budget: PrivacyBudget, partition_budget: PrivacyBudget
    ):
        """Tests using :func:`partition_and_create` to create a new ApproxDP session
        that supports PureDP partitions."""

        is_approxDP_starting_budget = isinstance(starting_budget, ApproxDPBudget)
        if is_approxDP_starting_budget and isinstance(partition_budget, PureDPBudget):
            partition_budget = ApproxDPBudget(partition_budget.epsilon, 0)

        self.test_partition_and_create_query(starting_budget, partition_budget)

    @pytest.mark.parametrize(
        "inf_budget,mechanism",
        [
            (PureDPBudget(float("inf")), CountMechanism.LAPLACE),
            (ApproxDPBudget(float("inf"), 0.5), CountMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), CountMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), CountMechanism.GAUSSIAN),
        ],
    )
    def test_partition_and_create_correct_answer(
        self, inf_budget: PrivacyBudget, mechanism: CountMechanism
    ):
        """Using :func:`partition_and_create` gives the correct answer if budget=inf."""
        session1 = Session.from_dataframe(
            privacy_budget=inf_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
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
        assert_frame_equal_with_sort(
            answer_session2.toPandas(), pd.DataFrame({"count": [3]})
        )
        answer_session3 = session3.evaluate(
            GroupByCount(
                child=PrivateSource("private1"), groupby_keys=KeySet.from_dict({})
            ),
            inf_budget,
        )
        assert_frame_equal_with_sort(
            answer_session3.toPandas(), pd.DataFrame({"count": [1]})
        )

    @pytest.mark.parametrize("output_measure", [(PureDP()), (ApproxDP()), (RhoZCDP())])
    def test_partitions_composed(
        self, output_measure: Union[PureDP, ApproxDP, RhoZCDP]
    ):
        """Smoke test for composing :func:`partition_and_create`."""
        starting_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        partition_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        second_partition_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        final_evaluate_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        if output_measure == PureDP():
            starting_budget = PureDPBudget(20)
            partition_budget = PureDPBudget(10)
            second_partition_budget = PureDPBudget(5)
            final_evaluate_budget = PureDPBudget(2)
        elif output_measure == ApproxDP():
            starting_budget = ApproxDPBudget(20, 0)
            partition_budget = ApproxDPBudget(10, 0)
            second_partition_budget = ApproxDPBudget(5, 0)
            final_evaluate_budget = ApproxDPBudget(2, 0)
        elif output_measure == RhoZCDP():
            starting_budget = RhoZCDPBudget(20)
            partition_budget = RhoZCDPBudget(10)
            second_partition_budget = RhoZCDPBudget(5)
            final_evaluate_budget = RhoZCDPBudget(2)
        else:
            pytest.fail(
                f"must use PureDP, ApproxDP, or RhoZCDP, found {output_measure}"
            )

        session1 = Session.from_dataframe(
            privacy_budget=starting_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        transformation_query1 = ReplaceNullAndNan(
            replace_with={},
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{}, {}],
                schema_new_columns=Schema({}),
                augment=True,
                max_num_rows=2,
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
        assert session1.remaining_privacy_budget == partition_budget
        assert session2.remaining_privacy_budget == partition_budget
        assert session2.private_sources == ["private0"]
        assert session2.get_schema("private0") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )
        assert session3.remaining_privacy_budget == partition_budget
        assert session3.private_sources == ["private1"]
        assert session3.get_schema("private1") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )

        transformation_query2 = ReplaceNullAndNan(
            replace_with={},
            child=FlatMap(
                child=PrivateSource("private0"),
                f=lambda row: [{}, {}, {}],
                schema_new_columns=Schema({}),
                augment=True,
                max_num_rows=2,
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
        assert session2.remaining_privacy_budget == second_partition_budget
        assert session4.remaining_privacy_budget == second_partition_budget
        assert session4.private_sources == ["private0"]
        assert session4.get_schema("private0") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )
        assert session5.remaining_privacy_budget == second_partition_budget
        assert session5.private_sources == ["private1"]
        assert session5.get_schema("private1") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )

        query = GroupByCount(
            child=PrivateSource("private0"), groupby_keys=KeySet.from_dict({})
        )
        session4.evaluate(query_expr=query, privacy_budget=final_evaluate_budget)

    @pytest.mark.parametrize(
        "starting_budget,partition_budget",
        [
            (PureDPBudget(20), PureDPBudget(10)),
            (ApproxDPBudget(20, 0.5), ApproxDPBudget(10, 0)),
            (RhoZCDPBudget(20), RhoZCDPBudget(10)),
        ],
    )
    def test_partition_execution_order(
        self, starting_budget: PrivacyBudget, partition_budget: PrivacyBudget
    ):
        """Tests behavior using :func:`partition_and_create` sessions out of order."""
        session1 = Session.from_dataframe(
            privacy_budget=starting_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        sessions = session1.partition_and_create(
            source_id="private",
            privacy_budget=partition_budget,
            column="A",
            splits={"private0": "0", "private1": "1"},
        )
        session2 = sessions["private0"]
        session3 = sessions["private1"]
        assert session1.remaining_privacy_budget == partition_budget
        assert session2.remaining_privacy_budget == partition_budget
        assert session2.private_sources == ["private0"]
        assert session2.get_schema("private0") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )
        assert session3.remaining_privacy_budget == partition_budget
        assert session3.private_sources == ["private1"]
        assert session3.get_schema("private1") == Schema(
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER, "X": ColumnType.INTEGER}
        )

        # pylint: disable=protected-access
        assert session1._accountant.state == PrivacyAccountantState.WAITING_FOR_CHILDREN
        assert session2._accountant.state == PrivacyAccountantState.ACTIVE
        assert session3._accountant.state == PrivacyAccountantState.WAITING_FOR_SIBLING

        # This should work, but it should also retire session2
        select_query3 = Select(columns=["A"], child=PrivateSource(source_id="private1"))
        session3.create_view(select_query3, "select_view", cache=False)
        assert session2._accountant.state == PrivacyAccountantState.RETIRED

        # Now trying to do operations on session2 should raise an error
        select_query2 = Select(columns=["A"], child=PrivateSource(source_id="private0"))
        with pytest.raises(
            RuntimeError,
            match=(
                "This session is no longer active, and no new queries can be performed"
            ),
        ):
            session2.create_view(select_query2, "select_view", cache=False)

        # This should work, but it should also retire session3
        select_query1 = Select(columns=["A"], child=PrivateSource(source_id="private"))
        session1.create_view(select_query1, "select_view", cache=False)
        assert session3._accountant.state == PrivacyAccountantState.RETIRED

        # Now trying to do operations on session3 should raise an error
        with pytest.raises(
            RuntimeError,
            match=(
                "This session is no longer active, and no new queries can be performed"
            ),
        ):
            session3.create_view(select_query3, "select_view_again", cache=False)

        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "budget", [(PureDPBudget(20)), (ApproxDPBudget(20, 0.5)), (RhoZCDPBudget(20))]
    )
    def test_partition_on_flatmap_grouping_column(self, budget: PrivacyBudget):
        """Tests that you can partition on columns created by grouping flat maps."""
        session = Session.from_dataframe(
            privacy_budget=budget, source_id="private", dataframe=self.sdf
        )
        grouping_flat_map = QueryBuilder("private").flat_map(
            f=lambda row: [{"new": 1}, {"new": 2}],
            new_column_types={"new": ColumnType.INTEGER},
            augment=True,
            grouping=True,
            max_num_rows=2,
        )
        session.create_view(grouping_flat_map, "duplicated", cache=False)
        new_sessions = session.partition_and_create(
            source_id="duplicated",
            privacy_budget=budget,
            column="new",
            splits={"new1": 1, "new2": 2},
        )
        new_sessions["new1"].evaluate(QueryBuilder("new1").count(), budget)
        new_sessions["new2"].evaluate(QueryBuilder("new2").count(), budget)

    @pytest.mark.parametrize(
        "budget", [(PureDPBudget(20)), (ApproxDPBudget(20, 0.5)), (RhoZCDPBudget(20))]
    )
    def test_partition_on_nongrouping_column(self, budget: PrivacyBudget):
        """Tests that you can partition on other columns after grouping flat maps."""
        session = Session.from_dataframe(
            privacy_budget=budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        grouping_flat_map = QueryBuilder("private").flat_map(
            f=lambda row: [{"new": 1}, {"new": 2}],
            new_column_types={"new": ColumnType.INTEGER},
            augment=True,
            grouping=True,
            max_num_rows=2,
        )
        session.create_view(grouping_flat_map, "duplicated", cache=False)
        new_sessions = session.partition_and_create(
            source_id="duplicated",
            privacy_budget=budget,
            column="A",
            splits={"zero": "0", "one": "1"},
        )
        keys = KeySet.from_dict({"new": [1, 2]})
        new_sessions["zero"].evaluate(
            QueryBuilder("zero").groupby(keys).count(), budget
        )
        new_sessions["one"].evaluate(QueryBuilder("one").groupby(keys).count(), budget)

    @pytest.mark.parametrize(
        "budget", [(PureDPBudget(20)), (ApproxDPBudget(20, 0.5)), (RhoZCDPBudget(20))]
    )
    def test_create_view_composed(self, budget: PrivacyBudget):
        """Composing views with :func:`create_view` works."""

        session = Session.from_dataframe(
            privacy_budget=budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        transformation_query1 = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{}, {}],
            schema_new_columns=Schema({}),
            augment=True,
            max_num_rows=2,
        )
        session.create_view(transformation_query1, "flatmap1", cache=False)
        # pylint: disable=protected-access
        assert session._accountant.d_in[NamedTable("flatmap1")] == 2
        # pylint: enable=protected-access

        transformation_query2 = FlatMap(
            child=PrivateSource("flatmap1"),
            f=lambda row: [{}, {}],
            schema_new_columns=Schema({}),
            augment=True,
            max_num_rows=3,
        )
        session.create_view(transformation_query2, "flatmap2", cache=False)
        # pylint: disable=protected-access
        assert session._accountant.d_in[NamedTable("flatmap2")] == 6
        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "budget", [(PureDPBudget(10)), (ApproxDPBudget(10, 0.5)), (RhoZCDPBudget(10))]
    )
    def test_create_view_composed_query(self, budget: PrivacyBudget):
        """Smoke test for composing views and querying."""
        session = Session.from_dataframe(
            privacy_budget=budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        transformation_query1 = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{}, {}],
            schema_new_columns=Schema({}),
            augment=True,
            max_num_rows=2,
        )
        session.create_view(transformation_query1, "flatmap1", cache=False)

        transformation_query2 = FlatMap(
            child=PrivateSource("flatmap1"),
            f=lambda row: [{}, {}],
            schema_new_columns=Schema({}),
            augment=True,
            max_num_rows=3,
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

    @pytest.mark.parametrize(
        "inf_budget,mechanism",
        [
            (PureDPBudget(float("inf")), SumMechanism.LAPLACE),
            (ApproxDPBudget(float("inf"), 0.5), SumMechanism.LAPLACE),
            (ApproxDPBudget(0.5, 1), SumMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), SumMechanism.LAPLACE),
            (RhoZCDPBudget(float("inf")), SumMechanism.GAUSSIAN),
        ],
    )
    def test_create_view_composed_correct_answer(
        self, inf_budget: PrivacyBudget, mechanism: SumMechanism
    ):
        """Composing :func:`create_view` gives the correct answer if budget=inf."""
        session = Session.from_dataframe(
            privacy_budget=inf_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        transformation_query1 = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            schema_new_columns=Schema({"Repeat": "INTEGER"}),
            augment=True,
            max_num_rows=1,
        )
        session.create_view(transformation_query1, "flatmap1", cache=False)
        transformation_query2 = FlatMap(
            child=PrivateSource("flatmap1"),
            f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            schema_new_columns=Schema({"i": "INTEGER"}),
            augment=False,
            max_num_rows=2,
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
        assert_frame_equal_with_sort(answer, expected)

    def test_caching(self, spark):
        """Tests that caching works as expected."""
        # pylint: disable=protected-access
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        # we need to add this to clear the cache in the spark session, since
        # with the addition of pytest all tests in a module share the same
        # spark context. Since there are views created in the previous test
        # the first assertion here will fail unless we clear the cache
        spark.catalog.clearCache()
        view1_query = QueryBuilder("private").filter("B = 0")
        view2_query = QueryBuilder("private").join_public(self.join_df)
        session.create_view(view1_query, "view1", cache=True)
        session.create_view(view2_query, "view2", cache=True)
        # Views have been created, but are lazy - nothing in cache yet
        assert len(list(spark.sparkContext._jsc.sc().getRDDStorageInfo())) == 0
        # Evaluate a query on view1
        session.evaluate(QueryBuilder("view1").count(), privacy_budget=PureDPBudget(1))
        assert len(list(spark.sparkContext._jsc.sc().getRDDStorageInfo())) == 1
        # Evaluate another query on view1
        session.evaluate(QueryBuilder("view1").count(), privacy_budget=PureDPBudget(1))
        assert len(list(spark.sparkContext._jsc.sc().getRDDStorageInfo())) == 1
        # Evaluate a query on view2
        session.evaluate(QueryBuilder("view2").count(), privacy_budget=PureDPBudget(1))
        assert len(list(spark.sparkContext._jsc.sc().getRDDStorageInfo())) == 2
        # Delete views
        session.delete_view("view1")
        assert len(list(spark.sparkContext._jsc.sc().getRDDStorageInfo())) == 1
        session.delete_view("view2")
        assert len(list(spark.sparkContext._jsc.sc().getRDDStorageInfo())) == 0

    # regression test for #2491
    def test_filter_regression(self, spark) -> None:
        """Regression tests for issue 2491.

        This issue caused incorrect results when joining a dataframe with
        another dataframe derived from the first (in this case, a KeySet
        derived from the private data).
        """
        sdf = spark.createDataFrame(
            pd.DataFrame(
                [["0", 1, 100000], ["1", 0, 20000], ["1", 2, 20000]],
                columns=["A", "B", "X"],
            )
        )
        total_budget = 10
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(total_budget),
            source_id="private",
            dataframe=sdf,
            protected_change=AddOneRow(),
        )

        all_keys = KeySet.from_dataframe(sdf)
        keyset = all_keys["A", "B"]
        budget_per_query = PureDPBudget(total_budget / 3)
        expected_a_b = pd.DataFrame([["0", 1], ["1", 0], ["1", 2]], columns=["A", "B"])

        count_query = QueryBuilder("private").filter("B == 2").groupby(keyset).count()
        count_result = session.evaluate(count_query, budget_per_query)
        count_a_b = count_result.select("A", "B")
        assert_frame_equal_with_sort(count_a_b.toPandas(), expected_a_b)

        median_query = (
            QueryBuilder("private")
            .filter("B == 2")
            .groupby(keyset)
            .median("X", 0, 10**6, "dp_median")
        )
        median_result = session.evaluate(median_query, budget_per_query)
        median_a_b = median_result.select("A", "B")
        assert_frame_equal_with_sort(median_a_b.toPandas(), expected_a_b)

        average_query = (
            QueryBuilder("private")
            .filter("B == 2")
            .groupby(keyset)
            .average("X", 0, 10**6, "dp_average")
        )
        average_result = session.evaluate(average_query, budget_per_query)
        average_a_b = average_result.select("A", "B")
        assert_frame_equal_with_sort(average_a_b.toPandas(), expected_a_b)

    def test_grouping_noninteger_stability(self, spark) -> None:
        """Test that zCDP grouping_column and non-integer stabilities work."""
        grouped_df = spark.createDataFrame(
            pd.DataFrame({"id": [7, 7, 8, 9], "group": [0, 1, 0, 1]})
        )
        ks = KeySet.from_dict({"group": [0, 1]})
        query = QueryBuilder("id").groupby(ks).count()

        session = Session.from_dataframe(
            RhoZCDPBudget(float("inf")),
            "id",
            grouped_df,
            stability=math.sqrt(2),
            grouping_column="group",
        )
        session.evaluate(query, RhoZCDPBudget(1))
