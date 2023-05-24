"""Tests for invalid session configurations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable= no-self-use

from typing import Dict, Tuple, Type, Union
from unittest.mock import Mock

import pytest
from pyspark.sql import DataFrame
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics._schema import Schema
from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)
from tmlt.analytics.protected_change import AddOneRow
from tmlt.analytics.query_expr import (
    FlatMap,
    GroupByBoundedSum,
    GroupByCount,
    PrivateSource,
    QueryExpr,
    Rename,
    ReplaceNullAndNan,
)
from tmlt.analytics.session import Session, _format_insufficient_budget_msg


@pytest.mark.usefixtures("session_data")
class TestInvalidSession:
    """Tests for Invalid Sessions."""

    sdf: DataFrame
    sdf_col_types: Dict[str, str]
    sdf_input_domain: SparkDataFrameDomain

    @pytest.mark.parametrize(
        "query_expr,error_type,expected_error_msg",
        [
            (
                GroupByCount(
                    child=PrivateSource("private_source_not_in_catalog"),
                    groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
                ),
                ValueError,
                "Query references nonexistent table 'private_source_not_in_catalog'",
            )
        ],
    )
    def test_invalid_queries_evaluate(
        self,
        query_expr: QueryExpr,
        error_type: Type[Exception],
        expected_error_msg: str,
    ):
        """evaluate raises error on invalid queries."""
        mock_accountant = Mock()
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric(
            {NamedTable("private"): SymmetricDifference()}
        )
        mock_accountant.input_domain = DictDomain(
            {NamedTable("private"): self.sdf_input_domain}
        )
        mock_accountant.d_in = {NamedTable("private"): ExactNumber(1)}
        mock_accountant.privacy_budget = ExactNumber(float("inf"))

        session = Session(accountant=mock_accountant, public_sources={})
        session.create_view(PrivateSource("private"), "view", cache=False)
        with pytest.raises(error_type, match=expected_error_msg):
            session.evaluate(query_expr, privacy_budget=PureDPBudget(float("inf")))

    @pytest.mark.parametrize(
        "requested,remaining,budget_type,expected_msg",
        [
            # TOD0(2476): Uncomment once we support Delta consumption
            # Also remove the first uncommented test. That msg will be incorrect
            # (
            #     (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
            #     (ExactNumber(2), ExactNumber.from_float(0.4, round_up=True)),
            #     ApproxDPBudget(2, 0.4),
            #   "\nRequested: ε=3.000, δ=0.500\nRemaining:"
            #     " ε=2.000, δ=0.400\nDifference: ε=1.000, δ=0.1000",
            # ),
            # (
            #     (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
            #     (ExactNumber(2), ExactNumber.from_float(0.5, round_up=True)),
            #     ApproxDPBudget(2, 0.5),
            #   "\nRequested: ε=3.000, δ=0.500\nRemaining:"
            #     " ε=2.000, δ=0.500\nDifference: ε=1.000",
            # ),
            # (
            #     (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
            #     (ExactNumber(3), ExactNumber.from_float(0.4, round_up=True)),
            #     ApproxDPBudget(3, 0.4),
            #   "\nRequested: ε=3.000, δ=0.500\nRemaining:"
            #     " ε=3.000, δ=0.400\nDifference: δ=0.100",
            # ),
            # (
            #     (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
            #     (ExactNumber(3), ExactNumber.from_float(0.41, round_up=True)),
            #     ApproxDPBudget(3, 0.41),
            #   "\nRequested: ε=3.000, δ=0.500\nRemaining:"
            #     " ε=3.000, δ=0.400\nDifference: δ=9.000e-02",
            # ),
            (
                (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
                (ExactNumber(2), ExactNumber.from_float(0.4, round_up=True)),
                ApproxDPBudget(2, 0.4),
                "\nRequested: ε=3.000\nRemaining: ε=2.000\nDifference: ε=1.000",
            ),
            (
                ExactNumber(3),
                ExactNumber(2),
                PureDPBudget(2),
                "\nRequested: ε=3.000\nRemaining: ε=2.000\nDifference: ε=1.000",
            ),
            (
                ExactNumber(3),
                ExactNumber.from_float(2.91, round_up=True),
                PureDPBudget(2.91),
                "\nRequested: ε=3.000\nRemaining: ε=2.910\nDifference: ε=9.000e-02",
            ),
            (
                ExactNumber(3),
                ExactNumber(2),
                RhoZCDPBudget(2),
                "\nRequested: 𝝆=3.000\nRemaining: 𝝆=2.000\nDifference: 𝝆=1.000",
            ),
            (
                ExactNumber(3),
                ExactNumber.from_float(2.91, round_up=True),
                RhoZCDPBudget(2.91),
                "\nRequested: 𝝆=3.000\nRemaining: 𝝆=2.910\nDifference: 𝝆=9.000e-02",
            ),
        ],
    )
    def test_format_insufficient_budget_msg(
        self,
        requested: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
        remaining: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
        budget_type: PrivacyBudget,
        expected_msg: str,
    ):
        """Tests that InsufficientBudgetError is formatted correctly."""
        assert repr(
            _format_insufficient_budget_msg(requested, remaining, budget_type)
        ) == repr(expected_msg)

    @pytest.mark.parametrize("output_measure", [(PureDP()), (ApproxDP()), (RhoZCDP())])
    def test_invalid_privacy_budget_evaluate_and_create(
        self, output_measure: Union[PureDP, RhoZCDP]
    ):
        """evaluate and create functions raise error on invalid privacy_budget."""
        one_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        two_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        if output_measure == PureDP():
            one_budget = PureDPBudget(1)
            two_budget = PureDPBudget(2)
        elif output_measure == ApproxDP():
            one_budget = ApproxDPBudget(1, 0.5)
            two_budget = ApproxDPBudget(2, 0.5)
        elif output_measure == RhoZCDP():
            one_budget = RhoZCDPBudget(1)
            two_budget = RhoZCDPBudget(2)
        else:
            pytest.fail(
                f"must use PureDP, ApproxDP, or RhoZCDP, found {output_measure}"
            )

        query_expr = GroupByCount(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}),
        )
        session = Session.from_dataframe(
            privacy_budget=one_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        with pytest.raises(
            RuntimeError,
            match="Cannot answer query without exceeding the Session privacy budget",
        ):
            session.evaluate(query_expr, privacy_budget=two_budget)

        with pytest.raises(
            RuntimeError,
            match="Cannot perform this partition without "
            "exceeding the Session privacy budget",
        ):
            session.partition_and_create(
                "private",
                privacy_budget=two_budget,
                column="A",
                splits={"part_0": "0", "part_1": "1"},
            )

    def test_invalid_grouping_with_view(self):
        """Tests that grouping flatmap + rename fails if not used in a later groupby."""
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        grouping_flatmap = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            schema_new_columns=Schema({"Repeat": "INTEGER"}, grouping_column="Repeat"),
            augment=True,
            max_num_rows=1,
        )
        session.create_view(
            Rename(child=grouping_flatmap, column_mapper={"Repeat": "repeated"}),
            "grouping_flatmap_renamed",
            cache=False,
        )

        with pytest.raises(
            ValueError,
            match=(
                "Column 'repeated' produced by grouping transformation is not in "
                r"groupby columns \['A'\]"
            ),
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
            protected_change=AddOneRow(),
        )

        grouping_flatmap = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            schema_new_columns=Schema({"Repeat": "INTEGER"}, grouping_column="Repeat"),
            augment=True,
            max_num_rows=1,
        )
        session.create_view(grouping_flatmap, "grouping_flatmap", cache=False)

        grouping_flatmap_2 = FlatMap(
            child=PrivateSource("grouping_flatmap"),
            f=lambda row: [{"i": row["X"]} for _ in range(row["Repeat"])],
            schema_new_columns=Schema({"i": "INTEGER"}, grouping_column="i"),
            augment=True,
            max_num_rows=2,
        )

        with pytest.raises(
            ValueError,
            match=(
                "Multiple grouping transformations are used in this query. "
                "Only one grouping transformation is allowed."
            ),
        ):
            session.create_view(grouping_flatmap_2, "grouping_flatmap_2", cache=False)
