"""Tests for Session with the AddMaxRowsInMaxGroups ProtectedChange.


Note that these tests are not intended to be exhaustive. They are intended to be a
sanity check that the Session is working correctly with AddMaxRowsInMaxGroups. More
thorough tests for Session are in
test/system/session/rows/test_add_max_rows.py."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=no-self-use

from typing import Any, Dict, List

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from tmlt.core.measures import RhoZCDP
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.parameters import calculate_noise_scale

from tmlt.analytics._noise_info import _NoiseMechanism
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.protected_change import AddMaxRowsInMaxGroups
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import (
    AverageMechanism,
    CountMechanism,
    GroupByBoundedAverage,
    GroupByCount,
    PrivateSource,
    QueryExpr,
)
from tmlt.analytics.session import Session


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

    @pytest.mark.parametrize("budget", [(PureDPBudget(20)), (RhoZCDPBudget(20))])
    def test_partition_on_grouping_column(self, spark, budget: PrivacyBudget):
        """Tests that you can partition on grouping columns."""
        grouping_df = spark.createDataFrame(pd.DataFrame({"new": [1, 2]}))
        session = Session.from_dataframe(
            privacy_budget=budget,
            source_id="private",
            dataframe=self.sdf.crossJoin(grouping_df),
            protected_change=AddMaxRowsInMaxGroups(
                grouping_column="new", max_groups=1, max_rows_per_group=1
            ),
        )
        new_sessions = session.partition_and_create(
            source_id="private",
            privacy_budget=budget,
            column="new",
            splits={"new1": 1, "new2": 2},
        )
        new_sessions["new1"].evaluate(QueryBuilder("new1").count(), budget)
        new_sessions["new2"].evaluate(QueryBuilder("new2").count(), budget)

    def test_max_rows_per_group_stability(self, spark) -> None:
        """MaxRowsPerGroup stability works with zCDP."""
        grouped_df = spark.createDataFrame(
            pd.DataFrame({"id": [7, 7, 8, 9], "group": [0, 1, 0, 1]})
        )
        ks = KeySet.from_dict({"group": [0, 1]})
        query = QueryBuilder("id").groupby(ks).count()

        session = Session.from_dataframe(
            RhoZCDPBudget(float("inf")),
            "id",
            grouped_df,
            protected_change=AddMaxRowsInMaxGroups(
                "group", max_groups=2, max_rows_per_group=1
            ),
        )
        session.evaluate(query, RhoZCDPBudget(1))

    @pytest.mark.parametrize(
        "query_expr,session_budget,query_budget,expected",
        [
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
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
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
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
            protected_change=AddMaxRowsInMaxGroups(
                "B", max_groups=1, max_rows_per_group=1
            ),
        )
        # pylint: disable=protected-access
        info = session._noise_info(query_expr, query_budget)
        # pylint: enable=protected-access
        assert info == expected
