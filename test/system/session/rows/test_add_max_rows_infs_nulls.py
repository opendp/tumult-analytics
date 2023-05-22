"""System tests for Sessions with Nulls and Infs."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Any, Dict, List, Mapping, Tuple, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from tmlt.core.measurements.interactive_measurements import SequentialQueryable

from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.protected_change import AddOneRow
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import AnalyticsDefault
from tmlt.analytics.session import Session
from tmlt.analytics.truncation_strategy import TruncationStrategy

from ....conftest import assert_frame_equal_with_sort


@pytest.mark.usefixtures("null_session_data")
class TestSessionWithNulls:
    """Tests for sessions with Nulls."""

    pdf: pd.DataFrame
    sdf: DataFrame

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
        assert_frame_equal_with_sort(self.pdf, self._expected_replace({}))
        assert_frame_equal_with_sort(expected, self._expected_replace(d))

    @pytest.mark.parametrize(
        "cols_to_defaults",
        [
            ({"A": "aaaaaaa"}),
            ({"I": 999}),
            (
                {
                    "A": "aaa",
                    "I": 999,
                    "X": -99.9,
                    "D": datetime.date.fromtimestamp(0),
                    "T": datetime.datetime.fromtimestamp(0),
                }
            ),
        ],
    )
    def test_replace_null_and_nan(
        self,
        cols_to_defaults: Mapping[
            str, Union[int, float, str, datetime.date, datetime.datetime]
        ],
    ) -> None:
        """Test Session.replace_null_and_nan."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")),
            "private",
            self.sdf,
            protected_change=AddOneRow(),
        )
        session.create_view(
            QueryBuilder("private").replace_null_and_nan(cols_to_defaults),
            "replaced",
            cache=False,
        )
        # pylint: disable=protected-access
        queryable = session._accountant._queryable
        assert isinstance(queryable, SequentialQueryable)
        data = queryable._data
        assert isinstance(data, dict)
        assert isinstance(data[NamedTable("replaced")], DataFrame)
        # pylint: enable=protected-access
        assert_frame_equal_with_sort(
            data[NamedTable("replaced")].toPandas(),
            self._expected_replace(cols_to_defaults),
        )

    @pytest.mark.parametrize(
        "public_df,keyset,expected",
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
        ],
    )
    def test_join_public(
        self, spark, public_df: pd.DataFrame, keyset: KeySet, expected: pd.DataFrame
    ) -> None:
        """Test that join_public creates the correct results.

        The query used to evaluate this is a GroupByCount on the new dataframe,
        using the keyset provided.
        """
        session = Session.from_dataframe(
            PureDPBudget(float("inf")),
            "private",
            self.sdf,
            protected_change=AddOneRow(),
        )
        session.add_public_dataframe("public", spark.createDataFrame(public_df))
        result = session.evaluate(
            QueryBuilder("private").join_public("public").groupby(keyset).count(),
            privacy_budget=PureDPBudget(float("inf")),
        )
        assert_frame_equal_with_sort(result.toPandas(), expected)

    @pytest.mark.parametrize(
        "private_df,keyset,expected",
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
        ],
    )
    def test_join_private(
        self, spark, private_df: pd.DataFrame, keyset: KeySet, expected: pd.DataFrame
    ) -> None:
        """Test that join_private creates the correct results.

        The query used to evaluate this is a GroupByCount on the joined dataframe,
        using the keyset provided.
        """
        session = (
            Session.Builder()
            .with_privacy_budget(PureDPBudget(float("inf")))
            .with_private_dataframe("private", self.sdf)
            .with_private_dataframe("private2", spark.createDataFrame(private_df))
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
        assert_frame_equal_with_sort(result.toPandas(), expected)


@pytest.mark.usefixtures("infs_test_data")
class TestSessionWithInfs:
    """Tests for Sessions with Infs."""

    pdf: pd.DataFrame
    sdf: DataFrame

    @pytest.mark.parametrize(
        "replace_with,",
        [
            ({}),
            ({"B": (-100.0, 100.0)}),
            ({"B": (123.45, 678.90)}),
            ({"B": (999.9, 111.1)}),
        ],
    )
    def test_replace_infinity(
        self, replace_with: Dict[str, Tuple[float, float]]
    ) -> None:
        """Test replace_infinity query."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")),
            "private",
            self.sdf,
            protected_change=AddOneRow(),
        )
        session.create_view(
            QueryBuilder("private").replace_infinity(replace_with),
            "replaced",
            cache=False,
        )
        # pylint: disable=protected-access
        queryable = session._accountant._queryable
        assert isinstance(queryable, SequentialQueryable)
        data = queryable._data
        assert isinstance(data, dict)
        assert isinstance(data[NamedTable("replaced")], DataFrame)
        # pylint: enable=protected-access
        (replace_negative, replace_positive) = replace_with.get(
            "B", (AnalyticsDefault.DECIMAL, AnalyticsDefault.DECIMAL)
        )
        expected = self.pdf.replace(float("-inf"), replace_negative).replace(
            float("inf"), replace_positive
        )
        assert_frame_equal_with_sort(data[NamedTable("replaced")].toPandas(), expected)

    @pytest.mark.parametrize(
        "replace_with,expected",
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
        ],
    )
    def test_sum(
        self, replace_with: Dict[str, Tuple[float, float]], expected: pd.DataFrame
    ) -> None:
        """Test GroupByBoundedSum after replacing infinite values."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")),
            "private",
            self.sdf,
            protected_change=AddOneRow(),
        )
        result = session.evaluate(
            QueryBuilder("private")
            .replace_infinity(replace_with)
            .groupby(KeySet.from_dict({"A": ["a0", "a1"]}))
            .sum("B", low=-1000, high=1000, name="sum"),
            PureDPBudget(float("inf")),
        )
        assert_frame_equal_with_sort(result.toPandas(), expected)

    def test_drop_infinity(self):
        """Test GroupByBoundedSum after dropping infinite values."""
        session = Session.from_dataframe(
            PureDPBudget(float("inf")),
            "private",
            self.sdf,
            protected_change=AddOneRow(),
        )
        result = session.evaluate(
            QueryBuilder("private")
            .drop_infinity(columns=["B"])
            .groupby(KeySet.from_dict({"A": ["a0", "a1"]}))
            .sum("B", low=-1000, high=1000, name="sum"),
            PureDPBudget(float("inf")),
        )
        expected = pd.DataFrame([["a0", 2.0], ["a1", 5.0]], columns=["A", "sum"])
        assert_frame_equal_with_sort(result.toPandas(), expected)
