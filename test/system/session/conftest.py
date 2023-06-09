"""Common fixtures for Session integration tests."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import pandas as pd
import pytest
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.protected_change import AddMaxRows, AddRowsWithID
from tmlt.analytics.session import Session

INF_BUDGET = PureDPBudget(float("inf"))
INF_BUDGET_ZCDP = RhoZCDPBudget(float("inf"))


def closest_value(value, collection):
    """Find the element of a collection numerically closest to a given value.

    Given a collection and a value, find the element of the collection that is
    closest to that value and return it. For numbers, the closest element is the
    one which has the smallest absolute difference with the value; for tuples of
    numbers, it is the one which has the smallest total absolute difference
    between corresponding pairs of values. If the collection is empty, None is
    returned.
    """
    if not collection:
        return None

    if isinstance(value, (int, float)):
        return min(collection, key=lambda c: abs(value - c))
    elif isinstance(value, tuple):
        return min(
            collection, key=lambda c: sum(abs(t[0] - t[1]) for t in zip(value, c))
        )
    else:
        raise AssertionError("Unknown input data type")


@pytest.fixture(scope="module")
def _session_data(spark):
    df_id1 = spark.createDataFrame(
        pd.DataFrame(
            [
                [1, "A", "X", 4, 4.0],
                [1, "A", "Y", 5, 5.0],
                [1, "A", "X", 6, 6.0],
                [2, "A", "Y", 7, 7.0],
                [3, "A", "X", 8, 8.0],
                [3, "B", "Y", 9, 9.0],
            ],
            columns=["id", "group", "group2", "n", "float_n"],
        )
    )
    df_id2 = spark.createDataFrame(
        pd.DataFrame(
            [
                [1, "A", 12],
                [1, "B", 15],
                [1, "A", 18],
                [2, "B", 21],
                [3, "A", 24],
                [3, "B", 27],
            ],
            columns=["id", "group", "x"],
        )
    )
    df_id3 = spark.createDataFrame(
        [
            [1, "A", 12],
            [None, "B", 15],
            [1, "A", 18],
            [2, "B", None],
            [3, "A", 24],
            [3, "B", 27],
            [None, "A", 30],
        ],
        schema=StructType(
            [
                StructField("id", IntegerType(), nullable=True),
                StructField("group", StringType(), nullable=False),
                StructField("x", LongType(), nullable=True),
            ]
        ),
    )
    # df with non-nullable id column
    df_id4 = spark.createDataFrame(
        [
            [1, "A", 12],
            [1, "B", 15],
            [1, "A", 18],
            [2, "B", 21],
            [3, "A", 24],
            [3, "B", 27],
        ],
        schema=StructType(
            [
                StructField("id", IntegerType(), nullable=False),
                StructField("group", StringType(), nullable=False),
                StructField("x", LongType(), nullable=False),
            ]
        ),
    )
    df_rows1 = spark.createDataFrame(
        [["0", 0, 0], ["0", 0, 1], ["0", 1, 2], ["1", 0, 3]],
        schema=StructType(
            [
                StructField("A", StringType(), nullable=False),
                StructField("B", LongType(), nullable=False),
                StructField("X", LongType(), nullable=False),
            ]
        ),
    )
    return {
        "id1": df_id1,
        "id2": df_id2,
        "id3": df_id3,
        "id4": df_id4,
        "rows1": df_rows1,
    }


@pytest.fixture
def session(_session_data, request):
    """A Session with some sample data.

    This fixture requires a parameter (typically passed by setting the
    `indirect` option to parametrize) specifying the privacy budget. Setting it
    up this way allows parametrizing tests to run with Sessions that use
    multiple privacy definitions without duplicating all of the test logic.
    """
    assert hasattr(
        request, "param"
    ), "The session fixture requires a parameter indicating its budget"
    budget = request.param
    assert isinstance(
        budget, PrivacyBudget
    ), "The session fixture parameter must be a PrivacyBudget"

    sess = (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_id_space("a")
        .with_id_space("b")
        # a and b use the same data, but they're still separate identifier
        # spaces; this is just to check that things like cross-ID-space joins
        # are detected.
        .with_private_dataframe(
            "id_a1", _session_data["id1"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a2", _session_data["id2"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a3", _session_data["id3"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a4", _session_data["id4"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_b1", _session_data["id1"], protected_change=AddRowsWithID("id", "b")
        )
        .with_private_dataframe(
            "id_b2", _session_data["id1"], protected_change=AddRowsWithID("id", "b")
        )
        .with_private_dataframe(
            "rows_1", _session_data["rows1"], protected_change=AddMaxRows(2)
        )
        .build()
    )
    return sess
