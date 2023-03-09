"""Common fixtures for Session integration tests."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import pandas as pd
import pytest

from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.protected_change import AddRowsWithID
from tmlt.analytics.session import Session

INF_BUDGET = PureDPBudget(float("inf"))


@pytest.fixture(scope="module")
def session(spark):
    """A Session with some sample data."""
    df_id1 = spark.createDataFrame(
        pd.DataFrame(
            [
                [1, "A", 4],
                [1, "A", 5],
                [1, "A", 6],
                [2, "A", 7],
                [3, "A", 8],
                [3, "B", 9],
            ],
            columns=["id", "group", "n"],
        )
    )
    df_id2 = spark.createDataFrame(
        pd.DataFrame(
            [[1, 12], [1, 15], [1, 18], [2, 21], [3, 24], [3, 27]], columns=["id", "x"]
        )
    )
    sess = (
        Session.Builder()
        .with_privacy_budget(INF_BUDGET)
        .with_primary_id("a")
        .with_primary_id("b")
        # a and b use the same data, but they're still separate identifier
        # spaces; this is just to check that things like cross-ID-space joins
        # are detected.
        .with_private_dataframe(
            "id_a1", df_id1, protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a2", df_id2, protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_b1", df_id1, protected_change=AddRowsWithID("id", "b")
        )
        .with_private_dataframe(
            "id_b2", df_id2, protected_change=AddRowsWithID("id", "b")
        )
        .build()
    )
    return sess
