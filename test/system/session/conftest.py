"""Common fixtures for Session integration tests."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import pandas as pd
import pytest

from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.protected_change import AddRowsWithID
from tmlt.analytics.session import Session

INF_BUDGET = PureDPBudget(float("inf"))
INF_BUDGET_ZCDP = RhoZCDPBudget(float("inf"))


def closest_value(value, collection):
    """Find the element of a collection numerically closest to a given value.

    Given a collection and a value, find the element of the collection that is
    closest to at value and return it. For numbers, the closest element is the
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
                [1, "A", "X", 4],
                [1, "A", "Y", 5],
                [1, "A", "X", 6],
                [2, "A", "Y", 7],
                [3, "A", "X", 8],
                [3, "B", "Y", 9],
            ],
            columns=["id", "group", "group2", "n"],
        )
    )
    df_id2 = spark.createDataFrame(
        pd.DataFrame(
            [[1, 12], [1, 15], [1, 18], [2, 21], [3, 24], [3, 27]], columns=["id", "x"]
        )
    )
    return {"id1": df_id1, "id2": df_id2}


@pytest.fixture(scope="module")
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
        .with_primary_id("a")
        .with_primary_id("b")
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
            "id_b1", _session_data["id1"], protected_change=AddRowsWithID("id", "b")
        )
        .with_private_dataframe(
            "id_b2", _session_data["id1"], protected_change=AddRowsWithID("id", "b")
        )
        .build()
    )
    return sess
