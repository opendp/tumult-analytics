"""Tests for Sessions that employ a mixture of IDs and non-IDs features.

These are not meant to be exhaustive, but rather to ensure that the Session
functions properly when used with a mixture of IDs and non-IDs protected changes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023


import pytest

from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics.constraints import (
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_builder import QueryBuilder

from ..conftest import INF_BUDGET, INF_BUDGET_ZCDP


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
def test_view_constraint(session):
    """Test that constraints are saved when creating views."""
    query = (
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(1))
        .enforce(MaxGroupsPerID("group", 1))
        .enforce(MaxRowsPerGroupPerID("group", 1))
    )
    session.create_view(query, "view", cache=False)
    # pylint: disable=protected-access
    assert session._table_constraints[NamedTable("view")] == [
        MaxRowsPerID(1),
        MaxGroupsPerID("group", 1),
        MaxRowsPerGroupPerID("group", 1),
    ]
    # pylint: enable=protected-access

    session.delete_view("view")
    # pylint: disable=protected-access
    assert NamedTable("view") not in session._table_constraints
    # pylint: enable=protected-access


# Test creating view, then doing (1) immediate aggregation and
# (2) continued IDs transformations
@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query1,condition,expected_first_res,expected_second_res",
    [
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(3)), "n > 4", 6, 3),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(1)), "n < 4", 3, 0),
        (QueryBuilder("id_a2").enforce(MaxRowsPerID(3)), "x < 24", 6, 2),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 3))
            .enforce(MaxRowsPerGroupPerID("group", 2)),
            "n > 4",
            5,
            3,
        ),
    ],
)
def test_evaluate_view(
    session,
    query1: QueryBuilder,
    condition: str,
    expected_first_res: int,
    expected_second_res: int,
):
    """Test that view can be used with immediate aggregation and IDs transformations."""
    session.create_view(query1, "query_view", cache=False)
    aggregation_res = session.evaluate(
        QueryBuilder("query_view").count(), session.remaining_privacy_budget
    ).toPandas()
    assert aggregation_res["count"][0] == expected_first_res

    # Invalidate constraints using flat_map, verifying that the flat map is
    # still in IDs (doesn't need truncation) and that aggregating without adding
    # new constraints doesn't work.
    query2 = (
        QueryBuilder("query_view")
        .flat_map(lambda _: [{}], {}, augment=True)
        .filter(condition)
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "A constraint on the number of rows contributed by each ID "
            "is needed to perform this query."
        ),
    ):
        session.evaluate(query2.count(), session.remaining_privacy_budget).toPandas()

    query2 = query2.enforce(MaxRowsPerID(1))
    transformation_res = session.evaluate(
        query2.count(), session.remaining_privacy_budget
    ).toPandas()
    assert transformation_res["count"][0] == expected_second_res

    session.delete_view("query_view")


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "ids_query,non_ids_query,ids_expected,non_ids_expected",
    [
        (
            QueryBuilder("id_a1").enforce(MaxRowsPerID(1)),
            QueryBuilder("rows_1").filter("X > 1"),
            3,
            2,
        ),
        (
            QueryBuilder("id_a2").filter("x < 24").enforce(MaxRowsPerID(3)),
            QueryBuilder("rows_1").groupby(KeySet.from_dict({"A": ["0", "1"]})),
            4,
            3,
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxRowsPerID(3))
            .enforce(MaxRowsPerGroupPerID("group", 2)),
            QueryBuilder("rows_1").groupby(KeySet.from_dict({"A": ["0", "1"]})),
            5,
            3,
        ),
    ],
)
def test_mixed_session(
    session,
    ids_query: QueryBuilder,
    non_ids_query: QueryBuilder,
    ids_expected: int,
    non_ids_expected: int,
):
    """Sanity check that a session can evaluate both ID and non-ID queries."""
    assert (
        session.evaluate(
            ids_query.count(), session.remaining_privacy_budget
        ).toPandas()["count"][0]
        == ids_expected
    )
    assert (
        session.evaluate(non_ids_query.count(), session.remaining_privacy_budget)
        .toPandas()
        .sort_values("count", ascending=False, ignore_index=True)["count"][0]
        == non_ids_expected
    )
