"""Integration tests for partition_and_create for IDs tables."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025


import pandas as pd
import pytest
import sympy as sp
from tmlt.core.metrics import AddRemoveKeys as CoreAddRemoveKeys
from tmlt.core.metrics import DictMetric, SymmetricDifference

from tmlt.analytics import (
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PureDPBudget,
    QueryBuilder,
)
from tmlt.analytics._table_identifier import NamedTable, TableCollection

from ....conftest import assert_frame_equal_with_sort
from ..conftest import INF_BUDGET, INF_BUDGET_ZCDP

_KEYSET = KeySet.from_dict({"group": ["A", "B"]})
_KEYSET2 = KeySet.from_dict({"group2": ["X", "Y"]})


@pytest.mark.parametrize("session", [INF_BUDGET], indirect=True, ids=["puredp"])
def test_invalid_constraint_partition_and_create(session):
    """Tests that :func:`partition_and_create` with invalid constraint errors."""
    with pytest.raises(
        ValueError,
        match=(
            "You must create a MaxGroupsPerID or MaxRowsPerID constraint before using "
            "partition_and_create on tables with the AddRowsWithID protected change."
        ),
    ):
        session.create_view(QueryBuilder("id_a1"), "truncated_ids", cache=True)
        session.partition_and_create(
            source_id="truncated_ids",
            privacy_budget=PureDPBudget(10),
            column="group",
            splits={"part0": "A", "part1": "B"},
        )

    with pytest.raises(
        ValueError,
        match=(
            "You must create a MaxGroupsPerID or MaxRowsPerID constraint before using "
            "partition_and_create on tables with the AddRowsWithID protected change."
        ),
    ):
        session.create_view(
            QueryBuilder("id_a1").enforce(MaxRowsPerGroupPerID("group", 5)),
            "truncated_ids2",
            cache=True,
        )
        session.partition_and_create(
            source_id="truncated_ids2",
            privacy_budget=PureDPBudget(10),
            column="group",
            splits={"part0": "A", "part1": "B"},
        )


@pytest.mark.parametrize(
    "session,table_stability",
    [(INF_BUDGET, 2), (INF_BUDGET_ZCDP, 2)],
    indirect=["session"],
    ids=["puredp", "zcdp"],
)
def test_partition_and_create_with_MaxRowsPerID(session, table_stability):
    """Test :func:`partition_and_create` on IDs table with MaxRowsPerID constraint."""
    session.create_view(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(2)), "truncated_ids1", cache=True
    )
    # Turns IDs into a partitioned non-IDs table
    new_sessions = session.partition_and_create(
        source_id="truncated_ids1",
        privacy_budget=session.remaining_privacy_budget,
        column="group",
        splits={"part0": "A", "part1": "B"},
    )

    # `id_a1` originally has 6 rows
    #     id group group2 n  float_n
    #     1     A      X  4      4.0
    #     1     A      Y  5      5.0
    #     1     A      X  6      6.0
    #     2     A      Y  7      7.0
    #     3     A      X  8      8.0
    #     3     B      Y  9      9.0

    # Since MaxRowsPerID(2) is enforced, it will have 5 rows
    #     id group group2 n  float_n
    #     1     A      X  4      4.0
    #     1     A      Y  5      5.0
    #     2     A      Y  7      7.0
    #     3     A      X  8      8.0
    #     3     B      Y  9      9.0

    # After partitioning by `group`,
    # `part0` with group = "A" with 4 rows (id = 1, 1, 2, 3)
    # and `part1` with group = "B" with 1 row (id = 3).
    assert len(new_sessions) == 2
    session2 = new_sessions["part0"]
    session3 = new_sessions["part1"]
    assert session2.private_sources == ["part0"]
    assert session3.private_sources == ["part1"]
    assert session2.get_id_column("part0") is None
    assert session3.get_id_column("part1") is None

    # Can't enforce MaxRowsPerID constraint on AddMaxRows protected change
    bad_query = QueryBuilder("part0").enforce(MaxRowsPerID(2)).count()
    with pytest.raises(
        ValueError,
        match=(
            r"Constraint MaxRowsPerID\(max=2\) can only be applied to tables"
            " with the AddRowsWithID protected change"
        ),
    ):
        session2.evaluate(bad_query, session.remaining_privacy_budget)

    answer_session2 = session2.evaluate(
        QueryBuilder("part0").count(),
        session.remaining_privacy_budget,
    )
    assert_frame_equal_with_sort(
        answer_session2.toPandas(), pd.DataFrame({"count": [4]})
    )
    answer_session3 = session3.evaluate(
        QueryBuilder("part1").count(),
        session.remaining_privacy_budget,
    )
    assert_frame_equal_with_sort(
        answer_session3.toPandas(), pd.DataFrame({"count": [1]})
    )

    assert session2._input_metric == DictMetric(
        {NamedTable("part0"): SymmetricDifference()}
    )
    assert session3._input_metric == DictMetric(
        {NamedTable("part1"): SymmetricDifference()}
    )
    assert session2._accountant.d_in == {NamedTable("part0"): table_stability}
    assert session3._accountant.d_in == {NamedTable("part1"): table_stability}


@pytest.mark.parametrize(
    "session,table_stability",
    [(INF_BUDGET, 2), (INF_BUDGET_ZCDP, sp.sqrt(2))],
    indirect=["session"],
    ids=["puredp", "zcdp"],
)
def test_partition_and_create_with_MaxGroupsPerID(session, table_stability):
    """Test :func:`partition_and_create` on IDs table with MaxGroupsPerID constraint."""
    # Since both MaxRowsPerID and MaxGroupsPerID are enforced,
    # MaxRowsPerID will be ignored
    session.create_view(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(5))
        .enforce(MaxGroupsPerID("group", 2)),
        "truncated_ids3",
        cache=True,
    )
    new_sessions = session.partition_and_create(
        source_id="truncated_ids3",
        privacy_budget=session.remaining_privacy_budget,
        column="group",
        splits={"part0": "A", "part1": "B"},
    )
    assert len(new_sessions) == 2
    session2 = new_sessions["part0"]
    session3 = new_sessions["part1"]
    # Turns an IDs table into a partitioned IDs table
    assert session2.get_id_column("part0") == "id"
    assert session3.get_id_column("part1") == "id"

    answer_session2 = session2.evaluate(
        QueryBuilder("part0").enforce(MaxRowsPerID(2)).count(),
        session.remaining_privacy_budget,
    )
    assert_frame_equal_with_sort(
        answer_session2.toPandas(), pd.DataFrame({"count": [4]})
    )
    answer_session3 = session3.evaluate(
        QueryBuilder("part1").enforce(MaxRowsPerID(2)).count(),
        session.remaining_privacy_budget,
    )
    assert_frame_equal_with_sort(
        answer_session3.toPandas(), pd.DataFrame({"count": [1]})
    )

    assert session2._input_metric == DictMetric(
        {TableCollection("a"): CoreAddRemoveKeys({NamedTable("part0"): "id"})}
    )
    assert session3._input_metric == DictMetric(
        {TableCollection("a"): CoreAddRemoveKeys({NamedTable("part1"): "id"})}
    )
    assert session2._accountant.d_in == {TableCollection("a"): table_stability}
    assert session3._accountant.d_in == {TableCollection("a"): table_stability}
