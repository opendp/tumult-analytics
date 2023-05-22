"""Integration tests for partition_and_create for IDs tables."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023


import pandas as pd
import pytest
import sympy as sp
from tmlt.core.metrics import AddRemoveKeys as CoreAddRemoveKeys
from tmlt.core.metrics import DictMetric

from tmlt.analytics._table_identifier import NamedTable, TableCollection
from tmlt.analytics.constraints import MaxGroupsPerID, MaxRowsPerID
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.query_builder import QueryBuilder

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
            "You must create MaxGroupsPerID constraint before using "
            "partition_and_create on tables with the AddRowsWithID protected change."
        ),
    ):
        session.create_view(QueryBuilder("id_a1"), "truncated_ids", cache=True)
        session.partition_and_create(
            source_id="truncated_ids",
            privacy_budget=PureDPBudget(10),
            column="group",
            splits={"part0": "0", "part1": "1"},
        )

    with pytest.raises(
        ValueError,
        match=(
            "You must create MaxGroupsPerID constraint before using "
            "partition_and_create on tables with the AddRowsWithID protected change."
        ),
    ):
        session.create_view(
            QueryBuilder("id_a1").enforce(MaxRowsPerID(5)), "truncated_ids2", cache=True
        )
        session.partition_and_create(
            source_id="truncated_ids2",
            privacy_budget=PureDPBudget(10),
            column="group2",
            splits={"part0": "0", "part1": "1"},
        )


@pytest.mark.parametrize(
    "session,table_stability",
    [(INF_BUDGET, 2), (INF_BUDGET_ZCDP, sp.sqrt(2))],
    indirect=["session"],
    ids=["puredp", "zcdp"],
)
def test_partition_and_create(session, table_stability):
    """Tests that :func:`partition_and_create` on IDs table."""
    session.create_view(
        QueryBuilder("id_a1").enforce(MaxGroupsPerID("group", 2)),
        "truncated_ids3",
        cache=True,
    )
    new_sessions = session.partition_and_create(
        source_id="truncated_ids3",
        privacy_budget=session.remaining_privacy_budget,
        column="group",
        splits={"part0": "A", "part1": "B"},
    )
    session2 = new_sessions["part0"]
    session3 = new_sessions["part1"]
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
    # pylint: disable=protected-access
    assert session2._input_metric == DictMetric(
        {TableCollection("a"): CoreAddRemoveKeys({NamedTable("part0"): "id"})}
    )
    assert session3._input_metric == DictMetric(
        {TableCollection("a"): CoreAddRemoveKeys({NamedTable("part1"): "id"})}
    )
    assert session2._accountant.d_in == {TableCollection("a"): table_stability}
    assert session3._accountant.d_in == {TableCollection("a"): table_stability}
    # pylint: enable=protected-access
