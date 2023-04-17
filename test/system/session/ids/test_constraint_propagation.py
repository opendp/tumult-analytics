"""Integration tests for constraint propagation."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import Dict, List

import pandas as pd
import pytest

from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.query_builder import ColumnType, QueryBuilder

from ..conftest import INF_BUDGET, INF_BUDGET_ZCDP

_CONSTRAINTS0 = [
    MaxRowsPerID(5),
    MaxGroupsPerID("group", 4),
    MaxGroupsPerID("group2", 3),
    MaxRowsPerGroupPerID("group", 2),
    MaxRowsPerGroupPerID("group2", 1),
]


def _test_propagation(query, expected_constraints, session):
    """Verify that the table resulting from a query has the expected constraints."""
    session.create_view(query, "view", cache=False)
    # pylint: disable=protected-access
    assert set(session._table_constraints[NamedTable("view")]) == set(
        expected_constraints
    )
    # pylint: enable=protected-access


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "column_mapper,constraints,expected_constraints",
    [
        (
            {"group": "g"},
            _CONSTRAINTS0,
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("g", 4),
                MaxGroupsPerID("group2", 3),
                MaxRowsPerGroupPerID("g", 2),
                MaxRowsPerGroupPerID("group2", 1),
            ],
        ),
        ({"id": "id2"}, _CONSTRAINTS0, _CONSTRAINTS0),
    ],
)
def test_rename(
    column_mapper: Dict[str, str],
    constraints: List[Constraint],
    expected_constraints: List[Constraint],
    session,
):
    """Propagation of constraints through renames works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.rename(column_mapper)
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints", [(_CONSTRAINTS0, _CONSTRAINTS0)]
)
def test_filter(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through filters works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.filter("n > 6")
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints",
    [
        (
            _CONSTRAINTS0,
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 4),
                MaxRowsPerGroupPerID("group", 2),
            ],
        )
    ],
)
def test_select(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through selects works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.select(["id", "group", "n"])
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints", [(_CONSTRAINTS0, _CONSTRAINTS0)]
)
def test_map(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through maps works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.map(
        lambda _: {"A": 1, "B": "c"},
        {"A": ColumnType.INTEGER, "B": ColumnType.VARCHAR},
        augment=True,
    )
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints",
    [(_CONSTRAINTS0, [MaxGroupsPerID("group", 4), MaxGroupsPerID("group2", 3)])],
)
def test_flat_map(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through flat maps works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.flat_map(
        lambda r: [{"A": i} for i in range(0, r["n"])],
        {"A": ColumnType.INTEGER},
        augment=True,
    )
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "left_constraints,right_constraints,expected_constraints",
    [
        ([MaxRowsPerID(1)], [], []),
        ([MaxRowsPerID(2)], [MaxRowsPerID(3)], [MaxRowsPerID(6)]),
        ([MaxGroupsPerID("group", 2)], [], [MaxGroupsPerID("group", 2)]),
        (
            [MaxGroupsPerID("group", 2)],
            [MaxRowsPerID(3)],
            [MaxGroupsPerID("group", 2), MaxRowsPerID(6)],
        ),
        (
            [MaxGroupsPerID("group2", 2)],
            [MaxRowsPerID(3)],
            [MaxGroupsPerID("group2", 2)],
        ),
        ([MaxRowsPerGroupPerID("group", 2)], [], []),
        (
            [MaxRowsPerGroupPerID("group", 2)],
            [MaxRowsPerID(3)],
            [MaxRowsPerGroupPerID("group", 6)],
        ),
        (
            [MaxRowsPerGroupPerID("group2", 2)],
            [MaxRowsPerID(3)],
            [MaxRowsPerGroupPerID("group2", 6)],
        ),
    ],
)
def test_join_private(
    left_constraints: List[Constraint],
    right_constraints: List[Constraint],
    expected_constraints: List[Constraint],
    session,
):
    """Propagation of constraints through private joins works as expected."""
    query = QueryBuilder("id_a1")
    for c in left_constraints:
        query = query.enforce(c)

    right_query = QueryBuilder("id_a2")
    for c in right_constraints:
        right_query = right_query.enforce(c)

    query = query.join_private(right_query)
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "left_constraints,right_constraints,expected_constraints",
    [
        ([MaxRowsPerID(2)], [MaxRowsPerID(3)], [MaxRowsPerID(6)]),
        (
            [MaxGroupsPerID("group", 2)],
            [MaxRowsPerID(3)],
            [MaxGroupsPerID("group_left", 2)],
        ),
        (
            [MaxRowsPerID(2)],
            [MaxGroupsPerID("group", 3)],
            [MaxGroupsPerID("group_right", 3)],
        ),
        ([MaxGroupsPerID("n", 2)], [MaxRowsPerID(3)], [MaxGroupsPerID("n", 2)]),
        ([MaxRowsPerID(2)], [MaxGroupsPerID("x", 3)], [MaxGroupsPerID("x", 3)]),
        (
            [MaxRowsPerGroupPerID("group", 2)],
            [MaxRowsPerID(3)],
            [MaxRowsPerGroupPerID("group_left", 6)],
        ),
        (
            [MaxRowsPerID(2)],
            [MaxRowsPerGroupPerID("group", 3)],
            [MaxRowsPerGroupPerID("group_right", 6)],
        ),
        (
            [MaxRowsPerGroupPerID("n", 2)],
            [MaxRowsPerID(3)],
            [MaxRowsPerGroupPerID("n", 6)],
        ),
        (
            [MaxRowsPerID(2)],
            [MaxRowsPerGroupPerID("x", 3)],
            [MaxRowsPerGroupPerID("x", 6)],
        ),
    ],
)
def test_join_private_disambiguation(
    left_constraints: List[Constraint],
    right_constraints: List[Constraint],
    expected_constraints: List[Constraint],
    session,
):
    """Propagation of constraints through private joins with column overlaps works."""
    query = QueryBuilder("id_a1")
    for c in left_constraints:
        query = query.enforce(c)

    right_query = QueryBuilder("id_a2")
    for c in right_constraints:
        right_query = right_query.enforce(c)

    query = query.join_private(right_query, join_columns=["id"])
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "public_df,constraints,expected_constraints",
    [
        (pd.DataFrame({"n": [1]}), [], []),
        (
            pd.DataFrame({"n": [1]}),
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
        ),
        (
            pd.DataFrame({"n": [1, 2]}),
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
        ),
        (
            pd.DataFrame({"n": [1, 1]}),
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
            [
                MaxRowsPerID(2),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 2),
            ],
        ),
    ],
)
def test_join_public(
    public_df: pd.DataFrame,
    constraints: List[Constraint],
    expected_constraints: List[Constraint],
    session,
    spark,
):
    """Propagation of constraints through private joins works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)

    query = query.join_public(spark.createDataFrame(public_df))
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "public_df,constraints,expected_constraints",
    [
        (pd.DataFrame({"n": [1], "group": ["A"]}), [], []),
        (
            pd.DataFrame({"n": [1], "group": ["A"]}),
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group_left", 1),
                MaxRowsPerGroupPerID("group_left", 1),
            ],
        ),
        (
            pd.DataFrame({"n": [1, 1], "group": ["A", "A"]}),
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
            [
                MaxRowsPerID(2),
                MaxGroupsPerID("group_left", 1),
                MaxRowsPerGroupPerID("group_left", 2),
            ],
        ),
        (
            pd.DataFrame({"n": [1, 1], "group": ["A", "B"]}),
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
            [
                MaxRowsPerID(2),
                MaxGroupsPerID("group_left", 1),
                MaxRowsPerGroupPerID("group_left", 2),
            ],
        ),
        (
            pd.DataFrame({"n": [1, 2], "group": ["A", "A"]}),
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group", 1),
                MaxRowsPerGroupPerID("group", 1),
            ],
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("group_left", 1),
                MaxRowsPerGroupPerID("group_left", 1),
            ],
        ),
    ],
)
def test_join_public_disambiguation(
    public_df: pd.DataFrame,
    constraints: List[Constraint],
    expected_constraints: List[Constraint],
    session,
    spark,
):
    """Propagation of constraints through private joins works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)

    query = query.join_public(spark.createDataFrame(public_df), join_columns=["n"])
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints",
    [
        (
            _CONSTRAINTS0,
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 4),
                MaxGroupsPerID("group2", 3),
                MaxRowsPerGroupPerID("group", 2),
            ],
        )
    ],
)
def test_replace_null_and_nan(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through replace nulls/nans works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.replace_null_and_nan({"group2": "Replacement"})
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints", [(_CONSTRAINTS0, _CONSTRAINTS0)]
)
def test_replace_infinity(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through replace infs works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.replace_infinity({"float_n": (0.0, 0.0)})
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints", [(_CONSTRAINTS0, _CONSTRAINTS0)]
)
def test_drop_null_and_nan(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through replace infs works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.drop_null_and_nan(["float_n"])
    _test_propagation(query, expected_constraints, session)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraints,expected_constraints", [(_CONSTRAINTS0, _CONSTRAINTS0)]
)
def test_drop_infinity(
    constraints: List[Constraint], expected_constraints: List[Constraint], session
):
    """Propagation of constraints through replace infs works as expected."""
    query = QueryBuilder("id_a1")
    for c in constraints:
        query = query.enforce(c)
    query = query.drop_infinity(["float_n"])
    _test_propagation(query, expected_constraints, session)
