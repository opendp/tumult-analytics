"""Integration tests for constraint propagation."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Callable, Dict, List

import pandas as pd
import pytest
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics import (
    AddRowsWithID,
    ColumnType,
    Constraint,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PureDPBudget,
    QueryBuilder,
    Session,
)
from tmlt.analytics._table_identifier import NamedTable

from ..conftest import INF_BUDGET, INF_BUDGET_ZCDP

_BASIC_CONSTRAINTS = [
    MaxRowsPerID(5),
    MaxGroupsPerID("group", 4),
    MaxGroupsPerID("group2", 3),
    MaxRowsPerGroupPerID("group", 2),
    MaxRowsPerGroupPerID("group2", 1),
]
_CONSTRAINTS_REQUIRING_SIMPLIFICATION = [
    MaxRowsPerID(5),
    MaxRowsPerID(1),
    MaxGroupsPerID("group", 4),
    MaxGroupsPerID("group", 2),
    MaxRowsPerGroupPerID("group2", 3),
    MaxRowsPerGroupPerID("group2", 1),
]
_SIMPLIFIED_CONSTRAINTS = [
    MaxRowsPerID(1),
    MaxGroupsPerID("group", 2),
    MaxRowsPerGroupPerID("group2", 1),
]


def _test_propagation(query, expected_constraints, session):
    """Verify that the table resulting from a query has the expected constraints."""
    session.create_view(query, "view", cache=False)
    assert set(session._table_constraints[NamedTable("view")]) == set(
        expected_constraints
    )


def _session_from_dataframes(dataframes: Dict[str, pd.DataFrame], spark) -> Session:
    """Construct a Session with all tables using the same ID space."""
    builder = Session.Builder().with_privacy_budget(PureDPBudget(float("inf")))
    builder = builder.with_id_space("ids")
    for source_id, dataframe in dataframes.items():
        builder = builder.with_private_dataframe(
            source_id,
            spark.createDataFrame(dataframe),
            protected_change=AddRowsWithID("id", "ids"),
        )
    return builder.build()


@parametrize(
    [
        Case("JoinPrivate")(
            dataframes={
                "left": pd.DataFrame({"id": [1, 1], "group": ["A", "A"]}),
                "right": pd.DataFrame({"id": [1, 1], "group": ["A", "A"]}),
            },
            query=(
                QueryBuilder("left")
                .enforce(MaxRowsPerID(2))
                .join_private(
                    QueryBuilder("right")
                    .enforce(MaxGroupsPerID("group", 1))
                    .enforce(MaxRowsPerGroupPerID("group", 2)),
                    join_columns=["id", "group"],
                )
                .select(["id", "group"])
            ),
        ),
    ]
)
def test_final_enforcement_is_noop(
    dataframes: Dict[str, pd.DataFrame],
    query: QueryBuilder,
    spark,
):
    """Propagated constraints do not cause additional truncation."""
    session = _session_from_dataframes(dataframes, spark)
    session.create_view(query, "view", cache=False)

    ks = KeySet.from_dict({"group": ["A"]})
    propagated = session.evaluate(
        QueryBuilder("view").groupby(ks).count(), PureDPBudget(float("inf"))
    )
    # Using flat_map_by_id erases the existing, propagated constraints, so it
    # can be used to check that applying the propagated constraints doesn't drop
    # additional rows.
    nonpropagated = session.evaluate(
        QueryBuilder("view")
        .flat_map_by_id(
            lambda rows: [{"group": row["group"]} for row in rows],
            {"group": ColumnType.VARCHAR},
        )
        .enforce(MaxRowsPerID(100))
        .groupby(ks)
        .count(),
        PureDPBudget(float("inf")),
    )

    assert_dataframe_equal(propagated, nonpropagated)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "column_mapper,constraints,expected_constraints",
    [
        (
            {"group": "g"},
            _BASIC_CONSTRAINTS,
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("g", 4),
                MaxGroupsPerID("group2", 3),
                MaxRowsPerGroupPerID("g", 2),
                MaxRowsPerGroupPerID("group2", 1),
            ],
        ),
        ({"id": "id2"}, _BASIC_CONSTRAINTS, _BASIC_CONSTRAINTS),
        (
            {"group": "g"},
            _CONSTRAINTS_REQUIRING_SIMPLIFICATION,
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("g", 2),
                MaxRowsPerGroupPerID("group2", 1),
            ],
        ),
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
    "constraints,expected_constraints",
    [
        (_BASIC_CONSTRAINTS, _BASIC_CONSTRAINTS),
        (_CONSTRAINTS_REQUIRING_SIMPLIFICATION, _SIMPLIFIED_CONSTRAINTS),
    ],
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
            _BASIC_CONSTRAINTS,
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 4),
                MaxRowsPerGroupPerID("group", 2),
            ],
        ),
        (
            _CONSTRAINTS_REQUIRING_SIMPLIFICATION,
            [MaxRowsPerID(1), MaxGroupsPerID("group", 2)],
        ),
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
    "constraints,expected_constraints",
    [
        (_BASIC_CONSTRAINTS, _BASIC_CONSTRAINTS),
        (_CONSTRAINTS_REQUIRING_SIMPLIFICATION, _SIMPLIFIED_CONSTRAINTS),
    ],
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
    [
        (_BASIC_CONSTRAINTS, [MaxGroupsPerID("group", 4), MaxGroupsPerID("group2", 3)]),
        (_CONSTRAINTS_REQUIRING_SIMPLIFICATION, [MaxGroupsPerID("group", 2)]),
    ],
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
            [MaxGroupsPerID("group", 2)],
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
            [MaxRowsPerGroupPerID("group", 6), MaxRowsPerID(6)],
        ),
        (
            [MaxRowsPerGroupPerID("group2", 2)],
            [MaxRowsPerID(3)],
            [MaxRowsPerGroupPerID("group2", 6)],
        ),
        (
            [MaxRowsPerID(5), MaxRowsPerID(1)],
            [MaxRowsPerID(3)],
            [MaxRowsPerID(3)],
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
        (
            [MaxGroupsPerID("group", 5), MaxGroupsPerID("group", 2)],
            [MaxRowsPerID(3)],
            [MaxGroupsPerID("group_left", 2)],
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
        (
            pd.DataFrame({"n": [1, 1]}),
            _CONSTRAINTS_REQUIRING_SIMPLIFICATION,
            [
                MaxRowsPerID(2),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group2", 2),
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
        (
            pd.DataFrame({"n": [1, 1], "group": ["A", "A"]}),
            _CONSTRAINTS_REQUIRING_SIMPLIFICATION,
            [
                MaxRowsPerID(2),
                MaxGroupsPerID("group_left", 2),
                MaxRowsPerGroupPerID("group2", 2),
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
            _BASIC_CONSTRAINTS,
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 4),
                MaxGroupsPerID("group2", 3),
                MaxRowsPerGroupPerID("group", 2),
            ],
        ),
        (
            _CONSTRAINTS_REQUIRING_SIMPLIFICATION,
            [MaxRowsPerID(1), MaxGroupsPerID("group", 2)],
        ),
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
    "constraints,expected_constraints",
    [
        (_BASIC_CONSTRAINTS, _BASIC_CONSTRAINTS),
        (_CONSTRAINTS_REQUIRING_SIMPLIFICATION, _SIMPLIFIED_CONSTRAINTS),
    ],
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
    "constraints,expected_constraints",
    [
        (_BASIC_CONSTRAINTS, _BASIC_CONSTRAINTS),
        (_CONSTRAINTS_REQUIRING_SIMPLIFICATION, _SIMPLIFIED_CONSTRAINTS),
    ],
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
    "constraints,expected_constraints",
    [
        (_BASIC_CONSTRAINTS, _BASIC_CONSTRAINTS),
        (_CONSTRAINTS_REQUIRING_SIMPLIFICATION, _SIMPLIFIED_CONSTRAINTS),
    ],
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
