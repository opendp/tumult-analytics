"""Integration tests for truncation constraints with multiple grouping columns."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023-2025, and the Tumult Analytics Contributors 2025-present

from typing import List

import pandas as pd
import pytest
from tmlt.core.utils.testing import assert_dataframe_equal

from tmlt.analytics import (
    AddRowsWithID,
    Constraint,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    PrivacyBudget,
    PureDPBudget,
    Query,
    QueryBuilder,
    RhoZCDPBudget,
    Session,
)

_ROW_TRUNCATION_DF = pd.DataFrame(
    [
        (1, "x", "blue"),
        (1, "x", "blue"),
        (1, "x", "red"),
        (1, "y", "blue"),
        (2, "x", "blue"),
        (2, "y", "red"),
        (2, "y", "red"),
        (2, "y", "red"),
        (3, "y", "red"),
    ],
    columns=["id", "letter", "color"],
)
"""Table with differing numbers of rows per (ID, letter, color) group.

No ID appears in more than three (letter, color) groups, so ``MaxGroupsPerID(..., 3)``
never drops any rows, and truncation is entirely determined by
``MaxRowsPerGroupPerID``. ID 1 has three groups, but only two letters and two colors,
so grouping by either column alone would produce different results.
"""

_GROUP_TRUNCATION_DF = pd.DataFrame(
    [
        (1, "x", "blue"),
        (1, "x", "blue"),
        (1, "x", "red"),
        (1, "x", "red"),
        (1, "y", "blue"),
        (1, "y", "blue"),
        (2, "x", "blue"),
        (2, "x", "blue"),
        (2, "y", "red"),
        (2, "y", "red"),
        (3, "y", "red"),
        (3, "y", "red"),
    ],
    columns=["id", "letter", "color"],
)
"""Table where every (ID, letter, color) group has exactly two rows.

Because all of an ID's groups are the same size, the number of rows kept per ID
does not depend on which groups truncation keeps. ID 1 has three (letter, color)
groups but only two distinct values of ``letter``, so truncating on ``letter``
or ``color`` alone would give different results.
"""

_KEYSET = KeySet.from_dict({"letter": ["x", "y"], "color": ["blue", "red"]})


@pytest.fixture(name="session")
def session_fixture(spark, request) -> Session:
    """A Session containing the test tables, with the given privacy budget."""
    budget = request.param
    assert isinstance(budget, PrivacyBudget)
    return (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_id_space("ids")
        .with_private_dataframe(
            "row_truncation",
            spark.createDataFrame(_ROW_TRUNCATION_DF),
            protected_change=AddRowsWithID("id", "ids"),
        )
        .with_private_dataframe(
            "group_truncation",
            spark.createDataFrame(_GROUP_TRUNCATION_DF),
            protected_change=AddRowsWithID("id", "ids"),
        )
        .build()
    )


_BUDGETS = pytest.mark.parametrize(
    "session",
    [PureDPBudget(float("inf")), RhoZCDPBudget(float("inf"))],
    indirect=True,
    ids=["puredp", "zcdp"],
)


@_BUDGETS
@pytest.mark.parametrize(
    "max_rows,expected",
    [
        (
            1,
            pd.DataFrame(
                {
                    "letter": ["x", "x", "y", "y"],
                    "color": ["blue", "red", "blue", "red"],
                    "count": [2, 1, 1, 2],
                }
            ),
        ),
        (
            2,
            pd.DataFrame(
                {
                    "letter": ["x", "x", "y", "y"],
                    "color": ["blue", "red", "blue", "red"],
                    "count": [3, 1, 1, 3],
                }
            ),
        ),
        (
            3,
            pd.DataFrame(
                {
                    "letter": ["x", "x", "y", "y"],
                    "color": ["blue", "red", "blue", "red"],
                    "count": [3, 1, 1, 4],
                }
            ),
        ),
    ],
)
def test_rows_per_group_grouped(
    max_rows: int, expected: pd.DataFrame, session: Session
):
    """Rows are truncated per (ID, a, b) group for grouped counts."""
    query = (
        QueryBuilder("row_truncation")
        .enforce(MaxGroupsPerID(["letter", "color"], 3))
        .enforce(MaxRowsPerGroupPerID(["letter", "color"], max_rows))
        .groupby(_KEYSET)
        .count()
    )
    assert_dataframe_equal(
        session.evaluate(query, session.remaining_privacy_budget).toPandas(), expected
    )


@_BUDGETS
@pytest.mark.parametrize("max_rows,expected", [(1, 6), (2, 8), (3, 9)])
def test_rows_per_group_ungrouped(max_rows: int, expected: int, session: Session):
    """Rows are truncated per (ID, a, b) group for ungrouped counts."""
    query = (
        QueryBuilder("row_truncation")
        .enforce(MaxGroupsPerID(["letter", "color"], 3))
        .enforce(MaxRowsPerGroupPerID(["letter", "color"], max_rows))
        .count()
    )
    assert_dataframe_equal(
        session.evaluate(query, session.remaining_privacy_budget).toPandas(),
        pd.DataFrame({"count": [expected]}),
    )


@_BUDGETS
@pytest.mark.parametrize(
    "max_groups,max_rows,expected",
    [
        # ID 1 keeps one of its (a, b) groups, rather than all of its rows with
        # one value of a, as it would when truncating on a alone.
        (1, 4, 6),
        (1, 1, 3),
        (2, 1, 5),
        (2, 2, 10),
        (3, 2, 12),
    ],
)
def test_groups_per_id(max_groups: int, max_rows: int, expected: int, session: Session):
    """The number of (a, b) groups per ID is truncated."""
    query = (
        QueryBuilder("group_truncation")
        .enforce(MaxGroupsPerID(["letter", "color"], max_groups))
        .enforce(MaxRowsPerGroupPerID(["letter", "color"], max_rows))
        .count()
    )
    assert_dataframe_equal(
        session.evaluate(query, session.remaining_privacy_budget).toPandas(),
        pd.DataFrame({"count": [expected]}),
    )


@_BUDGETS
def test_column_order_does_not_matter(session: Session):
    """Constraints on the same columns in different orders are used together."""
    query = (
        QueryBuilder("row_truncation")
        .enforce(MaxGroupsPerID(["letter", "color"], 3))
        .enforce(MaxRowsPerGroupPerID(["color", "letter"], 1))
        .groupby(_KEYSET)
        .count()
    )
    assert_dataframe_equal(
        session.evaluate(query, session.remaining_privacy_budget).toPandas(),
        pd.DataFrame(
            {
                "letter": ["x", "x", "y", "y"],
                "color": ["blue", "red", "blue", "red"],
                "count": [2, 1, 1, 2],
            }
        ),
    )


@_BUDGETS
@pytest.mark.parametrize(
    "constraints",
    [
        [MaxGroupsPerID(["letter", "color"], 1), MaxRowsPerGroupPerID("letter", 1)],
        [MaxGroupsPerID("letter", 1), MaxRowsPerGroupPerID(["letter", "color"], 1)],
        [MaxGroupsPerID(["letter", "color"], 1), MaxRowsPerGroupPerID("color", 1)],
    ],
)
def test_mismatched_grouping_columns(constraints: List[Constraint], session: Session):
    """Constraints with different sets of grouping columns can't be combined."""
    query = QueryBuilder("row_truncation")
    for c in constraints:
        query = query.enforce(c)
    with pytest.raises(
        RuntimeError, match="A constraint on the number of rows contributed.*"
    ):
        session.evaluate(query.count(), session.remaining_privacy_budget)


@_BUDGETS
def test_count_distinct_grouped(session: Session):
    """Count-distinct of IDs grouped on all grouping columns needs only L0.

    This relies on the count-distinct optimization; without it, a
    MaxRowsPerGroupPerID constraint would also be required.
    """
    query = (
        QueryBuilder("row_truncation")
        .enforce(MaxGroupsPerID(["letter", "color"], 3))
        .groupby(_KEYSET)
        .count_distinct(["id"])
    )
    assert_dataframe_equal(
        session.evaluate(query, session.remaining_privacy_budget).toPandas(),
        pd.DataFrame(
            {
                "letter": ["x", "x", "y", "y"],
                "color": ["blue", "red", "blue", "red"],
                "count_distinct(id)": [2, 1, 1, 2],
            }
        ),
    )


_NOISE_BASE = (
    QueryBuilder("row_truncation")
    .enforce(MaxGroupsPerID(["letter", "color"], 2))
    .enforce(MaxRowsPerGroupPerID(["letter", "color"], 3))
)


@pytest.mark.parametrize(
    "session", [PureDPBudget(float("inf"))], indirect=True, ids=["puredp"]
)
@pytest.mark.parametrize(
    "query,expected_noise",
    [
        (_NOISE_BASE.count(), 6),
        (_NOISE_BASE.groupby(_KEYSET).count(), 6),
        (_NOISE_BASE.groupby(KeySet.from_dict({"letter": ["x", "y"]})).count(), 6),
    ],
)
def test_noise_scale_puredp(query: Query, expected_noise: float, session: Session):
    """With pure DP, noise scales with the product of the truncation bounds."""
    noise_info = session._noise_info(query, PureDPBudget(1))
    assert [info["noise_parameter"] for info in noise_info] == [expected_noise]


@pytest.mark.parametrize(
    "session", [RhoZCDPBudget(float("inf"))], indirect=True, ids=["zcdp"]
)
@pytest.mark.parametrize(
    "query,expected_noise",
    [
        # Sensitivity is 2 * 3 = 6, so the noise parameter is 6^2 / 2 = 18.
        (_NOISE_BASE.count(), 18),
        # Grouping on all of the constraints' grouping columns allows using
        # L2 sensitivity, sqrt(2) * 3, so the noise parameter is 18 / 2 = 9.
        (_NOISE_BASE.groupby(_KEYSET).count(), 9),
        # Grouping on only some of the grouping columns does not.
        (_NOISE_BASE.groupby(KeySet.from_dict({"letter": ["x", "y"]})).count(), 18),
    ],
)
def test_noise_scale_zcdp(query: Query, expected_noise: float, session: Session):
    """With zCDP, L2 sensitivity is used when grouping on all grouping columns."""
    noise_info = session._noise_info(query, RhoZCDPBudget(1))
    noise = [info["noise_parameter"] for info in noise_info]
    assert noise == [pytest.approx(expected_noise)]
