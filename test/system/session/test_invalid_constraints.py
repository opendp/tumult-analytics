"""Tests for invalid constraint enforcement."""

import pytest

from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import QueryExpr

from .conftest import INF_BUDGET, INF_BUDGET_ZCDP


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "constraint",
    [MaxRowsPerID(5), MaxGroupsPerID("A", 5), MaxRowsPerGroupPerID("A", 5)],
)
def test_constraint_non_ids(constraint: Constraint, session):
    """Applying constraints to non-IDs tables raises an exception."""
    with pytest.raises(
        ValueError,
        match="Constraint.*can only be applied.*AddRowsWithID protected change",
    ):
        session.evaluate(
            QueryBuilder("rows_1").enforce(constraint).count(),
            session.remaining_privacy_budget,
        )


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,expected_message",
    [
        (
            QueryBuilder("id_a1").enforce(MaxGroupsPerID("id", 1)).count(),
            "The grouping column of constraint MaxGroups.* cannot be the ID column.*",
        ),
        (
            QueryBuilder("id_a1").enforce(MaxRowsPerGroupPerID("id", 1)).count(),
            "The grouping column of constraint MaxRowsPer.* cannot be the ID column.*",
        ),
        (
            QueryBuilder("id_a1").enforce(MaxGroupsPerID("none", 1)).count(),
            "The grouping column of constraint MaxGroups.* does not exist in.*",
        ),
        (
            QueryBuilder("id_a1").enforce(MaxRowsPerGroupPerID("none", 1)).count(),
            "The grouping column of constraint MaxRowsPer.* does not exist in.*",
        ),
    ],
)
def test_invalid_grouping_column(query: QueryExpr, expected_message: str, session):
    """Enforcing constraints with incompatible grouping columns raises an exception."""
    with pytest.raises(ValueError, match=expected_message):
        session.evaluate(query, session.remaining_privacy_budget)
