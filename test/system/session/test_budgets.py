"""Tests for passing different types of budgets and querying with them."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import sys

import pytest

from tmlt.analytics.constraints import MaxRowsPerID
from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import QueryExpr


@pytest.mark.parametrize(
    "session,query_budget",
    [
        (PureDPBudget(2), PureDPBudget(2.000000001)),
        (PureDPBudget(2), PureDPBudget(1.999999999)),
        (PureDPBudget(2.000000001), PureDPBudget(2)),
        (PureDPBudget(1.999999999), PureDPBudget(2)),
        (PureDPBudget(sys.float_info.max), PureDPBudget(0.1)),
        (PureDPBudget(sys.float_info.max), PureDPBudget(sys.float_info.max)),
        (PureDPBudget(float("inf")), PureDPBudget(1)),
        (PureDPBudget(float("inf")), PureDPBudget(1.5)),
        (PureDPBudget(float("inf")), PureDPBudget(float("inf"))),
        (RhoZCDPBudget(2), RhoZCDPBudget(2.000000001)),
        (RhoZCDPBudget(2), RhoZCDPBudget(1.999999999)),
        (RhoZCDPBudget(2.000000001), RhoZCDPBudget(2)),
        (RhoZCDPBudget(1.999999999), RhoZCDPBudget(2)),
        (RhoZCDPBudget(sys.float_info.max), RhoZCDPBudget(0.1)),
        (RhoZCDPBudget(sys.float_info.max), RhoZCDPBudget(sys.float_info.max)),
        (RhoZCDPBudget(float("inf")), RhoZCDPBudget(1)),
        (RhoZCDPBudget(float("inf")), RhoZCDPBudget(1.5)),
        (RhoZCDPBudget(float("inf")), RhoZCDPBudget(float("inf"))),
        (ApproxDPBudget(2, 0.1), ApproxDPBudget(2.000000001, 0.1)),
        (ApproxDPBudget(2, 0.1), ApproxDPBudget(1.999999999, 0.1)),
        (ApproxDPBudget(2.000000001, 0.1), ApproxDPBudget(2, 0.1)),
        (ApproxDPBudget(1.999999999, 0.1), ApproxDPBudget(2, 0.1)),
        (ApproxDPBudget(sys.float_info.max, 0.1), ApproxDPBudget(0.1, 0.1)),
        (
            ApproxDPBudget(sys.float_info.max, 0.1),
            ApproxDPBudget(sys.float_info.max, 0.1),
        ),
        (ApproxDPBudget(1, 1), ApproxDPBudget(1, 0.1)),
        (ApproxDPBudget(1, 1), ApproxDPBudget(1.5, 0.1)),
        (ApproxDPBudget(1, 1), ApproxDPBudget(1.5, 1)),
        (ApproxDPBudget(1, 1), ApproxDPBudget(float("inf"), 0.1)),
        (ApproxDPBudget(1, 1), ApproxDPBudget(float("inf"), 1)),
        (ApproxDPBudget(float("inf"), 0.1), ApproxDPBudget(1, 0.1)),
        (ApproxDPBudget(float("inf"), 0.1), ApproxDPBudget(1.5, 0.1)),
        (ApproxDPBudget(float("inf"), 0.1), ApproxDPBudget(1.5, 1)),
        (ApproxDPBudget(float("inf"), 0.1), ApproxDPBudget(float("inf"), 0.1)),
        (ApproxDPBudget(float("inf"), 0.1), ApproxDPBudget(float("inf"), 1)),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(1, 0.1)),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(1.5, 0.1)),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(1.5, 1)),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(float("inf"), 0.1)),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(float("inf"), 1)),
        (ApproxDPBudget(2, 0.1), PureDPBudget(2.000000001)),
        (ApproxDPBudget(2, 0.1), PureDPBudget(1.999999999)),
        (ApproxDPBudget(2.000000001, 0.1), PureDPBudget(2)),
        (ApproxDPBudget(1.999999999, 0.1), PureDPBudget(2)),
        (ApproxDPBudget(float("inf"), 0.1), PureDPBudget(2)),
        (ApproxDPBudget(1, 1), PureDPBudget(2)),
        (ApproxDPBudget(float("inf"), 0.1), PureDPBudget(float("inf"))),
        (ApproxDPBudget(1, 1), PureDPBudget(float("inf"))),
    ],
    indirect=["session"],
)
@pytest.mark.parametrize(
    "query",
    [
        QueryBuilder("rows_1").count(),
        # Other queries are unlikely to catch errors that this first count
        # doesn't, and they take a while to run all of the above test cases, so
        # they're only run in the nightly.
        pytest.param(QueryBuilder("rows_1").sum("X", 0, 10), marks=pytest.mark.slow),
        pytest.param(
            QueryBuilder("id_a1").enforce(MaxRowsPerID(2)).count(),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            QueryBuilder("id_a1").enforce(MaxRowsPerID(2)).sum("n", 0, 10),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_query(query_budget: PrivacyBudget, query: QueryExpr, session):
    """Queries with unusual budget combinations work as expected."""
    session.evaluate(query, query_budget)
