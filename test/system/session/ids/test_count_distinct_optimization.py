"""Tests for constraint inference and optimization on count-distinct queries."""

from typing import List

import pandas as pd
import pytest

from tmlt.analytics.constraints import MaxGroupsPerID, MaxRowsPerID
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget, RhoZCDPBudget
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import QueryExpr

from ....conftest import assert_frame_equal_with_sort
from ..conftest import INF_BUDGET, INF_BUDGET_ZCDP

_KEYSET = KeySet.from_dict({"group": ["A", "B"]})
_KEYSET2 = KeySet.from_dict({"group2": ["X", "Y"]})


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "base_query",
    [
        QueryBuilder("id_a1"),
        QueryBuilder("id_a1").enforce(MaxRowsPerID(1)),
        QueryBuilder("id_a1").enforce(MaxGroupsPerID("group", 2)),
    ],
)
def test_id_only(base_query: QueryBuilder, session):
    """Test ungrouped inference of count-distinct constraints."""
    res = session.evaluate(
        base_query.count_distinct(["id"]), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    assert res["count_distinct(id)"][0] == 3


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,expected_res",
    [
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            pd.DataFrame({"group": ["A", "B"], "count_distinct(id)": [3, 1]}),
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group2", 2))
            .groupby(_KEYSET2)
            .count_distinct(["id"]),
            pd.DataFrame({"group2": ["X", "Y"], "count_distinct(id)": [2, 3]}),
        ),
    ],
)
def test_id_only_grouped(query: QueryBuilder, expected_res: pd.DataFrame, session):
    """Test grouped inference of count-distinct constraints."""
    res = session.evaluate(query, session.remaining_privacy_budget).toPandas()

    assert_frame_equal_with_sort(res, expected_res)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query",
    [
        # Ungrouped count-distinct with additional distinct columns
        QueryBuilder("id_a1").count_distinct(["id", "group"]),
        # Grouped count-distinct without MaxGroupsPerID
        QueryBuilder("id_a1").groupby(_KEYSET).count_distinct(["id"]),
        # Grouped count-distinct with MaxGroupsPerID on different column
        QueryBuilder("id_a1")
        .enforce(MaxGroupsPerID("group2", 2))
        .groupby(_KEYSET)
        .count_distinct(["id"]),
        # Grouped count-distinct with multiple grouping columns
        QueryBuilder("id_a1")
        .enforce(MaxGroupsPerID("group", 2))
        .enforce(MaxGroupsPerID("group2", 2))
        .groupby(_KEYSET * _KEYSET2)
        .count_distinct(["id", "group", "group2"]),
    ],
)
def test_insufficient_constraints(query: QueryBuilder, session):
    """Test that constraint inference doesn't happen when it doesn't apply."""
    with pytest.raises(
        RuntimeError,
        match="^A constraint on the number of rows contributed by each ID.*",
    ):
        session.evaluate(query, session.remaining_privacy_budget)


@pytest.mark.parametrize("session", [INF_BUDGET], indirect=True, ids=["puredp"])
@pytest.mark.parametrize(
    "query,expected_noise",
    [
        (QueryBuilder("id_a1").count_distinct(["id"]), [1]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(1)).count_distinct(["id"]), [1]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(5)).count_distinct(["id"]), [1]),
        (
            QueryBuilder("id_a1")
            .enforce(MaxRowsPerID(1))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [1],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [2],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 1))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [1],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [2],
        ),
    ],
)
def test_noise_scale_puredp(query: QueryExpr, expected_noise: List[float], session):
    """Noise scales are adjusted correctly for different truncations with pure DP."""
    # pylint: disable=protected-access
    noise_info = session._noise_info(query, PureDPBudget(1))
    # pylint: enable=protected-access
    noise = [info["noise_parameter"] for info in noise_info]
    assert noise == expected_noise


@pytest.mark.parametrize("session", [INF_BUDGET_ZCDP], indirect=True, ids=["zcdp"])
@pytest.mark.parametrize(
    "query,expected_noise",
    [
        (QueryBuilder("id_a1").count_distinct(["id"]), [0.5]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(1)).count_distinct(["id"]), [0.5]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(5)).count_distinct(["id"]), [0.5]),
        (
            QueryBuilder("id_a1")
            .enforce(MaxRowsPerID(1))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [0.5],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [2],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 1))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [0.5],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .groupby(_KEYSET)
            .count_distinct(["id"]),
            [1],
        ),
    ],
)
def test_noise_scale_zcdp(query: QueryExpr, expected_noise: List[float], session):
    """Noise scales are adjusted correctly for different truncations with zCDP."""
    # pylint: disable=protected-access
    noise_info = session._noise_info(query, RhoZCDPBudget(1))
    # pylint: enable=protected-access
    noise = [info["noise_parameter"] for info in noise_info]
    assert noise == expected_noise
