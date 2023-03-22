"""Integration tests for aggregations using L0 + L-inf truncation."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import statistics
from typing import List, Optional, Set, Tuple

import pytest

from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget, RhoZCDPBudget
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import QueryExpr

from ..conftest import INF_BUDGET, INF_BUDGET_ZCDP, closest_value

_TRUNCATED_N = {
    1: [{4, 7, 8}, {4, 7, 9}, {5, 7, 8}, {5, 7, 9}, {6, 7, 8}, {6, 7, 9}],
    2: [{4, 5, 7, 8, 9}, {4, 6, 7, 8, 9}, {5, 6, 7, 8, 9}],
    3: [{4, 5, 6, 7, 8, 9}],
}
"""Possible sets of values of n at different truncation thresholds."""

_BASE_QUERIES = [
    QueryBuilder("id_a1")
    .enforce(MaxGroupsPerID("group", 1))
    .enforce(MaxRowsPerGroupPerID("group", 1)),
    QueryBuilder("id_a1")
    .enforce(MaxGroupsPerID("group", 2))
    .enforce(MaxRowsPerGroupPerID("group", 1)),
    QueryBuilder("id_a1")
    .enforce(MaxGroupsPerID("group", 1))
    .enforce(MaxRowsPerGroupPerID("group", 2)),
    QueryBuilder("id_a1")
    .enforce(MaxGroupsPerID("group", 2))
    .enforce(MaxRowsPerGroupPerID("group", 2)),
    QueryBuilder("id_a1")
    .enforce(MaxGroupsPerID("group", 1))
    .enforce(MaxRowsPerGroupPerID("group", 3)),
    QueryBuilder("id_a1")
    .enforce(MaxGroupsPerID("group", 2))
    .enforce(MaxRowsPerGroupPerID("group", 3)),
]
_BASE_QUERY_NS_GROUPED: List[Set[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = [
    {
        ((4, 7, 8), ()),
        ((5, 7, 8), ()),
        ((6, 7, 8), ()),
        ((4, 7), (9,)),
        ((5, 7), (9,)),
        ((6, 7), (9,)),
    },
    {((4, 7, 8), (9,)), ((5, 7, 8), (9,)), ((6, 7, 8), (9,))},
    {
        ((4, 5, 7, 8), ()),
        ((4, 6, 7, 8), ()),
        ((5, 6, 7, 8), ()),
        ((4, 5, 7), (9,)),
        ((4, 6, 7), (9,)),
        ((5, 6, 7), (9,)),
    },
    {((4, 5, 7, 8), (9,)), ((4, 6, 7, 8), (9,)), ((5, 6, 7, 8), (9,))},
    {((4, 5, 6, 7, 8), ()), ((4, 5, 6, 7), (9,))},
    {((4, 5, 6, 7, 8), (9,))},
]
_BASE_QUERY_NS = [{n[0] + n[1] for n in ns} for ns in _BASE_QUERY_NS_GROUPED]
_KEYSET = KeySet.from_dict({"group": ["A", "B"]})
_KEYSET2 = KeySet.from_dict({"group2": ["X", "Y"]})


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS))
def test_count(base_query: QueryBuilder, ns: Set[Tuple[int, ...]], session):
    """Ungrouped counts on tables with IDs work using L2 truncation."""
    res = session.evaluate(
        base_query.count(), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    assert res["count"][0] in {len(n) for n in ns}


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS_GROUPED))
def test_count_grouped(
    base_query: QueryBuilder, ns: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], session
):
    """Grouped counts on tables with IDs work using L2 truncation."""
    res = session.evaluate(
        base_query.groupby(_KEYSET).count(), session.remaining_privacy_budget
    ).toPandas()

    counts = tuple(
        res.loc[res["group"] == group]["count"].values[0] for group in ["A", "B"]
    )
    assert counts in {(len(n[0]), len(n[1])) for n in ns}


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "base_query,expected", zip(_BASE_QUERIES, [{3}, {3}, {3}, {3}, {3}, {3}])
)
def test_count_distinct(base_query: QueryBuilder, expected: Set[int], session):
    """Ungrouped count-distincts on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.count_distinct(["id"]), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    assert res["count_distinct(id)"][0] in expected


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "base_query,expected",
    zip(
        _BASE_QUERIES,
        [
            {(2, 1), (3, 0)},
            {(3, 1)},
            {(2, 1), (3, 0)},
            {(3, 1)},
            {(2, 1), (3, 0)},
            {(3, 1)},
        ],
    ),
)
def test_count_distinct_grouped(
    base_query: QueryBuilder, expected: Set[Tuple[int, int]], session
):
    """Grouped count-distincts on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.groupby(_KEYSET).count_distinct(["id"]),
        session.remaining_privacy_budget,
    ).toPandas()

    counts = tuple(
        res.loc[res["group"] == group]["count_distinct(id)"].values[0]
        for group in ["A", "B"]
    )
    assert counts in expected


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS))
def test_quantile(base_query: QueryBuilder, ns: Set[Tuple[int, ...]], session):
    """Ungrouped quantiles on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.quantile("n", 0.5, 0, 10), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    value = res["n_quantile(0.5)"][0]
    closest = closest_value(value, {statistics.median(n) for n in ns})
    # Our quantile algorithm can be significantly off from the true median on
    # low numbers of records, even with infinite budget, so use a huge absolute
    # tolerance.
    assert value == pytest.approx(closest, abs=2)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query", _BASE_QUERIES)
def test_quantile_grouped(base_query: QueryBuilder, session):
    """Grouped quantiles on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.groupby(_KEYSET).quantile("n", 0.5, 0, 10),
        session.remaining_privacy_budget,
    ).toPandas()

    quantiles = tuple(
        res.loc[res["group"] == group]["n_quantile(0.5)"].values[0]
        for group in ["A", "B"]
    )
    # Because of the inaccuracy in the quantile on the now even-smaller number
    # of records per group, checking its output is even harder. Just ensure that
    # the results it gives aren't completely absurd.
    assert all(0 <= q <= 10 for q in quantiles)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS))
def test_sum(base_query: QueryBuilder, ns: Set[Tuple[int, ...]], session):
    """Ungrouped sums on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.sum("n", 0, 10), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    assert res["n_sum"][0] in {sum(n) for n in ns}


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS_GROUPED))
def test_sum_grouped(
    base_query: QueryBuilder, ns: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], session
):
    """Grouped sums on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.groupby(_KEYSET).sum("n", 0, 10), session.remaining_privacy_budget
    ).toPandas()

    sums = tuple(
        res.loc[res["group"] == group]["n_sum"].values[0] for group in ["A", "B"]
    )
    assert sums in {tuple(sum(g) for g in n) for n in ns}


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS))
def test_average(base_query: QueryBuilder, ns: Set[Tuple[int, ...]], session):
    """Ungrouped averages on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.average("n", 0, 10), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    assert res["n_average"][0] in {statistics.mean(n) for n in ns}


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS_GROUPED))
def test_average_grouped(
    base_query: QueryBuilder, ns: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], session
):
    """Grouped averages on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.groupby(_KEYSET).average("n", 0, 10),
        session.remaining_privacy_budget,
    ).toPandas()

    averages = tuple(
        res.loc[res["group"] == group]["n_average"].values[0] for group in ["A", "B"]
    )
    assert averages in {tuple(statistics.mean(g) if g else 5 for g in n) for n in ns}


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS))
def test_variance(base_query: QueryBuilder, ns: Set[Tuple[int, ...]], session):
    """Ungrouped variances on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.variance("n", 0, 10), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    # There's some floating-point imprecision at play here, so find the closest
    # value and use approx() for the comparison. Our variance algorithm always
    # produces (width of bounds / 2)**2 as the variance of an empty collection
    # of rows, so fill that in to prevent pvariance from raising an exception in
    # that case.
    value = res["n_variance"][0]
    closest = closest_value(value, {statistics.pvariance(n) if n else 25 for n in ns})
    assert value == pytest.approx(closest)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS_GROUPED))
def test_variance_grouped(
    base_query: QueryBuilder, ns: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], session
):
    """Grouped variances on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.groupby(_KEYSET).variance("n", 0, 10),
        session.remaining_privacy_budget,
    ).toPandas()
    expected_variances = {
        tuple(statistics.pvariance(g) if len(g) > 1 else 25 for g in n) for n in ns
    }
    # There's some floating-point imprecision at play here, so find the closest
    # value and use approx() for the comparison.
    value = tuple(
        res.loc[res["group"] == group]["n_variance"].values[0] for group in ["A", "B"]
    )
    closest = closest_value(value, expected_variances)
    assert value == pytest.approx(closest)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS))
def test_stdev(base_query: QueryBuilder, ns: Set[Tuple[int, ...]], session):
    """Ungrouped stdevs on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.stdev("n", 0, 10), session.remaining_privacy_budget
    ).toPandas()
    assert len(res) == 1
    # There's some floating-point imprecision at play here, so find the closest
    # value and use approx() for the comparison. Our stdev algorithm always
    # produces (width of bounds / 2) as the stdev (before noise) if no data
    # points are available, so fill that in to prevent pstdev from raising an
    # exception in that case.
    value = res["n_stdev"][0]
    closest = closest_value(value, {statistics.pstdev(n) if n else 25 for n in ns})
    assert value == pytest.approx(closest)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize("base_query,ns", zip(_BASE_QUERIES, _BASE_QUERY_NS_GROUPED))
def test_stdev_grouped(
    base_query: QueryBuilder, ns: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], session
):
    """Grouped stdevs on tables with IDs work using L0 + L-inf."""
    res = session.evaluate(
        base_query.groupby(_KEYSET).stdev("n", 0, 10), session.remaining_privacy_budget
    ).toPandas()
    expected_stdevs = {
        tuple(statistics.pstdev(g) if len(g) > 1 else 5 for g in n) for n in ns
    }
    value = tuple(
        res.loc[res["group"] == group]["n_stdev"].values[0] for group in ["A", "B"]
    )
    closest = closest_value(value, expected_stdevs)
    assert value == pytest.approx(closest)


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
def test_mismatched_grouping_columns(session):
    """Constraints with different grouping columns can't be used for truncation."""
    query = (
        QueryBuilder("id_a1")
        .enforce(MaxGroupsPerID("group", 1))
        .enforce(MaxRowsPerGroupPerID("group2", 1))
        .count()
    )
    with pytest.raises(
        RuntimeError, match="A constraint on the number of rows contributed.*"
    ):
        session.evaluate(query, session.remaining_privacy_budget)


@pytest.mark.parametrize("session", [INF_BUDGET], indirect=True, ids=["puredp"])
@pytest.mark.parametrize(
    "query,expected_noise",
    [
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 1))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .count(),
            [1],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 1))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .groupby(_KEYSET)
            .count(),
            [1],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .count(),
            [2],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .groupby(_KEYSET)
            .count(),
            [2],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 3))
            .average("n", 0, 10),
            [60, 12],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 3))
            .groupby(_KEYSET)
            .average("n", 0, 10),
            [60, 12],
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
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 1))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .count(),
            [0.5],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 1))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .groupby(_KEYSET)
            .count(),
            [0.5],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .count(),
            [2],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 1))
            .groupby(_KEYSET)
            .count(),
            [1],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 3))
            .average("n", 0, 10),
            [900, 36],
        ),
        (
            QueryBuilder("id_a1")
            .enforce(MaxGroupsPerID("group", 2))
            .enforce(MaxRowsPerGroupPerID("group", 3))
            .groupby(_KEYSET)
            .average("n", 0, 10),
            [450, 18],
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


@pytest.mark.parametrize("session", [INF_BUDGET], indirect=True, ids=["puredp"])
@pytest.mark.parametrize(
    "constraints,group,expected_noise",
    [
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
            ],
            None,
            [5],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
            ],
            _KEYSET,
            [5],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 2),
            ],
            _KEYSET,
            [4],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 2),
            ],
            _KEYSET2,
            [4],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
                MaxGroupsPerID("group2", 2),
                MaxRowsPerGroupPerID("group2", 2),
            ],
            _KEYSET * _KEYSET2,
            [4],
        ),
    ],
)
def test_constraint_selection_puredp(
    constraints: List[Constraint],
    group: Optional[KeySet],
    expected_noise: List[float],
    session,
):
    """Lowest-noise truncations are used when multiple are available with pure DP."""
    base_query = QueryBuilder("id_a1")
    for c in constraints:
        base_query.enforce(c)
    query = base_query.groupby(group).count() if group else base_query.count()

    # pylint: disable=protected-access
    noise_info = session._noise_info(query, PureDPBudget(1))
    # pylint: enable=protected-access
    noise = [info["noise_parameter"] for info in noise_info]
    assert noise == expected_noise


@pytest.mark.parametrize("session", [INF_BUDGET_ZCDP], indirect=True, ids=["zcdp"])
@pytest.mark.parametrize(
    "constraints,group,expected_noise",
    [
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
            ],
            None,
            [25 / 2],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
            ],
            _KEYSET,
            [9],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
            ],
            _KEYSET2,
            [25 / 2],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
                MaxGroupsPerID("group2", 3),
                MaxRowsPerGroupPerID("group2", 2),
            ],
            _KEYSET,
            [9],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
                MaxGroupsPerID("group2", 3),
                MaxRowsPerGroupPerID("group2", 2),
            ],
            _KEYSET2,
            [6],
        ),
        (
            [
                MaxRowsPerID(5),
                MaxGroupsPerID("group", 2),
                MaxRowsPerGroupPerID("group", 3),
                MaxGroupsPerID("group2", 3),
                MaxRowsPerGroupPerID("group2", 2),
            ],
            _KEYSET * _KEYSET2,
            [6],
        ),
    ],
)
def test_constraint_selection_zcdp(
    constraints: List[Constraint],
    group: Optional[KeySet],
    expected_noise: List[float],
    session,
):
    """Lowest-noise truncations are used when multiple are available with zCDP."""
    base_query = QueryBuilder("id_a1")
    for c in constraints:
        base_query.enforce(c)
    query = base_query.groupby(group).count() if group else base_query.count()

    # pylint: disable=protected-access
    noise_info = session._noise_info(query, RhoZCDPBudget(1))
    # pylint: enable=protected-access
    noise = [info["noise_parameter"] for info in noise_info]
    assert noise == expected_noise
