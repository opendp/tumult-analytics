"""Integration tests for aggregations using the MaxRowPerID constraint."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import math
import statistics
from typing import List, Set, Tuple

import pytest

from tmlt.analytics.constraints import MaxRowsPerID
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import QueryExpr

from ..conftest import INF_BUDGET

_TRUNCATED_N = {
    1: [{4, 7, 8}, {4, 7, 9}, {5, 7, 8}, {5, 7, 9}, {6, 7, 8}, {6, 7, 9}],
    2: [{4, 5, 7, 8, 9}, {4, 6, 7, 8, 9}, {5, 6, 7, 8, 9}],
    3: [{4, 5, 6, 7, 8, 9}],
}
"""Possible sets of values of n at different truncation thresholds."""


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_count(threshold: int, session):
    """Ungrouped counts on tables with IDs work using MaxRowsPerID."""
    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).count(), INF_BUDGET
    ).toPandas()
    assert len(res) == 1
    assert res["count"][0] in {len(n) for n in _TRUNCATED_N[threshold]}


@pytest.mark.parametrize(
    "threshold,expected", [(1, {(3, 0), (2, 1)}), (2, {(4, 1)}), (3, {(5, 1)})]
)
def test_count_grouped(threshold: int, expected: Set[Tuple[int, int]], session):
    """Grouped counts on tables with IDs work using MaxRowsPerID."""
    ks = KeySet.from_dict({"group": ["A", "B"]})
    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).groupby(ks).count(),
        INF_BUDGET,
    ).toPandas()

    counts = tuple(
        res.loc[res["group"] == group]["count"].values[0] for group in ["A", "B"]
    )
    assert counts in expected


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_count_distinct(threshold: int, session):
    """Ungrouped count-distincts on tables with IDs work using MaxRowsPerID."""
    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).count_distinct(["n"]),
        INF_BUDGET,
    ).toPandas()
    assert len(res) == 1
    assert res["count_distinct(n)"][0] in {len(n) for n in _TRUNCATED_N[threshold]}

    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).count_distinct(["id"]),
        INF_BUDGET,
    ).toPandas()
    assert len(res) == 1
    assert res["count_distinct(id)"][0] == 3


@pytest.mark.parametrize(
    "threshold,expected", [(1, {(3, 0), (2, 1)}), (2, {(3, 1)}), (3, {(3, 1)})]
)
def test_count_distinct_grouped(
    threshold: int, expected: Set[Tuple[int, int]], session
):
    """Grouped count-distincts on tables with IDs work using MaxRowsPerID."""
    ks = KeySet.from_dict({"group": ["A", "B"]})
    res = session.evaluate(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(threshold))
        .groupby(ks)
        .count_distinct(["id"]),
        INF_BUDGET,
    ).toPandas()

    counts = tuple(
        res.loc[res["group"] == group]["count_distinct(id)"].values[0]
        for group in ["A", "B"]
    )
    assert counts in expected


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_quantile(threshold: int, session):
    """Ungrouped quantiles on tables with IDs work using MaxRowsPerID."""
    res = session.evaluate(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(threshold))
        .quantile("n", 0.5, 0, 10),
        INF_BUDGET,
    ).toPandas()
    assert len(res) == 1
    # Our quantile algorithm can be significantly off from the true median on
    # low numbers of records, even with infinite budget, so use isclose with a
    # huge tolerance.
    assert any(
        math.isclose(res["n_quantile(0.5)"][0], statistics.median(n), abs_tol=2)
        for n in _TRUNCATED_N[threshold]
    )


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_quantile_grouped(threshold: int, session):
    """Grouped quantiles on tables with IDs work using MaxRowsPerID."""
    ks = KeySet.from_dict({"group": ["A", "B"]})
    res = session.evaluate(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(threshold))
        .groupby(ks)
        .quantile("n", 0.5, 0, 10),
        INF_BUDGET,
    ).toPandas()

    quantiles = tuple(
        res.loc[res["group"] == group]["n_quantile(0.5)"].values[0]
        for group in ["A", "B"]
    )
    # Because of the inaccuracy in the quantile on the now even-smaller number
    # of records per group, checking its output is even harder. Just check that
    # the results it gives aren't completely absurd.
    assert all(0 <= q <= 10 for q in quantiles)


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_sum(threshold: int, session):
    """Ungrouped sums on tables with IDs work using MaxRowsPerID."""
    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).sum("n", 0, 10),
        INF_BUDGET,
    ).toPandas()
    assert len(res) == 1
    assert res["n_sum"][0] in {sum(n) for n in _TRUNCATED_N[threshold]}


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (1, {(11, 9), (12, 9), (13, 9), (19, 0), (20, 0), (21, 0)}),
        (2, {(24, 9), (25, 9), (26, 9)}),
        (3, {(30, 9)}),
    ],
)
def test_sum_grouped(threshold: int, expected: Set[Tuple[int, int]], session):
    """Grouped sums on tables with IDs work using MaxRowsPerID."""
    ks = KeySet.from_dict({"group": ["A", "B"]})
    res = session.evaluate(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(threshold))
        .groupby(ks)
        .sum("n", 0, 10),
        INF_BUDGET,
    ).toPandas()

    sums = tuple(
        res.loc[res["group"] == group]["n_sum"].values[0] for group in ["A", "B"]
    )
    assert sums in expected


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_average(threshold: int, session):
    """Ungrouped averages on tables with IDs work using MaxRowsPerID."""
    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).average("n", 0, 10),
        INF_BUDGET,
    ).toPandas()
    assert len(res) == 1
    assert res["n_average"][0] in {statistics.mean(n) for n in _TRUNCATED_N[threshold]}


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (1, {(11 / 2, 9), (6, 9), (13 / 2, 9), (19 / 3, 5), (20 / 3, 5), (7, 5)}),
        (2, {(24 / 4, 9), (25 / 4, 9), (26 / 4, 9)}),
        (3, {(30 / 5, 9)}),
    ],
)
def test_average_grouped(threshold: int, expected: Set[Tuple[int, int]], session):
    """Grouped averages on tables with IDs work using MaxRowsPerID."""
    ks = KeySet.from_dict({"group": ["A", "B"]})
    res = session.evaluate(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(threshold))
        .groupby(ks)
        .average("n", 0, 10),
        INF_BUDGET,
    ).toPandas()

    averages = tuple(
        res.loc[res["group"] == group]["n_average"].values[0] for group in ["A", "B"]
    )
    assert averages in expected


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_variance(threshold: int, session):
    """Ungrouped variances on tables with IDs work using MaxRowsPerID."""
    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).variance("n", 0, 10),
        INF_BUDGET,
    ).toPandas()
    assert len(res) == 1
    # There's some floating-point imprecision at play here, so use isclose.
    assert any(
        math.isclose(res["n_variance"][0], statistics.pvariance(n))
        for n in _TRUNCATED_N[threshold]
    )


@pytest.mark.parametrize(
    "threshold,A_ns",
    [
        (1, {(4, 7), (4, 7, 8), (5, 7), (5, 7, 8), (6, 7), (6, 7, 8)}),
        (2, {(4, 5, 7, 8), (4, 6, 7, 8), (5, 6, 7, 8)}),
        (3, {(4, 5, 6, 7, 8)}),
    ],
)
def test_variance_grouped(threshold: int, A_ns: Set[Tuple[int, ...]], session):
    """Grouped variances on tables with IDs work using MaxRowsPerID."""
    ks = KeySet.from_dict({"group": ["A", "B"]})
    res = session.evaluate(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(threshold))
        .groupby(ks)
        .variance("n", 0, 10),
        INF_BUDGET,
    ).toPandas()

    variances = tuple(
        res.loc[res["group"] == group]["n_variance"].values[0] for group in ["A", "B"]
    )
    # Our variance algorithm always produces (width of bounds / 2)**2 as the
    # variance for a single data point (before noise).
    expected_variances = {(statistics.pvariance(d), (10 / 2) ** 2) for d in A_ns}
    assert variances in expected_variances


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_stdev(threshold: int, session):
    """Ungrouped stdevs on tables with IDs work using MaxRowsPerID."""
    res = session.evaluate(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(threshold)).stdev("n", 0, 10),
        INF_BUDGET,
    ).toPandas()
    assert len(res) == 1
    # There's some floating-point imprecision at play here, so use isclose.
    assert any(
        math.isclose(res["n_stdev"][0], statistics.pstdev(n))
        for n in _TRUNCATED_N[threshold]
    )


@pytest.mark.parametrize(
    "threshold,A_ns",
    [
        (1, {(4, 7), (4, 7, 8), (5, 7), (5, 7, 8), (6, 7), (6, 7, 8)}),
        (2, {(4, 5, 7, 8), (4, 6, 7, 8), (5, 6, 7, 8)}),
        (3, {(4, 5, 6, 7, 8)}),
    ],
)
def test_stdev_grouped(threshold: int, A_ns: Set[Tuple[int, ...]], session):
    """Grouped stdevs on tables with IDs work using MaxRowsPerID."""
    ks = KeySet.from_dict({"group": ["A", "B"]})
    res = session.evaluate(
        QueryBuilder("id_a1")
        .enforce(MaxRowsPerID(threshold))
        .groupby(ks)
        .stdev("n", 0, 10),
        INF_BUDGET,
    ).toPandas()

    stdevs = tuple(
        res.loc[res["group"] == group]["n_stdev"].values[0] for group in ["A", "B"]
    )
    expected_stdevs = {(statistics.pstdev(d), 10 / 2) for d in A_ns}
    assert stdevs in expected_stdevs


@pytest.mark.parametrize(
    "query,expected_noise",
    [
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(1)).count(), [1]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(2)).count(), [2]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(5)).count(), [5]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(1)).count_distinct(["id"]), [1]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(2)).count_distinct(["id"]), [2]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(5)).count_distinct(["id"]), [5]),
        # Two aggregations, a sum and a count, each with half the budget
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(1)).average("n", 0, 10), [10, 2]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(5)).average("n", 0, 10), [50, 10]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(1)).average("n", 0, 20), [20, 2]),
        (QueryBuilder("id_a1").enforce(MaxRowsPerID(5)).average("n", 0, 20), [100, 10]),
    ],
)
def test_noise_scale(query: QueryExpr, expected_noise: List[float], session):
    """Noise scales are adjusted correctly with different truncation thresholds."""
    # pylint: disable=protected-access
    noise_info = session._noise_info(query, PureDPBudget(1))
    # pylint: enable=protected-access
    noise = [info["noise_parameter"] for info in noise_info]
    assert noise == expected_noise
