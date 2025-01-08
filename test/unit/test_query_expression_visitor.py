"""Tests for QueryExprVisitor."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import pytest

from tmlt.analytics import KeySet, MaxRowsPerID, TruncationStrategy
from tmlt.analytics._query_expr import (
    DropInfinity,
    DropNullAndNan,
    EnforceConstraint,
    Filter,
    FlatMap,
    FlatMapByID,
    GetBounds,
    GetGroups,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    JoinPrivate,
    JoinPublic,
    Map,
    PrivateSource,
    QueryExpr,
    QueryExprVisitor,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
    SuppressAggregates,
)
from tmlt.analytics._schema import FrozenDict, Schema


class QueryExprIdentifierVisitor(QueryExprVisitor):
    """A simple QueryExprVisitor for testing."""

    def visit_private_source(self, expr):
        return "PrivateSource"

    def visit_rename(self, expr):
        return "Rename"

    def visit_filter(self, expr):
        return "Filter"

    def visit_select(self, expr):
        return "Select"

    def visit_map(self, expr):
        return "Map"

    def visit_flat_map(self, expr):
        return "FlatMap"

    def visit_flat_map_by_id(self, expr):
        return "FlatMapByID"

    def visit_join_private(self, expr):
        return "JoinPrivate"

    def visit_join_public(self, expr):
        return "JoinPublic"

    def visit_replace_null_and_nan(self, expr):
        return "ReplaceNullAndNan"

    def visit_replace_infinity(self, expr):
        return "ReplaceInfinity"

    def visit_drop_infinity(self, expr):
        return "DropInfinity"

    def visit_drop_null_and_nan(self, expr):
        return "DropNullAndNan"

    def visit_enforce_constraint(self, expr):
        return "EnforceConstraint"

    def visit_get_groups(self, expr):
        return "GetGroups"

    def visit_get_bounds(self, expr):
        return "GetBounds"

    def visit_groupby_count(self, expr):
        return "GroupByCount"

    def visit_groupby_count_distinct(self, expr):
        return "GroupByCountDistinct"

    def visit_groupby_quantile(self, expr):
        return "GroupByQuantile"

    def visit_groupby_bounded_sum(self, expr):
        return "GroupByBoundedSum"

    def visit_groupby_bounded_average(self, expr):
        return "GroupByBoundedAverage"

    def visit_groupby_bounded_variance(self, expr):
        return "GroupByBoundedVariance"

    def visit_groupby_bounded_stdev(self, expr):
        return "GroupByBoundedSTDEV"

    def visit_suppress_aggregates(self, expr):
        return "SuppressAggregates"


@pytest.mark.parametrize(
    "expr,expected",
    [
        (PrivateSource("P"), "PrivateSource"),
        (Rename(PrivateSource("P"), FrozenDict.from_dict({"A": "B"})), "Rename"),
        (Filter(PrivateSource("P"), "A<B"), "Filter"),
        (Select(PrivateSource("P"), tuple("A")), "Select"),
        (Map(PrivateSource("P"), lambda r: r, Schema({"A": "VARCHAR"}), True), "Map"),
        (
            FlatMap(
                PrivateSource("P"), lambda r: [r], Schema({"A": "VARCHAR"}), True, 1
            ),
            "FlatMap",
        ),
        (
            FlatMapByID(PrivateSource("P"), lambda rs: rs, Schema({"A": "VARCHAR"})),
            "FlatMapByID",
        ),
        (
            JoinPrivate(
                PrivateSource("P"),
                PrivateSource("Q"),
                TruncationStrategy.DropNonUnique(),
                TruncationStrategy.DropNonUnique(),
            ),
            "JoinPrivate",
        ),
        (JoinPublic(PrivateSource("P"), "Q"), "JoinPublic"),
        (
            ReplaceNullAndNan(
                PrivateSource("P"), FrozenDict.from_dict({"column": "default"})
            ),
            "ReplaceNullAndNan",
        ),
        (
            ReplaceInfinity(
                PrivateSource("P"), FrozenDict.from_dict({"column": (-100.0, 100.0)})
            ),
            "ReplaceInfinity",
        ),
        (DropInfinity(PrivateSource("P"), tuple("column")), "DropInfinity"),
        (DropNullAndNan(PrivateSource("P"), tuple("column")), "DropNullAndNan"),
        (
            EnforceConstraint(
                PrivateSource("P"), MaxRowsPerID(5), FrozenDict.from_dict({})
            ),
            "EnforceConstraint",
        ),
        (GetGroups(PrivateSource("P"), tuple("column")), "GetGroups"),
        (
            GetBounds(PrivateSource("P"), KeySet.from_dict({}), "A", "lower", "upper"),
            "GetBounds",
        ),
        (GroupByCount(PrivateSource("P"), KeySet.from_dict({})), "GroupByCount"),
        (
            GroupByCountDistinct(PrivateSource("P"), KeySet.from_dict({})),
            "GroupByCountDistinct",
        ),
        (
            GroupByQuantile(PrivateSource("P"), KeySet.from_dict({}), "A", 0.5, 0, 1),
            "GroupByQuantile",
        ),
        (
            GroupByBoundedSum(PrivateSource("P"), KeySet.from_dict({}), "A", 0, 1),
            "GroupByBoundedSum",
        ),
        (
            GroupByBoundedAverage(PrivateSource("P"), KeySet.from_dict({}), "A", 0, 1),
            "GroupByBoundedAverage",
        ),
        (
            GroupByBoundedVariance(PrivateSource("P"), KeySet.from_dict({}), "A", 0, 1),
            "GroupByBoundedVariance",
        ),
        (
            GroupByBoundedSTDEV(PrivateSource("P"), KeySet.from_dict({}), "A", 0, 1),
            "GroupByBoundedSTDEV",
        ),
        (
            SuppressAggregates(
                GroupByCount(
                    PrivateSource("P"),
                    KeySet.from_dict({}),
                    output_column="count",
                ),
                column="count",
                threshold=10,
            ),
            "SuppressAggregates",
        ),
    ],
)
def test_visitor(expr: QueryExpr, expected: str):
    """Verify that QueryExprs dispatch the correct methods in QueryExprVisitor."""
    visitor = QueryExprIdentifierVisitor()
    assert expr.accept(visitor) == expected
