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
        (PrivateSource(source_id="P"), "PrivateSource"),
        (Rename(child=PrivateSource(source_id="P"), column_mapper=FrozenDict.from_dict({"A": "B"})), "Rename"),
        (Filter(child=PrivateSource(source_id="P"), condition="A<B"), "Filter"),
        (Select(child=PrivateSource(source_id="P"), columns=tuple("A")), "Select"),
        (Map(child=PrivateSource(source_id="P"), f=lambda r: r, schema_new_columns=Schema({"A": "VARCHAR"}), augment=True), "Map"),
        (
            FlatMap(
                child=PrivateSource(source_id="P"), f=lambda r: [r],
                schema_new_columns=Schema({"A": "VARCHAR"}), augment=True,
                max_rows=1
            ),
            "FlatMap",
        ),
        (
            FlatMapByID(child=PrivateSource(source_id="P"), f=lambda rs: rs,
                        schema_new_columns=Schema({"A": "VARCHAR"})),
            "FlatMapByID",
        ),
        (
            JoinPrivate(
                child=PrivateSource(source_id="P"),
                right_operand_expr=PrivateSource(source_id="Q"),
                truncation_strategy_left=TruncationStrategy.DropNonUnique(),
                truncation_strategy_right=TruncationStrategy.DropNonUnique(),
            ),
            "JoinPrivate",
        ),
        (JoinPublic(child=PrivateSource(source_id="P"), public_table="Q"), "JoinPublic"),
        (
            ReplaceNullAndNan(
                child=PrivateSource(source_id="P"), replace_with=FrozenDict.from_dict({"column": "default"})
            ),
            "ReplaceNullAndNan",
        ),
        (
            ReplaceInfinity(
                child=PrivateSource(source_id="P"), replace_with=FrozenDict.from_dict({"column": (-100.0, 100.0)})
            ),
            "ReplaceInfinity",
        ),
        (DropInfinity(child=PrivateSource(source_id="P"), columns=tuple("column")), "DropInfinity"),
        (DropNullAndNan(child=PrivateSource(source_id="P"), columns=tuple("column")), "DropNullAndNan"),
        (
            EnforceConstraint(
                child=PrivateSource(source_id="P"), constraint=MaxRowsPerID(5),
                options=FrozenDict.from_dict({})
            ),
            "EnforceConstraint",
        ),
        (GetGroups(child=PrivateSource(source_id="P"), columns=tuple("column")), "GetGroups"),
        (
            GetBounds(child=PrivateSource(source_id="P"),
                      groupby_keys=KeySet.from_dict({}), measure_column="A",
                      lower_bound_column="lower", upper_bound_column="upper"),
            "GetBounds",
        ),
        (GroupByCount(child=PrivateSource(source_id="P"), groupby_keys=KeySet.from_dict({})), "GroupByCount"),
        (
            GroupByCountDistinct(child=PrivateSource(source_id="P"),
                                 groupby_keys=KeySet.from_dict({})),
            "GroupByCountDistinct",
        ),
        (
            GroupByQuantile(child=PrivateSource(source_id="P"),
                            groupby_keys=KeySet.from_dict({}),
                            measure_column="A", quantile=0.5, low=0, high=1),
            "GroupByQuantile",
        ),
        (
            GroupByBoundedSum(child=PrivateSource(source_id="P"),
                              groupby_keys=KeySet.from_dict({}),
                              measure_column="A", low=0, high=1),
            "GroupByBoundedSum",
        ),
        (
            GroupByBoundedAverage(child=PrivateSource(source_id="P"),
                                  groupby_keys=KeySet.from_dict({}),
                                  measure_column="A", low=0, high=1),
            "GroupByBoundedAverage",
        ),
        (
            GroupByBoundedVariance(child=PrivateSource(source_id="P"),
                                   groupby_keys=KeySet.from_dict({}),
                                   measure_column="A", low=0, high=1),
            "GroupByBoundedVariance",
        ),
        (
            GroupByBoundedSTDEV(child=PrivateSource(source_id="P"),
                                groupby_keys=KeySet.from_dict({}),
                                measure_column="A", low=0, high=1),
            "GroupByBoundedSTDEV",
        ),
        (
            SuppressAggregates(
                child=GroupByCount(
                    child=PrivateSource(source_id="P"),
                    groupby_keys=KeySet.from_dict({}),
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
