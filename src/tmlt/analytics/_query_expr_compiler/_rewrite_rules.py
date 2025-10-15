"""Rules for rewriting QueryExprs."""

# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from functools import wraps
from typing import Callable

from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._query_expr import (
    AverageMechanism,
    CompilationInfo,
    CountDistinctMechanism,
    CountMechanism,
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
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)

EXPRS_WITH_ONE_CHILD = (
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
    JoinPublic,
    Map,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
    SuppressAggregates,
)
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._schema import ColumnType

def depth_first(func: Callable[[QueryExpr], QueryExpr]) -> Callable[[QueryExpr], QueryExpr]:
    """Recursively applies the given method to a QueryExpr, depth-first."""

    @wraps(func)
    def wrapped(expr: QueryExpr) -> QueryExpr:
        if isinstance(expr, PrivateSource):
            return func(expr)
        if isinstance(expr, EXPRS_WITH_ONE_CHILD):
            child=wrapped(expr.child)
            return func(replace(expr, child=child))
        elif isinstance(expr, JoinPrivate):
            left = wrapped(expr.child)
            right = wrapped(expr.right_operand_expr)
            return func(replace(expr, child=left, right_operand_expr=right))
        else:
            raise AnalyticsInternalError(
                    f"Unrecognized QueryExpr subtype {type(expr).__qualname__}."
            )

    return wrapped


def add_compilation_info(info: CompilationInfo) -> Callable[QueryExpr, QueryExpr]:
    """Adds the compilation info to each node of the QueryExpr."""

    @depth_first
    def add_info(expr: QueryExpr) -> QueryExpr:
        return replace(expr, compilation_info=info)

    return add_info


def select_noise_mechanism(expr: QueryExpr) -> QueryExpr:
    """Changes the default noise type into a concrete noise type for aggregations.

    This requires the QueryExpr to have been annotated with compilation info."""
    output_measure = expr.compilation_info.output_measure

    if isinstance(expr, SuppressAggregates):
        return replace(expr, child=select_noise_mechanism(expr.child))

    if isinstance(expr, (GroupByCount, GroupByCountDistinct)):
        if expr.mechanism in (CountMechanism.DEFAULT, CountDistinctMechanism.DEFAULT):
            core_mechanism = (
                NoiseMechanism.GEOMETRIC
                if isinstance(output_measure, (PureDP, ApproxDP))
                else NoiseMechanism.DISCRETE_GAUSSIAN

            )
        elif expr.mechanism in (CountMechanism.LAPLACE, CountDistinctMechanism.LAPLACE):
            core_mechanism = NoiseMechanism.GEOMETRIC
        elif expr.mechanism in (CountMechanism.GAUSSIAN, CountDistinctMechanism.GAUSSIAN):
            core_mechanism = NoiseMechanism.DISCRETE_GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize the mechanism name {expr.mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE, and GAUSSIAN."
            )
        return replace(expr, core_mechanism=core_mechanism)

    if isinstance(expr, (
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
    )):
        # Distinguish between Laplace/Geometric or (Discrete) Gaussian.
        # Assume floating-point output column type at first
        if expr.mechanism in (
            SumMechanism.DEFAULT,
            AverageMechanism.DEFAULT,
            VarianceMechanism.DEFAULT,
            StdevMechanism.DEFAULT,
        ):
            core_mechanism = (
                NoiseMechanism.LAPLACE
                if isinstance(output_measure, (PureDP, ApproxDP))
                else NoiseMechanism.GAUSSIAN
            )
        elif expr.mechanism in (
            SumMechanism.LAPLACE,
            AverageMechanism.LAPLACE,
            VarianceMechanism.LAPLACE,
            StdevMechanism.LAPLACE,
        ):
            core_mechanism = NoiseMechanism.LAPLACE
        elif expr.mechanism in (
            SumMechanism.GAUSSIAN,
            AverageMechanism.GAUSSIAN,
            VarianceMechanism.GAUSSIAN,
            StdevMechanism.GAUSSIAN,
        ):
            core_mechanism = NoiseMechanism.GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize requested mechanism {expr.mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE,  and GAUSSIAN."
            )

        # If the output column type is integer, use integer noise distributions instead
        catalog = expr.compilation_info.catalog
        schema = expr.child.accept(OutputSchemaVisitor(catalog))
        measure_column_type = schema[expr.measure_column].column_type
        if measure_column_type == ColumnType.INTEGER:
            core_mechanism = (
                NoiseMechanism.GEOMETRIC
                if core_mechanism == NoiseMechanism.LAPLACE
                else NoiseMechanism.DISCRETE_GAUSSIAN
            )

        return replace(expr, core_mechanism=core_mechanism)

    # All other aggregations don't use Core's NoiseMechanism, so they stay unchanged.
    return expr


def rewrite(info: CompilationInfo, expr: QueryExpr) -> QueryExpr:
    """Rewrites the given QueryExpr into a QueryExpr that can be compiled."""
    rewrite_rules = [
        add_compilation_info(info),
        select_noise_mechanism,
    ]
    for rule in rewrite_rules:
        expr = rule(expr)
    return expr
