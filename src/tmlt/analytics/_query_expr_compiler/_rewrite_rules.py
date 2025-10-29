"""Rules for rewriting QueryExprs.

These are executed at the beginning of the query compilation process, and each rewrite
rule corresponds to one compilation step. The rewritten QueryExpr is then visited by the
MeasurementVisitor to be converted to a Core measurement.
"""

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from functools import wraps
from typing import Callable, Union

from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
    DropInfinity,
    DropNullAndNan,
    FrozenDict,
    GetBounds,
    GroupByBoundedAverage,
    GroupByBoundedStdev,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    JoinPrivate,
    PrivateSource,
    QueryExpr,
    ReplaceInfinity,
    SingleChildQueryExpr,
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)
from tmlt.analytics._schema import ColumnType


@dataclass(frozen=True)
class CompilationInfo:
    """Contextual information used by rewrite rules during compilation."""

    output_measure: Union[PureDP, ApproxDP, RhoZCDP]
    """The output measure used by this query."""

    catalog: Catalog
    """The Catalog of the Session this query is executed on."""


def depth_first(
    func: Callable[[QueryExpr], QueryExpr]
) -> Callable[[QueryExpr], QueryExpr]:
    """Recursively applies the given method to a QueryExpr, depth-first."""

    @wraps(func)
    def wrapped(expr: QueryExpr) -> QueryExpr:
        if isinstance(expr, PrivateSource):
            return func(expr)
        if isinstance(expr, SuppressAggregates):
            child = wrapped(expr.child)
            if not isinstance(child, GroupByCount):
                raise AnalyticsInternalError(
                    "Rewriting rule should have produced a QueryExpr of type "
                    "GroupByCount as a child for SuppressAggregates, got type "
                    f"{type(child).__qualname__} instead."
                )
            return func(replace(expr, child=child))
        if isinstance(expr, SingleChildQueryExpr):
            child = wrapped(expr.child)
            return func(replace(expr, child=child))
        if isinstance(expr, JoinPrivate):
            left = wrapped(expr.child)
            right = wrapped(expr.right_operand_expr)
            return func(replace(expr, child=left, right_operand_expr=right))
        else:
            raise AnalyticsInternalError(
                f"Unrecognized QueryExpr subtype {type(expr).__qualname__}."
            )

    return wrapped


def select_noise_mechanism(info: CompilationInfo) -> Callable[[QueryExpr], QueryExpr]:
    """Changes the default noise type into a concrete noise type for aggregations."""

    def select_noise_for_count(
        info: CompilationInfo, expr: Union[GroupByCount, GroupByCountDistinct]
    ) -> QueryExpr:
        mechanism = expr.mechanism
        if mechanism in (CountMechanism.DEFAULT, CountDistinctMechanism.DEFAULT):
            core_mechanism = (
                NoiseMechanism.GEOMETRIC
                if isinstance(info.output_measure, (PureDP, ApproxDP))
                else NoiseMechanism.DISCRETE_GAUSSIAN
            )
        elif mechanism in (CountMechanism.LAPLACE, CountDistinctMechanism.LAPLACE):
            core_mechanism = NoiseMechanism.GEOMETRIC
        elif mechanism in (
            CountMechanism.GAUSSIAN,
            CountDistinctMechanism.GAUSSIAN,
        ):
            if not isinstance(info.output_measure, RhoZCDP):
                raise ValueError(
                    "Gaussian noise is only supported when using a RhoZCDP budget. "
                    "Use Laplace noise instead, or switch to RhoZCDP."
                )
            core_mechanism = NoiseMechanism.DISCRETE_GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize the mechanism name {mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE, and GAUSSIAN."
            )
        return replace(expr, core_mechanism=core_mechanism)

    def select_noise_for_non_count(
        info: CompilationInfo,
        expr: Union[
            GroupByBoundedAverage,
            GroupByBoundedStdev,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ],
    ) -> QueryExpr:
        mechanism = expr.mechanism
        # Distinguish between Laplace/Geometric or (Discrete) Gaussian.
        # Assume floating-point output column type at first
        if mechanism in (
            SumMechanism.DEFAULT,
            AverageMechanism.DEFAULT,
            VarianceMechanism.DEFAULT,
            StdevMechanism.DEFAULT,
        ):
            core_mechanism = (
                NoiseMechanism.LAPLACE
                if isinstance(info.output_measure, (PureDP, ApproxDP))
                else NoiseMechanism.GAUSSIAN
            )
        elif mechanism in (
            SumMechanism.LAPLACE,
            AverageMechanism.LAPLACE,
            VarianceMechanism.LAPLACE,
            StdevMechanism.LAPLACE,
        ):
            core_mechanism = NoiseMechanism.LAPLACE
        elif mechanism in (
            SumMechanism.GAUSSIAN,
            AverageMechanism.GAUSSIAN,
            VarianceMechanism.GAUSSIAN,
            StdevMechanism.GAUSSIAN,
        ):
            if not isinstance(info.output_measure, RhoZCDP):
                raise ValueError(
                    "Gaussian noise is only supported when using a RhoZCDP budget. "
                    "Use Laplace noise instead, or switch to RhoZCDP."
                )
            core_mechanism = NoiseMechanism.GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize requested mechanism {mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE, and GAUSSIAN."
            )

        # If the measure column type is integer, use integer noise distributions
        schema = expr.child.schema(info.catalog)
        measure_column_type = schema[expr.measure_column].column_type
        if measure_column_type == ColumnType.INTEGER:
            core_mechanism = (
                NoiseMechanism.GEOMETRIC
                if core_mechanism == NoiseMechanism.LAPLACE
                else NoiseMechanism.DISCRETE_GAUSSIAN
            )

        return replace(expr, core_mechanism=core_mechanism)

    @depth_first
    def select_noise(expr: QueryExpr) -> QueryExpr:
        if isinstance(expr, (GroupByCount, GroupByCountDistinct)):
            return select_noise_for_count(info, expr)
        if isinstance(
            expr,
            (
                GroupByBoundedAverage,
                GroupByBoundedStdev,
                GroupByBoundedSum,
                GroupByBoundedVariance,
            ),
        ):
            return select_noise_for_non_count(info, expr)
        return expr

    return select_noise


def add_special_value_handling(
    info: CompilationInfo,
) -> Callable[[QueryExpr], QueryExpr]:
    """Rewrites the query to handle nulls, NaNs and infinite values.

    If the measure column allows nulls or NaNs, the rewritten query will drop those
    values. If the measure column allows infinite values, the new query will replace
    those values with the clamping bounds specified in the query, or drop these values
    for :meth:`~tmlt.analytics.QueryBuilder.get_bounds`.
    """

    @depth_first
    def handle_special_values(expr: QueryExpr) -> QueryExpr:
        if not isinstance(
            expr,
            (
                GroupByBoundedAverage,
                GroupByBoundedStdev,
                GroupByBoundedSum,
                GroupByBoundedVariance,
                GroupByQuantile,
                GetBounds,
            ),
        ):
            return expr
        schema = expr.child.schema(info.catalog)
        measure_desc = schema[expr.measure_column]
        # Remove nulls/NaN if necessary
        if measure_desc.allow_null or (
            measure_desc.column_type == ColumnType.DECIMAL and measure_desc.allow_nan
        ):
            expr = replace(
                expr,
                child=DropNullAndNan(child=expr.child, columns=(expr.measure_column,)),
            )
        # Remove infinities if necessary
        if measure_desc.column_type == ColumnType.DECIMAL and measure_desc.allow_inf:
            if isinstance(expr, GetBounds):
                return replace(
                    expr,
                    child=DropInfinity(
                        child=expr.child, columns=(expr.measure_column,)
                    ),
                )
            return replace(
                expr,
                child=ReplaceInfinity(
                    child=expr.child,
                    replace_with=FrozenDict.from_dict(
                        {expr.measure_column: (expr.low, expr.high)}
                    ),
                ),
            )
        return expr

    return handle_special_values


def rewrite(info: CompilationInfo, expr: QueryExpr) -> QueryExpr:
    """Rewrites the given QueryExpr into a QueryExpr that can be compiled."""
    rewrite_rules = [
        add_special_value_handling(info),
        select_noise_mechanism(info),
    ]
    for rule in rewrite_rules:
        expr = rule(expr)
    return expr
