"""Rewrite rules for handling special values (nulls, NaNs, infinities)."""

# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Callable

from tmlt.analytics._query_expr import (
    DropInfinity,
    DropNullAndNan,
    FrozenDict,
    GetBounds,
    GroupByBoundedAverage,
    GroupByBoundedStdev,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByQuantile,
    QueryExpr,
    ReplaceInfinity,
)
from tmlt.analytics._query_expr_compiler._rewrite_rules._utils import (
    CompilationInfo,
    depth_first,
)
from tmlt.analytics._schema import ColumnType


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
