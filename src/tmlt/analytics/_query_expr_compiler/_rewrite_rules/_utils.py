"""Common utilities for rewrite rules."""

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from functools import wraps
from typing import Callable, Union

from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr import (
    GroupByCount,
    JoinPrivate,
    PrivateSource,
    QueryExpr,
    SingleChildQueryExpr,
    SuppressAggregates,
)


@dataclass(frozen=True)
class CompilationInfo:
    """Contextual information used by rewrite rules during compilation."""

    output_measure: Union[PureDP, ApproxDP, RhoZCDP]
    """The output measure used by this query."""

    catalog: Catalog
    """The Catalog of the Session this query is executed on."""


def depth_first(
    func: Callable[[QueryExpr], QueryExpr],
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
            left = wrapped(expr.left_child)
            right = wrapped(expr.right_child)
            return func(replace(expr, left_child=left, right_child=right))
        else:
            raise AnalyticsInternalError(
                f"Unrecognized QueryExpr subtype {type(expr).__qualname__}."
            )

    return wrapped
