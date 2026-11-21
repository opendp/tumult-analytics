"""Rewrite rules for resolving automatic column selections."""

# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from tmlt.analytics._query_expr import AutoColumnQueryExpr, QueryExpr

from ._base import CompilationInfo, depth_first


def resolve_columns(info: CompilationInfo) -> Callable[[QueryExpr], QueryExpr]:
    """Make column selections explicit, resolving children before their parents."""

    @depth_first
    def resolve(expr: QueryExpr) -> QueryExpr:
        if isinstance(expr, AutoColumnQueryExpr):
            return expr.resolve_columns(info.catalog)
        return expr

    return resolve
