"""Defines a visitor for creating a transformation from a query expression."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from tmlt.analytics._query_expr_compiler._base_transformation_visitor import (
    BaseTransformationVisitor,
)
from tmlt.analytics.query_expr import FlatMap as FlatMapExpr


class TransformationVisitor(BaseTransformationVisitor):
    """A visitor to create a transformation from a DP query expression."""

    def visit_flat_map(self, expr: FlatMapExpr) -> BaseTransformationVisitor.Output:
        """Create a transformation from a FlatMap query expression."""
        return self._visit_flat_map(expr, expr.max_rows)
