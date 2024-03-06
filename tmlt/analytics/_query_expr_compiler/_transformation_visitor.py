"""Defines a visitor for creating a transformation from a query expression."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from tmlt.analytics._query_expr_compiler._base_transformation_visitor import (
    BaseTransformationVisitor,
)


class TransformationVisitor(BaseTransformationVisitor):
    """A visitor to create a transformation from a DP query expression."""
