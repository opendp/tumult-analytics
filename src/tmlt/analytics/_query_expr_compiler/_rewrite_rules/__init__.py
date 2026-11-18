r"""Rules for rewriting QueryExprs.

This module defines a collection of rules for transforming ``QueryExpr``\ s,
each of which maintains the meaning of the expression's result while potentially
modifying how that result is computed.
"""

# SPDX-License-Identifier: Apache-2.0

from tmlt.analytics._query_expr import QueryExpr

from ._base import CompilationInfo
from ._noise import select_noise_mechanism
from ._special_values import add_special_value_handling


def rewrite(info: CompilationInfo, expr: QueryExpr) -> QueryExpr:
    """Rewrites the given QueryExpr into a QueryExpr that can be compiled."""
    rewrite_rules = [
        add_special_value_handling(info),
        select_noise_mechanism(info),
    ]
    for rule in rewrite_rules:
        expr = rule(expr)
    return expr
