r"""Rules for rewriting QueryExprs.

This module defines a collection of rules for transforming ``QueryExpr``\ s,
each of which maintains the meaning of the expression's result while potentially
modifying how that result is computed.

Rewriting occurs in two stages:
* Column resolution: all expressions that automatically detect the columns they
  apply to, e.g. joins without explicitly-specified join columns, are replaced
  with equivalent expressions where these columns are explicit. In this stage,
  expressions' output schemas and constraints can only safely be determined if
  they (and thus their children) have already had their columns resolved.
* Implicit option selection, query restructuring, and optimization: the bulk of
  the rewrite rules, covering everything other than column resolution. At this
  point, schemas and constraints can be freely determined. These rewrite rules
  must not introduce any subclasses of AutoColumnQueryExpr back into the
  expression.
"""

# SPDX-License-Identifier: Apache-2.0

from tmlt.analytics._query_expr import QueryExpr

from ._base import CompilationInfo
from ._columns import resolve_columns
from ._noise import select_noise_mechanism
from ._special_values import add_special_value_handling


def rewrite(info: CompilationInfo, expr: QueryExpr) -> QueryExpr:
    """Rewrite and validate a QueryExpr so it can be compiled."""
    expr = resolve_columns(info)(expr)
    # Validate before the remaining rules assume that referenced columns exist.
    expr.schema(info.catalog)

    rewrite_rules = [
        add_special_value_handling(info),
        select_noise_mechanism(info),
    ]
    for rule in rewrite_rules:
        expr = rule(expr)
    return expr
