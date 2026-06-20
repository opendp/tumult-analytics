"""Rules for rewriting QueryExprs.

These are executed at the beginning of the query compilation process, and each rewrite
rule corresponds to one compilation step. The rewritten QueryExpr is then visited by the
MeasurementVisitor to be converted to a Core measurement.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Union

from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr import QueryExpr
from tmlt.analytics._query_expr_compiler._rewrite_rules._select_noise import (
    select_noise_mechanism,
)
from tmlt.analytics._query_expr_compiler._rewrite_rules._special_values import (
    add_special_value_handling,
)
from tmlt.analytics._query_expr_compiler._rewrite_rules._utils import CompilationInfo


def rewrite(
    expr: QueryExpr,
    *,
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    catalog: Catalog,
) -> QueryExpr:
    """Rewrites the given QueryExpr into a QueryExpr that can be compiled."""
    info = CompilationInfo(output_measure=output_measure, catalog=catalog)
    rewrite_rules = [
        add_special_value_handling(info),
        select_noise_mechanism(info),
    ]
    for rule in rewrite_rules:
        expr = rule(expr)
    return expr
