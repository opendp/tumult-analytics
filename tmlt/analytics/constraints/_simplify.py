"""Tools for simplifying constraints."""

from typing import List

from ._base import Constraint
from ._truncation import simplify_truncation_constraints


def simplify_constraints(constraints: List[Constraint]) -> List[Constraint]:
    """Remove redundant constraints from a list of constraints.

    Given a list of the constraints on a table, produce a copy which simplifies
    it as much as possible by removing or combining constraints which provide
    overlapping information. The original list is not modified.
    """
    # This .copy() just guarantees that nothing that happens in this function
    # can modify the original list.
    constraints = simplify_truncation_constraints(constraints.copy())
    return constraints
