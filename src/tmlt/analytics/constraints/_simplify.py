"""Tools for simplifying constraints."""
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Iterable

from ._base import Constraint
from ._truncation import simplify_truncation_constraints


def simplify_constraints(constraints: Iterable[Constraint]) -> frozenset[Constraint]:
    """Remove redundant constraints from a set of constraints.

    Given a set of the constraints on a table, produce a copy which simplifies
    it as much as possible by removing or combining constraints which provide
    overlapping information.
    """
    return simplify_truncation_constraints(constraints)
