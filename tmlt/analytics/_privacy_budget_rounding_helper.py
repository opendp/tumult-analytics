"""Helper functions for dealing with budget floating point imprecision."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Union

import sympy as sp
from typeguard import typechecked

from tmlt.core.utils.exact_number import ExactNumber

BUDGET_RELATIVE_TOLERANCE: sp.Expr = sp.Pow(10, 9)


@typechecked
def get_adjusted_budget(
    requested_budget: Union[int, float], remaining_budget: ExactNumber
) -> ExactNumber:
    """Converts a requested int or float budget into an adjusted budget.

    If the requested budget is "slightly larger" than the remaining budget, as
    determined by some threshold, then we round down and consume all remaining
    budget. The goal is to accommodate some degree of floating point imprecision by
    erring on the side of providing a slightly stronger privacy guarantee
    rather than declining the request altogether.

    Args:
        requested_budget: The numeric value of the requested budget.
        remaining_budget: How much budget we have left.
    """
    requested_budget_exact: ExactNumber
    if isinstance(requested_budget, int):
        requested_budget_exact = ExactNumber(requested_budget)
    else:
        # Round down to err on the side of better privacy.
        requested_budget_exact = ExactNumber.from_float(
            value=requested_budget, round_up=False
        )
    if requested_budget_is_slightly_higher_than_remaining(
        requested_budget_exact, remaining_budget
    ):
        return remaining_budget
    else:
        return requested_budget_exact


def requested_budget_is_slightly_higher_than_remaining(
    requested_budget: ExactNumber, remaining_budget: ExactNumber
) -> bool:
    """Returns True if requested budget is slightly larger than remaining.

    This check uses a relative tolerance, i.e., it determines if the requested
    budget is within X% of the remaining budget.

    Args:
        requested_budget: Exact representation of requested budget.
        remaining_budget: Exact representation of how much budget we have left.
    """
    if not remaining_budget.is_finite:
        return False

    diff = remaining_budget - requested_budget
    if diff >= 0:
        return False
    return abs(diff) <= remaining_budget / BUDGET_RELATIVE_TOLERANCE
