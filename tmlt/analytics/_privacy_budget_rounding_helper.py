"""Helper functions for dealing with budget floating point imprecision."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import sympy as sp
from tmlt.core.utils.exact_number import ExactNumber
from typeguard import typechecked

from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)

BUDGET_RELATIVE_TOLERANCE: sp.Expr = sp.Pow(10, 9)


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


@typechecked
def get_adjusted_budget_number(
    requested_budget: ExactNumber, remaining_budget: ExactNumber
) -> ExactNumber:
    """Converts a requested int or float budget into an adjusted budget.

    If the requested budget is "slightly larger" than the remaining budget, as
    determined by some threshold, then we round down and consume all remaining
    budget. The goal is to accommodate some degree of floating point imprecision by
    erring on the side of providing a slightly stronger privacy guarantee
    rather than declining the request altogether.

    Args:
        requested_budget: The numeric value of the requested budget.
        remaining_budget: The numeric value of how much budget we have left.
    """
    if requested_budget_is_slightly_higher_than_remaining(
        requested_budget, remaining_budget
    ):
        return remaining_budget

    return requested_budget


@typechecked
def get_adjusted_budget(
    requested_privacy_budget: PrivacyBudget, remaining_privacy_budget: PrivacyBudget
) -> PrivacyBudget:
    """Converts a requested privacy budget into an adjusted privacy budget.

    For each term in the privacy budget, calls get_adjusted_budget_number to adjust
    the requested budget slightly if it's close enough to the remaining budget.

    Args:
        requested_privacy_budget: The requested privacy budget.
        remaining_privacy_budget: How much privacy budget we have left.
    """
    # pylint: disable=protected-access
    if isinstance(requested_privacy_budget, PureDPBudget) and isinstance(
        remaining_privacy_budget, PureDPBudget
    ):
        adjusted_epsilon = get_adjusted_budget_number(
            requested_privacy_budget._epsilon, remaining_privacy_budget._epsilon
        )
        return PureDPBudget(adjusted_epsilon)

    elif isinstance(requested_privacy_budget, ApproxDPBudget) and isinstance(
        remaining_privacy_budget, ApproxDPBudget
    ):
        adjusted_epsilon = get_adjusted_budget_number(
            requested_privacy_budget._epsilon, remaining_privacy_budget._epsilon
        )
        adjusted_delta = get_adjusted_budget_number(
            requested_privacy_budget._delta, remaining_privacy_budget._delta
        )
        return ApproxDPBudget(adjusted_epsilon, adjusted_delta)

    elif isinstance(requested_privacy_budget, RhoZCDPBudget) and isinstance(
        remaining_privacy_budget, RhoZCDPBudget
    ):
        adjusted_rho = get_adjusted_budget_number(
            requested_privacy_budget._rho, remaining_privacy_budget._rho
        )
        return RhoZCDPBudget(adjusted_rho)
    # pylint: enable=protected-access
    else:
        raise ValueError(
            "Unable to compute a privacy budget with the requested budget "
            f"of {requested_privacy_budget} and a remaining budget of "
            f"{remaining_privacy_budget}."
        )
