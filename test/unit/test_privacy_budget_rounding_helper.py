"""Tests for :mod:`tmlt.analytics._privacy_budget_rounding_helper`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
# pylint: disable=pointless-string-statement


from tmlt.core.utils.exact_number import ExactNumber
from typeguard import typechecked

from tmlt.analytics._privacy_budget_rounding_helper import (
    BUDGET_RELATIVE_TOLERANCE,
    get_adjusted_budget,
    requested_budget_is_slightly_higher_than_remaining,
)
from tmlt.analytics.privacy_budget import PureDPBudget

"""Tests for converting a numeric budget into symbolic representation."""
FUDGE_FACTOR = 1 / 1e9
INT_100 = 100
PURE_DP_99 = PureDPBudget(99)
PURE_DP_100 = PureDPBudget(100)
PURE_DP_101 = PureDPBudget(101)


def test_int_request():
    """Make sure int requests are handled properly.

    If the requested and remaining budgets are both reasonable integral values
    (i.e., nowhere near 10^9, which would be ridiculous for privacy parameters),
    we should never run into the tolerance threshold issue. This means the
    requested budget should be returned in all cases.
    """

    adjusted = get_adjusted_budget(PURE_DP_99, PURE_DP_100)
    assert adjusted == PURE_DP_99
    adjusted = get_adjusted_budget(PURE_DP_101, PURE_DP_100)
    assert adjusted == PURE_DP_101


def test_float_request():
    """Make sure float requests are handled properly.

    The only time the remaining budget should be returned is if the
    requested budget is slightly less than the remaining.
    """
    # We should never round up.
    adjusted = get_adjusted_budget(PureDPBudget(99.1), PURE_DP_100)
    assert adjusted == PureDPBudget(99.1)

    # Even if request is only slightly less, we still should not round up.
    requested = PureDPBudget(INT_100 - FUDGE_FACTOR)
    adjusted = get_adjusted_budget(requested, PURE_DP_100)
    assert adjusted == requested

    # Slightly greater than the remaining budget means we should round down.
    requested = PureDPBudget(INT_100 + FUDGE_FACTOR)
    adjusted = get_adjusted_budget(requested, PURE_DP_100)
    assert adjusted == PURE_DP_100

    # Up to the threshold, we should still round down.
    requested = PureDPBudget(INT_100 + (INT_100 * FUDGE_FACTOR))
    adjusted = get_adjusted_budget(requested, PURE_DP_100)
    assert adjusted == PURE_DP_100

    # But past the threshold, we assume this is not a rounding error, and we let
    # the requested budget proceed deeper into the system (to ultimately be caught
    # and inform the user they requested too much).
    requested = PureDPBudget(INT_100 + (INT_100 * FUDGE_FACTOR * 2))
    adjusted = get_adjusted_budget(requested, PURE_DP_100)
    assert adjusted == requested


"""Tests that our 'slightly higher' check works as intended."""


def test_requested_budget_much_higher():
    """Make sure a requested budget that is much higher than remaining fails.

    We should not round down in this case. The queryable's internal math will
    detect that we requested too much budget and raise an error.
    """
    requested = ExactNumber(100)
    remaining = ExactNumber(50)
    _compare_budgets(requested, remaining, False)


def test_requested_budget_much_lower():
    """Make sure a requested budget that is much lower than remaining fails.

    We should not consume all remaining budget when the request is way lower.
    """
    requested = ExactNumber(50)
    remaining = ExactNumber(100)
    _compare_budgets(requested, remaining, False)


def test_requested_budget_equals_remaining():
    """No need to perform any rounding if request and response are equal."""
    requested = ExactNumber(50)
    remaining = ExactNumber(50)
    _compare_budgets(requested, remaining, False)


def test_requested_budget_slightly_lower():
    """Make sure a requested budget that is slightly lower than remaining fails.

    We should never round UP to consume remaining budget, as this would be a
    privacy violation.
    """
    remaining = ExactNumber(10)
    fudge_factor = ExactNumber(1 / BUDGET_RELATIVE_TOLERANCE)
    requested = remaining + ((remaining + 1) * fudge_factor)
    _compare_budgets(requested, remaining, False)


def test_requested_budget_slightly_higher():
    """Makre sure a requested budget that is slightly higher than remaining works.

    We should successfully round DOWN To consume remaining budget.
    """
    remaining = ExactNumber(10)
    fudge_factor = ExactNumber(1 / BUDGET_RELATIVE_TOLERANCE)
    # Just below the tolerance threshold.
    requested = remaining + ((remaining - 1) * fudge_factor)
    _compare_budgets(requested, remaining, True)
    # Exactly equal to the tolerance threshold
    requested = remaining + (remaining * fudge_factor)
    _compare_budgets(requested, remaining, True)


def test_infinite_request_and_remaining():
    """Confirm that the comparison of infinite budgets works correctly."""
    _compare_budgets(ExactNumber(float("inf")), ExactNumber(float("inf")), False)


@typechecked
def _compare_budgets(
    requested: ExactNumber, remaining: ExactNumber, expected_comparison: bool
):
    """Compare 2 budgets and check against an expected result.

    Args:
        requested: The requested budget.
        remaining: The remaining budget.
        expected_comparison: True if requested should be higher than remaining.
    """
    comparison = requested_budget_is_slightly_higher_than_remaining(
        requested_budget=requested, remaining_budget=remaining
    )
    assert comparison == expected_comparison
