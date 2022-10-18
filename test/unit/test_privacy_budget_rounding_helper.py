"""Tests for :mod:`tmlt.analytics._privacy_budget_rounding_helper`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
# pylint: disable=pointless-string-statement


from typeguard import typechecked

from tmlt.analytics._privacy_budget_rounding_helper import (
    BUDGET_RELATIVE_TOLERANCE,
    get_adjusted_budget,
    requested_budget_is_slightly_higher_than_remaining,
)
from tmlt.core.utils.exact_number import ExactNumber

"""Tests for converting a numeric budget into symbolic representation."""
REMAINING_INT = 100
REMAINING_EXACT = ExactNumber(100)


def test_int_request():
    """Make sure int requests are handled properly.

    If the requested and remaining budgets are both reasonable integral values
    (i.e., nowhere near 10^9, which would be ridiculous for privacy parameters),
    we should never run into the tolerance threshold issue. This means the
    requested budget should be returned in all cases.
    """
    adjusted = get_adjusted_budget(99, REMAINING_EXACT)
    assert adjusted == 99
    adjusted = get_adjusted_budget(101, REMAINING_EXACT)
    assert adjusted == 101


def test_float_request():
    """Make sure float requests are handled properly.

    The only time the remaining budget should be returned is if the
    requested budget is slightly less than the remaining.
    """
    # We should never round up.
    adjusted = get_adjusted_budget(99.1, REMAINING_EXACT)
    assert adjusted == ExactNumber.from_float(99.1, False)
    fudge_factor = 1 / 1e9
    # Even if request is only slightly less, we still should not round up.
    requested = REMAINING_INT - fudge_factor
    adjusted = get_adjusted_budget(requested, REMAINING_EXACT)
    assert adjusted == ExactNumber.from_float(requested, False)
    # Slightly greater than the remaining budget means we should round down.
    requested = REMAINING_INT + fudge_factor
    adjusted = get_adjusted_budget(requested, REMAINING_EXACT)
    assert adjusted == REMAINING_EXACT
    # Up to the threshold, we should still round down.
    requested = REMAINING_INT + (REMAINING_INT * fudge_factor)
    adjusted = get_adjusted_budget(requested, REMAINING_EXACT)
    assert adjusted == REMAINING_EXACT
    # But past the threshold, we assume this is not a rounding error, and we let
    # the requested budget proceed deeper into the system (to ultimately be caught
    # and inform the user they requested too much).
    requested = REMAINING_INT + (REMAINING_INT * fudge_factor * 2)
    adjusted = get_adjusted_budget(requested, REMAINING_EXACT)
    assert adjusted == ExactNumber.from_float(requested, False)


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
