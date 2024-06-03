"""Tests for :mod:`tmlt.analytics.privacy_budget`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024
# pylint: disable=pointless-string-statement

from typing import List

import pytest
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)

"""Tests for :class:`tmlt.analytics.privacy_budget.PureDPBudget`."""


def test_constructor_success_nonnegative_int():
    """Tests that construction succeeds with nonnegative ints."""
    budget = PureDPBudget(2)
    assert budget.epsilon == 2
    budget = PureDPBudget(0)
    assert budget.epsilon == 0


def test_constructor_success_nonnegative_float():
    """Tests that construction succeeds with nonnegative floats."""
    budget = PureDPBudget(2.5)
    assert budget.epsilon == 2.5
    budget = PureDPBudget(0.0)
    assert budget.epsilon == 0.0


def test_constructor_fail_negative_int():
    """Tests that construction fails with a negative int."""
    with pytest.raises(ValueError, match="Epsilon must be non-negative."):
        PureDPBudget(-1)


def test_constructor_fail_negative_float():
    """Tests that construction fails with a negative float."""
    with pytest.raises(ValueError, match="Epsilon must be non-negative."):
        PureDPBudget(-1.5)


def test_constructor_fail_bad_epsilon_type():
    """Tests that construction fails with epsilon that is not an int or float."""
    with pytest.raises(TypeError):
        PureDPBudget("1.5")  # type: ignore


def test_constructor_fail_nan():
    """Tests that construction fails with epsilon that is a NaN."""
    with pytest.raises(ValueError, match="Epsilon cannot be a NaN."):
        PureDPBudget(float("nan"))


"""Tests for :class:`tmlt.analytics.privacy_budget.ApproxDPBudget`."""


def test_constructor_success_nonnegative_int_ApproxDP():
    """Tests that construction succeeds with nonnegative ints."""
    budget = ApproxDPBudget(2, 0.1)
    assert budget.epsilon == 2
    assert budget.delta == 0.1

    budget = ApproxDPBudget(0, 0)
    assert budget.epsilon == 0
    assert budget.delta == 0


def test_constructor_success_nonnegative_int_and_float_ApproxDP():
    """Tests that construction succeeds with mix of nonnegative ints and floats."""
    budget = ApproxDPBudget(0.5, 0)
    assert budget.epsilon == 0.5
    assert budget.delta == 0

    budget = ApproxDPBudget(2, 0.5)
    assert budget.epsilon == 2
    assert budget.delta == 0.5


def test_constructor_success_nonnegative_float_ApproxDP():
    """Tests that construction succeeds with nonnegative floats."""
    budget = ApproxDPBudget(2.5, 0.5)
    assert budget.epsilon == 2.5
    assert budget.delta == 0.5


def test_constructor_fail_epsilon_negative_int_ApproxDP():
    """Tests that construction fails with a negative int epsilon."""
    with pytest.raises(ValueError, match="Epsilon must be non-negative."):
        ApproxDPBudget(-1, 0.5)


def test_constructor_fail_delta_negative_int_ApproxDP():
    """Tests that construction fails with a negative int delta."""
    with pytest.raises(ValueError, match="Delta must be between 0 and 1"):
        ApproxDPBudget(0.5, -1)


def test_constructor_fail_epsilon_negative_float_ApproxDP():
    """Tests that construction fails with a negative float epsilon."""
    with pytest.raises(ValueError, match="Epsilon must be non-negative."):
        ApproxDPBudget(-1.5, 0.5)


def test_constructor_fail_delta_negative_float_ApproxDP():
    """Tests that construction fails with a negative float delta."""
    with pytest.raises(ValueError, match="Delta must be between 0 and 1"):
        ApproxDPBudget(0.5, -1.5)


def test_constructor_fail_bad_epsilon_type_ApproxDP():
    """Tests that construction fails with epsilon that is not an int or float."""
    with pytest.raises(TypeError):
        ApproxDPBudget("1.5", 0.5)  # type: ignore


def test_constructor_fail_bad_delta_type_ApproxDP():
    """Tests that construction fails with delta that is not an int or float."""
    with pytest.raises(TypeError):
        ApproxDPBudget(0.5, "1.5")  # type: ignore


def test_constructor_fail_epsilon_nan_ApproxDP():
    """Tests that construction fails with epsilon that is a NaN."""
    with pytest.raises(ValueError, match="Epsilon cannot be a NaN."):
        ApproxDPBudget(float("nan"), 0.5)


def test_constructor_fail_delta_nan_ApproxDP():
    """Tests that construction fails with delta that is a NaN."""
    with pytest.raises(ValueError, match="Delta cannot be a NaN."):
        ApproxDPBudget(0.5, float("nan"))


"""Tests for :class:`tmlt.analytics.privacy_budget.RhoZCDPBudget`."""


def test_constructor_success_nonnegative_int_ZCDP():
    """Tests that construction succeeds with nonnegative ints."""
    budget = RhoZCDPBudget(2)
    assert budget.rho == 2
    budget = RhoZCDPBudget(0)
    assert budget.rho == 0


def test_constructor_success_nonnegative_float_ZCDP():
    """Tests that construction succeeds with nonnegative floats."""
    budget = RhoZCDPBudget(2.5)
    assert budget.rho == 2.5
    budget = RhoZCDPBudget(0.0)
    assert budget.rho == 0.0


def test_constructor_fail_negative_int_ZCDP():
    """Tests that construction fails with negative ints."""
    with pytest.raises(ValueError, match="Rho must be non-negative."):
        RhoZCDPBudget(-1)


def test_constructor_fail_negative_float_ZCDP():
    """Tests that construction fails with negative floats."""
    with pytest.raises(ValueError, match="Rho must be non-negative."):
        RhoZCDPBudget(-1.5)


def test_constructor_fail_bad_rho_type_ZCDP():
    """Tests that construction fails with rho that is not an int or float."""
    with pytest.raises(TypeError):
        RhoZCDPBudget("1.5")  # type: ignore


def test_constructor_fail_nan_ZCDP():
    """Tests that construction fails with rho that is a NaN."""
    with pytest.raises(ValueError, match="Rho cannot be a NaN."):
        RhoZCDPBudget(float("nan"))


@pytest.mark.parametrize(
    "budget,inf_bool",
    [
        # Handles all ApproxDP Inf Options
        (ApproxDPBudget(float("inf"), 1), True),
        (ApproxDPBudget(1, 1), True),
        (ApproxDPBudget(float("inf"), 0), True),
        # Handles all ApproxDP Non-Inf Options
        (ApproxDPBudget(1, 0.1), False),
        (ApproxDPBudget(1, 0), False),
        # Handles all RhoZCDP Options
        (RhoZCDPBudget(float("inf")), True),
        (RhoZCDPBudget(1), False),
        # Handles all PureDP Options
        (PureDPBudget(float("inf")), True),
        (PureDPBudget(1), False),
    ],
)
def test_is_infinite(budget: PrivacyBudget, inf_bool: bool):
    """Tests the is_infinite function for each budget."""
    assert budget.is_infinite == inf_bool


@pytest.mark.parametrize(
    "budgets",
    [
        # Tests with normal budget values
        [PureDPBudget(1)],
        [ApproxDPBudget(0.5, 1e-10)],
        [RhoZCDPBudget(1)],
        # Tests with infinite budget values
        [PureDPBudget(float("inf"))],
        [ApproxDPBudget(float("inf"), 1)],
        [
            RhoZCDPBudget(float("inf")),
        ],
        # Tests that no budgets are confused with each other.
        [PureDPBudget(1), ApproxDPBudget(1, 1e-10), RhoZCDPBudget(1)],
        [
            PureDPBudget(float("inf")),
            ApproxDPBudget(float("inf"), 1),
            RhoZCDPBudget(float("inf")),
        ],
        [PureDPBudget(1), PureDPBudget(2), PureDPBudget(3)],
        [ApproxDPBudget(1, 1e-10), ApproxDPBudget(2, 1e-10), ApproxDPBudget(3, 1e-10)],
        [ApproxDPBudget(1, 1e-10), ApproxDPBudget(1, 1e-11), ApproxDPBudget(1, 1e-12)],
        [RhoZCDPBudget(1), RhoZCDPBudget(2), RhoZCDPBudget(3)],
    ],
)
def test_hashing(budgets: List[PrivacyBudget]):
    """Tests that each privacy budget is hashable."""
    # Add each budget to a dictionary
    budgets_dict = {budget: budget.value for budget in budgets}

    # Check that the budgets are correctly mapped.
    for budget in budgets:
        assert budgets_dict[budget] == budget.value


# pylint: disable=protected-access
def test_PureDPBudget_immutability():
    """Tests that the PureDPBudget is immutable."""

    with pytest.raises(AttributeError):
        PureDPBudget(1)._epsilon = 2  # type: ignore


def test_ApproxDPBudget_immutability():
    """Tests that the ApproxDPBudget is immutable."""

    with pytest.raises(AttributeError):
        ApproxDPBudget(1, 0.1)._epsilon = 2  # type: ignore
    with pytest.raises(AttributeError):
        ApproxDPBudget(1, 0.1)._delta = 0.2  # type: ignore


def test_RhoZCDPBudget_immutability():
    """Tests that the RhoZCDPBudget is immutable."""

    with pytest.raises(AttributeError):
        RhoZCDPBudget(1)._rho = 2  # type: ignore


# pylint: enable=protected-access


@pytest.mark.parametrize(
    "budget_a, budget_b, equal",
    [
        # PureDPBudget Tests
        (PureDPBudget(1), PureDPBudget(1), True),
        (PureDPBudget(1), PureDPBudget(2), False),
        (PureDPBudget(1), ApproxDPBudget(1, 1e-10), False),
        (PureDPBudget(1), RhoZCDPBudget(1), False),
        (PureDPBudget(1), ApproxDPBudget(1, 0), False),
        # ApproxDPBudget Tests
        (ApproxDPBudget(1, 1e-10), ApproxDPBudget(1, 1e-10), True),
        (ApproxDPBudget(1, 1e-10), ApproxDPBudget(2, 1e-10), False),
        (ApproxDPBudget(1, 1e-10), ApproxDPBudget(1, 1e-11), False),
        (ApproxDPBudget(1, 1e-10), PureDPBudget(1), False),
        (ApproxDPBudget(1, 1e-10), RhoZCDPBudget(1), False),
        (ApproxDPBudget(1, 0), PureDPBudget(1), False),
        # RhoZCDPBudget Tests
        (RhoZCDPBudget(1), RhoZCDPBudget(1), True),
        (RhoZCDPBudget(1), RhoZCDPBudget(2), False),
        (RhoZCDPBudget(1), PureDPBudget(1), False),
        (RhoZCDPBudget(1), ApproxDPBudget(1, 1e-10), False),
        # Tests with infinite budgets
        (PureDPBudget(float("inf")), PureDPBudget(float("inf")), True),
        (PureDPBudget(1), PureDPBudget(float("inf")), False),
        (PureDPBudget(float("inf")), PureDPBudget(1), False),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(float("inf"), 1), True),
        (ApproxDPBudget(1, 1), ApproxDPBudget(float("inf"), 1), True),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(1, 1), True),
        (ApproxDPBudget(0, 1), ApproxDPBudget(float("inf"), 1), True),
        (ApproxDPBudget(float("inf"), 1), ApproxDPBudget(0, 1), True),
        (RhoZCDPBudget(float("inf")), RhoZCDPBudget(float("inf")), True),
        (RhoZCDPBudget(1), RhoZCDPBudget(float("inf")), False),
        (RhoZCDPBudget(float("inf")), RhoZCDPBudget(1), False),
        # Tests with different input types.
        (PureDPBudget(1), PureDPBudget(ExactNumber("1.0")), True),
        (PureDPBudget(1), PureDPBudget(1.0), True),
        (PureDPBudget(1), PureDPBudget(1.1), False),
        (
            ApproxDPBudget(1, 1e-10),
            ApproxDPBudget(
                ExactNumber("1.0"), ExactNumber.from_float(1e-10, round_up=False)
            ),
            True,
        ),
        (
            ApproxDPBudget(
                ExactNumber("1.0"), ExactNumber.from_float(1e-10, round_up=False)
            ),
            ApproxDPBudget(1, 1e-10),
            True,
        ),
        (ApproxDPBudget(1, 1e-10), ApproxDPBudget(1.0, 1e-11), False),
        (ApproxDPBudget(1.1, 1e-10), ApproxDPBudget(1.0, 1e-10), False),
        (RhoZCDPBudget(1), RhoZCDPBudget(ExactNumber("1.0")), True),
        (RhoZCDPBudget(1), RhoZCDPBudget(1.0), True),
        (RhoZCDPBudget(1), RhoZCDPBudget(1.1), False),
    ],
)
def test_budget_equality(budget_a: PrivacyBudget, budget_b: PrivacyBudget, equal: bool):
    """Tests that two budgets are equal if they have the same value."""
    assert (budget_a == budget_b) == equal
