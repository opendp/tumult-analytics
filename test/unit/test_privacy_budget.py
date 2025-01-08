"""Tests for :mod:`tmlt.analytics.privacy_budget`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025
# pylint: disable=pointless-string-statement

import math
from typing import List, Type, Union

import pytest
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import Case, parametrize
from typeguard import TypeCheckError

from tmlt.analytics import ApproxDPBudget, PrivacyBudget, PureDPBudget, RhoZCDPBudget

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
    with pytest.raises(TypeCheckError):
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
    with pytest.raises(TypeCheckError):
        ApproxDPBudget("1.5", 0.5)  # type: ignore


def test_constructor_fail_bad_delta_type_ApproxDP():
    """Tests that construction fails with delta that is not an int or float."""
    with pytest.raises(TypeCheckError):
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
    with pytest.raises(TypeCheckError):
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
def test_hashing_dict_value(budgets: List[PrivacyBudget]):
    """Tests that each privacy budget is hashable."""
    # Add each budget to a dictionary
    budgets_dict = {budget: budget.value for budget in budgets}

    # Check that the budgets are correctly mapped.
    for budget in budgets:
        assert budgets_dict[budget] == budget.value


@pytest.mark.parametrize(
    "budgets, equal",
    [
        ([PureDPBudget(1), PureDPBudget(1)], True),
        ([PureDPBudget(1), RhoZCDPBudget(1)], False),
        ([PureDPBudget(float("inf")), RhoZCDPBudget(float("inf"))], False),
        ([RhoZCDPBudget(1), RhoZCDPBudget(10)], False),
        ([PureDPBudget(1), ApproxDPBudget(1, 1e-10)], False),
        ([ApproxDPBudget(1, 1e-10), ApproxDPBudget(1, 1e-10)], True),
        (
            [ApproxDPBudget(float("inf"), 1), ApproxDPBudget(float("inf"), 1)],
            True,
        ),
    ],
)
def test_budget_hashing(budgets: List[PrivacyBudget], equal: bool):
    """Tests that each privacy budget is hashable."""
    # Add each budget to a dictionary
    budget0_hash = hash(budgets[0])
    budget1_hash = hash(budgets[1])
    if equal:
        assert budget0_hash == budget1_hash
    else:
        assert budget0_hash != budget1_hash


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


@parametrize(
    Case("puredp_ints")(
        budget=PureDPBudget(1),
        divisor=2,
        expected=PureDPBudget(0.5),
    ),
    Case("puredp_floats")(
        budget=PureDPBudget(1.0),
        divisor=2.0,
        expected=PureDPBudget(0.5),
    ),
    Case("puredp_non_half")(
        budget=PureDPBudget(2),
        divisor=5,
        expected=PureDPBudget(0.4),
    ),
    Case("puredp_bad_type")(
        budget=PureDPBudget(2),
        divisor={},
        expected=TypeError,
    ),
    Case("puredp_zero_divisor")(
        budget=PureDPBudget(2),
        divisor=0,
        expected=ValueError,
    ),
    Case("puredp_negative_divisor")(
        budget=PureDPBudget(2),
        divisor=-1,
        expected=ValueError,
    ),
    Case("puredp_inf_divisor")(
        budget=PureDPBudget(2),
        divisor=float("inf"),
        expected=ValueError,
    ),
    Case("puredp_nan_divisor")(
        budget=PureDPBudget(2),
        divisor=math.nan,
        expected=ValueError,
    ),
    Case("zcdp_ints")(
        budget=RhoZCDPBudget(1),
        divisor=2,
        expected=RhoZCDPBudget(0.5),
    ),
    Case("zcdp_floats")(
        budget=RhoZCDPBudget(1.0),
        divisor=2.0,
        expected=RhoZCDPBudget(0.5),
    ),
    Case("zcdp_non_half")(
        budget=RhoZCDPBudget(2),
        divisor=5,
        expected=RhoZCDPBudget(0.4),
    ),
    Case("zcdp_bad_type")(
        budget=RhoZCDPBudget(2),
        divisor={},
        expected=TypeError,
    ),
    Case("zcdp_zero_divisor")(
        budget=RhoZCDPBudget(2),
        divisor=0,
        expected=ValueError,
    ),
    Case("zcdp_negative_divisor")(
        budget=RhoZCDPBudget(2),
        divisor=-1,
        expected=ValueError,
    ),
    Case("zcdp_inf_divisor")(
        budget=RhoZCDPBudget(2),
        divisor=float("inf"),
        expected=ValueError,
    ),
    Case("zcdp_nan_divisor")(
        budget=RhoZCDPBudget(2),
        divisor=math.nan,
        expected=ValueError,
    ),
    Case("approxdp_ints")(
        budget=ApproxDPBudget(1, 0.1),
        divisor=2,
        expected=ApproxDPBudget(0.5, 0.05),
    ),
    Case("approxdp_floats")(
        budget=ApproxDPBudget(1.0, 0.1),
        divisor=2.0,
        expected=ApproxDPBudget(0.5, 0.05),
    ),
    Case("approxdp_non_half")(
        budget=ApproxDPBudget(2, 0.2),
        divisor=5,
        expected=ApproxDPBudget(0.4, 0.04),
    ),
    Case("approxdp_bad_type")(
        budget=ApproxDPBudget(2, 0.1),
        divisor={},
        expected=TypeError,
    ),
    Case("approxdp_zero_divisor")(
        budget=ApproxDPBudget(2, 0.1),
        divisor=0,
        expected=ValueError,
    ),
    Case("approxdp_negative_divisor")(
        budget=ApproxDPBudget(2, 0.1),
        divisor=-1,
        expected=ValueError,
    ),
    Case("approxdp_inf_divisor")(
        budget=ApproxDPBudget(2, 0.1),
        divisor=float("inf"),
        expected=ValueError,
    ),
    Case("approxdp_nan_divisor")(
        budget=ApproxDPBudget(2, 0.1),
        divisor=math.nan,
        expected=ValueError,
    ),
)
def test_budget_division(
    budget: PrivacyBudget,
    divisor: Union[int, float],
    expected: Union[PrivacyBudget, Type[Exception]],
):
    """Tests that division works correctly on privacy budgets."""
    if isinstance(expected, PrivacyBudget):
        assert (budget / divisor) == expected
    else:
        with pytest.raises(expected):
            _ = budget / divisor


@parametrize(
    Case("puredp_ints")(
        budget=PureDPBudget(1),
        multiplier=2,
        expected=PureDPBudget(2),
    ),
    Case("puredp_floats")(
        budget=PureDPBudget(0.5),
        multiplier=2.0,
        expected=PureDPBudget(1.0),
    ),
    Case("puredp_non_half")(
        budget=PureDPBudget(2),
        multiplier=5,
        expected=PureDPBudget(10),
    ),
    Case("puredp_bad_type")(
        budget=PureDPBudget(2),
        multiplier={},
        expected=TypeError,
    ),
    Case("puredp_zero_multiplier")(
        budget=PureDPBudget(2),
        multiplier=0,
        expected=PureDPBudget(0),
    ),
    Case("puredp_negative_multiplier")(
        budget=PureDPBudget(2),
        multiplier=-1,
        expected=ValueError,
    ),
    Case("puredp_inf_multiplier")(
        budget=PureDPBudget(2),
        multiplier=float("inf"),
        expected=ValueError,
    ),
    Case("puredp_nan_multiplier")(
        budget=PureDPBudget(2),
        multiplier=math.nan,
        expected=ValueError,
    ),
    Case("zcdp_ints")(
        budget=RhoZCDPBudget(1),
        multiplier=2,
        expected=RhoZCDPBudget(2),
    ),
    Case("zcdp_floats")(
        budget=RhoZCDPBudget(0.5),
        multiplier=2.0,
        expected=RhoZCDPBudget(1.0),
    ),
    Case("zcdp_non_half")(
        budget=RhoZCDPBudget(2),
        multiplier=5,
        expected=RhoZCDPBudget(10),
    ),
    Case("zcdp_bad_type")(
        budget=RhoZCDPBudget(2),
        multiplier={},
        expected=TypeError,
    ),
    Case("zcdp_zero_multiplier")(
        budget=RhoZCDPBudget(2),
        multiplier=0,
        expected=RhoZCDPBudget(0),
    ),
    Case("zcdp_negative_multiplier")(
        budget=RhoZCDPBudget(2),
        multiplier=-1,
        expected=ValueError,
    ),
    Case("zcdp_inf_multiplier")(
        budget=RhoZCDPBudget(2),
        multiplier=float("inf"),
        expected=ValueError,
    ),
    Case("zcdp_nan_multiplier")(
        budget=RhoZCDPBudget(2),
        multiplier=math.nan,
        expected=ValueError,
    ),
    Case("approxdp_ints")(
        budget=ApproxDPBudget(1, 0.1),
        multiplier=2,
        expected=ApproxDPBudget(2, 0.2),
    ),
    Case("approxdp_floats")(
        budget=ApproxDPBudget(0.5, 0.05),
        multiplier=2.0,
        expected=ApproxDPBudget(1.0, 0.1),
    ),
    Case("approxdp_non_half")(
        budget=ApproxDPBudget(2, 0.02),
        multiplier=5,
        expected=ApproxDPBudget(10, 0.1),
    ),
    Case("approxdp_bad_type")(
        budget=ApproxDPBudget(2, 0.1),
        multiplier={},
        expected=TypeError,
    ),
    Case("approxdp_zero_multiplier")(
        budget=ApproxDPBudget(2, 0.1),
        multiplier=0,
        expected=ApproxDPBudget(0, 0),
    ),
    Case("approxdp_negative_multiplier")(
        budget=ApproxDPBudget(2, 0.1),
        multiplier=-1,
        expected=ValueError,
    ),
    Case("approxdp_inf_multiplier")(
        budget=ApproxDPBudget(2, 0.1),
        multiplier=float("inf"),
        expected=ValueError,
    ),
    Case("approxdp_nan_multiplier")(
        budget=ApproxDPBudget(2, 0.1),
        multiplier=math.nan,
        expected=ValueError,
    ),
    Case("approxdp_overflow")(
        budget=ApproxDPBudget(1, 0.5),
        multiplier=5,
        expected=ApproxDPBudget(5, 1.0),
    ),
)
def test_budget_multiplication(
    budget: PrivacyBudget,
    multiplier: Union[int, float],
    expected: Union[PrivacyBudget, Type[Exception]],
):
    """Tests that division works correctly on privacy budgets."""
    if isinstance(expected, PrivacyBudget):
        assert (budget * multiplier) == expected
    else:
        with pytest.raises(expected):
            _ = budget * multiplier


@parametrize(
    Case("puredp_ints")(
        budget_a=PureDPBudget(1),
        budget_b=PureDPBudget(2),
        expected=PureDPBudget(3),
    ),
    Case("puredp_floats")(
        budget_a=PureDPBudget(1.5),
        budget_b=PureDPBudget(2.5),
        expected=PureDPBudget(4.0),
    ),
    Case("puredp_inf_plus_finite")(
        budget_a=PureDPBudget(float("inf")),
        budget_b=PureDPBudget(2),
        expected=PureDPBudget(float("inf")),
    ),
    Case("puredp_inf_plus_inf")(
        budget_a=PureDPBudget(float("inf")),
        budget_b=PureDPBudget(float("inf")),
        expected=PureDPBudget(float("inf")),
    ),
    Case("zcdp_ints")(
        budget_a=RhoZCDPBudget(1),
        budget_b=RhoZCDPBudget(2),
        expected=RhoZCDPBudget(3),
    ),
    Case("zcdp_floats")(
        budget_a=RhoZCDPBudget(1.5),
        budget_b=RhoZCDPBudget(2.5),
        expected=RhoZCDPBudget(4.0),
    ),
    Case("zcdp_inf_plus_finite")(
        budget_a=RhoZCDPBudget(float("inf")),
        budget_b=RhoZCDPBudget(2),
        expected=RhoZCDPBudget(float("inf")),
    ),
    Case("zcdp_inf_plus_inf")(
        budget_a=RhoZCDPBudget(float("inf")),
        budget_b=RhoZCDPBudget(float("inf")),
        expected=RhoZCDPBudget(float("inf")),
    ),
    Case("approxdp_ints")(
        budget_a=ApproxDPBudget(1, 0.25),
        budget_b=ApproxDPBudget(2, 0.25),
        expected=ApproxDPBudget(3, 0.5),
    ),
    Case("approxdp_floats")(
        budget_a=ApproxDPBudget(1.5, 0.15),
        budget_b=ApproxDPBudget(2.5, 0.25),
        expected=ApproxDPBudget(4.0, 0.4),
    ),
    Case("approxdp_delta_overflow")(
        budget_a=ApproxDPBudget(1, 0.8),
        budget_b=ApproxDPBudget(2, 0.3),
        expected=ApproxDPBudget(3, 1.0),
    ),
    Case("approxdp_plus_puredp")(
        budget_a=ApproxDPBudget(1, 0.5),
        budget_b=PureDPBudget(2),
        expected=ApproxDPBudget(3, 0.5),
    ),
    Case("puredp_plus_approxdp")(
        budget_a=PureDPBudget(2),
        budget_b=ApproxDPBudget(1, 0.5),
        expected=ApproxDPBudget(3, 0.5),
    ),
    Case("approxdp_inf_plus_finite")(
        budget_a=ApproxDPBudget(float("inf"), 1.0),
        budget_b=ApproxDPBudget(2, 0.1),
        expected=ApproxDPBudget(float("inf"), 1.0),
    ),
    Case("approxdp_inf_plus_inf")(
        budget_a=ApproxDPBudget(float("inf"), 1.0),
        budget_b=ApproxDPBudget(float("inf"), 1.0),
        expected=ApproxDPBudget(float("inf"), 1.0),
    ),
    Case("approxdp_delta_inf_plus_finite")(
        budget_a=ApproxDPBudget(3, 1.0),
        budget_b=ApproxDPBudget(2, 0.1),
        expected=ApproxDPBudget(5, 1.0),
    ),
    Case("approxdp_delta_inf_plus_inf")(
        budget_a=ApproxDPBudget(3, 1.0),
        budget_b=ApproxDPBudget(2, 1.0),
        expected=ApproxDPBudget(5, 1.0),
    ),
    Case("approxdp_epsilon_inf_plus_finite")(
        budget_a=ApproxDPBudget(float("inf"), 0.2),
        budget_b=ApproxDPBudget(2, 0.1),
        expected=ApproxDPBudget(float("inf"), 0.3),
    ),
    Case("approxdp_epsilon_inf_plus_inf")(
        budget_a=ApproxDPBudget(float("inf"), 0.2),
        budget_b=ApproxDPBudget(float("inf"), 0.1),
        expected=ApproxDPBudget(float("inf"), 0.3),
    ),
)
def test_budget_addition(
    budget_a: PrivacyBudget,
    budget_b: PrivacyBudget,
    expected: Union[PrivacyBudget, Type[Exception]],
):
    """Tests that two budgets added together yield the expected result."""
    if isinstance(expected, PrivacyBudget):
        assert (budget_a + budget_b) == expected
    else:
        with pytest.raises(expected):
            _ = budget_a + budget_b


@parametrize(
    Case("puredp_ints")(
        budget_a=PureDPBudget(3),
        budget_b=PureDPBudget(2),
        expected=PureDPBudget(1),
    ),
    Case("puredp_floats")(
        budget_a=PureDPBudget(4.0),
        budget_b=PureDPBudget(2.5),
        expected=PureDPBudget(1.5),
    ),
    Case("puredp_inf_minus_finite")(
        budget_a=PureDPBudget(float("inf")),
        budget_b=PureDPBudget(2),
        expected=PureDPBudget(float("inf")),
    ),
    Case("puredp_inf_minus_inf")(
        budget_a=PureDPBudget(float("inf")),
        budget_b=PureDPBudget(float("inf")),
        expected=PureDPBudget(float("inf")),
    ),
    Case("puredp_rounds")(
        budget_a=PureDPBudget(1),
        budget_b=PureDPBudget(1.000000000001),
        expected=PureDPBudget(0),
    ),
    Case("zcdp_ints")(
        budget_a=RhoZCDPBudget(3),
        budget_b=RhoZCDPBudget(2),
        expected=RhoZCDPBudget(1),
    ),
    Case("zcdp_floats")(
        budget_a=RhoZCDPBudget(4.0),
        budget_b=RhoZCDPBudget(2.5),
        expected=RhoZCDPBudget(1.5),
    ),
    Case("zcdp_inf_minus_finite")(
        budget_a=RhoZCDPBudget(float("inf")),
        budget_b=RhoZCDPBudget(2),
        expected=RhoZCDPBudget(float("inf")),
    ),
    Case("zcdp_inf_minus_inf")(
        budget_a=RhoZCDPBudget(float("inf")),
        budget_b=RhoZCDPBudget(float("inf")),
        expected=RhoZCDPBudget(float("inf")),
    ),
    Case("zcdp_rounds")(
        budget_a=RhoZCDPBudget(1),
        budget_b=RhoZCDPBudget(1.000000000001),
        expected=RhoZCDPBudget(0),
    ),
    Case("approxdp_ints")(
        budget_a=ApproxDPBudget(3, 0.5),
        budget_b=ApproxDPBudget(2, 0.25),
        expected=ApproxDPBudget(1, 0.25),
    ),
    Case("approxdp_floats")(
        budget_a=ApproxDPBudget(4.0, 0.5),
        budget_b=ApproxDPBudget(2.5, 0.25),
        expected=ApproxDPBudget(1.5, 0.25),
    ),
    Case("puredp_underflow")(
        budget_a=PureDPBudget(2),
        budget_b=PureDPBudget(3),
        expected=ValueError,
    ),
    Case("zcdp_underflow")(
        budget_a=RhoZCDPBudget(2),
        budget_b=RhoZCDPBudget(3),
        expected=ValueError,
    ),
    Case("approxdp_epsilon_underflow")(
        budget_a=ApproxDPBudget(2, 0.5),
        budget_b=ApproxDPBudget(3, 0.5),
        expected=ValueError,
    ),
    Case("approxdp_delta_underflow")(
        budget_a=ApproxDPBudget(3, 0.5),
        budget_b=ApproxDPBudget(2, 0.6),
        expected=ValueError,
    ),
    Case("approxdp_minus_puredp")(
        budget_a=ApproxDPBudget(3, 0.5),
        budget_b=PureDPBudget(1),
        expected=ApproxDPBudget(2, 0.5),
    ),
    Case("puredp_minus_approxdp")(
        budget_a=PureDPBudget(3),
        budget_b=ApproxDPBudget(1, 0.5),
        expected=TypeError,
    ),
    Case("approxdp_inf_minus_finite")(
        budget_a=ApproxDPBudget(float("inf"), 1.0),
        budget_b=ApproxDPBudget(2, 0.1),
        expected=ApproxDPBudget(float("inf"), 1.0),
    ),
    Case("approxdp_inf_minus_inf")(
        budget_a=ApproxDPBudget(float("inf"), 1.0),
        budget_b=ApproxDPBudget(float("inf"), 1.0),
        expected=ApproxDPBudget(float("inf"), 1.0),
    ),
    Case("approx_rounds_epsilon")(
        budget_a=ApproxDPBudget(1, 0.9),
        budget_b=ApproxDPBudget(1.00000000001, 0.9),
        expected=ApproxDPBudget(0, 0.0),
    ),
    Case("approx_rounds_delta")(
        budget_a=ApproxDPBudget(1, 0.1),
        budget_b=ApproxDPBudget(1, 0.100000000001),
        expected=ApproxDPBudget(0, 0.0),
    ),
)
def test_budget_subtraction(
    budget_a: PrivacyBudget,
    budget_b: PrivacyBudget,
    expected: Union[PrivacyBudget, Type[Exception]],
):
    """Tests that two budgets added together yield the expected result."""
    if isinstance(expected, PrivacyBudget):
        assert (budget_a - budget_b) == expected
    else:
        with pytest.raises(expected):
            _ = budget_a - budget_b
