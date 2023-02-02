"""Tests for :mod:`tmlt.analytics.privacy_budget`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
# pylint: disable=pointless-string-statement

import pytest

from tmlt.analytics.privacy_budget import ApproxDPBudget, PureDPBudget, RhoZCDPBudget

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
    budget = ApproxDPBudget(2, 1)
    assert budget.epsilon == 2
    assert budget.delta == 1

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
