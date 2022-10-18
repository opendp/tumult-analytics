"""Tests for :mod:`tmlt.analytics.privacy_budget`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
# pylint: disable=pointless-string-statement

import pytest

from tmlt.analytics.privacy_budget import PureDPBudget, RhoZCDPBudget

"""Tests for :class:`tmlt.analytics.privacy_budget.PureDPBudget`."""


def test_constructor_success_nonnegative_int():
    """Tests that construction succeeds with a nonnegative int."""
    budget = PureDPBudget(2)
    assert budget.epsilon == 2
    budget = PureDPBudget(0)
    assert budget.epsilon == 0


def test_constructor_success_nonnegative_float():
    """Tests that construction succeeds with a nonnegative float."""
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
        PureDPBudget("1.5")


def test_constructor_fail_nan():
    """Tests that construction fails with epsilon that is a NAN."""
    with pytest.raises(ValueError, match="Epsilon cannot be a NaN."):
        PureDPBudget(float("nan"))


"""Tests for :class:`tmlt.analytics.privacy_budget.RhoZCDPBudget`."""


def test_constructor_success_nonnegative_int_ZCDP():
    """Tests that construction succeeds with a nonnegative int."""
    budget = RhoZCDPBudget(2)
    assert budget.rho == 2
    budget = RhoZCDPBudget(0)
    assert budget.rho == 0


def test_constructor_success_nonnegative_float_ZCDP():
    """Tests that construction succeeds with a nonnegative float."""
    budget = RhoZCDPBudget(2.5)
    assert budget.rho == 2.5
    budget = RhoZCDPBudget(0.0)
    assert budget.rho == 0.0


def test_constructor_fail_negative_int_ZCDP():
    """Tests that construction fails with a negative int."""
    with pytest.raises(ValueError, match="Rho must be non-negative."):
        RhoZCDPBudget(-1)


def test_constructor_fail_negative_float_ZCDP():
    """Tests that construction fails with a negative float."""
    with pytest.raises(ValueError, match="Rho must be non-negative."):
        RhoZCDPBudget(-1.5)


def test_constructor_fail_bad_rho_type_ZCDP():
    """Tests that construction fails with rho that is not an int or float."""
    with pytest.raises(TypeError):
        RhoZCDPBudget("1.5")


def test_constructor_fail_nan_ZCDP():
    """Tests that construction fails with rho that is a NAN."""
    with pytest.raises(ValueError, match="Rho cannot be a NaN."):
        RhoZCDPBudget(float("nan"))
