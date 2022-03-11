"""Tests for :mod:`tmlt.analytics.privacy_budget`."""

# <placeholder: boilerplate>
from unittest import TestCase

from tmlt.analytics.privacy_budget import PureDPBudget, RhoZCDPBudget


class TestPureDPBudget(TestCase):
    """Tests for :class:`tmlt.analytics.privacy_budget.PureDPBudget`."""

    def test_constructor_success_nonnegative_int(self):
        """Tests that construction succeeds with a nonnegative int."""
        budget = PureDPBudget(2)
        self.assertEqual(budget.epsilon, 2)

        budget = PureDPBudget(0)
        self.assertEqual(budget.epsilon, 0)

    def test_constructor_success_nonnegative_float(self):
        """Tests that construction succeeds with a nonnegative float."""
        budget = PureDPBudget(2.5)
        self.assertEqual(budget.epsilon, 2.5)

        budget = PureDPBudget(0.0)
        self.assertEqual(budget.epsilon, 0.0)

    def test_constructor_fail_negative_int(self):
        """Tests that construction fails with a negative int."""
        with self.assertRaisesRegex(ValueError, "Epsilon must be non-negative."):
            PureDPBudget(-1)

    def test_constructor_fail_negative_float(self):
        """Tests that construction fails with a negative float."""
        with self.assertRaisesRegex(ValueError, "Epsilon must be non-negative."):
            PureDPBudget(-1.5)

    def test_constructor_fail_bad_epsilon_type(self):
        """Tests that construction fails with epsilon that is not an int or float."""
        with self.assertRaises(TypeError):
            PureDPBudget("1.5")

    def test_constructor_fail_nan(self):
        """Tests that construction fails with epsilon that is a NAN."""
        with self.assertRaisesRegex(ValueError, "Epsilon cannot be a NaN."):
            PureDPBudget(float("nan"))


class TestRhoZCDPBudget(TestCase):
    """Tests for :class:`tmlt.analytics.privacy_budget.RhoZCDPBudget`."""

    def test_constructor_success_nonnegative_int(self):
        """Tests that construction succeeds with a nonnegative int."""
        budget = RhoZCDPBudget(2)
        self.assertEqual(budget.rho, 2)

        budget = RhoZCDPBudget(0)
        self.assertEqual(budget.rho, 0)

    def test_constructor_success_nonnegative_float(self):
        """Tests that construction succeeds with a nonnegative float."""
        budget = RhoZCDPBudget(2.5)
        self.assertEqual(budget.rho, 2.5)

        budget = RhoZCDPBudget(0.0)
        self.assertEqual(budget.rho, 0.0)

    def test_constructor_fail_negative_int(self):
        """Tests that construction fails with a negative int."""
        with self.assertRaisesRegex(ValueError, "Rho must be non-negative."):
            RhoZCDPBudget(-1)

    def test_constructor_fail_negative_float(self):
        """Tests that construction fails with a negative float."""
        with self.assertRaisesRegex(ValueError, "Rho must be non-negative."):
            RhoZCDPBudget(-1.5)

    def test_constructor_fail_bad_rho_type(self):
        """Tests that construction fails with rho that is not an int or float."""
        with self.assertRaises(TypeError):
            RhoZCDPBudget("1.5")

    def test_constructor_fail_nan(self):
        """Tests that construction fails with rho that is a NAN."""
        with self.assertRaisesRegex(ValueError, "Rho cannot be a NaN."):
            RhoZCDPBudget(float("nan"))
