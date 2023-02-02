"""Classes for specifying privacy budgets.

For a full introduction to privacy budgets, see the
:ref:`privacy budget topic guide<Privacy budget fundamentals>`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
import math
from abc import ABC
from typing import Union

from typeguard import typechecked

from tmlt.core.utils.exact_number import ExactNumber


class PrivacyBudget(ABC):
    """Base class for specifying privacy parameters.

    A PrivacyBudget is a privacy definition, along with its associated parameters.
    The choice of a PrivacyBudget has an impact on the accuracy of query
    results. Smaller parameters correspond to a stronger privacy guarantee, and
    usually lead to less accurate results.

    .. note::
        An "infinite" privacy budget means that the chosen DP algorithm will use
        parameters that do not guarantee privacy. This is not always exactly equivalent
        to evaluating the query without applying differential privacy.
        Please see the individual subclasses of PrivacyBudget for details on how to
        appropriately specify infinite budgets.
    """


class PureDPBudget(PrivacyBudget):
    """A privacy budget under pure differential privacy.

    This privacy definition is also known as epsilon-differential privacy, and the
    associated value is the epsilon privacy parameter. The privacy definition can
    be found `here <https://en.wikipedia.org/wiki/Differential_privacy#Definition_of_%CE%B5-differential_privacy>`__.
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(self, epsilon: Union[int, float]):
        """Construct a new PureDPBudget.

        Args:
            epsilon: The epsilon privacy parameter. Must be non-negative
                and cannot be NaN.
                To specify an infinite budget, set epsilon equal to float('inf').
        """
        if math.isnan(epsilon):
            raise ValueError("Epsilon cannot be a NaN.")
        if epsilon < 0:
            raise ValueError(
                "Epsilon must be non-negative. "
                f"Cannot construct a PureDPBudget with epsilon of {epsilon}."
            )
        self._epsilon = epsilon

    @property
    def epsilon(self) -> Union[int, float]:
        """Returns the value of epsilon."""
        return self._epsilon

    def __repr__(self) -> str:
        """Returns string representation of this PureDPBudget."""
        return f"PureDPBudget(epsilon={self.epsilon})"

    def __eq__(self, other) -> bool:
        """Returns whether or not two PureDPBudgets are equivalent."""
        if isinstance(other, PureDPBudget):
            return ExactNumber.from_float(
                self.epsilon, False
            ) == ExactNumber.from_float(other.epsilon, False)
        return False


class ApproxDPBudget(PrivacyBudget):
    """A privacy budget under approximate differential privacy.

    This privacy definition is also known as (ε, δ)-differential privacy, and the
    associated privacy parameters are epsilon and delta. The formal definition can
    be found `here <https://desfontain.es/privacy/almost-differential-privacy.html#formal-definition>`__.
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(self, epsilon: Union[int, float], delta: float):
        """Construct a new ApproxDPBudget.

        Args:
            epsilon: The epsilon privacy parameter. Must be non-negative.
                To specify an infinite budget, set epsilon equal to float('inf').
            delta: The delta privacy parameter. Must be between 0 and 1 (inclusive).
                If delta is 0, this is equivalent to PureDP.
        """
        if math.isnan(epsilon):
            raise ValueError("Epsilon cannot be a NaN.")
        if math.isnan(delta):
            raise ValueError("Delta cannot be a NaN.")
        if epsilon < 0:
            raise ValueError(
                "Epsilon must be non-negative. "
                f"Cannot construct an ApproxDPBudget with epsilon of {epsilon}."
            )
        if delta < 0 or delta > 1:
            raise ValueError(
                "Delta must be between 0 and 1 (inclusive). "
                f"Cannot construct an ApproxDPBudget with delta of {delta}."
            )
        self._epsilon = epsilon
        self._delta = delta

    @property
    def epsilon(self) -> Union[int, float]:
        """Returns the value of epsilon."""
        return self._epsilon

    @property
    def delta(self) -> float:
        """Returns the value of delta."""
        return self._delta

    @property
    def is_infinite(self) -> bool:
        """Returns true if epsilon is float('inf') or delta is 1."""
        return self.epsilon == float("inf") or self.delta == 1

    def __repr__(self) -> str:
        """Returns the string representation of this ApproxDPBudget."""
        return f"ApproxDPBudget(epsilon={self.epsilon}, delta={self.delta})"

    def __eq__(self, other) -> bool:
        """Returns whether two ApproxDPBudgets are equivalent.

        ApproxDPBudgets that provide no privacy guarantee are considered equal (for example, if one has an
        epsilon of float('inf') and the other has a delta of 1).
        """
        if isinstance(other, ApproxDPBudget):
            are_both_infinite = self.is_infinite and other.is_infinite
            is_same_epsilon = ExactNumber.from_float(
                self.epsilon, False
            ) == ExactNumber.from_float(other.epsilon, False)
            is_same_delta = ExactNumber.from_float(
                self.delta, False
            ) == ExactNumber.from_float(other.delta, False)
            return are_both_infinite or (is_same_epsilon and is_same_delta)
        return False


class RhoZCDPBudget(PrivacyBudget):
    """A privacy budget under rho-zero-concentrated differential privacy.

    The definition of rho-zCDP can be found in
    `this <https://arxiv.org/pdf/1605.02065.pdf>`_ paper under Definition 1.1.
    """

    @typechecked()
    def __init__(self, rho: Union[int, float]):
        """Construct a new RhoZCDPBudget.

        Args:
            rho: The rho privacy parameter.
                Rho must be non-negative and cannot be NaN.
                To specify an infinite budget, set rho equal to float('inf').
        """
        if math.isnan(rho):
            raise ValueError("Rho cannot be a NaN.")
        if rho < 0:
            raise ValueError(
                "Rho must be non-negative. "
                f"Cannot construct a RhoZCDPBudget with rho of {rho}."
            )
        self._rho = rho

    @property
    def rho(self) -> Union[int, float]:
        """Returns the value of rho."""
        return self._rho

    def __repr__(self) -> str:
        """Returns string representation of this RhoZCDPBudget."""
        return f"RhoZCDPBudget(rho={self.rho})"

    def __eq__(self, other) -> bool:
        """Returns whether or not two RhoZCDPBudgets are equivalent."""
        if isinstance(other, RhoZCDPBudget):
            return ExactNumber.from_float(self.rho, False) == ExactNumber.from_float(
                other.rho, False
            )
        return False
