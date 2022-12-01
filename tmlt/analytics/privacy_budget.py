"""Classes for specifying privacy budgets.

For a full introduction to privacy budgets, see the
:ref:`privacy budget topic guide<Privacy budget fundamentals>`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
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
    be found `here <https://en.wikipedia.org/wiki/Differential_privacy#Definition_of_%CE%B5-differential_privacy>`_
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
