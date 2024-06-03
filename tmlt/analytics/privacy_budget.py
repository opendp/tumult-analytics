"""Classes for specifying privacy budgets.

For a full introduction to privacy budgets, see the
:ref:`privacy budget topic guide<Privacy budget fundamentals>`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import sympy as sp
from tmlt.core.utils.exact_number import ExactNumber
from typeguard import typechecked


def _is_exact_number_from_integer(value: ExactNumber) -> bool:
    """Returns True if the ExactNumber is an integer."""
    return isinstance(value.expr, sp.Integer)


def _to_int_or_float(value: ExactNumber) -> Union[int, float]:
    """Converts an ExactNumber to an int or float."""
    if _is_exact_number_from_integer(value):
        return int(value.expr)
    else:
        return float(value.expr)


def _to_exact_number(value: Union[int, float, ExactNumber]) -> ExactNumber:
    """Converts a value to an ExactNumber."""
    if isinstance(value, ExactNumber):
        return value
    elif isinstance(value, int):
        return ExactNumber(value)
    elif isinstance(value, float):
        return ExactNumber.from_float(value, round_up=False)
    else:
        raise ValueError(
            f"Cannot convert value of type {type(value)} to an ExactNumber."
        )


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

    @property
    @abstractmethod
    def value(self) -> Union[ExactNumber, Tuple[ExactNumber, ExactNumber]]:
        """Return the value of the privacy budget."""

    @property
    @abstractmethod
    def is_infinite(self) -> bool:
        """Returns true if the privacy budget is infinite."""


@dataclass(frozen=True, init=False, unsafe_hash=True)
class PureDPBudget(PrivacyBudget):
    """A privacy budget under pure differential privacy.

    This privacy definition is also known as epsilon-differential privacy, and the
    associated value is the epsilon privacy parameter. The privacy definition can
    be found `here <https://en.wikipedia.org/wiki/Differential_privacy#Definition_of_%CE%B5-differential_privacy>`__.
    """  # pylint: disable=line-too-long

    _epsilon: ExactNumber

    @typechecked
    def __init__(self, epsilon: Union[int, float, ExactNumber]):
        """Construct a new PureDPBudget.

        Args:
            epsilon: The epsilon privacy parameter. Must be non-negative
                and cannot be NaN.
                To specify an infinite budget, set epsilon equal to float('inf').
        """
        if not isinstance(epsilon, ExactNumber) and math.isnan(epsilon):
            raise ValueError("Epsilon cannot be a NaN.")
        if epsilon < 0:
            raise ValueError(
                "Epsilon must be non-negative. "
                f"Cannot construct a PureDPBudget with epsilon of {epsilon}."
            )
        # The class is frozen, so we need to subvert it to update epsilon.
        object.__setattr__(self, "_epsilon", _to_exact_number(epsilon))

    @property
    def value(self) -> ExactNumber:
        """Return the value of the privacy budget as an ExactNumber.

        For printing purposes, you should use the epsilon property instead, as it will
        represent the same value, but be more human readable.
        """
        return self._epsilon

    @property
    def epsilon(self) -> Union[int, float]:
        """Returns the value of epsilon as an int or float.

        This is helpful for human readability. If you need to use the epsilon value in
        a computation, you should use self.value instead.
        """
        return _to_int_or_float(self._epsilon)

    @property
    def is_infinite(self) -> bool:
        """Returns true if epsilon is float('inf')."""
        return self._epsilon == float("inf")

    def __repr__(self) -> str:
        """Returns string representation of this PureDPBudget."""
        return f"PureDPBudget(epsilon={self.epsilon})"


@dataclass(frozen=True, init=False, eq=False, unsafe_hash=False)
class ApproxDPBudget(PrivacyBudget):
    """A privacy budget under approximate differential privacy.

    This privacy definition is also known as (ε, δ)-differential privacy, and the
    associated privacy parameters are epsilon and delta. The formal definition can
    be found `here <https://desfontain.es/privacy/almost-differential-privacy.html#formal-definition>`__.
    """  # pylint: disable=line-too-long

    _epsilon: ExactNumber
    _delta: ExactNumber

    @typechecked
    def __init__(
        self,
        epsilon: Union[int, float, ExactNumber],
        delta: Union[int, float, ExactNumber],
    ):
        """Construct a new ApproxDPBudget.

        Args:
            epsilon: The epsilon privacy parameter. Must be non-negative.
                To specify an infinite budget, set epsilon equal to float('inf').
            delta: The delta privacy parameter. Must be between 0 and 1 (inclusive).
                If delta is 0, this is equivalent to PureDP.
        """
        if not isinstance(epsilon, ExactNumber) and math.isnan(epsilon):
            raise ValueError("Epsilon cannot be a NaN.")
        if not isinstance(delta, ExactNumber) and math.isnan(delta):
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

        # The class is frozen, so we need to subvert it to update epsilon and delta.
        object.__setattr__(self, "_epsilon", _to_exact_number(epsilon))
        object.__setattr__(self, "_delta", _to_exact_number(delta))

    @property
    def value(self) -> Tuple[ExactNumber, ExactNumber]:
        """Returns self._epsilon and self._delta as an ExactNumber tuple.

        For printing purposes, you might want to use the epsilon and delta properties
        instead, as they will represent the same values, but be more human readable.
        """
        return (self._epsilon, self._delta)

    @property
    def epsilon(self) -> Union[int, float]:
        """Returns the value of epsilon as an int or float.

        This is helpful for human readability. If you need to use the epsilon value in
        a computation, you should use self.value[0] instead.
        """
        return _to_int_or_float(self._epsilon)

    @property
    def delta(self) -> Union[int, float]:
        """Returns the value of delta as an int or float.

        This is helpful for human readability. If you need to use the delta value in
        a computation, you should use self.value[1] instead.
        """
        return _to_int_or_float(self._delta)

    @property
    def is_infinite(self) -> bool:
        """Returns true if epsilon is float('inf') or delta is 1."""
        return self._epsilon == float("inf") or self._delta == 1

    def __repr__(self) -> str:
        """Returns the string representation of this ApproxDPBudget."""
        return f"ApproxDPBudget(epsilon={self.epsilon}, delta={self.delta})"

    def __eq__(self, other) -> bool:
        """Returns True if both ApproxDPBudgets are infinite or have equal values."""
        if isinstance(other, ApproxDPBudget):
            if self.is_infinite and other.is_infinite:
                return True
            else:
                return self.value == other.value
        return False

    def __hash__(self):
        """Hashes on the values, but infinite budgets hash to the same value."""
        if self.is_infinite:
            return hash((float("inf"), float("inf")))
        return hash(self.value)


@dataclass(frozen=True, init=False, unsafe_hash=True)
class RhoZCDPBudget(PrivacyBudget):
    """A privacy budget under rho-zero-concentrated differential privacy.

    The definition of rho-zCDP can be found in
    `this <https://arxiv.org/pdf/1605.02065.pdf>`_ paper under Definition 1.1.
    """

    _rho: ExactNumber

    @typechecked()
    def __init__(self, rho: Union[int, float, ExactNumber]):
        """Construct a new RhoZCDPBudget.

        Args:
            rho: The rho privacy parameter.
                Rho must be non-negative and cannot be NaN.
                To specify an infinite budget, set rho equal to float('inf').
        """
        if not isinstance(rho, ExactNumber) and math.isnan(rho):
            raise ValueError("Rho cannot be a NaN.")
        if rho < 0:
            raise ValueError(
                "Rho must be non-negative. "
                f"Cannot construct a RhoZCDPBudget with rho of {rho}."
            )
        # The class is frozen, so we need to subvert it to update rho.
        object.__setattr__(self, "_rho", _to_exact_number(rho))

    @property
    def value(self) -> ExactNumber:
        """Return the value of the privacy budget as an ExactNumber.

        For printing purposes, you should use the rho property instead, as it will
        represent the same value, but be more human readable.
        """
        return self._rho

    @property
    def rho(self) -> Union[int, float]:
        """Returns the value of rho as an int or float.

        This is helpful for human readability. If you need to use the rho value in
        a computation, you should use self.value instead.
        """
        return _to_int_or_float(self._rho)

    @property
    def is_infinite(self) -> bool:
        """Returns true if rho is float('inf')."""
        return self._rho == float("inf")

    def __repr__(self) -> str:
        """Returns string representation of this RhoZCDPBudget."""
        return f"RhoZCDPBudget(rho={self.rho})"
