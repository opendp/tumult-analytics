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
    """Base class for specifying the maximal privacy loss of a Session or a query.

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

    @abstractmethod
    def __truediv__(self, other) -> "PrivacyBudget":
        """Budgets can be divided by finite integer/float values > 0."""

    @abstractmethod
    def __mul__(self, other) -> "PrivacyBudget":
        """Budgets can be multiplied by finite integer/float values >= 0."""

    @abstractmethod
    def __add__(self, other) -> "PrivacyBudget":
        """Budgets can be added to other budgets of compatible types.

        Addition is only supported so long as the result is still a valid budget
        (i.e. all parameters fall within valid ranges).
        """

    @abstractmethod
    def __sub__(self, other) -> "PrivacyBudget":
        """Budgets can be subtracted from other budgets of compatible types.

        Subtraction is only supported so long as the result is still a valid budget
        (i.e. all parameters fall within valid ranges).

        Subtracting anything from an infinite budget will return an infinite budget.
        """

    @classmethod
    @abstractmethod
    def inf(cls) -> "PrivacyBudget":
        """Get an infinite budget of this type."""


@dataclass(frozen=True, init=False)
class PureDPBudget(PrivacyBudget):
    """A privacy budget under pure differential privacy.

    This privacy definition is also known as epsilon-differential privacy, and the
    associated value is the epsilon privacy parameter. The privacy definition can
    be found `here <https://en.wikipedia.org/wiki/Differential_privacy#Definition>`__.
    """

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

    def __hash__(self):
        """Hashes on type and value."""
        return hash((type(self), self.epsilon))

    def __truediv__(self, other) -> "PureDPBudget":
        """Divide this budget by a finite integer/float value > 0."""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot divide a PureDPBudget by a {type(other)}.")
        if other <= 0 or math.isnan(other) or math.isinf(other):
            raise ValueError(
                f"Tried to divide a privacy budget by {other}, but can only "
                "divide by non-infinite numbers >0."
            )
        return PureDPBudget(self.epsilon / other)

    def __mul__(self, other) -> "PureDPBudget":
        """Multiply this budget by a finite integer/float value >= 0."""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot multiply a PureDPBudget by a {type(other)}.")
        if other < 0 or math.isnan(other) or math.isinf(other):
            raise ValueError(
                f"Tried to multiply a privacy budget by {other}, but can only "
                "multiply by non-infinite numbers >=0."
            )
        return PureDPBudget(self.epsilon * other)

    def __add__(self, other) -> PrivacyBudget:
        """Add this budget to another PureDPBudget or an ApproxDPBudget.

        The resulting epsilon must be greater than zero, and the
        resulting delta (if any) must be in [0, 1].
        """
        if isinstance(other, ApproxDPBudget):
            return ApproxDPBudget(self.epsilon, 0) + other
        elif not isinstance(other, PureDPBudget):
            raise TypeError(f"Cannot add a PureDPBudget to a {type(other)}.")
        if self.is_infinite:
            return self
        if other.is_infinite:
            return other
        return PureDPBudget(self.epsilon + other.epsilon)

    def __sub__(self, other) -> "PureDPBudget":
        """Subtract a PureDPBudget from this budget.

        The resulting epsilon must greater than zero. Subtracting anything from an
        infinite budget will return an infinite budget.

        Note that you cannot subtract an ApproxDPBudget from a PureDPBudget, though
        the reverse is allowed.
        """
        if not isinstance(other, PureDPBudget):
            raise TypeError(f"Cannot subtract a {type(other)} from a PureDPBudget.")
        if self.is_infinite:
            return self
        adjusted_other = _get_adjusted_budget(other, self)
        assert isinstance(adjusted_other, PureDPBudget)
        return PureDPBudget(self.epsilon - adjusted_other.epsilon)

    @classmethod
    def inf(cls) -> "PureDPBudget":
        """Get an infinite budget of this type."""
        return PureDPBudget(ExactNumber.from_float(float("inf"), round_up=False))


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

    def __truediv__(self, other) -> "ApproxDPBudget":
        """Divide this budget by a finite integer/float value > 0."""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot divide a ApproxDPBudget by a {type(other)}.")
        if other <= 0 or math.isnan(other) or math.isinf(other):
            raise ValueError(
                f"Tried to divide a privacy budget by {other}, but can only "
                "divide by non-infinite numbers >0."
            )
        return ApproxDPBudget(self.epsilon / other, self.delta / other)

    def __mul__(self, other) -> "ApproxDPBudget":
        """Multiply this budget by a finite integer/float value >= 0."""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot multiply a ApproxDPBudget by a {type(other)}.")
        if other < 0 or math.isnan(other) or math.isinf(other):
            raise ValueError(
                f"Tried to multiply a privacy budget by {other}, but can only "
                "multiply by non-infinite numbers >=0."
            )
        return ApproxDPBudget(self.epsilon * other, min(self.delta * other, 1.0))

    def __add__(self, other) -> "ApproxDPBudget":
        """Add this budget to another ApproxDPBudget or a PureDPBudget.

        The resulting epsilon must greater than zero. If the resulting delta is >1 it
        will be rounded down to 1.

        Addition is performed using basic composition, and the sum is therefore a
        (possibly loose) upper bound on the privacy loss from running two queries.
        """
        if isinstance(other, PureDPBudget):
            return self + ApproxDPBudget(other.epsilon, 0)
        elif not isinstance(other, ApproxDPBudget):
            raise TypeError(f"Cannot add a ApproxDPBudget to a {type(other)}.")
        if self.is_infinite:
            return self
        if other.is_infinite:
            return other
        return ApproxDPBudget(
            self.epsilon + other.epsilon, min(self.delta + other.delta, 1.0)
        )

    def __sub__(self, other) -> "ApproxDPBudget":
        """Subtract a PureDPBudget or ApproxDPBudget from this budget.

        The resulting epsilon greater than zero, and the resulting
        delta must be in [0, 1]. Subtracting anything from an infinite
        budget will return an infinite budget.

        Note that you can subtract a PureDPBudget from an ApproxDPBudget, though
        the reverse is not allowed.
        """
        if isinstance(other, PureDPBudget):
            return self - ApproxDPBudget(other.epsilon, 0)
        elif not isinstance(other, ApproxDPBudget):
            raise TypeError(f"Cannot subtract a {type(other)} from a ApproxDPBudget.")
        if self.is_infinite:
            return self
        adjusted_other = _get_adjusted_budget(other, self)
        assert isinstance(adjusted_other, ApproxDPBudget)
        return ApproxDPBudget(
            self.epsilon - adjusted_other.epsilon, self.delta - adjusted_other.delta
        )

    @classmethod
    def inf(cls) -> "ApproxDPBudget":
        """Get an infinite budget of this type."""
        return ApproxDPBudget(ExactNumber.from_float(float("inf"), round_up=False), 0)


@dataclass(frozen=True, init=False)
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

    def __hash__(self):
        """Hashes on type and value."""
        return hash((type(self), self.rho))

    def __truediv__(self, other) -> "RhoZCDPBudget":
        """Divide this budget by a finite integer/float value > 0."""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot divide a RhoZCDPBudget by a {type(other)}.")
        if other <= 0 or math.isnan(other) or math.isinf(other):
            raise ValueError(
                f"Tried to divide a privacy budget by {other}, but can only "
                "divide by non-infinite numbers >0."
            )
        return RhoZCDPBudget(self.rho / other)

    def __mul__(self, other) -> "RhoZCDPBudget":
        """Multiply this budget by a finite integer/float value >= 0."""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot multiply a RhoZCDPBudget by a {type(other)}.")
        if other < 0 or math.isnan(other) or math.isinf(other):
            raise ValueError(
                f"Tried to multiply a privacy budget by {other}, but can only "
                "multiply by non-infinite numbers >=0."
            )
        return RhoZCDPBudget(self.rho * other)

    def __add__(self, other) -> "RhoZCDPBudget":
        """Add this budget to another RhoZCDPBudget.

        The resulting rho must be greater than zero.
        """
        if not isinstance(other, RhoZCDPBudget):
            raise TypeError(f"Cannot add a RhoZCDPBudget to a {type(other)}.")
        if self.is_infinite:
            return self
        if other.is_infinite:
            return other
        return RhoZCDPBudget(self.rho + other.rho)

    def __sub__(self, other) -> "RhoZCDPBudget":
        """Subtract a RhoZCDPBudget from this budget.

        The resulting rho must be greater than zero. Subtracting anything
        from an infinite budget will return an infinite budget.
        """
        if not isinstance(other, RhoZCDPBudget):
            raise TypeError(f"Cannot subtract a {type(other)} from a RhoZCDPBudget.")
        if self.is_infinite:
            return self
        adjusted_other = _get_adjusted_budget(other, self)
        assert isinstance(adjusted_other, RhoZCDPBudget)
        return RhoZCDPBudget(self.rho - adjusted_other.rho)

    @classmethod
    def inf(cls) -> "RhoZCDPBudget":
        """Get an infinite budget of this type."""
        return RhoZCDPBudget(ExactNumber.from_float(float("inf"), round_up=False))


_BUDGET_RELATIVE_TOLERANCE: sp.Expr = sp.Pow(10, 9)


def _requested_budget_is_slightly_higher_than_remaining(
    requested_budget: ExactNumber, remaining_budget: ExactNumber
) -> bool:
    """Returns True if requested budget is slightly larger than remaining.

    This check uses a relative tolerance, i.e., it determines if the requested
    budget is within X% of the remaining budget.

    Args:
        requested_budget: Exact representation of requested budget.
        remaining_budget: Exact representation of how much budget we have left.
    """
    if not remaining_budget.is_finite:
        return False

    diff = remaining_budget - requested_budget
    if diff >= 0:
        return False
    return abs(diff) <= remaining_budget / _BUDGET_RELATIVE_TOLERANCE


@typechecked
def _get_adjusted_budget_number(
    requested_budget: ExactNumber, remaining_budget: ExactNumber
) -> ExactNumber:
    """Converts a requested int or float budget into an adjusted budget.

    If the requested budget is "slightly larger" than the remaining budget, as
    determined by some threshold, then we round down and consume all remaining
    budget. The goal is to accommodate some degree of floating point imprecision by
    erring on the side of providing a slightly stronger privacy guarantee
    rather than declining the request altogether.

    Args:
        requested_budget: The numeric value of the requested budget.
        remaining_budget: The numeric value of how much budget we have left.
    """
    if _requested_budget_is_slightly_higher_than_remaining(
        requested_budget, remaining_budget
    ):
        return remaining_budget

    return requested_budget


@typechecked
def _get_adjusted_budget(
    requested_privacy_budget: PrivacyBudget, remaining_privacy_budget: PrivacyBudget
) -> PrivacyBudget:
    """Converts a requested privacy budget into an adjusted privacy budget.

    For each term in the privacy budget, calls _get_adjusted_budget_number to adjust
    the requested budget slightly if it's close enough to the remaining budget.

    Args:
        requested_privacy_budget: The requested privacy budget.
        remaining_privacy_budget: How much privacy budget we have left.
    """
    # pylint: disable=protected-access
    if isinstance(requested_privacy_budget, PureDPBudget) and isinstance(
        remaining_privacy_budget, PureDPBudget
    ):
        adjusted_epsilon = _get_adjusted_budget_number(
            requested_privacy_budget._epsilon, remaining_privacy_budget._epsilon
        )
        return PureDPBudget(adjusted_epsilon)

    elif isinstance(requested_privacy_budget, ApproxDPBudget) and isinstance(
        remaining_privacy_budget, ApproxDPBudget
    ):
        adjusted_epsilon = _get_adjusted_budget_number(
            requested_privacy_budget._epsilon, remaining_privacy_budget._epsilon
        )
        adjusted_delta = _get_adjusted_budget_number(
            requested_privacy_budget._delta, remaining_privacy_budget._delta
        )
        return ApproxDPBudget(adjusted_epsilon, adjusted_delta)

    elif isinstance(requested_privacy_budget, RhoZCDPBudget) and isinstance(
        remaining_privacy_budget, RhoZCDPBudget
    ):
        adjusted_rho = _get_adjusted_budget_number(
            requested_privacy_budget._rho, remaining_privacy_budget._rho
        )
        return RhoZCDPBudget(adjusted_rho)
    # pylint: enable=protected-access
    else:
        raise ValueError(
            "Unable to compute a privacy budget with the requested budget "
            f"of {requested_privacy_budget} and a remaining budget of "
            f"{remaining_privacy_budget}."
        )
