"""Get noise scale from a Measurement."""

from copy import deepcopy
from enum import Enum
from functools import singledispatch
from typing import Any, Dict, List, Set, Tuple, Union

from pyspark.sql import DataFrame
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.series import NoisyQuantile
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber


class _NoiseMechanism(Enum):
    LAPLACE = 1
    GEOMETRIC = 2
    DISCRETE_GAUSSIAN = 3
    EXPONENTIAL = 4
    GAUSSIAN = 5

    def to_cls(self):
        """Returns the appropriate measurement class for this enum value."""
        if self.value == _NoiseMechanism.LAPLACE.value:
            return AddLaplaceNoise
        if self.value == _NoiseMechanism.GEOMETRIC.value:
            return AddGeometricNoise
        if self.value == _NoiseMechanism.DISCRETE_GAUSSIAN.value:
            return AddDiscreteGaussianNoise
        if self.value == _NoiseMechanism.EXPONENTIAL.value:
            return NoisyQuantile
        if self.value == _NoiseMechanism.GAUSSIAN.value:
            return AddGaussianNoise
        raise KeyError("Unknown measurement type.")


@singledispatch
def _get_info(a: Any) -> Any:
    """Get information from a measurement or transformation.

    Output will look a lot like input's __dict__, except that:
    - Measurements become dictionaries (via _get_info)
    - Transformations become dictionaries (via _get_info)
    - ExactNumbers become floats (via .to_float(round_up=False))

    Deep-copies are used, so you can change elements of this informational
    dictionary without changing the original object.
    """
    if hasattr(a, "__dict__"):
        return {k: _get_info(v) for k, v in a.__dict__.items()}
    return deepcopy(a)


@_get_info.register(Measurement)
@_get_info.register(Transformation)
def _(mt: Union[Measurement, Transformation]) -> Dict[str, Any]:
    """Get a dictionary of information about a measurement or transformation."""
    d: Dict[str, Any] = {k: _get_info(v) for k, v in mt.__dict__.items()}
    d["name"] = type(mt).__name__
    return d


@_get_info.register(ExactNumber)
def _(e: ExactNumber) -> float:
    return e.to_float(round_up=False)


@_get_info.register(list)
def _(l: List[Any]) -> List[Any]:
    return [_get_info(e) for e in l]


@_get_info.register(dict)
def _(d: Dict[Any, Any]) -> Dict[Any, Any]:
    return {k: _get_info(v) for k, v in d.items()}


@_get_info.register(tuple)
def _(t: Tuple) -> Tuple:
    return tuple(_get_info(e) for e in t)


@_get_info.register(set)
def _(s: Set) -> Set:
    return set(_get_info(e) for e in s)


@_get_info.register(DataFrame)
def _(df: DataFrame) -> str:
    # Deepcopying dataframes doesn't work
    # (If you try, you'll see an error about being unable to
    # pickle threads.)
    # for now, just report that there was a DataFrame here
    return f"<a Spark DataFrame with columns: {df.columns}>"


def _noise_from_measurement(m: Measurement) -> List[Dict[str, Any]]:
    """Get noise information from a measurement.

    Each dictionary will look like:
    {"noise_mechanism": _NoiseMechanism.LAPLACE, "noise_parameter": 1}
    """
    return _noise_from_info(_get_info(m))


def _inverse_cdf(noise_info: Dict[str, Any], p: float) -> float:
    """Get the inverse cdf of a measurement at a probability.

    Args:
        noise_info: A dictionary of the type returned by _noise_from_measurement.
        p: The probability at which to calculate the inverse cdf.
    """
    noise_cls = noise_info["noise_mechanism"].to_cls()
    return noise_cls.inverse_cdf(noise_info["noise_parameter"], p)


@singledispatch
def _noise_from_info(
    info: Any,  # pylint: disable=unused-argument
) -> List[Dict[str, Any]]:
    """Get noise information from info (for a measurement).

    Each dictionary will look like:
    {"noise_mechanism": _NoiseMechanism.LAPLACE, "noise_parameter": 1}
    """
    return []


@_noise_from_info.register(dict)
def _(info: Dict[Any, Any]) -> List[Dict[str, Any]]:
    name = info.get("name")
    if name == "AddLaplaceNoise":
        return [
            {
                "noise_mechanism": _NoiseMechanism.LAPLACE,
                "noise_parameter": info["_scale"],
            }
        ]
    if name == "AddDiscreteGaussianNoise":
        return [
            {
                "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                "noise_parameter": info["_sigma_squared"],
            }
        ]
    if name == "AddGaussianNoise":
        return [
            {
                "noise_mechanism": _NoiseMechanism.GAUSSIAN,
                "noise_parameter": info["_sigma_squared"],
            }
        ]
    if name == "AddGeometricNoise":
        return [
            {
                "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                "noise_parameter": info["_alpha"],
            }
        ]
    if name == "NoisyQuantile":
        return [
            {
                "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                "noise_parameter": info["_epsilon"],
            }
        ]
    out: List[Dict[str, Any]] = []
    for v in info.values():
        out += _noise_from_info(v)
    return out


@_noise_from_info.register(list)
@_noise_from_info.register(tuple)
@_noise_from_info.register(set)
def _(l: Union[List[Any], Tuple, Set[Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in l:
        out += _noise_from_info(e)
    return out
