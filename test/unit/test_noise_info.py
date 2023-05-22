"""Unit tests for noise info."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import Dict

import pytest
import sympy as sp
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.series import NoisyQuantile
from tmlt.core.measures import PureDP

from tmlt.analytics._noise_info import (
    _inverse_cdf,
    _noise_from_measurement,
    _NoiseMechanism,
)


@pytest.mark.parametrize(
    "measurement,expected",
    [
        (
            AddLaplaceNoise(NumpyIntegerDomain(), scale=sp.Rational(2.5)),
            [{"noise_mechanism": _NoiseMechanism.LAPLACE, "noise_parameter": 2.5}],
        ),
        (
            AddGeometricNoise(alpha=sp.Rational(3.5)),
            [{"noise_mechanism": _NoiseMechanism.GEOMETRIC, "noise_parameter": 3.5}],
        ),
        (
            AddDiscreteGaussianNoise(sigma_squared=sp.Rational(4.5)),
            [
                {
                    "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                    "noise_parameter": 4.5,
                }
            ],
        ),
        (
            NoisyQuantile(
                PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                PureDP(),
                quantile=0.5,
                lower=0,
                upper=10,
                epsilon=sp.Rational(5.5),
            ),
            [{"noise_mechanism": _NoiseMechanism.EXPONENTIAL, "noise_parameter": 5.5}],
        ),
    ],
)
def test_noise_from_measurement(measurement: Measurement, expected: Dict):
    """Get noise from measurement."""
    noise_info = _noise_from_measurement(measurement)
    assert noise_info == expected


@pytest.mark.parametrize(
    "noise_info,p,expected",
    [
        ({"noise_mechanism": _NoiseMechanism.LAPLACE, "noise_parameter": 1}, 0.5, 0.0),
        (
            {"noise_mechanism": _NoiseMechanism.LAPLACE, "noise_parameter": 1},
            0.75,
            0.693147,
        ),
        (
            {"noise_mechanism": _NoiseMechanism.GEOMETRIC, "noise_parameter": 1},
            0.5,
            0.0,
        ),
        ({"noise_mechanism": _NoiseMechanism.GEOMETRIC, "noise_parameter": 1}, 0.75, 1),
        (
            {
                "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                "noise_parameter": 1,
            },
            0.5,
            0.0,
        ),
        (
            {
                "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                "noise_parameter": 1,
            },
            0.75,
            1,
        ),
    ],
)
def test_inverse_cdf(noise_info: Dict, p: float, expected: float):
    """Inverse CDF from noise_info."""
    result = _inverse_cdf(noise_info, p)
    assert result == pytest.approx(expected)
