"""Rewrite rules for selecting aggregation noise mechanisms."""

# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Callable, Union

from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP

from tmlt.analytics._query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
    GroupByBoundedAverage,
    GroupByBoundedStdev,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    QueryExpr,
    StdevMechanism,
    SumMechanism,
    VarianceMechanism,
    sums_as_float,
)

from ._base import CompilationInfo, depth_first


def select_noise_mechanism(info: CompilationInfo) -> Callable[[QueryExpr], QueryExpr]:
    """Changes the default noise type into a concrete noise type for aggregations."""

    def select_noise_for_count(
        info: CompilationInfo, expr: Union[GroupByCount, GroupByCountDistinct]
    ) -> QueryExpr:
        mechanism = expr.mechanism
        if mechanism in (CountMechanism.DEFAULT, CountDistinctMechanism.DEFAULT):
            core_mechanism = (
                NoiseMechanism.GEOMETRIC
                if isinstance(info.output_measure, (PureDP, ApproxDP))
                else NoiseMechanism.DISCRETE_GAUSSIAN
            )
        elif mechanism in (CountMechanism.LAPLACE, CountDistinctMechanism.LAPLACE):
            core_mechanism = NoiseMechanism.GEOMETRIC
        elif mechanism in (
            CountMechanism.GAUSSIAN,
            CountDistinctMechanism.GAUSSIAN,
        ):
            if not isinstance(info.output_measure, RhoZCDP):
                raise ValueError(
                    "Gaussian noise is only supported when using a RhoZCDP budget. "
                    "Use Laplace noise instead, or switch to RhoZCDP."
                )
            core_mechanism = NoiseMechanism.DISCRETE_GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize the mechanism name {mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE, and GAUSSIAN."
            )
        return replace(expr, core_mechanism=core_mechanism)

    def select_noise_for_non_count(
        info: CompilationInfo,
        expr: Union[
            GroupByBoundedAverage,
            GroupByBoundedStdev,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ],
    ) -> QueryExpr:
        mechanism = expr.mechanism
        # Distinguish between Laplace/Geometric or (Discrete) Gaussian.
        # Assume floating-point output column type at first
        if mechanism in (
            SumMechanism.DEFAULT,
            AverageMechanism.DEFAULT,
            VarianceMechanism.DEFAULT,
            StdevMechanism.DEFAULT,
        ):
            core_mechanism = (
                NoiseMechanism.LAPLACE
                if isinstance(info.output_measure, (PureDP, ApproxDP))
                else NoiseMechanism.GAUSSIAN
            )
        elif mechanism in (
            SumMechanism.LAPLACE,
            AverageMechanism.LAPLACE,
            VarianceMechanism.LAPLACE,
            StdevMechanism.LAPLACE,
        ):
            core_mechanism = NoiseMechanism.LAPLACE
        elif mechanism in (
            SumMechanism.GAUSSIAN,
            AverageMechanism.GAUSSIAN,
            VarianceMechanism.GAUSSIAN,
            StdevMechanism.GAUSSIAN,
        ):
            if not isinstance(info.output_measure, RhoZCDP):
                raise ValueError(
                    "Gaussian noise is only supported when using a RhoZCDP budget. "
                    "Use Laplace noise instead, or switch to RhoZCDP."
                )
            core_mechanism = NoiseMechanism.GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize requested mechanism {mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE, and GAUSSIAN."
            )

        # Sums of integer columns with small enough clamping bounds use integer
        # noise; other sums, average, variance, and stdev use continuous noise.
        if isinstance(expr, GroupByBoundedSum) and not sums_as_float(
            expr, expr.child.schema(info.catalog)
        ):
            core_mechanism = (
                NoiseMechanism.GEOMETRIC
                if core_mechanism == NoiseMechanism.LAPLACE
                else NoiseMechanism.DISCRETE_GAUSSIAN
            )

        return replace(expr, core_mechanism=core_mechanism)

    @depth_first
    def select_noise(expr: QueryExpr) -> QueryExpr:
        if isinstance(expr, (GroupByCount, GroupByCountDistinct)):
            return select_noise_for_count(info, expr)
        if isinstance(
            expr,
            (
                GroupByBoundedAverage,
                GroupByBoundedStdev,
                GroupByBoundedSum,
                GroupByBoundedVariance,
            ),
        ):
            return select_noise_for_non_count(info, expr)
        return expr

    return select_noise
