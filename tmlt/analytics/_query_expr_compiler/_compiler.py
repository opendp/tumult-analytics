"""Defines :class:`QueryExprCompiler` for compiling query expressions into a measurement.
"""  # pylint: disable=line-too-long
# TODO(#746): Check for UnaryExpr.
# TODO(#1044): Put the PureDPToZCDP converter directly on Vector measurements.
# TODO(#1547): Associate metric with output measure instead of algorithm for
#              adding noise.

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Dict, List, Sequence, Union

import sympy as sp
from pyspark.sql import DataFrame

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._measurement_visitor import MeasurementVisitor
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics.query_expr import QueryExpr
from tmlt.core.domains.collections import DictDomain
from tmlt.core.measurements.aggregations import NoiseMechanism as CoreNoiseMechanism
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.composition import Composition
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric
from tmlt.core.transformations.base import Transformation

DEFAULT_MECHANISM = "DEFAULT"
"""Constant used for DEFAULT noise mechanism"""

LAPLACE_MECHANISM = "LAPLACE"
"""Constant used for LAPLACE noise mechanism"""

GAUSSIAN_MECHANISM = "GAUSSIAN"
"""Constant used for GAUSSIAN noise mechanism"""


class QueryExprCompiler:
    r"""Compiles a list of query expressions to a single measurement object.

    Requires that each query is a groupby-aggregation on a sequence of transformations
    on a PrivateSource or PrivateView. If there is a PrivateView, the stability of the
    view is handled when the noise scale is calculated.

    A QueryExprCompiler object compiles a list of
    :class:`~tmlt.analytics.query_expr.QueryExpr` objects into
    a single  object (based on the privacy framework). The
    :class:`~tmlt.core.measurements.base.Measurement` object can be
    run with a private data source to obtain DP answers to supplied queries.

    Supported :class:`~tmlt.analytics.query_expr.QueryExpr`\ s:

    * :class:`~tmlt.analytics.query_expr.PrivateSource`
    * :class:`~tmlt.analytics.query_expr.Filter`
    * :class:`~tmlt.analytics.query_expr.FlatMap`
    * :class:`~tmlt.analytics.query_expr.Map`
    * :class:`~tmlt.analytics.query_expr.Rename`
    * :class:`~tmlt.analytics.query_expr.Select`
    * :class:`~tmlt.analytics.query_expr.JoinPublic`
    * :class:`~tmlt.analytics.query_expr.JoinPrivate`
    * :class:`~tmlt.analytics.query_expr.GroupByCount`
    * :class:`~tmlt.analytics.query_expr.GroupByCountDistinct`
    * :class:`~tmlt.analytics.query_expr.GroupByBoundedSum`
    * :class:`~tmlt.analytics.query_expr.GroupByBoundedAverage`
    * :class:`~tmlt.analytics.query_expr.GroupByBoundedSTDEV`
    * :class:`~tmlt.analytics.query_expr.GroupByBoundedVariance`
    * :class:`~tmlt.analytics.query_expr.GroupByQuantile`
    """

    def __init__(self, output_measure: Union[PureDP, RhoZCDP] = PureDP()):
        """Constructor.

        Args:
            output_measure: Distance measure for measurement's output.
        """
        # TODO(#1547): Can be removed when issue is resolved.
        self._mechanism = (
            CoreNoiseMechanism.LAPLACE
            if isinstance(output_measure, PureDP)
            else CoreNoiseMechanism.DISCRETE_GAUSSIAN
        )
        self._output_measure = output_measure

    @property
    def mechanism(self) -> CoreNoiseMechanism:
        """Return the value of Core noise mechanism."""
        return self._mechanism

    @mechanism.setter
    def mechanism(self, value):
        """Set the value of Core noise mechanism."""
        self._mechanism = value

    @property
    def output_measure(self) -> Union[PureDP, RhoZCDP]:
        """Return the distance measure for the measurement's output."""
        return self._output_measure

    def __call__(
        self,
        queries: Sequence[QueryExpr],
        privacy_budget: sp.Expr,
        stability: Dict[str, sp.Expr],
        input_domain: DictDomain,
        input_metric: DictMetric,
        public_sources: Dict[str, DataFrame],
        catalog: Catalog,
    ) -> Measurement:
        """Returns a compiled DP measurement.

        Args:
            queries: Queries representing measurements to compile.
            privacy_budget: The total privacy budget for answering the queries.
            stability: The stability of the input to compiled query.
            input_domain: The input domain of the compiled query.
            input_metric: The input metric of the compiled query.
            public_sources: Public data sources for the queries.
            catalog: The catalog, used only for query validation.
        """
        if len(queries) == 0:
            raise ValueError("At least one query needs to be provided")

        measurements: List[Measurement] = []
        per_query_privacy_budget = privacy_budget / len(queries)
        visitor = MeasurementVisitor(
            per_query_privacy_budget=per_query_privacy_budget,
            stability=stability,
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=self._output_measure,
            default_mechanism=self._mechanism,
            public_sources=public_sources,
            catalog=catalog,
        )
        for query in queries:
            query_measurement = query.accept(visitor)
            if not isinstance(query_measurement, Measurement):
                raise AssertionError(
                    "This query did not create a measurement. "
                    "This is probably a bug; please let us know so we can fix it!"
                )
            if (
                query_measurement.privacy_function(stability)
                != per_query_privacy_budget
            ):
                raise AssertionError(
                    "Query measurement privacy function does not match "
                    "per-query privacy budget. This is probably a bug; "
                    "please let us know so we can fix it!"
                )
            measurements.append(query_measurement)

        measurement = Composition(measurements)
        if measurement.privacy_function(stability) != privacy_budget:
            raise AssertionError(
                "Measurement privacy function does not match "
                "privacy budget. This is probably a bug; "
                "please let us know so we can fix it!"
            )
        return measurement

    def build_transformation(
        self,
        query: QueryExpr,
        input_domain: DictDomain,
        input_metric: DictMetric,
        public_sources: Dict[str, DataFrame],
        catalog: Catalog,
    ) -> Transformation:
        r"""Returns a transformation for the query.

        Supported
        :class:`~tmlt.analytics.query_expr.QueryExpr`\ s:

        * :class:`~tmlt.analytics.query_expr.Filter`
        * :class:`~tmlt.analytics.query_expr.FlatMap`
        * :class:`~tmlt.analytics.query_expr.JoinPrivate`
        * :class:`~tmlt.analytics.query_expr.JoinPublic`
        * :class:`~tmlt.analytics.query_expr.Map`
        * :class:`~tmlt.analytics.query_expr.PrivateSource`
        * :class:`~tmlt.analytics.query_expr.Rename`
        * :class:`~tmlt.analytics.query_expr.Select`

        Args:
            query: A query representing a transformation to compile.
            input_domain: The input domain of the compiled query.
            input_metric: The input metric of the compiled query.
            public_sources: Public data sources for the queries.
            catalog: The catalog, used only for query validation.
        """
        query.accept(OutputSchemaVisitor(catalog))

        transformation_visitor = TransformationVisitor(
            input_domain=input_domain,
            input_metric=input_metric,
            mechanism=self.mechanism,
            public_sources=public_sources,
        )
        transformation = query.accept(transformation_visitor)
        if not isinstance(transformation, Transformation):
            raise AssertionError(
                "Unable to create transformation. This is probably "
                "a bug; please let us know about it so we can fix it!"
            )
        transformation_visitor.validate_transformation(query, transformation, catalog)
        return transformation
