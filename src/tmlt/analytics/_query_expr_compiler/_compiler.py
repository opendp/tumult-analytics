"""Defines :class:`QueryExprCompiler` for building transformations from query exprs."""
# TODO(#746): Check for UnaryExpr.
# TODO(#1044): Put the PureDPToZCDP converter directly on Vector measurements.
# TODO(#1547): Associate metric with output measure instead of algorithm for
#              adding noise.

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Dict, List, Tuple, Union

from pyspark.sql import DataFrame
from tmlt.core.domains.collections import DictDomain
from tmlt.core.measurements.aggregations import NoiseMechanism as CoreNoiseMechanism
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric
from tmlt.core.transformations.base import Transformation

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._noise_info import NoiseInfo
from tmlt.analytics._query_expr import QueryExpr
from tmlt.analytics._query_expr_compiler._measurement_visitor import MeasurementVisitor
from tmlt.analytics._query_expr_compiler._rewrite_rules import CompilationInfo, rewrite
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._schema import Schema
from tmlt.analytics._table_identifier import Identifier
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics.constraints import Constraint
from tmlt.analytics.privacy_budget import PrivacyBudget

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
    :class:`~tmlt.analytics._query_expr.QueryExpr` objects into
    a single  object (based on the privacy framework). The
    :class:`~tmlt.core.measurements.base.Measurement` object can be
    run with a private data source to obtain DP answers to supplied queries.

    Supported :class:`~tmlt.analytics._query_expr.QueryExpr`\ s:

    * :class:`~tmlt.analytics._query_expr.PrivateSource`
    * :class:`~tmlt.analytics._query_expr.Filter`
    * :class:`~tmlt.analytics._query_expr.FlatMap`
    * :class:`~tmlt.analytics._query_expr.Map`
    * :class:`~tmlt.analytics._query_expr.Rename`
    * :class:`~tmlt.analytics._query_expr.Select`
    * :class:`~tmlt.analytics._query_expr.JoinPublic`
    * :class:`~tmlt.analytics._query_expr.JoinPrivate`
    * :class:`~tmlt.analytics._query_expr.GroupByCount`
    * :class:`~tmlt.analytics._query_expr.GroupByCountDistinct`
    * :class:`~tmlt.analytics._query_expr.GroupByBoundedSum`
    * :class:`~tmlt.analytics._query_expr.GroupByBoundedAverage`
    * :class:`~tmlt.analytics._query_expr.GroupByBoundedStdev`
    * :class:`~tmlt.analytics._query_expr.GroupByBoundedVariance`
    * :class:`~tmlt.analytics._query_expr.GroupByQuantile`
    """

    def __init__(self, output_measure: Union[PureDP, ApproxDP, RhoZCDP] = PureDP()):
        """Constructor.

        Args:
            output_measure: Distance measure for measurement's output.
        """
        # TODO(#1547): Can be removed when issue is resolved.
        self._mechanism = (
            CoreNoiseMechanism.LAPLACE
            if isinstance(output_measure, (PureDP, ApproxDP))
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
    def output_measure(self) -> Union[PureDP, ApproxDP, RhoZCDP]:
        """Return the distance measure for the measurement's output."""
        return self._output_measure

    @staticmethod
    def query_schema(query: QueryExpr, catalog: Catalog) -> Schema:
        """Return the schema created by a given query."""
        schema = query.schema(catalog)
        if not isinstance(schema, Schema):
            raise AnalyticsInternalError(
                "Schema for this query is not a Schema but is instead a(n) "
                f"{type(schema)}."
            )
        return schema

    def __call__(
        self,
        query: QueryExpr,
        privacy_budget: PrivacyBudget,
        stability: Any,
        input_domain: DictDomain,
        input_metric: DictMetric,
        public_sources: Dict[str, DataFrame],
        catalog: Catalog,
        table_constraints: Dict[Identifier, List[Constraint]],
    ) -> Tuple[Measurement, NoiseInfo]:
        """Returns a compiled DP measurement and its noise information.

        Args:
            query: The query representing a measurement to compile.
            privacy_budget: The total privacy budget for answering the queries.
            stability: The stability of the input to compiled query.
            input_domain: The input domain of the compiled query.
            input_metric: The input metric of the compiled query.
            public_sources: Public data sources for the queries.
            catalog: The catalog, used only for query validation.
            table_constraints: A mapping of tables to the existing constraints on them.
        """
        # Computing the schema validates that the query is well-formed.
        query.schema(catalog)

        # Compilation happens in two stages: first, we apply rewrite rules...
        compilation_info = CompilationInfo(
            output_measure=self._output_measure,
            catalog=catalog,
        )
        query = rewrite(compilation_info, query)

        # ... then we visit the query.
        visitor = MeasurementVisitor(
            privacy_budget=privacy_budget,
            stability=stability,
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=self._output_measure,
            default_mechanism=self._mechanism,
            public_sources=public_sources,
            catalog=catalog,
            table_constraints=table_constraints,
        )

        measurement, noise_info = query.accept(visitor)
        if not isinstance(measurement, Measurement):
            raise AnalyticsInternalError("This query did not create a measurement.")

        if isinstance(visitor.adjusted_budget.value, tuple):
            # TODO(#2754): add a log message.
            privacy_function_budget_mismatch = any(
                x > y
                for x, y in zip(
                    measurement.privacy_function(stability),
                    visitor.adjusted_budget.value,
                )
            )
        else:
            privacy_function_budget_mismatch = (
                measurement.privacy_function(stability) != visitor.adjusted_budget.value
            )
        if privacy_function_budget_mismatch:
            raise AnalyticsInternalError(
                "Query measurement privacy function does not match "
                "privacy budget value."
            )

        return measurement, noise_info

    def build_transformation(
        self,
        query: QueryExpr,
        input_domain: DictDomain,
        input_metric: DictMetric,
        public_sources: Dict[str, DataFrame],
        catalog: Catalog,
        table_constraints: Dict[Identifier, List[Constraint]],
    ) -> Tuple[Transformation, TableReference, List[Constraint]]:
        r"""Returns a transformation and reference for the query.

        Supported
        :class:`~tmlt.analytics._query_expr.QueryExpr`\ s:

        * :class:`~tmlt.analytics._query_expr.Filter`
        * :class:`~tmlt.analytics._query_expr.FlatMap`
        * :class:`~tmlt.analytics._query_expr.JoinPrivate`
        * :class:`~tmlt.analytics._query_expr.JoinPublic`
        * :class:`~tmlt.analytics._query_expr.Map`
        * :class:`~tmlt.analytics._query_expr.PrivateSource`
        * :class:`~tmlt.analytics._query_expr.Rename`
        * :class:`~tmlt.analytics._query_expr.Select`

        Args:
            query: A query representing a transformation to compile.
            input_domain: The input domain of the compiled query.
            input_metric: The input metric of the compiled query.
            public_sources: Public data sources for the queries.
            catalog: The catalog, used only for query validation.
            table_constraints: A mapping of tables to the existing constraints on them.
        """
        # Computing the schema validates that the query is well-formed. It's useful to
        # perform this check here in addition to __call__ so validation errors can be
        # raised at view creation, not just query evaluation.
        query.schema(catalog)

        transformation_visitor = TransformationVisitor(
            input_domain=input_domain,
            input_metric=input_metric,
            mechanism=self.mechanism,
            public_sources=public_sources,
            table_constraints=table_constraints,
        )
        transformation, reference, constraints = query.accept(transformation_visitor)
        if not isinstance(transformation, Transformation):
            raise AnalyticsInternalError(
                "Unable to create transformation for this query."
            )
        transformation_visitor.validate_transformation(
            query, transformation, reference, catalog
        )
        return transformation, reference, constraints
