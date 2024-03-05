"""Defines a visitor for creating noisy measurements from query expressions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from typing import List, Tuple

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_partition_selection_measurement,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.metrics import HammingDistance, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.converters import UnwrapIfGroupedBy
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.misc import get_nonconflicting_string

from tmlt.analytics._noise_info import NoiseInfo, _noise_from_measurement
from tmlt.analytics._query_expr_compiler._base_measurement_visitor import (
    BaseMeasurementVisitor,
)
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics._transformation_utils import get_table_from_ref
from tmlt.analytics.constraints import Constraint
from tmlt.analytics.privacy_budget import ApproxDPBudget
from tmlt.analytics.query_expr import GetGroups, QueryExpr


class MeasurementVisitor(BaseMeasurementVisitor):
    """A visitor to create a measurement from a DP query expression."""

    def _visit_child_transformation(
        self, expr: QueryExpr, mechanism: NoiseMechanism
    ) -> Tuple[Transformation, TableReference, List[Constraint]]:
        """Visit a child transformation, producing a transformation."""
        tv = TransformationVisitor(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            mechanism=mechanism,
            public_sources=self.public_sources,
            table_constraints=self.table_constraints,
        )
        child, reference, constraints = expr.accept(tv)

        tv.validate_transformation(expr, child, reference, self.catalog)

        return child, reference, constraints

    def _handle_enforce(
        self,
        constraint: Constraint,
        child_transformation: Transformation,
        child_ref: TableReference,
        **kwargs,
    ) -> Tuple[Transformation, TableReference]:
        """Enforce a constraint after a child transformation."""
        return constraint._enforce(  # pylint: disable=protected-access
            child_transformation, child_ref, **kwargs
        )

    def visit_get_groups(self, expr: GetGroups) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GetGroups query expression."""
        if not isinstance(self.budget, ApproxDPBudget):
            raise ValueError("GetGroups is only supported with ApproxDPBudgets.")

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))

        schema = expr.child.accept(OutputSchemaVisitor(self.catalog))

        # Set the columns if no columns were provided.
        if expr.columns:
            columns = expr.columns
        else:
            columns = [
                col for col in schema.column_descs.keys() if col != schema.id_column
            ]

        # Check if ID column is one of the columns in get_groups
        # Note: if get_groups columns is None or empty, all of the columns in the table
        # is used for partition selection, hence that needs to be checked as well
        if schema.id_column and (not columns or (schema.id_column in columns)):
            raise RuntimeError(
                "get_groups cannot be used on the privacy ID column"
                f" ({schema.id_column}) of a table with the AddRowsWithID protected"
                " change."
            )

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, NoiseMechanism.GEOMETRIC),
            grouping_columns=[],
        )

        transformation = get_table_from_ref(child_transformation, child_ref)
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)

        # squares the sensitivity in zCDP, which is a worst-case analysis
        # that we may be able to improve.
        if isinstance(transformation.output_metric, IfGroupedBy):
            transformation |= UnwrapIfGroupedBy(
                transformation.output_domain, transformation.output_metric
            )

        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        transformation |= SelectTransformation(
            transformation.output_domain, transformation.output_metric, columns
        )

        mid_stability = transformation.stability_function(self.stability)
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        count_column = "count"
        if count_column in set(transformation.output_domain.schema):
            count_column = get_nonconflicting_string(
                list(transformation.output_domain.schema)
            )

        epsilon, delta = self.budget.value
        agg = create_partition_selection_measurement(
            input_domain=transformation.output_domain,
            epsilon=epsilon,
            delta=delta,
            d_in=mid_stability,
            count_column=count_column,
        )

        self._validate_measurement(agg, mid_stability)

        measurement = PostProcess(
            transformation | agg, lambda result: result.drop(count_column)
        )
        noise_info = _noise_from_measurement(measurement)
        return measurement, noise_info
