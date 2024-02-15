"""Defines a visitor for creating noisy measurements from query expressions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import dataclasses
from typing import List, Optional, Tuple

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_average_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_partition_selection_measurement,
    create_quantile_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
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

from tmlt.analytics._query_expr_compiler._base_measurement_visitor import (
    BaseMeasurementVisitor,
)
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._schema import ColumnType, Schema
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics._transformation_utils import get_table_from_ref
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.privacy_budget import ApproxDPBudget
from tmlt.analytics.query_expr import (
    CountDistinctMechanism,
    CountMechanism,
    DropNullAndNan,
    EnforceConstraint,
    GetGroups,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    QueryExpr,
    ReplaceInfinity,
)


def _generate_constrained_count_distinct(
    query: GroupByCountDistinct, schema: Schema, constraints: List[Constraint]
) -> Optional[GroupByCount]:
    """Return a more optimal query for the given count-distinct, if one exists.

    This method handles inferring additional constraints on a
    GroupByCountDistinct query and using those constraints to generate more
    optimal queries. This is possible in two cases, both on IDs tables:

    - Only the ID column is being counted, and no groupby is performed. When
      this happens, each ID can contribute at most once to the resulting count,
      equivalent to a ``MaxRowsPerID(1)`` constraint.

    - Only the ID column is being counted, and the result is grouped on exactly
      one column which has a MaxGroupsPerID constraint on it. In this case, each
      ID can contribute at most once to the count of each group, equivalent to a
      ``MaxRowsPerGroupPerID(other_column, 1)`` constraint.

    In both of these cases, a performance optimization is also possible: because
    enforcing the constraints drops all but one of the rows per ID in the first
    case or per (ID, group) value pair in the second, a normal count query will
    produce the same result and should run faster because it doesn't need to
    handle deduplicating the values.
    """
    columns_to_count = set(query.columns_to_count or schema.columns)
    groupby_columns = query.groupby_keys.dataframe().columns

    # For non-IDs cases or cases where columns other than the ID column must be
    # distinct, there's no optimization to make.
    if schema.id_column is None or columns_to_count != {schema.id_column}:
        return None

    mechanism = (
        CountMechanism.DEFAULT
        if query.mechanism == CountDistinctMechanism.DEFAULT
        else CountMechanism.LAPLACE
        if query.mechanism == CountDistinctMechanism.LAPLACE
        else CountMechanism.GAUSSIAN
        if query.mechanism == CountDistinctMechanism.GAUSSIAN
        else None
    )
    if mechanism is None:
        raise AssertionError(
            f"Unknown mechanism {query.mechanism}. This is probably a bug; "
            "please let us know about it so we can fix it!"
        )

    if not groupby_columns:
        # No groupby is performed; this is equivalent to a MaxRowsPerID(1)
        # constraint on the table.
        return GroupByCount(
            EnforceConstraint(query.child, MaxRowsPerID(1)),
            groupby_keys=query.groupby_keys,
            output_column=query.output_column,
            mechanism=mechanism,
        )
    elif len(groupby_columns) == 1:
        # A groupby on exactly one column is performed; if that column has a
        # MaxGroupsPerID constraint, then this is equivalent to a
        # MaxRowsPerGroupsPerID(grouping_column, 1) constraint.
        grouping_column = groupby_columns[0]
        constraint = next(
            (
                c
                for c in constraints
                if isinstance(c, MaxGroupsPerID)
                and c.grouping_column == grouping_column
            ),
            None,
        )
        if constraint is not None:
            return GroupByCount(
                EnforceConstraint(
                    query.child, MaxRowsPerGroupPerID(constraint.grouping_column, 1)
                ),
                groupby_keys=query.groupby_keys,
                output_column=query.output_column,
                mechanism=mechanism,
            )

    # If none of the above cases are true, no optimization is possible.
    return None


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

    def visit_get_groups(self, expr: GetGroups) -> Measurement:
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
        return measurement

    def visit_groupby_count(self, expr: GroupByCount) -> Measurement:
        """Create a measurement from a GroupByCount query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_count(expr)
        mechanism = self._pick_noise_for_count(expr)
        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, mechanism),
            grouping_columns=expr.groupby_keys.dataframe().columns,
        )

        transformation = get_table_from_ref(child_transformation, child_ref)
        # _visit_child_transformation already raises an error if these aren't true
        # these are just here for MyPy's benefit
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )
        mid_stability = transformation.stability_function(self.stability)
        groupby = self._build_groupby(
            transformation.output_domain,
            transformation.output_metric,
            expr.groupby_keys,
            mechanism,
        )

        agg = create_count_measurement(
            input_domain=transformation.output_domain,
            input_metric=transformation.output_metric,
            noise_mechanism=mechanism,
            d_in=mid_stability,
            d_out=self.adjusted_budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            count_column=expr.output_column,
        )
        self._validate_measurement(agg, mid_stability)
        return transformation | agg

    def visit_groupby_count_distinct(self, expr: GroupByCountDistinct) -> Measurement:
        """Create a measurement from a GroupByCountDistinct query expression."""
        self._validate_approxDP_and_adjust_budget(expr)
        mechanism = self._pick_noise_for_count(expr)
        (
            child_transformation,
            child_ref,
            child_constraints,
        ) = self._visit_child_transformation(expr.child, mechanism)
        constrained_query = _generate_constrained_count_distinct(
            expr,
            expr.child.accept(OutputSchemaVisitor(self.catalog)),
            child_constraints,
        )
        if constrained_query is not None:
            return constrained_query.accept(self)

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_count_distinct(expr)

        child_transformation, child_ref = self._truncate_table(
            child_transformation,
            child_ref,
            child_constraints,
            grouping_columns=expr.groupby_keys.dataframe().columns,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)

        # _visit_child_transformation already raises an error if these aren't true
        # these are just here for MyPy's benefit
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )
        # If not counting all columns, drop the ones that are neither counted
        # nor grouped on.
        if expr.columns_to_count:
            groupby_columns = list(expr.groupby_keys.schema().keys())
            transformation |= SelectTransformation(
                transformation.output_domain,
                transformation.output_metric,
                list(set(expr.columns_to_count + groupby_columns)),
            )
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        mid_stability = transformation.stability_function(self.stability)
        groupby = self._build_groupby(
            transformation.output_domain,
            transformation.output_metric,
            expr.groupby_keys,
            mechanism,
        )

        agg = create_count_distinct_measurement(
            input_domain=transformation.output_domain,
            input_metric=transformation.output_metric,
            noise_mechanism=mechanism,
            d_in=mid_stability,
            d_out=self.adjusted_budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            count_column=expr.output_column,
        )
        self._validate_measurement(agg, mid_stability)
        return transformation | agg

    def visit_groupby_quantile(self, expr: GroupByQuantile) -> Measurement:
        """Create a measurement from a GroupByQuantile query expression.

        This method also checks to see if the schema allows invalid values
        (nulls, NaNs, and infinite values) on the measure column; if so,
        the query has DropNullAndNan and/or ReplaceInfinity queries
        inserted immediately before it is executed.
        """
        child_schema: Schema = expr.child.accept(OutputSchemaVisitor(self.catalog))
        # Check the measure column for nulls/NaNs/infs (which aren't allowed)
        try:
            measure_desc = child_schema[expr.measure_column]
        except KeyError as e:
            raise KeyError(
                f"Measure column '{expr.measure_column}' is not in the input schema."
            ) from e
        # If null or NaN values are allowed ...
        if measure_desc.allow_null or (
            measure_desc.column_type == ColumnType.DECIMAL and measure_desc.allow_nan
        ):
            # Those values aren't allowed! Drop them
            # (without mutating the original QueryExpr)
            drop_null_and_nan_query = DropNullAndNan(
                child=expr.child, columns=[expr.measure_column]
            )
            expr = dataclasses.replace(expr, child=drop_null_and_nan_query)

        # If infinite values are allowed ...
        if measure_desc.column_type == ColumnType.DECIMAL and measure_desc.allow_inf:
            # Clamp those values
            # (without mutating the original QueryExpr)
            replace_infinity_query = ReplaceInfinity(
                child=expr.child,
                replace_with={expr.measure_column: (expr.low, expr.high)},
            )
            expr = dataclasses.replace(expr, child=replace_infinity_query)

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_quantile(expr)

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, self.default_mechanism),
            grouping_columns=expr.groupby_keys.dataframe().columns,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)
        # _visit_child_transformation already raises an error if these aren't true
        # these are just here for MyPy's benefit
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )
        mid_stability = transformation.stability_function(self.stability)
        groupby = self._build_groupby(
            transformation.output_domain,
            transformation.output_metric,
            expr.groupby_keys,
            self.default_mechanism,
        )

        # For ApproxDP keep epsilon value, but always pass 0 for delta
        self.adjusted_budget = (
            ApproxDPBudget(self.budget.value[0], 0)
            if isinstance(self.budget, ApproxDPBudget)
            else self.budget
        )

        agg = create_quantile_measurement(
            input_domain=transformation.output_domain,
            input_metric=transformation.output_metric,
            measure_column=expr.measure_column,
            quantile=expr.quantile,
            lower=expr.low,
            upper=expr.high,
            d_in=mid_stability,
            d_out=self.adjusted_budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            quantile_column=expr.output_column,
        )
        self._validate_measurement(agg, mid_stability)
        return transformation | agg

    def visit_groupby_bounded_sum(self, expr: GroupByBoundedSum) -> Measurement:
        """Create a measurement from a GroupByBoundedSum query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_sum(expr)

        info = self._build_common(expr)
        # _build_common already checks these;
        # these asserts are just for mypy's benefit
        assert isinstance(info.transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            info.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        agg = create_sum_measurement(
            input_domain=info.transformation.output_domain,
            input_metric=info.transformation.output_metric,
            measure_column=expr.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.adjusted_budget.value,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            sum_column=expr.output_column,
        )
        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg

    def visit_groupby_bounded_average(self, expr: GroupByBoundedAverage) -> Measurement:
        """Create a measurement from a GroupByBoundedAverage query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_average(expr)
        info = self._build_common(expr)
        # _visit_child_transformation already raises an error if these aren't true
        # these are just here for MyPy's benefit
        assert isinstance(info.transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            info.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        agg = create_average_measurement(
            input_domain=info.transformation.output_domain,
            input_metric=info.transformation.output_metric,
            measure_column=expr.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.adjusted_budget.value,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            average_column=expr.output_column,
        )
        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg

    def visit_groupby_bounded_variance(
        self, expr: GroupByBoundedVariance
    ) -> Measurement:
        """Create a measurement from a GroupByBoundedVariance query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_variance(expr)
        info = self._build_common(expr)
        # _visit_child_transformation already raises an error if these aren't true
        # these are just here for MyPy's benefit
        assert isinstance(info.transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            info.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        agg = create_variance_measurement(
            input_domain=info.transformation.output_domain,
            input_metric=info.transformation.output_metric,
            measure_column=expr.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.adjusted_budget.value,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            variance_column=expr.output_column,
        )
        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg

    def visit_groupby_bounded_stdev(self, expr: GroupByBoundedSTDEV) -> Measurement:
        """Create a measurement from a GroupByBoundedStdev query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_stdev(expr)
        info = self._build_common(expr)
        # _visit_child_transformation already raises an error if these aren't true
        # these are just here for MyPy's benefit
        assert isinstance(info.transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            info.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        agg = create_standard_deviation_measurement(
            input_domain=info.transformation.output_domain,
            input_metric=info.transformation.output_metric,
            measure_column=expr.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.adjusted_budget.value,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            standard_deviation_column=expr.output_column,
        )

        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg
