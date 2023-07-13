"""Defines a visitor for creating a transformation from a query expression."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import dataclasses
import datetime
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

from pyspark.sql import DataFrame
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain, SparkRowDomain
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.metrics import (
    AddRemoveKeys,
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    Metric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.converters import HammingDistanceToSymmetricDifference
from tmlt.core.transformations.dictionary import (
    AugmentDictTransformation,
    CreateDictFromValue,
    Subset,
    create_copy_and_transform_value,
)
from tmlt.core.transformations.identity import Identity as IdentityTransformation
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    DropInfsValue as DropInfsValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    DropNaNsValue as DropNaNsValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    DropNullsValue as DropNullsValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    FilterValue as FilterValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    FlatMapValue as FlatMapValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    MapValue as MapValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    PublicJoinValue as PublicJoinValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    RenameValue as RenameValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    ReplaceInfsValue as ReplaceInfsValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    ReplaceNaNsValue as ReplaceNaNsValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    ReplaceNullsValue as ReplaceNullsValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    SelectValue as SelectValueTransformation,
)
from tmlt.core.transformations.spark_transformations.filter import (
    Filter as FilterTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoin as PrivateJoinTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoinOnKey as PrivateJoinOnKeyTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PublicJoin as PublicJoinTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    TruncationStrategy as CoreTruncationStrategy,
)
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap as FlatMapTransformation,
)
from tmlt.core.transformations.spark_transformations.map import GroupingFlatMap
from tmlt.core.transformations.spark_transformations.map import Map as MapTransformation
from tmlt.core.transformations.spark_transformations.map import (
    RowToRowsTransformation,
    RowToRowTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropInfs as DropInfTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropNaNs as DropNaNsTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropNulls as DropNullsTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    ReplaceInfs as ReplaceInfsTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    ReplaceNaNs as ReplaceNaNsTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    ReplaceNulls as ReplaceNullsTransformation,
)
from tmlt.core.transformations.spark_transformations.rename import (
    Rename as RenameTransformation,
)
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._constraint_propagation import (
    propagate_flat_map,
    propagate_join_private,
    propagate_join_public,
    propagate_map,
    propagate_rename,
    propagate_replace,
    propagate_select,
    propagate_unmodified,
)
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_dataframe_domain_to_analytics_columns,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics._table_identifier import Identifier, TemporaryTable
from tmlt.analytics._table_reference import (
    TableReference,
    find_named_tables,
    find_reference,
    lookup_domain,
    lookup_metric,
)
from tmlt.analytics._transformation_utils import generate_nested_transformation
from tmlt.analytics.constraints import Constraint, simplify_constraints
from tmlt.analytics.query_expr import AnalyticsDefault
from tmlt.analytics.query_expr import DropInfinity as DropInfExpr
from tmlt.analytics.query_expr import DropNullAndNan, EnforceConstraint
from tmlt.analytics.query_expr import Filter as FilterExpr
from tmlt.analytics.query_expr import FlatMap as FlatMapExpr
from tmlt.analytics.query_expr import (
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
)
from tmlt.analytics.query_expr import JoinPrivate as JoinPrivateExpr
from tmlt.analytics.query_expr import JoinPublic as JoinPublicExpr
from tmlt.analytics.query_expr import Map as MapExpr
from tmlt.analytics.query_expr import QueryExpr, QueryExprVisitor
from tmlt.analytics.query_expr import Rename as RenameExpr
from tmlt.analytics.query_expr import ReplaceInfinity, ReplaceNullAndNan
from tmlt.analytics.query_expr import Select as SelectExpr
from tmlt.analytics.truncation_strategy import TruncationStrategy


class TransformationVisitor(QueryExprVisitor):
    """A visitor to create a transformation from a query expression."""

    class Output(NamedTuple):
        """A container for the outputs of the visitor."""

        transformation: Transformation
        reference: TableReference
        constraints: List[Constraint]

    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: DictMetric,
        mechanism: NoiseMechanism,
        public_sources: Dict[str, DataFrame],
        table_constraints: Dict[Identifier, List[Constraint]],
    ):
        """Constructor for a TransformationVisitor.

        Args:
            input_domain: The input domain that the transformation should have.
            input_metric: The input metric that the transformation should have.
            mechanism: The noise mechanism (only used for FlatMaps).
            public_sources: Public sources to use for JoinPublic queries.
            table_constraints: A mapping of tables to the existing constraints on them.
        """
        self.input_domain = input_domain
        self.input_metric = input_metric
        self.mechanism = mechanism
        self.public_sources = public_sources
        self.table_constraints = table_constraints

    def validate_transformation(
        self,
        query: QueryExpr,
        transformation: Transformation,
        reference: TableReference,
        catalog: Catalog,
    ):
        """Ensure that a query's transformation is valid on a given catalog."""
        expected_schema = query.accept(OutputSchemaVisitor(catalog))
        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )

        expected_output_metric: Metric
        if (
            expected_schema.grouping_column is not None
            and expected_schema.id_column is not None
        ):
            raise AssertionError(
                "Output schema from a transformation had both an ID column "
                "and a grouping column, which is not allowed. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )

        if expected_schema.grouping_column is not None:
            expected_output_metric = IfGroupedBy(
                expected_schema.grouping_column, self.inner_metric()
            )
        elif expected_schema.id_column is not None:
            expected_output_metric = IfGroupedBy(
                expected_schema.id_column, SymmetricDifference()
            )
        else:
            expected_output_metric = SymmetricDifference()

        if (
            lookup_domain(transformation.output_domain, reference)
            != expected_output_domain
        ):
            raise AssertionError(
                "Unexpected output domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        if (
            lookup_metric(transformation.output_metric, reference)
            != expected_output_metric
        ):
            raise AssertionError(
                "Unexpected output metric. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )

    def inner_metric(self) -> Union[SumOf, RootSumOfSquared]:
        """Get the inner metric used by this TransformationVisitor."""
        if self.mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            return SumOf(SymmetricDifference())
        elif self.mechanism in (
            NoiseMechanism.DISCRETE_GAUSSIAN,
            NoiseMechanism.GAUSSIAN,
        ):
            return RootSumOfSquared(SymmetricDifference())
        else:
            raise RuntimeError(
                f"Unsupported mechanism {self.mechanism}. "
                "Supported mechanisms are "
                f"{NoiseMechanism.GAUSSIAN},"
                f"{NoiseMechanism.DISCRETE_GAUSSIAN}, "
                f"{NoiseMechanism.LAPLACE}, and"
                f"{NoiseMechanism.GEOMETRIC}."
            )

    def _visit_child(self, child: QueryExpr) -> Output:
        """Visit a child query and raise assertion errors if needed."""
        transformation, reference, constraints = child.accept(self)
        if not isinstance(transformation, Transformation):
            raise AssertionError(
                "Child query did not create a transformation. "
                "This is probably a bug; please let us know about it so "
                "we can fix it!"
            )
        input_domain = lookup_domain(transformation.output_domain, reference)
        input_metric = lookup_metric(transformation.output_metric, reference)
        if not isinstance(input_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Child query has an invalid output domain. "
                "This is probably a bug; please let us know about it so "
                "we can fix it!"
            )
        if not isinstance(
            input_metric, (IfGroupedBy, SymmetricDifference, HammingDistance)
        ):
            raise AssertionError(
                "Child query does not have a recognized output metric. "
                "This is probably a bug; please let us know about "
                "it so we can fix it!"
            )
        return self.Output(transformation, reference, constraints)

    @classmethod
    def _ensure_not_hamming(
        cls,
        transformation: Transformation,
        reference: TableReference,
        constraints: List[Constraint],
    ) -> Output:
        """Convert transformation to one with a SymmetricDifference() output metric."""
        input_domain = lookup_domain(transformation.output_domain, reference)
        input_metric = lookup_metric(transformation.output_metric, reference)
        if not isinstance(input_metric, HammingDistance):
            return cls.Output(transformation, reference, constraints)

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Cannot convert this transformation to one with a "
                    "SymmetricDifference output metric. This is probably "
                    "a bug; please let us know about it so we can fix it!"
                )
            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                reference.identifier,
                target,
                HammingDistanceToSymmetricDifference(input_domain),
                lambda *args: None,
            )

        transformation_generator: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric
        }

        return cls.Output(
            *generate_nested_transformation(
                transformation, reference.parent, transformation_generator
            ),
            constraints,
        )

    def visit_private_source(self, expr) -> Output:
        """Create a transformation from a PrivateSource query expression."""
        ref = find_reference(expr.source_id, self.input_domain)
        if ref is None:
            named_tables = find_named_tables(self.input_domain)
            raise ValueError(
                f"Table '{expr.source_id}' does not exist. "
                f"Available tables are: {named_tables}"
            )
        transformation = IdentityTransformation(self.input_metric, self.input_domain)
        try:
            constraints = self.table_constraints[ref.identifier]
        except KeyError as e:
            raise AssertionError(
                f"Table {ref.identifier} not present in constraints dictionary. "
                "This is probably a bug; please let us know about it so we can fix it!"
            ) from e
        return self.Output(transformation, ref, constraints)

    def visit_rename(self, expr: RenameExpr) -> Output:
        """Create a transformation from a Rename query expression."""
        child_transformation, child_ref, child_constraints = expr.child.accept(self)

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            input_domain = lookup_domain(child_transformation.output_domain, child_ref)
            input_metric = lookup_metric(child_transformation.output_metric, child_ref)
            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            if not isinstance(input_metric, (SymmetricDifference, IfGroupedBy)):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            nonexistent_columns = set(expr.column_mapper) - set(input_domain.schema)
            if nonexistent_columns:
                raise ValueError(
                    f"Nonexistent columns in rename query: {nonexistent_columns}"
                )
            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                RenameTransformation(input_domain, input_metric, expr.column_mapper),
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            return RenameValueTransformation(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                expr.column_mapper,
            )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_rename(expr, child_constraints)),
        )

    def visit_filter(self, expr: FilterExpr) -> Output:
        """Create a transformation from a FilterExpr query expression."""
        child_transformation, child_ref, child_constraints = expr.child.accept(self)

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            input_domain = lookup_domain(child_transformation.output_domain, child_ref)
            input_metric = lookup_metric(child_transformation.output_metric, child_ref)
            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            if not isinstance(input_metric, (IfGroupedBy, SymmetricDifference)):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                FilterTransformation(input_domain, input_metric, expr.condition),
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            return FilterValueTransformation(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                expr.condition,
            )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_unmodified(expr, child_constraints)),
        )

    def visit_select(self, expr: SelectExpr) -> Output:
        """Create a transformation from a Select query expression."""
        child_transformation, child_ref, child_constraints = expr.child.accept(self)

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            input_domain = lookup_domain(child_transformation.output_domain, child_ref)
            input_metric = lookup_metric(child_transformation.output_metric, child_ref)

            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            if not isinstance(
                input_metric, (IfGroupedBy, SymmetricDifference, HammingDistance)
            ):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            nonexistent_columns = set(expr.columns) - set(input_domain.schema)
            if nonexistent_columns:
                raise ValueError(
                    f"Nonexistent columns in select query: {nonexistent_columns}"
                )
            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                SelectTransformation(input_domain, input_metric, expr.columns),
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            return SelectValueTransformation(
                parent_domain, parent_metric, child_ref.identifier, target, expr.columns
            )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_select(expr, child_constraints)),
        )

    def visit_map(self, expr: MapExpr) -> Output:
        """Create a transformation from a Map query expression."""
        child_transformation, child_ref, child_constraints = expr.child.accept(self)

        input_domain = lookup_domain(child_transformation.output_domain, child_ref)
        if not isinstance(input_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Unrecognized input domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        transformer_input_domain = SparkRowDomain(input_domain.schema)
        # Any new column created by Map could contain a null value
        spark_columns_descriptor = {
            # all of the Spark<Type>ColumnDescriptor classes are dataclasses,
            # but the SparkColumnDescriptor base class isn't;
            # hence the "type: ignore" here
            k: dataclasses.replace(v, allow_null=True)  # type: ignore
            for k, v in analytics_to_spark_columns_descriptor(
                expr.schema_new_columns
            ).items()
        }

        if expr.augment:
            output_schema = {
                **transformer_input_domain.schema,
                **spark_columns_descriptor,
            }
        else:
            output_schema = spark_columns_descriptor

        output_domain = SparkRowDomain(output_schema)
        # If you change `getattr(query, "f")` below to `query.f`,
        # mypy will be upset
        transformer = RowToRowTransformation(
            input_domain=transformer_input_domain,
            output_domain=output_domain,
            trusted_f=getattr(expr, "f"),
            augment=expr.augment,
        )

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            input_metric = lookup_metric(child_transformation.output_metric, child_ref)
            if not isinstance(
                input_metric, (IfGroupedBy, SymmetricDifference, HammingDistance)
            ):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )

            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                MapTransformation(metric=input_metric, row_transformer=transformer),
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            if not expr.augment:
                raise ValueError(
                    "Maps on tables with the AddRowsWithID protected change "
                    "must be augmenting"
                )
            return MapValueTransformation(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                row_transformer=transformer,
            )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_map(expr, child_constraints)),
        )

    def visit_flat_map(self, expr: FlatMapExpr) -> Output:
        """Create a transformation from a FlatMap query expression."""
        child_transformation, child_ref, child_constraints = self._ensure_not_hamming(
            *expr.child.accept(self)
        )

        input_domain = lookup_domain(child_transformation.output_domain, child_ref)
        if not isinstance(input_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Unrecognized input domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        transformer_input_domain = SparkRowDomain(input_domain.schema)
        # Any new column created by FlatMap could contain a null value
        spark_columns_descriptor = {
            # all of the Spark<Type>ColumnDescriptor classes are dataclasses,
            # but the SparkColumnDescriptor base class isn't;
            # hence the "type: ignore" here
            k: dataclasses.replace(v, allow_null=True)  # type: ignore
            for k, v in analytics_to_spark_columns_descriptor(
                expr.schema_new_columns
            ).items()
        }

        if expr.augment:
            output_schema = {
                **transformer_input_domain.schema,
                **spark_columns_descriptor,
            }
        else:
            output_schema = spark_columns_descriptor

        output_domain = ListDomain(SparkRowDomain(output_schema))

        row_transformer = RowToRowsTransformation(
            input_domain=transformer_input_domain,
            output_domain=output_domain,
            trusted_f=getattr(expr, "f"),
            augment=expr.augment,
        )

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            if expr.max_num_rows is None:
                raise ValueError(
                    "Flat maps on tables without IDs must have a defined max_num_rows"
                    " parameter."
                )
            input_metric = lookup_metric(child_transformation.output_metric, child_ref)
            if not isinstance(input_metric, (IfGroupedBy, SymmetricDifference)):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            transformation: Transformation
            if expr.schema_new_columns.grouping_column is not None:
                transformation = GroupingFlatMap(
                    output_metric=self.inner_metric(),
                    row_transformer=row_transformer,
                    max_num_rows=expr.max_num_rows,
                )
            else:
                transformation = FlatMapTransformation(
                    metric=input_metric,
                    row_transformer=row_transformer,
                    max_num_rows=expr.max_num_rows,
                )

            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                transformation,
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            if not expr.augment:
                raise ValueError(
                    "Flat maps on tables with the AddRowsWithID protected change "
                    "must be augmenting"
                )
            if expr.schema_new_columns.grouping_column is not None:
                raise ValueError(
                    "Flat maps on tables with the AddRowsWithID protected "
                    "change cannot be grouping"
                )
            if expr.max_num_rows is not None:
                warnings.warn(
                    "When performing a flat map on a table with the AddRowsWithID "
                    "ProtectedChange(), the max_num_rows parameter "
                    "is not required and will be ignored."
                )
            return FlatMapValueTransformation(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                row_transformer=row_transformer,
                max_num_rows=expr.max_num_rows,
            )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_flat_map(expr, child_constraints)),
        )

    def visit_join_private(self, expr: JoinPrivateExpr) -> Output:
        """Create a transformation from a JoinPrivate query expression."""
        left_transformation, left_ref, left_constraints = expr.child.accept(self)
        right_visitor = TransformationVisitor(
            left_transformation.output_domain,
            left_transformation.output_metric,
            self.mechanism,
            self.public_sources,
            self.table_constraints,
        )
        (
            right_transformation,
            right_ref,
            right_constraints,
        ) = expr.right_operand_expr.accept(right_visitor)

        if left_ref.parent != right_ref.parent:
            raise ValueError(
                "Left and right tables must be initialized with compatible "
                f"ProtectedChange() descriptions, but {left_ref} and {right_ref} "
                "were not."
            )

        child_transformation = left_transformation | right_transformation
        left_domain = lookup_domain(child_transformation.output_domain, left_ref)
        right_domain = lookup_domain(child_transformation.output_domain, right_ref)
        if not isinstance(left_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Unrecognized input domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        if not isinstance(right_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Unrecognized input domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )

        def get_truncation_params(
            strategy: TruncationStrategy.Type,
        ) -> Tuple[CoreTruncationStrategy, int]:
            if isinstance(strategy, TruncationStrategy.DropExcess):
                return CoreTruncationStrategy.TRUNCATE, strategy.max_records
            elif isinstance(strategy, TruncationStrategy.DropNonUnique):
                return CoreTruncationStrategy.DROP, 1
            else:
                raise ValueError(
                    f"Truncation strategy type {strategy.__class__.__qualname__} "
                    "is not supported."
                )

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            if (
                expr.truncation_strategy_left is None
                or expr.truncation_strategy_right is None
            ):
                raise ValueError(
                    "When joining without IDs, truncation strategies are required."
                )
            l_trunc_strat, l_trunc_threshold = get_truncation_params(
                expr.truncation_strategy_left
            )
            r_trunc_strat, r_trunc_threshold = get_truncation_params(
                expr.truncation_strategy_right
            )
            subset = Subset(
                parent_domain,
                parent_metric,
                [left_ref.identifier, right_ref.identifier],
            )
            if not isinstance(subset.output_domain, DictDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            join = PrivateJoinTransformation(
                input_domain=subset.output_domain,
                left_key=left_ref.identifier,
                right_key=right_ref.identifier,
                left_truncation_strategy=l_trunc_strat,
                right_truncation_strategy=r_trunc_strat,
                left_truncation_threshold=l_trunc_threshold,
                right_truncation_threshold=r_trunc_threshold,
                join_cols=expr.join_columns,
                join_on_nulls=True,
            )
            create_dict = CreateDictFromValue(
                join.output_domain, join.output_metric, key=target
            )
            return AugmentDictTransformation(subset | join | create_dict)

        def gen_transformation_ark(parent_domain, parent_metric, target):
            if (expr.truncation_strategy_left is not None) or (
                expr.truncation_strategy_right is not None
            ):
                warnings.warn(
                    "When joining with IDs, truncation strategies are not required."
                    " Provided truncation parameters will be ignored."
                )
            return PrivateJoinOnKeyTransformation(
                parent_domain,
                parent_metric,
                left_ref.identifier,
                right_ref.identifier,
                target,
                expr.join_columns,
                join_on_nulls=True,
            )

        common_cols = set(left_domain.schema) & set(right_domain.schema)
        join_cols = set(expr.join_columns or common_cols)
        overlapping_cols = common_cols - join_cols
        constraints = propagate_join_private(
            join_cols, overlapping_cols, left_constraints, right_constraints
        )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }
        return self.Output(
            *generate_nested_transformation(
                child_transformation,
                # right_ref.parent and left_ref.parent must be the same,
                # so either works here.
                right_ref.parent,
                transformation_generators,
            ),
            simplify_constraints(constraints),
        )

    def visit_join_public(self, expr: JoinPublicExpr) -> Output:
        """Create a transformation from a JoinPublic query expression."""
        child_transformation, child_ref, child_constraints = self._ensure_not_hamming(
            *self._visit_child(expr.child)
        )

        public_df: DataFrame
        if isinstance(expr.public_table, str):
            try:
                public_df = self.public_sources[expr.public_table]
            except KeyError as e:
                raise ValueError(
                    f"There is no public source with the identifier {expr.public_table}"
                ) from e
        else:
            public_df = expr.public_table

        public_df_schema = Schema(spark_schema_to_analytics_columns(public_df.schema))

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            input_domain = lookup_domain(child_transformation.output_domain, child_ref)
            input_metric = lookup_metric(child_transformation.output_metric, child_ref)
            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            if not isinstance(input_metric, (IfGroupedBy, SymmetricDifference)):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )

            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                PublicJoinTransformation(
                    input_domain=SparkDataFrameDomain(input_domain.schema),
                    public_df=public_df,
                    public_df_domain=SparkDataFrameDomain(
                        analytics_to_spark_columns_descriptor(public_df_schema)
                    ),
                    join_cols=expr.join_columns if expr.join_columns else None,
                    metric=input_metric,
                    join_on_nulls=True,
                ),
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            return PublicJoinValueTransformation(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                public_df,
                SparkDataFrameDomain(
                    analytics_to_spark_columns_descriptor(public_df_schema)
                ),
                expr.join_columns,
                join_on_nulls=True,
            )

        child_domain = lookup_domain(child_transformation.output_domain, child_ref)
        if not isinstance(child_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Unrecognized input domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )

        common_cols = set(child_domain.schema) & set(public_df_schema)
        join_cols = set(expr.join_columns or common_cols)
        overlapping_cols = common_cols - join_cols
        constraints = propagate_join_public(
            join_cols, overlapping_cols, public_df, child_constraints
        )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }
        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(constraints),
        )

    @staticmethod
    def _get_replace_with(
        expr: ReplaceNullAndNan,
        analytics_schema: Dict[str, ColumnDescriptor],
        grouping_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        replace_with: Dict[str, Any] = dict(expr.replace_with).copy()
        if len(replace_with) == 0:
            for col in Schema(analytics_schema).column_descs:
                if col == grouping_column:
                    continue
                if not analytics_schema[col].allow_null and not (
                    analytics_schema[col].allow_nan
                ):
                    continue
                if analytics_schema[col].column_type == ColumnType.INTEGER:
                    replace_with[col] = int(AnalyticsDefault.INTEGER)
                elif analytics_schema[col].column_type == ColumnType.DECIMAL:
                    replace_with[col] = float(AnalyticsDefault.DECIMAL)
                elif analytics_schema[col].column_type == ColumnType.VARCHAR:
                    replace_with[col] = str(AnalyticsDefault.VARCHAR)
                elif analytics_schema[col].column_type == ColumnType.DATE:
                    date: datetime.date = AnalyticsDefault.DATE
                    replace_with[col] = date
                elif analytics_schema[col].column_type == ColumnType.TIMESTAMP:
                    dt: datetime.datetime = AnalyticsDefault.TIMESTAMP
                    replace_with[col] = dt
                else:
                    raise RuntimeError(
                        f"Analytics does not have a default value for column {col} of"
                        f" type {analytics_schema[col].column_type}, and no default"
                        " value was provided"
                    )

        else:
            # Check that all columns exist
            for col in replace_with:
                if not col in analytics_schema:
                    raise ValueError(
                        f"Cannot replace values in column {col}, because it is not in"
                        " the schema"
                    )
            # Make sure all DECIMAL replacement values are floats
            for col in replace_with.keys():
                if analytics_schema[col].column_type == ColumnType.DECIMAL:
                    replace_with[col] = float(replace_with[col])

        return replace_with

    def visit_replace_null_and_nan(self, expr: ReplaceNullAndNan) -> Output:
        """Create a transformation from a ReplaceNullAndNan query expression."""
        child_transformation, child_ref, child_constraints = self._visit_child(
            expr.child
        )
        input_domain = lookup_domain(child_transformation.output_domain, child_ref)
        input_metric = lookup_metric(child_transformation.output_metric, child_ref)
        if not isinstance(input_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Unrecognized input domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        if not isinstance(
            input_metric, (IfGroupedBy, HammingDistance, SymmetricDifference)
        ):
            raise AssertionError(
                "Unrecognized input metric. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        grouping_column: Optional[str] = None
        if isinstance(input_metric, IfGroupedBy):
            grouping_column = input_metric.column
            if grouping_column in expr.replace_with:
                raise ValueError(
                    "Cannot replace null values in column"
                    f" {input_metric.column}, because it is being used as a"
                    " grouping column"
                )
        analytics_schema = spark_dataframe_domain_to_analytics_columns(input_domain)

        replace_with = self._get_replace_with(expr, analytics_schema, grouping_column)

        replace_null = any(analytics_schema[col].allow_null for col in replace_with)

        replace_nan = any(
            analytics_schema[col].column_type == ColumnType.DECIMAL
            and analytics_schema[col].allow_nan
            for col in replace_with.keys()
        )

        null_replace_map = {
            col: val
            for col, val in replace_with.items()
            if analytics_schema[col].allow_null
        }

        nan_replace_map = {
            col: replace_with[col]
            for col in replace_with.keys()
            if (
                analytics_schema[col].column_type == ColumnType.DECIMAL
                and analytics_schema[col].allow_nan
            )
        }

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            transformation: Transformation = IdentityTransformation(
                input_metric, input_domain
            )
            if replace_null:
                assert isinstance(input_domain, SparkDataFrameDomain)
                assert isinstance(
                    input_metric, (IfGroupedBy, HammingDistance, SymmetricDifference)
                )
                transformation |= ReplaceNullsTransformation(
                    input_domain=input_domain,
                    metric=input_metric,
                    replace_map=null_replace_map,
                )
            if replace_nan:
                assert isinstance(transformation.output_domain, SparkDataFrameDomain)
                assert isinstance(
                    transformation.output_metric,
                    (IfGroupedBy, HammingDistance, SymmetricDifference),
                )
                transformation |= ReplaceNaNsTransformation(
                    input_domain=transformation.output_domain,
                    metric=transformation.output_metric,
                    replace_map=nan_replace_map,
                )

            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                transformation,
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            transformation: Transformation = IdentityTransformation(
                parent_metric, parent_domain
            )
            temp_table_id = TemporaryTable()

            if replace_null:
                transformation |= ReplaceNullsValueTransformation(
                    parent_domain,
                    parent_metric,
                    child_ref.identifier,
                    temp_table_id if replace_nan else target,
                    replace_map=null_replace_map,
                )

            if replace_nan:
                assert isinstance(transformation.output_domain, DictDomain)
                assert isinstance(transformation.output_metric, AddRemoveKeys)
                transformation |= ReplaceNaNsValueTransformation(
                    transformation.output_domain,
                    transformation.output_metric,
                    temp_table_id if replace_null else child_ref.identifier,
                    target,
                    nan_replace_map,
                )

            if (not replace_null) and (not replace_nan):
                # Rename that essentially does nothing
                assert isinstance(transformation.output_domain, DictDomain)
                assert isinstance(transformation.output_metric, AddRemoveKeys)
                transformation |= RenameValueTransformation(
                    transformation.output_domain,
                    transformation.output_metric,
                    child_ref.identifier,
                    target,
                    {},
                )
            return transformation

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_replace(expr, child_constraints)),
        )

    def visit_replace_infinity(self, expr: ReplaceInfinity) -> Output:
        """Create a transformation from a ReplaceInfinity query expression."""
        child_transformation, child_ref, child_constraints = self._visit_child(
            expr.child
        )
        input_domain = lookup_domain(child_transformation.output_domain, child_ref)
        input_metric = lookup_metric(child_transformation.output_metric, child_ref)

        analytics_schema = Schema(
            spark_dataframe_domain_to_analytics_columns(input_domain)
        )
        replace_with = expr.replace_with.copy()
        if len(replace_with) == 0:
            replace_with = {
                col: (AnalyticsDefault.DECIMAL, AnalyticsDefault.DECIMAL)
                for col in analytics_schema.column_descs
                if analytics_schema[col].column_type == ColumnType.DECIMAL
            }

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            if not isinstance(
                input_metric, (IfGroupedBy, HammingDistance, SymmetricDifference)
            ):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                ReplaceInfsTransformation(
                    input_domain=input_domain,
                    metric=input_metric,
                    replace_map=replace_with,
                ),
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            return ReplaceInfsValueTransformation(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                replace_map=replace_with,
            )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_replace(expr, child_constraints)),
        )

    def visit_drop_infinity(self, expr: DropInfExpr) -> Output:
        """Create a transformation from a DropInfinity query expression."""
        child_transformation, child_ref, child_constraints = self._ensure_not_hamming(
            *self._visit_child(expr.child)
        )
        input_domain = lookup_domain(child_transformation.output_domain, child_ref)
        input_metric = lookup_metric(child_transformation.output_metric, child_ref)
        analytics_schema = Schema(
            spark_dataframe_domain_to_analytics_columns(input_domain)
        )

        columns = expr.columns.copy()
        if len(columns) == 0:
            columns = [
                col
                for col, cd in analytics_schema.column_descs.items()
                if (cd.column_type == ColumnType.DECIMAL and cd.allow_inf)
            ]
        else:
            for col in columns:
                if analytics_schema.column_descs[col].column_type != ColumnType.DECIMAL:
                    raise ValueError(
                        f"Cannot drop infinite values from column {col}, because that"
                        " column's type is not DECIMAL"
                    )

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            if not isinstance(input_metric, (IfGroupedBy, SymmetricDifference)):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                DropInfTransformation(
                    input_domain=input_domain, metric=input_metric, columns=columns
                ),
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            return DropInfsValueTransformation(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                columns=columns,
            )

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_unmodified(expr, child_constraints)),
        )

    def visit_drop_null_and_nan(self, expr: DropNullAndNan) -> Output:
        """Create a transformation from a DropNullAndNan query expression."""
        child_transformation, child_ref, child_constraints = self._ensure_not_hamming(
            *self._visit_child(expr.child)
        )
        input_domain = lookup_domain(child_transformation.output_domain, child_ref)
        input_metric = lookup_metric(child_transformation.output_metric, child_ref)
        analytics_schema = Schema(
            spark_dataframe_domain_to_analytics_columns(input_domain)
        )

        # TODO(2702): This should be supported for IDs tables
        grouping_column: Optional[str] = None
        if isinstance(input_metric, IfGroupedBy):
            grouping_column = input_metric.column
            if grouping_column in expr.columns:
                raise ValueError(
                    "Cannot drop null values in column"
                    f" {input_metric.column}, because it is being used as a"
                    " grouping column"
                )

        columns = expr.columns.copy()
        if len(columns) == 0:
            columns = [
                col
                for col, cd in analytics_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan) and not (col == grouping_column)
            ]

        null_columns = [col for col in columns if analytics_schema[col].allow_null]

        nan_columns = [col for col in columns if analytics_schema[col].allow_nan]

        def gen_transformation_dictmetric(parent_domain, parent_metric, target):
            if not isinstance(input_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized input domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            if not isinstance(input_metric, (IfGroupedBy, SymmetricDifference)):
                raise AssertionError(
                    "Unrecognized input metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            transformation: Transformation = IdentityTransformation(
                input_metric, input_domain
            )

            if null_columns:
                transformation |= DropNullsTransformation(
                    input_domain, input_metric, null_columns
                )

            if nan_columns:
                assert isinstance(transformation.output_domain, SparkDataFrameDomain)
                assert isinstance(
                    transformation.output_metric, (IfGroupedBy, SymmetricDifference)
                )
                transformation |= DropNaNsTransformation(
                    transformation.output_domain,
                    transformation.output_metric,
                    nan_columns,
                )

            return create_copy_and_transform_value(
                parent_domain,
                parent_metric,
                child_ref.identifier,
                target,
                transformation,
                lambda *args: None,
            )

        def gen_transformation_ark(parent_domain, parent_metric, target):
            transformation: Transformation = IdentityTransformation(
                parent_metric, parent_domain
            )
            temp_table_id = TemporaryTable()

            if null_columns:
                transformation |= DropNullsValueTransformation(
                    parent_domain,
                    parent_metric,
                    child_ref.identifier,
                    temp_table_id if nan_columns else target,
                    null_columns,
                )

            if nan_columns:
                assert isinstance(transformation.output_domain, DictDomain)
                assert isinstance(transformation.output_metric, AddRemoveKeys)
                transformation |= DropNaNsValueTransformation(
                    transformation.output_domain,
                    transformation.output_metric,
                    temp_table_id if null_columns else child_ref.identifier,
                    target,
                    nan_columns,
                )

            if (not null_columns) and (not nan_columns):
                # Rename that essentially does nothing
                assert isinstance(transformation.output_domain, DictDomain)
                assert isinstance(transformation.output_metric, AddRemoveKeys)
                transformation |= RenameValueTransformation(
                    transformation.output_domain,
                    transformation.output_metric,
                    child_ref.identifier,
                    target,
                    {},
                )
            return transformation

        transformation_generators: Dict[Type[Metric], Callable] = {
            DictMetric: gen_transformation_dictmetric,
            AddRemoveKeys: gen_transformation_ark,
        }

        return self.Output(
            *generate_nested_transformation(
                child_transformation, child_ref.parent, transformation_generators
            ),
            simplify_constraints(propagate_unmodified(expr, child_constraints)),
        )

    def visit_enforce_constraint(self, expr: EnforceConstraint) -> Output:
        """Create a transformation from an EnforceConstraint query expression."""
        # Note: at present, enforcing one constraint can never invalidate
        # another constraint, so just adding the new constraint to the list of
        # constraints is perfectly fine. If a new constraint is added that can
        # invalidate other constraints, this will have to be broken out into
        # per-constraint-type logic.
        child_transformation, child_ref, child_constraints = self._visit_child(
            expr.child
        )
        # pylint: disable=protected-access
        transformation, ref = expr.constraint._enforce(
            child_transformation, child_ref, *expr.options
        )
        # pylint: enable=protected-access
        return self.Output(
            transformation,
            ref,
            simplify_constraints(child_constraints + [expr.constraint]),
        )

    # None of the queries that produce measurements are implemented
    def visit_groupby_count(self, expr: GroupByCount) -> Any:
        """Visit a GroupByCount query expression (raises an error)."""
        raise NotImplementedError

    def visit_groupby_count_distinct(self, expr: GroupByCountDistinct) -> Any:
        """Visit a GroupByCountDistinct query expression (raises an error)."""
        raise NotImplementedError

    def visit_groupby_quantile(self, expr: GroupByQuantile) -> Any:
        """Visit a GroupByQuantile query expression (raises an error)."""
        raise NotImplementedError

    def visit_groupby_bounded_sum(self, expr: GroupByBoundedSum) -> Any:
        """Visit a GroupByBoundedSum query expression (raises an error)."""
        raise NotImplementedError

    def visit_groupby_bounded_average(self, expr: GroupByBoundedAverage) -> Any:
        """Visit a GroupByBoundedAverage query expression (raises an error)."""
        raise NotImplementedError

    def visit_groupby_bounded_variance(self, expr: GroupByBoundedVariance) -> Any:
        """Visit a GroupByBoundedVariance query expression (raises an error)."""
        raise NotImplementedError

    def visit_groupby_bounded_stdev(self, expr: GroupByBoundedSTDEV) -> Any:
        """Visit a GroupByBoundedSTDEV query expression (raises an error)."""
        raise NotImplementedError
