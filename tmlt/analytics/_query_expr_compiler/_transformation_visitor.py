"""Defines a visitor for creating a transformation from a query expression."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import dataclasses
import datetime
from typing import Any, Dict, Optional, Union

from pyspark.sql import DataFrame

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._schema import (
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_dataframe_domain_to_analytics_columns,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics.query_expr import AnalyticsDefault
from tmlt.analytics.query_expr import DropInfinity as DropInfExpr
from tmlt.analytics.query_expr import DropNullAndNan
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
from tmlt.analytics.query_expr import PrivateSource, QueryExpr, QueryExprVisitor
from tmlt.analytics.query_expr import Rename as RenameExpr
from tmlt.analytics.query_expr import ReplaceInfinity, ReplaceNullAndNan
from tmlt.analytics.query_expr import Select as SelectExpr
from tmlt.analytics.truncation_strategy import TruncationStrategy
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain, SparkRowDomain
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.metrics import (
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.converters import HammingDistanceToSymmetricDifference
from tmlt.core.transformations.dictionary import (
    AugmentDictTransformation,
    CreateDictFromValue,
    GetValue,
    Subset,
)
from tmlt.core.transformations.spark_transformations.filter import (
    Filter as FilterTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoin as PrivateJoinTransformation,
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
    DropNaNs,
    DropNulls,
    ReplaceInfs,
    ReplaceNaNs,
    ReplaceNulls,
)
from tmlt.core.transformations.spark_transformations.rename import (
    Rename as RenameTransformation,
)
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.misc import get_nonconflicting_string


class TransformationVisitor(QueryExprVisitor):
    """A visitor to create a transformation from a query expression."""

    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: DictMetric,
        mechanism: NoiseMechanism,
        public_sources: Dict[str, DataFrame],
    ):
        """Constructor for a TransformationVisitor.

        Args:
            input_domain: The input domain that the transformation should have.
            input_metric: The input metric that the transformation should have.
            mechanism: The noise mechanism (only used for FlatMaps).
            public_sources: Public sources to use for JoinPublic queries.
        """
        self.input_domain = input_domain
        self.input_metric = input_metric
        self.mechanism = mechanism
        self.public_sources = public_sources

    def validate_transformation(
        self, query: QueryExpr, transformation: Transformation, catalog: Catalog
    ):
        """Ensure that a query's transformation is valid on a given catalog."""
        expected_schema = query.accept(OutputSchemaVisitor(catalog))
        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        expected_output_metric = (
            SymmetricDifference()
            if expected_schema.grouping_column is None
            else IfGroupedBy(expected_schema.grouping_column, self.inner_metric())
        )
        if transformation.output_domain != expected_output_domain:
            raise AssertionError(
                "Unexpected output domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        if transformation.output_metric != expected_output_metric:
            raise AssertionError(
                "Unexpected output metric. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )

    # TODO(#2172): update so it no longer assumes source ID is in domain
    def visit_private_source(self, expr: PrivateSource) -> Transformation:
        """Create a transformation from a PrivateSource query expression."""
        # check if the source ID is in the input domain's keys. If not,
        # it likely exists in a dict one level into the input domain
        # if expr.source_id in self.input_domain.key_to_domain:
        return GetValue(self.input_domain, self.input_metric, expr.source_id)

    def _visit_child(self, child: QueryExpr) -> Transformation:
        """Visit a child query and raise assertion errors if needed."""
        transformation = child.accept(self)
        if not isinstance(transformation, Transformation):
            raise AssertionError(
                "Child query did not create a transformation. "
                "This is probably a bug; please let us know about it so "
                "we can fix it!"
            )
        if not isinstance(transformation.output_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Child query has an invalid output domain. "
                "This is probably a bug; please let us know about it so "
                "we can fix it!"
            )
        if not isinstance(
            transformation.output_metric,
            (IfGroupedBy, SymmetricDifference, HammingDistance),
        ):
            raise AssertionError(
                "Child query does not have a recognized output metric. "
                "This is probably a bug; please let us know about "
                "it so we can fix it!"
            )
        return transformation

    def visit_rename(self, query: RenameExpr) -> Transformation:
        """Create a transformation from a Rename query expression."""
        if not isinstance(query.column_mapper, dict):
            raise ValueError(
                "Rename query's column_mapper must be "
                "a dictionary mapping old column names "
                "to new column names."
            )
        child = self._visit_child(query.child)
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(
            child.output_metric, (IfGroupedBy, SymmetricDifference, HammingDistance)
        )
        rename_transformation = RenameTransformation(
            input_domain=child.output_domain,
            metric=child.output_metric,
            rename_mapping=query.column_mapper,
        )
        return child | rename_transformation

    @staticmethod
    def _ensure_not_hamming(transformation: Transformation) -> Transformation:
        """Convert transformation to one with a SymmetricDifference() output metric."""
        if not isinstance(transformation.output_metric, HammingDistance):
            return transformation
        if not isinstance(transformation.output_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Cannot convert this transformation to one with a "
                "SymmetricDifference output metric. This is probably "
                "a bug; please let us know about it so we can fix it!"
            )
        return transformation | HammingDistanceToSymmetricDifference(
            transformation.output_domain
        )

    def visit_filter(self, query: FilterExpr) -> Transformation:
        """Create a transformation from a FilterExpr query expression."""
        child = self._ensure_not_hamming(self._visit_child(query.child))
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(child.output_metric, (IfGroupedBy, SymmetricDifference))
        transformation = FilterTransformation(
            domain=child.output_domain,
            metric=child.output_metric,
            filter_expr=query.predicate,
        )
        return child | transformation

    def visit_select(self, query: SelectExpr) -> Transformation:
        """Create a transformation from a Select query expression."""
        child = self._visit_child(query.child)
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(
            child.output_metric, (IfGroupedBy, SymmetricDifference, HammingDistance)
        )
        transformation = SelectTransformation(
            input_domain=child.output_domain,
            metric=child.output_metric,
            columns=list(query.columns),
        )
        return child | transformation

    def visit_map(self, query: MapExpr) -> Transformation:
        """Create a transformation from a Map query expression."""
        child = self._visit_child(query.child)
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(
            child.output_metric, (IfGroupedBy, HammingDistance, SymmetricDifference)
        )
        transformer_input_domain = SparkRowDomain(child.output_domain.schema)
        # Any new column created by Map could contain a null value
        spark_columns_descriptor = {
            k: dataclasses.replace(v, allow_null=True)
            for k, v in analytics_to_spark_columns_descriptor(
                query.schema_new_columns
            ).items()
        }
        if query.augment:
            output_schema = {
                **transformer_input_domain.schema,
                **spark_columns_descriptor,
            }
        else:
            output_schema = spark_columns_descriptor
        output_domain = SparkRowDomain(output_schema)

        # If you change `getattr(query, "f")` below to `query.f`,
        # mypy will complain at you
        transformation = MapTransformation(
            metric=child.output_metric,
            row_transformer=RowToRowTransformation(
                input_domain=transformer_input_domain,
                output_domain=output_domain,
                trusted_f=getattr(query, "f"),
                augment=query.augment,
            ),
        )
        return child | transformation

    def inner_metric(self) -> Union[SumOf, RootSumOfSquared]:
        """Get the inner metric used by this TransformationVisitor."""
        if self.mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            return SumOf(SymmetricDifference())
        else:
            if self.mechanism != NoiseMechanism.DISCRETE_GAUSSIAN:
                raise RuntimeError(
                    f"Unsupported mechanism {self.mechanism}. "
                    "Supported mechanisms are "
                    f"{NoiseMechanism.DISCRETE_GAUSSIAN}, "
                    f"{NoiseMechanism.LAPLACE}, and"
                    f"{NoiseMechanism.GEOMETRIC}."
                )
            return RootSumOfSquared(SymmetricDifference())

    def visit_flat_map(self, query: FlatMapExpr) -> Transformation:
        """Create a transformation from a FlatMap query expression."""
        child = self._visit_child(query.child)
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(
            child.output_metric, (IfGroupedBy, HammingDistance, SymmetricDifference)
        )

        transformer_input_domain = SparkRowDomain(child.output_domain.schema)
        # Any new column created by FlatMap could contain a null value
        spark_columns_descriptor = {
            k: dataclasses.replace(v, allow_null=True)
            for k, v in analytics_to_spark_columns_descriptor(
                query.schema_new_columns
            ).items()
        }
        if query.augment:
            output_schema = {
                **transformer_input_domain.schema,
                **spark_columns_descriptor,
            }
        else:
            output_schema = spark_columns_descriptor
        output_domain = ListDomain(SparkRowDomain(output_schema))

        # If you change `getattr(query, "f")` below to `query.f`,
        # mypy will complain at you
        row_transformer = RowToRowsTransformation(
            input_domain=transformer_input_domain,
            output_domain=output_domain,
            trusted_f=getattr(query, "f"),
            augment=query.augment,
        )

        transformation: Transformation
        if query.schema_new_columns.grouping_column is not None:
            transformation = GroupingFlatMap(
                output_metric=self.inner_metric(),  # (sqrt) sum of (squared) symm diff
                row_transformer=row_transformer,
                max_num_rows=query.max_num_rows,
            )
        else:
            child = self._ensure_not_hamming(child)
            assert isinstance(child.output_domain, SparkDataFrameDomain)
            assert isinstance(child.output_metric, (IfGroupedBy, SymmetricDifference))

            transformation = FlatMapTransformation(
                metric=child.output_metric,
                row_transformer=row_transformer,
                max_num_rows=query.max_num_rows,
            )
        return child | transformation

    def visit_join_private(self, query: JoinPrivateExpr) -> Transformation:
        """Create a transformation from a JoinPrivate query expression."""
        child = self._visit_child(query.child)
        left_metric, left_transformation = (child.output_metric, child)

        # Check that left metrics are correct
        if isinstance(left_metric, IfGroupedBy):
            raise ValueError(
                "Left operand used a grouping transformation. "
                "This is not yet supported for private joins."
            )
        if isinstance(left_metric, HammingDistance):
            if not isinstance(left_transformation.output_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Left operand has an unsupported "
                    "output domain. This is probably a bug; please let us "
                    "know about it so we can fix it!"
                )
            left_transformation = (
                left_transformation
                | HammingDistanceToSymmetricDifference(
                    left_transformation.output_domain
                )
            )
        if left_transformation.output_metric != SymmetricDifference():
            raise ValueError(
                "Left operand has an unsupported output metric. "
                "The only supported output metrics are "
                f"{SymmetricDifference()} and {HammingDistance()}"
            )

        current_keys = [str(k) for k in self.input_domain.key_to_domain.keys()]
        left_output_key = get_nonconflicting_string(current_keys)
        right_output_key = get_nonconflicting_string(current_keys + [left_output_key])

        add_left_transformation = AugmentDictTransformation(
            left_transformation
            | CreateDictFromValue(
                input_domain=left_transformation.output_domain,
                input_metric=left_transformation.output_metric,
                key=left_output_key,
            )
        )
        # input = {left_input, right_input},
        # output = {left_input, right_input, left_output}

        if not isinstance(add_left_transformation.output_domain, DictDomain):
            raise AssertionError(
                "Left transformation output domain has the wrong type. "
                "This is probably a bug; please let us know so we can "
                "fix it!"
            )
        if not isinstance(add_left_transformation.output_metric, DictMetric):
            raise AssertionError(
                "Left transformation output metric has the wrong type. "
                "This is probably a bug; please let us know so we can "
                "fix it!"
            )
        # Get right operand transformation
        right_visitor = TransformationVisitor(
            input_domain=add_left_transformation.output_domain,
            input_metric=add_left_transformation.output_metric,
            mechanism=self.mechanism,
            public_sources=self.public_sources,
        )
        right_transformation = query.right_operand_expr.accept(right_visitor)
        if not isinstance(right_transformation, Transformation):
            raise AssertionError(
                "Right operand does not produce a transformation. "
                "This is probably a bug; please let us know so we can "
                "fix it!"
            )
        # Check that right metrics are correct
        if isinstance(right_transformation.output_metric, IfGroupedBy):
            raise ValueError(
                "Right operand used a grouping transformation. "
                "This is not yet supported for private joins."
            )
        if not isinstance(right_transformation.output_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Right operand has an output domain other than "
                "SparkDataFrameDomain. This is probably a bug; "
                "please let us know so we can fix it!"
            )
        if isinstance(right_transformation.output_metric, HammingDistance):
            right_transformation = (
                right_transformation
                | HammingDistanceToSymmetricDifference(
                    right_transformation.output_domain
                )
            )
        if right_transformation.output_metric != SymmetricDifference():
            raise AssertionError(
                "Right operand has an output metric other than "
                "SymmetricDifference. This is probably a bug; "
                "please let us know so we can fix it!"
            )

        add_right_transformation = AugmentDictTransformation(
            right_transformation
            | CreateDictFromValue(
                input_domain=right_transformation.output_domain,
                input_metric=right_transformation.output_metric,
                key=right_output_key,
            )
        )
        # input = {left_input, right_input, left_output},
        # output = {left_input, right_input, left_output, right_output}

        combined_transformations = add_left_transformation | add_right_transformation
        # input = {left_input, right_input},
        # output = {left_input, right_input, left_output, right_output}

        if not isinstance(combined_transformations.output_domain, DictDomain):
            raise AssertionError(
                "Combined transformation has an unrecognized "
                "output domain. This is probably a bug; "
                "please let us know so we can fix it! "
            )
        if not isinstance(combined_transformations.output_metric, DictMetric):
            raise AssertionError(
                "Combined transformation has an unrecognized "
                "output metric. This is probably a bug; "
                "please let us know so we can fix it!"
            )
        previous_transformation = combined_transformations | Subset(
            input_domain=combined_transformations.output_domain,
            input_metric=combined_transformations.output_metric,
            keys=[left_output_key, right_output_key],
        )
        # input = {left_input, right_input}, output = {left_output, right_output}

        # Create the PrivateJoin transformation
        previous_domain = previous_transformation.output_domain
        if not isinstance(previous_domain, DictDomain):
            raise AssertionError("This is a bug. Please let us know so we can fix it!")
        left_domain = previous_domain.key_to_domain[left_output_key]
        right_domain = previous_domain.key_to_domain[right_output_key]
        if not isinstance(left_domain, SparkDataFrameDomain):
            raise ValueError(
                "Left operand has an output domain that is not a SparkDataFrameDomain."
            )
        if not isinstance(right_domain, SparkDataFrameDomain):
            raise ValueError(
                "Right operand has an output domain that is not a SparkDataFrameDomain."
            )

        def get_core_truncation_strategy(
            strategy: TruncationStrategy.Type,
        ) -> CoreTruncationStrategy:
            if isinstance(strategy, TruncationStrategy.DropExcess):
                return CoreTruncationStrategy.TRUNCATE
            elif isinstance(strategy, TruncationStrategy.DropNonUnique):
                return CoreTruncationStrategy.DROP
            else:
                # This will be triggered if an end user tries to implement their own
                # subclass of TruncationStrategy, or if this function is not updated
                # when a new TruncationStrategy variant is added to the
                # library. Unfortunately, because TruncationStrategy is not an enum
                # it isn't possible to use the mypy assert_never trick to check that
                # this is exhaustive.
                raise ValueError(
                    f"Truncation strategy type {strategy.__class__.__qualname__} "
                    "is not supported."
                )

        def get_truncation_threshold(strategy: TruncationStrategy.Type) -> int:
            if isinstance(strategy, TruncationStrategy.DropExcess):
                return strategy.max_records
            elif isinstance(strategy, TruncationStrategy.DropNonUnique):
                return 1
            else:
                raise ValueError(
                    f"Truncation strategy type {strategy.__class__.__qualname__} "
                    "is not supported."
                )

        join_on_nulls: bool = any(
            [v.allow_null for v in dict(left_domain.schema).values()]
            + [v.allow_null for v in dict(right_domain.schema).values()]
        )
        if query.join_columns is not None:
            join_on_nulls = any(
                [
                    (left_domain[col].allow_null and right_domain[col].allow_null)
                    for col in query.join_columns
                ]
            )

        transformation = PrivateJoinTransformation(
            input_domain=previous_domain,
            left_key=left_output_key,
            right_key=right_output_key,
            left_truncation_strategy=get_core_truncation_strategy(
                query.truncation_strategy_left
            ),
            right_truncation_strategy=get_core_truncation_strategy(
                query.truncation_strategy_right
            ),
            left_truncation_threshold=get_truncation_threshold(
                query.truncation_strategy_left
            ),
            right_truncation_threshold=get_truncation_threshold(
                query.truncation_strategy_right
            ),
            join_cols=query.join_columns,
            join_on_nulls=join_on_nulls,
        )
        return previous_transformation | transformation

    def visit_join_public(self, query: JoinPublicExpr) -> Transformation:
        """Create a transformation from a JoinPublic query expression."""
        public_df: DataFrame
        if isinstance(query.public_table, str):
            try:
                public_df = self.public_sources[query.public_table]
            except KeyError:
                raise ValueError(
                    "There is no public source with the identifier "
                    f"{query.public_table}"
                )
        else:
            public_df = query.public_table

        child = self._ensure_not_hamming(self._visit_child(query.child))
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(child.output_metric, (IfGroupedBy, SymmetricDifference))

        public_df_schema = Schema(spark_schema_to_analytics_columns(public_df.schema))
        transformation = PublicJoinTransformation(
            input_domain=SparkDataFrameDomain(child.output_domain.schema),
            public_df=public_df,
            public_df_domain=SparkDataFrameDomain(
                analytics_to_spark_columns_descriptor(public_df_schema)
            ),
            join_cols=list(query.join_columns) if query.join_columns else None,
            metric=child.output_metric,
            join_on_nulls=any(
                [
                    public_df_schema[col].allow_null
                    for col in list(public_df_schema.keys())
                ]
            ),
        )
        return child | transformation

    def visit_replace_null_and_nan(self, query: ReplaceNullAndNan) -> Transformation:
        """Create a transformation from a ReplaceNullAndNan query expression."""
        child = self._visit_child(query.child)
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(
            child.output_metric, (IfGroupedBy, HammingDistance, SymmetricDifference)
        )
        grouping_column: Optional[str] = None
        if isinstance(child.output_metric, IfGroupedBy):
            grouping_column = child.output_metric.column
            if grouping_column in query.replace_with:
                raise ValueError(
                    "Cannot replace null values in column"
                    f" {child.output_metric.column}, because it is being used as a"
                    " grouping column"
                )
        analytics_schema = spark_dataframe_domain_to_analytics_columns(
            child.output_domain
        )

        replace_with: Dict[str, Any] = dict(query.replace_with).copy()
        if len(replace_with) == 0:
            for col in list(Schema(analytics_schema).column_descs.keys()):
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
            for col in list(replace_with.keys()):
                if analytics_schema[col].column_type == ColumnType.DECIMAL:
                    replace_with[col] = float(replace_with[col])

        also_replace_nan = any(
            [
                (
                    analytics_schema[col].column_type == ColumnType.DECIMAL
                    and analytics_schema[col].allow_nan
                )
                for col in list(replace_with.keys())
            ]
        )

        transformation: Transformation = child

        if any([analytics_schema[col].allow_null for col in replace_with]):
            transformation = transformation | ReplaceNulls(
                input_domain=child.output_domain,
                metric=child.output_metric,
                replace_map={
                    col: val
                    for col, val in replace_with.items()
                    if analytics_schema[col].allow_null
                },
            )
        if also_replace_nan:
            # These assertions are here to make MyPy happy
            # If they fail, something is very wrong in Core
            assert isinstance(transformation.output_domain, SparkDataFrameDomain)
            assert isinstance(
                transformation.output_metric,
                (IfGroupedBy, HammingDistance, SymmetricDifference),
            )
            replace_nan = ReplaceNaNs(
                input_domain=transformation.output_domain,
                metric=transformation.output_metric,
                replace_map={
                    col: replace_with[col]
                    for col in list(replace_with.keys())
                    if (
                        analytics_schema[col].column_type == ColumnType.DECIMAL
                        and analytics_schema[col].allow_nan
                    )
                },
            )
            transformation = transformation | replace_nan
        return transformation

    def visit_replace_infinity(self, query: ReplaceInfinity) -> Transformation:
        """Create a transformation from a ReplaceInfinity query expression."""
        child = self._visit_child(query.child)
        assert isinstance(child.output_domain, SparkDataFrameDomain)
        assert isinstance(
            child.output_metric, (IfGroupedBy, HammingDistance, SymmetricDifference)
        )
        analytics_schema = Schema(
            spark_dataframe_domain_to_analytics_columns(child.output_domain)
        )
        replace_with = query.replace_with.copy()
        if len(replace_with) == 0:
            replace_with = {
                col: (AnalyticsDefault.DECIMAL, AnalyticsDefault.DECIMAL)
                for col in analytics_schema.column_descs
                if analytics_schema[col].column_type == ColumnType.DECIMAL
            }
        transformation = ReplaceInfs(
            input_domain=child.output_domain,
            metric=child.output_metric,
            replace_map=replace_with,
        )
        return child | transformation

    def visit_drop_infinity(self, query: DropInfExpr) -> Transformation:
        """Create a transformation from a DropInfinity expression."""
        child = self._visit_child(query.child)
        analytics_schema = Schema(
            spark_dataframe_domain_to_analytics_columns(child.output_domain)
        )

        grouping_column: Optional[str] = None
        if isinstance(child.output_metric, IfGroupedBy):
            grouping_column = child.output_metric.column
            if grouping_column in query.columns:
                raise ValueError(
                    "Cannot drop infinite values in column"
                    f" {child.output_metric.column}, because it is being used as a"
                    " grouping column"
                )

        columns = query.columns.copy()
        if len(columns) == 0:
            columns = [
                col
                for col, cd in analytics_schema.column_descs.items()
                if (cd.column_type == ColumnType.DECIMAL and cd.allow_inf)
                and not (col == grouping_column)
            ]
        else:
            for col in columns:
                if analytics_schema.column_descs[col].column_type != ColumnType.DECIMAL:
                    raise ValueError(
                        "Cannot drop infinite values from column {col}, because that"
                        " column's type is not DECIMAL"
                    )

        transformation: Transformation = self._ensure_not_hamming(child)
        # visit_child will raise an exception if these aren't true;
        # these asserts are for mypy
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric, (IfGroupedBy, SymmetricDifference)
        )
        transformation = transformation | DropInfTransformation(
            transformation.output_domain, transformation.output_metric, columns
        )

        return transformation

    def visit_drop_null_and_nan(self, query: DropNullAndNan) -> Transformation:
        """Create a transformation from a DropNullAndNan expression."""
        child = self._visit_child(query.child)
        analytics_schema = Schema(
            spark_dataframe_domain_to_analytics_columns(child.output_domain)
        )

        grouping_column: Optional[str] = None
        if isinstance(child.output_metric, IfGroupedBy):
            grouping_column = child.output_metric.column
            if grouping_column in query.columns:
                raise ValueError(
                    "Cannot drop null values in column"
                    f" {child.output_metric.column}, because it is being used as a"
                    " grouping column"
                )

        columns = query.columns.copy()
        if len(columns) == 0:
            columns = [
                col
                for col, cd in analytics_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan) and not (col == grouping_column)
            ]

        transformation: Transformation = self._ensure_not_hamming(child)
        # visit_child will raise an exception if these aren't true;
        # these asserts are for mypy
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric, (IfGroupedBy, SymmetricDifference)
        )
        null_columns = [col for col in columns if analytics_schema[col].allow_null]
        if len(null_columns) != 0:
            transformation = transformation | DropNulls(
                transformation.output_domain, transformation.output_metric, null_columns
            )
        nan_columns = [col for col in columns if analytics_schema[col].allow_nan]
        if len(nan_columns) != 0:
            # Either a DropNulls transformation was created - for which these
            # should always be true - or it wasn't, in which case we already
            # checked these.
            # These asserts are just here so mypy knows what types to expect.
            assert isinstance(transformation.output_domain, SparkDataFrameDomain)
            assert isinstance(
                transformation.output_metric, (IfGroupedBy, SymmetricDifference)
            )
            transformation = transformation | DropNaNs(
                transformation.output_domain, transformation.output_metric, nan_columns
            )

        return transformation

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
