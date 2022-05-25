"""Defines a visitor for creating noisy measurements from query expressions."""

# <placeholder: boilerplate>

import dataclasses
from typing import Any, Dict, List, Tuple, Union

import sympy as sp
from pyspark.sql import DataFrame

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._schema import (
    ColumnType,
    analytics_to_spark_columns_descriptor,
    spark_dataframe_domain_to_analytics_columns,
)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
    Filter,
    FlatMap,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    JoinPrivate,
    JoinPublic,
    Map,
    PrivateSource,
    QueryExpr,
    QueryExprVisitor,
    Rename,
    ReplaceNullAndNan,
    Select,
    StdevMechanism,
    SumMechanism,
    VarianceMechanism,
)
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_average_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_quantile_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.nan import ReplaceNaNs
from tmlt.core.utils.exact_number import ExactNumber


def _get_query_bounds(
    query: Union[
        GroupByBoundedAverage,
        GroupByBoundedSTDEV,
        GroupByBoundedSum,
        GroupByBoundedVariance,
        GroupByQuantile,
    ]
) -> Tuple[ExactNumber, ExactNumber]:
    """Returns lower and upper clamping bounds of a query as :class:`~.ExactNumbers`."""
    if query.high == query.low:
        bound = ExactNumber.from_float(query.high, round_up=True)
        return (bound, bound)
    lower_ceiling = ExactNumber.from_float(query.low, round_up=True)
    upper_floor = ExactNumber.from_float(query.high, round_up=False)
    return (lower_ceiling, upper_floor)


class MeasurementVisitor(QueryExprVisitor):
    """A visitor to create a measurement from a query expression."""

    def __init__(
        self,
        per_query_privacy_budget: sp.Expr,
        stability: Dict[str, sp.Expr],
        input_domain: DictDomain,
        input_metric: DictMetric,
        output_measure: Union[PureDP, RhoZCDP],
        default_mechanism: NoiseMechanism,
        public_sources: Dict[str, DataFrame],
        catalog: Catalog,
    ):
        """Constructor for MeasurementVisitor."""
        self.budget = per_query_privacy_budget
        self.stability = stability
        self.input_domain = input_domain
        self.input_metric = input_metric
        self.default_mechanism = default_mechanism
        self.public_sources = public_sources
        self.output_measure = output_measure
        self.catalog = catalog

    def _visit_child_transformation(
        self, expr: QueryExpr, mechanism: NoiseMechanism
    ) -> Transformation:
        """Visit a child transformation, producing a transformation."""
        tv = TransformationVisitor(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            mechanism=mechanism,
            public_sources=self.public_sources,
        )
        transformation = expr.accept(tv)
        if not isinstance(transformation, Transformation):
            raise AssertionError(
                "Expression failed to produce a transformation. "
                "This is probably a bug; please let us know about it "
                "so we can fix it!"
            )
        tv.validate_transformation(expr, transformation, self.catalog)

        if not isinstance(
            transformation.output_metric,
            (SymmetricDifference, HammingDistance, IfGroupedBy),
        ):
            raise AssertionError(
                "Unrecognized output metric. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )
        if not isinstance(transformation.output_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Unrecognized output domain. This is probably a bug; "
                "please let us know about it so we can fix it!"
            )

        return transformation

    def _pick_noise_for_count(
        self, query: Union[GroupByCount, GroupByCountDistinct]
    ) -> NoiseMechanism:
        """Pick the noise mechanism to use for a count or count-distinct query."""
        requested_mechanism: NoiseMechanism
        if query.mechanism in (CountMechanism.DEFAULT, CountDistinctMechanism.DEFAULT):
            if isinstance(self.output_measure, PureDP):
                requested_mechanism = NoiseMechanism.LAPLACE
            else:  # output measure is RhoZCDP
                requested_mechanism = NoiseMechanism.DISCRETE_GAUSSIAN
        elif query.mechanism in (
            CountMechanism.LAPLACE,
            CountDistinctMechanism.LAPLACE,
        ):
            requested_mechanism = NoiseMechanism.LAPLACE
        elif query.mechanism in (
            CountMechanism.GAUSSIAN,
            CountDistinctMechanism.GAUSSIAN,
        ):
            requested_mechanism = NoiseMechanism.DISCRETE_GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize the mechanism name {query.mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE, and GAUSSIAN."
            )

        if requested_mechanism == NoiseMechanism.LAPLACE:
            return NoiseMechanism.GEOMETRIC
        elif requested_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
            return NoiseMechanism.DISCRETE_GAUSSIAN
        else:
            # This should never happen
            raise AssertionError(
                f"Did not recognize the requested mechanism {requested_mechanism}."
                " This is probably a bug; please let us know about it so we can fix it!"
            )

    def _pick_noise_for_non_count(
        self,
        query: Union[
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ],
        measure_column_type: SparkColumnDescriptor,
    ) -> NoiseMechanism:
        """Pick the noise mechnaism for non-count queries.

        GroupByQuantile only supports one noise mechanism, so it is not
        included here.
        """
        requested_mechanism: NoiseMechanism
        if query.mechanism in (
            SumMechanism.DEFAULT,
            AverageMechanism.DEFAULT,
            VarianceMechanism.DEFAULT,
            StdevMechanism.DEFAULT,
        ):
            requested_mechanism = (
                NoiseMechanism.LAPLACE
                if isinstance(self.output_measure, PureDP)
                else NoiseMechanism.DISCRETE_GAUSSIAN
            )
        elif query.mechanism in (
            SumMechanism.LAPLACE,
            AverageMechanism.LAPLACE,
            VarianceMechanism.LAPLACE,
            StdevMechanism.LAPLACE,
        ):
            requested_mechanism = NoiseMechanism.LAPLACE
        elif query.mechanism in (
            SumMechanism.GAUSSIAN,
            AverageMechanism.GAUSSIAN,
            VarianceMechanism.GAUSSIAN,
            StdevMechanism.GAUSSIAN,
        ):
            requested_mechanism = NoiseMechanism.DISCRETE_GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize requested mechanism {query.mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE,  and GAUSSIAN."
            )

        # If the query requested a Laplace measure ...
        if requested_mechanism == NoiseMechanism.LAPLACE:
            if isinstance(measure_column_type, SparkIntegerColumnDescriptor):
                return NoiseMechanism.GEOMETRIC
            elif isinstance(measure_column_type, SparkFloatColumnDescriptor):
                return NoiseMechanism.LAPLACE
            else:
                raise AssertionError(
                    "Query's measure column should be numeric. This should"
                    " not happen and is probably a bug;  please let us know"
                    " so we can fix it!"
                )

        # If the query requested a Gaussian measure...
        elif requested_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
            if isinstance(measure_column_type, SparkIntegerColumnDescriptor):
                return NoiseMechanism.DISCRETE_GAUSSIAN
            else:
                raise NotImplementedError(
                    f"{NoiseMechanism.DISCRETE_GAUSSIAN} noise is not yet"
                    " compatible with floating-point values."
                )

        # The requested_mechanism should be either LAPLACE or
        # DISCRETE_GAUSSIAN, so something has gone awry
        else:
            raise AssertionError(
                f"Did not recognize requested mechanism {requested_mechanism}."
                " This is probably a bug; please let us know about it so we can fix it!"
            )

    @staticmethod
    def _build_groupby(
        input_domain: SparkDataFrameDomain,
        input_metric: Union[HammingDistance, SymmetricDifference, IfGroupedBy],
        groupby_keys: KeySet,
        mechanism: NoiseMechanism,
    ) -> GroupBy:
        """Build a groupby query from the parameters provided.

        This groupby query will run after the provided Transformation.
        """
        # TODO(#1044 and #1547): Update condition to when issue is resolved.
        # isinstance(self._output_measure, RhoZCDP)
        use_l2 = mechanism == NoiseMechanism.DISCRETE_GAUSSIAN

        groupby_df: DataFrame = groupby_keys.dataframe()

        return GroupBy(
            input_domain=input_domain,
            input_metric=input_metric,
            use_l2=use_l2,
            group_keys=groupby_df,
        )

    @dataclasses.dataclass
    class _AggInfo:
        """All the information you need for some query exprs.

        Supported types:
        - GroupByBoundedAverage
        - GroupByBoundedSTDEV
        - GroupByBoundedSum
        - GroupByBoundedVariance
        """

        mechanism: NoiseMechanism
        transformation: Transformation
        mid_stability: sp.Expr
        groupby: GroupBy
        lower_bound: ExactNumber
        upper_bound: ExactNumber

    def _build_common(
        self,
        query: Union[
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ],
    ) -> _AggInfo:
        """Everything you need to build a measurement for these query types."""
        lower_bound, upper_bound = _get_query_bounds(query)

        expected_schema = query.child.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        measure_column_type = expected_output_domain[query.measure_column]

        mechanism = self._pick_noise_for_non_count(query, measure_column_type)
        transformation = self._visit_child_transformation(query.child, mechanism)
        # _visit_child_transformation already raises an error if these aren't true
        # these assert statements are just here for MyPy's benefit
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )
        mid_stability = transformation.stability_function(self.stability)
        groupby = self._build_groupby(
            transformation.output_domain,
            transformation.output_metric,
            query.groupby_keys,
            mechanism,
        )
        return MeasurementVisitor._AggInfo(
            mechanism=mechanism,
            transformation=transformation,
            mid_stability=mid_stability,
            groupby=groupby,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def _validate_measurement(self, measurement: Measurement, mid_stability: sp.Expr):
        """Validate a measurement."""
        if measurement.privacy_function(mid_stability) != self.budget:
            raise AssertionError(
                "Privacy function does not match per-query privacy budget. "
                "This is probably a bug; please let us know so we can "
                "fix it!"
            )

    def visit_groupby_count(self, query: GroupByCount) -> Measurement:
        """Create a measurement from a GroupByCount query expression."""
        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_count(query)
        mechanism = self._pick_noise_for_count(query)
        transformation = self._visit_child_transformation(query.child, mechanism)
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
            query.groupby_keys,
            mechanism,
        )

        agg = create_count_measurement(
            input_domain=transformation.output_domain,
            input_metric=transformation.output_metric,
            noise_mechanism=mechanism,
            d_in=mid_stability,
            d_out=self.budget,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            count_column=query.output_column,
        )
        self._validate_measurement(agg, mid_stability)
        return transformation | agg

    def visit_groupby_count_distinct(self, query: GroupByCountDistinct) -> Measurement:
        """Create a measurement from a GroupByCountDistinct query expression."""
        # Yes, you need both of these:
        # columns_to_count=[] means something different from
        # columns_to_count=None
        if query.columns_to_count is not None and len(query.columns_to_count) > 0:
            # select all relevant columns
            groupby_columns: List[str] = list(query.groupby_keys.schema().keys())
            # select_cols = all columns to count + groupby_columns
            select_query = Select(
                child=query.child, columns=query.columns_to_count + groupby_columns
            )
            # Use of dataclasses.replace guarantees that a copy is created,
            # rather than mutating the original QueryExpr.
            query = dataclasses.replace(
                query, child=select_query, columns_to_count=None
            )

        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_count_distinct(query)
        mechanism = self._pick_noise_for_count(query)
        transformation = self._visit_child_transformation(query.child, mechanism)
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
            query.groupby_keys,
            mechanism,
        )

        agg = create_count_distinct_measurement(
            input_domain=transformation.output_domain,
            input_metric=transformation.output_metric,
            noise_mechanism=mechanism,
            d_in=mid_stability,
            d_out=self.budget,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            count_column=query.output_column,
        )
        self._validate_measurement(agg, mid_stability)
        return transformation | agg

    def visit_groupby_quantile(self, query: GroupByQuantile) -> Measurement:
        """Create a measurement from a GroupByQuantile query expression."""
        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_quantile(query)

        transformation = self._visit_child_transformation(
            query.child, self.default_mechanism
        )
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
            query.groupby_keys,
            self.default_mechanism,
        )

        agg = create_quantile_measurement(
            input_domain=transformation.output_domain,
            input_metric=transformation.output_metric,
            measure_column=query.measure_column,
            quantile=query.quantile,
            lower=query.low,
            upper=query.high,
            d_in=mid_stability,
            d_out=self.budget,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            quantile_column=query.output_column,
        )
        self._validate_measurement(agg, mid_stability)
        return transformation | agg

    def visit_groupby_bounded_sum(self, query: GroupByBoundedSum) -> Measurement:
        """Create a measurement from a GroupByBoundedSum query expression."""
        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_sum(query)
        info = self._build_common(query)

        # _visit_child_transformation already raises an error if these aren't true
        # these are just here for MyPy's benefit
        assert isinstance(info.transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            info.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        schema = spark_dataframe_domain_to_analytics_columns(
            info.transformation.output_domain
        )
        nan_cols = [
            k
            for k in list(schema.keys())
            if schema[k].column_type == ColumnType.DECIMAL and schema[k].allow_nan
        ]
        if len(nan_cols) != 0:
            info.transformation = info.transformation | ReplaceNaNs(
                input_domain=info.transformation.output_domain,
                metric=info.transformation.output_metric,
                replace_map={k: float(0) for k in nan_cols},
            )
            info.mid_stability = info.transformation.stability_function(self.stability)
            if not isinstance(info.transformation.output_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unexpected output domain. This is probably a bug; please let us"
                    " know so we can fix it!"
                )
            if not isinstance(
                info.transformation.output_metric,
                (IfGroupedBy, HammingDistance, SymmetricDifference),
            ):
                raise AssertionError(
                    "Unexpected output metric. This is probably a bug; please let us"
                    " know so we can fix it!"
                )
            info.groupby = self._build_groupby(
                mechanism=info.mechanism,
                input_domain=info.transformation.output_domain,
                input_metric=info.transformation.output_metric,
                groupby_keys=query.groupby_keys,
            )
        # It is impossible to reach this code without this being true,
        # but if you remove these asserts, mypy will complain about it
        assert isinstance(info.transformation.output_domain, SparkDataFrameDomain)
        assert isinstance(
            info.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )

        agg = create_sum_measurement(
            input_domain=info.transformation.output_domain,
            input_metric=info.transformation.output_metric,
            measure_column=query.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.budget,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            sum_column=query.output_column,
        )
        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg

    def visit_groupby_bounded_average(
        self, query: GroupByBoundedAverage
    ) -> Measurement:
        """Create a measurement from a GroupByBoundedAverage query expression."""
        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_average(query)
        info = self._build_common(query)

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
            measure_column=query.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.budget,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            average_column=query.output_column,
        )
        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg

    def visit_groupby_bounded_variance(
        self, query: GroupByBoundedVariance
    ) -> Measurement:
        """Create a measurement from a GroupByBoundedVariance query expression."""
        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_variance(query)
        info = self._build_common(query)

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
            measure_column=query.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.budget,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            variance_column=query.output_column,
        )
        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg

    def visit_groupby_bounded_stdev(self, query: GroupByBoundedSTDEV) -> Measurement:
        """Create a measurement from a GroupByBoundedStdev query expression."""
        # Peek at the schema, to see if there are errors there
        OutputSchemaVisitor(self.catalog).visit_groupby_bounded_stdev(query)
        info = self._build_common(query)

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
            measure_column=query.measure_column,
            lower=info.lower_bound,
            upper=info.upper_bound,
            noise_mechanism=info.mechanism,
            d_in=info.mid_stability,
            d_out=self.budget,
            output_measure=self.output_measure,
            groupby_transformation=info.groupby,
            standard_deviation_column=query.output_column,
        )
        self._validate_measurement(agg, info.mid_stability)
        return info.transformation | agg

    # None of these produce measurements, so they all return a NotImplementedError
    def visit_private_source(self, expr: PrivateSource) -> Any:
        """Visit a PrivateSource query expression (raises an error)."""
        raise NotImplementedError

    def visit_rename(self, expr: Rename) -> Any:
        """Visit a Rename query expression (raises an error)."""
        raise NotImplementedError

    def visit_filter(self, expr: Filter) -> Any:
        """Visit a Filter query expression (raises an error)."""
        raise NotImplementedError

    def visit_select(self, expr: Select) -> Any:
        """Visit a Select query expression (raises an error)."""
        raise NotImplementedError

    def visit_map(self, expr: Map) -> Any:
        """Visit a Map query expression (raises an error)."""
        raise NotImplementedError

    def visit_flat_map(self, expr: FlatMap) -> Any:
        """Visit a FlatMap query expression (raises an error)."""
        raise NotImplementedError

    def visit_join_private(self, expr: JoinPrivate) -> Any:
        """Visit a JoinPrivate query expression (raises an error)."""
        raise NotImplementedError

    def visit_join_public(self, expr: JoinPublic) -> Any:
        """Visit a JoinPublic query expression (raises an error)."""
        raise NotImplementedError

    def visit_replace_null_and_nan(self, expr: ReplaceNullAndNan) -> Any:
        """Visit a ReplaceNullAndNan query expression (raises an error)."""
        raise NotImplementedError
