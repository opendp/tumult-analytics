"""Defines :class:`QueryExprCompiler` for compiling query expressions into a measurement.
"""  # pylint: disable=line-too-long
# TODO(#746): Check for UnaryExpr.
# TODO(#1044): Put the PureDPToZCDP converter directly on Vector measurements.
# TODO(#1547): Associate metric with output measure instead of algorithm for
#              adding noise.

# <placeholder: boilerplate>

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import sympy as sp
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._schema import (
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_schema_to_analytics_columns,
)
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
from tmlt.analytics.query_expr import PrivateSource, QueryExpr
from tmlt.analytics.query_expr import Rename as RenameExpr
from tmlt.analytics.query_expr import Select as SelectExpr
from tmlt.analytics.truncation_strategy import TruncationStrategy
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.measurements.aggregations import NoiseMechanism as CoreNoiseMechanism
from tmlt.core.measurements.aggregations import (
    create_average_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_quantile_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.composition import Composition
from tmlt.core.measures import PureDP, RhoZCDP
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
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.join import (
    DropAllTruncation,
    HashTopKTruncation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoin as PrivateJoinTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PublicJoin as PublicJoinTransformation,
)
from tmlt.core.transformations.spark_transformations.join import Truncation
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap as FlatMapTransformation,
)
from tmlt.core.transformations.spark_transformations.map import GroupingFlatMap
from tmlt.core.transformations.spark_transformations.map import Map as MapTransformation
from tmlt.core.transformations.spark_transformations.map import (
    RowToRowsTransformation,
    RowToRowTransformation,
)
from tmlt.core.transformations.spark_transformations.rename import (
    Rename as RenameTransformation,
)
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.type_utils import assert_never

DEFAULT_MECHANISM = "DEFAULT"
"""Constant used for DEFAULT noise mechanism"""

LAPLACE_MECHANISM = "LAPLACE"
"""Constant used for LAPLACE noise mechanism"""

GAUSSIAN_MECHANISM = "GAUSSIAN"
"""Constant used for GAUSSIAN noise mechanism"""


def _get_chaining_hint(value: Any) -> Callable:
    """Returns hint function that returns a fixed value.

    This can be used to avoid cell-var-in-loop errors.
    """
    return lambda _, __: value


def _get_negative_clamper(col: str) -> Callable:
    """Returns a function to clamp negative values of a column to 0.

    Args:
        col: Name of column being clamped.
    """

    def clamp_negatives(sdf: DataFrame) -> DataFrame:
        # pylint: disable=no-member
        return sdf.withColumn(col, sf.when(sf.col(col) < 0, 0).otherwise(sf.col(col)))

    return clamp_negatives


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
        for source_id, dataframe in public_sources.items():
            if any(
                dataframe.schema[column_name].nullable
                for column_name in dataframe.columns
            ):
                raise ValueError(
                    f"Public source ({source_id}) contains nullable columns."
                )

        measurements: List[Measurement] = []
        per_query_privacy_budget = privacy_budget / len(queries)
        for query in queries:
            if not isinstance(  # TODO(#746): Check for GroupbyAggregate
                query,
                (
                    GroupByBoundedAverage,
                    GroupByBoundedSTDEV,
                    GroupByBoundedSum,
                    GroupByBoundedVariance,
                    GroupByCount,
                    GroupByCountDistinct,
                    GroupByQuantile,
                ),
            ):
                raise NotImplementedError(query)

            # GroupByCountDistinct requires some special handling
            # to select certain columns
            if (
                isinstance(query, GroupByCountDistinct)
                and query.columns_to_count is not None
                and len(query.columns_to_count) > 0
            ):
                # select all the relevant columns
                # TODO(#1707): Remove _public_id handling
                groupby_columns: List[str]
                if query.groupby_keys._public_id is not None:
                    groupby_columns = list(
                        public_sources[query.groupby_keys._public_id].columns
                    )
                else:
                    groupby_columns = list(query.groupby_keys.schema().keys())
                # select_cols = query.columns_to_count + groupby.groupby_columns
                select_query = SelectExpr(
                    child=query.child, columns=query.columns_to_count + groupby_columns
                )
                query.child = select_query
                query.columns_to_count = None

            # This validates the query, the actual output schema is ignored
            query.accept(OutputSchemaVisitor(catalog))

            # Until TODO(#1547): Do an initial peek at the query to get the output schema and figure out what noise mechanism to be used
            expected_schema = query.child.accept(OutputSchemaVisitor(catalog))
            expected_output_domain = SparkDataFrameDomain(
                analytics_to_spark_columns_descriptor(expected_schema)
            )

            if isinstance(
                query,
                (
                    GroupByBoundedAverage,
                    GroupByBoundedSTDEV,
                    GroupByBoundedSum,
                    GroupByBoundedVariance,
                    GroupByCount,
                    GroupByCountDistinct,
                ),
            ):
                default_noise_mechanism = None
                if query.mechanism.name == DEFAULT_MECHANISM:
                    default_noise_mechanism = (
                        LAPLACE_MECHANISM
                        if isinstance(self._output_measure, PureDP)
                        else GAUSSIAN_MECHANISM
                    )

                if LAPLACE_MECHANISM in (query.mechanism.name, default_noise_mechanism):
                    if isinstance(query, (GroupByCount, GroupByCountDistinct)):
                        noise_mechanism = CoreNoiseMechanism.GEOMETRIC
                    else:
                        if isinstance(
                            expected_output_domain[query.measure_column],
                            SparkIntegerColumnDescriptor,
                        ):
                            noise_mechanism = CoreNoiseMechanism.GEOMETRIC
                        elif isinstance(
                            expected_output_domain[query.measure_column],
                            SparkFloatColumnDescriptor,
                        ):
                            noise_mechanism = CoreNoiseMechanism.LAPLACE
                        else:
                            raise AssertionError(
                                "Query's measure column should be numeric. This should"
                                " not happen and is probably a bug;  please let us know"
                                " so we can fix it!"
                            )
                elif GAUSSIAN_MECHANISM in (
                    query.mechanism.name,
                    default_noise_mechanism,
                ):
                    if isinstance(query, (GroupByCount, GroupByCountDistinct)):
                        noise_mechanism = CoreNoiseMechanism.DISCRETE_GAUSSIAN
                    else:
                        if isinstance(
                            expected_output_domain[query.measure_column],
                            SparkIntegerColumnDescriptor,
                        ):
                            noise_mechanism = CoreNoiseMechanism.DISCRETE_GAUSSIAN
                        else:
                            raise NotImplementedError(
                                f"{GAUSSIAN_MECHANISM} noise is not yet compatible with"
                                " floating-point values."
                            )
                else:
                    raise AssertionError(
                        "Unrecognized mechanism name. Supported mechanisms: "
                        f" '{DEFAULT_MECHANISM}', '{LAPLACE_MECHANISM}' or "
                        f"'{GAUSSIAN_MECHANISM}'. This should not happen and is "
                        " probably a bug; please let us know so we can fix it!"
                    )
                self._mechanism = noise_mechanism

            noise_mechanism = self._mechanism
            transformation = self.build_transformation(
                query=query.child,
                input_domain=input_domain,
                input_metric=input_metric,
                public_sources=public_sources,
                catalog=catalog,
            )
            mid_stability = transformation.stability_function(stability)
            if not isinstance(
                transformation.output_metric,
                (SymmetricDifference, HammingDistance, IfGroupedBy),
            ):
                raise AssertionError(
                    "Unrecognized output metric. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            mid_domain = transformation.output_domain
            if not isinstance(mid_domain, SparkDataFrameDomain):
                raise AssertionError(
                    "Unrecognized output domain. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )

            # TODO(#1044 and #1547): Update condition to when issue is resolved.
            # isinstance(self._output_measure, RhoZCDP)
            output_metric: Union[RootSumOfSquared, SumOf] = (
                RootSumOfSquared(SymmetricDifference())
                if noise_mechanism == CoreNoiseMechanism.DISCRETE_GAUSSIAN
                else SumOf(SymmetricDifference())
            )
            # TODO(#1707): Remove _public_id handling
            groupby_df: DataFrame
            if query.groupby_keys._public_id is not None:
                groupby_df = public_sources[query.groupby_keys._public_id]
            else:
                groupby_df = query.groupby_keys.dataframe()
            groupby = GroupBy(
                input_domain=mid_domain,
                input_metric=transformation.output_metric,
                output_metric=output_metric,
                group_keys=groupby_df,
            )
            noisy_aggregation: Measurement
            if isinstance(
                query,
                (
                    GroupByBoundedAverage,
                    GroupByBoundedSTDEV,
                    GroupByBoundedSum,
                    GroupByBoundedVariance,
                ),
            ):
                lower, upper = _get_query_bounds(query)
                if isinstance(query, GroupByBoundedAverage):
                    noisy_aggregation = create_average_measurement(
                        input_domain=mid_domain,
                        input_metric=transformation.output_metric,
                        measure_column=query.measure_column,
                        lower=lower,
                        upper=upper,
                        noise_mechanism=noise_mechanism,
                        d_in=mid_stability,
                        d_out=per_query_privacy_budget,
                        output_measure=self.output_measure,
                        groupby_transformation=groupby,
                        average_column=query.output_column,
                    )
                elif isinstance(query, GroupByBoundedSTDEV):
                    noisy_aggregation = create_standard_deviation_measurement(
                        input_domain=mid_domain,
                        input_metric=transformation.output_metric,
                        measure_column=query.measure_column,
                        lower=lower,
                        upper=upper,
                        noise_mechanism=noise_mechanism,
                        d_in=mid_stability,
                        d_out=per_query_privacy_budget,
                        output_measure=self.output_measure,
                        groupby_transformation=groupby,
                        standard_deviation_column=query.output_column,
                    )
                elif isinstance(query, GroupByBoundedSum):
                    noisy_aggregation = create_sum_measurement(
                        input_domain=mid_domain,
                        input_metric=transformation.output_metric,
                        measure_column=query.measure_column,
                        lower=lower,
                        upper=upper,
                        noise_mechanism=noise_mechanism,
                        d_in=mid_stability,
                        d_out=per_query_privacy_budget,
                        output_measure=self.output_measure,
                        groupby_transformation=groupby,
                        sum_column=query.output_column,
                    )
                elif isinstance(query, GroupByBoundedVariance):
                    noisy_aggregation = create_variance_measurement(
                        input_domain=mid_domain,
                        input_metric=transformation.output_metric,
                        measure_column=query.measure_column,
                        lower=lower,
                        upper=upper,
                        noise_mechanism=noise_mechanism,
                        d_in=mid_stability,
                        d_out=per_query_privacy_budget,
                        output_measure=self.output_measure,
                        groupby_transformation=groupby,
                        variance_column=query.output_column,
                    )
                else:
                    assert_never(query)
            elif isinstance(query, GroupByCount):
                noisy_aggregation = create_count_measurement(
                    input_domain=mid_domain,
                    input_metric=transformation.output_metric,
                    noise_mechanism=noise_mechanism,
                    d_in=mid_stability,
                    d_out=per_query_privacy_budget,
                    output_measure=self.output_measure,
                    groupby_transformation=groupby,
                    count_column=query.output_column,
                )
            elif isinstance(query, GroupByCountDistinct):
                noisy_aggregation = create_count_distinct_measurement(
                    input_domain=mid_domain,
                    input_metric=transformation.output_metric,
                    noise_mechanism=noise_mechanism,
                    d_in=mid_stability,
                    d_out=per_query_privacy_budget,
                    output_measure=self.output_measure,
                    groupby_transformation=groupby,
                    count_column=query.output_column,
                )

            elif isinstance(query, GroupByQuantile):
                lower, upper = _get_query_bounds(query)
                noisy_aggregation = create_quantile_measurement(
                    input_domain=mid_domain,
                    input_metric=transformation.output_metric,
                    measure_column=query.measure_column,
                    quantile=query.quantile,
                    lower=lower,
                    upper=upper,
                    d_in=mid_stability,
                    d_out=per_query_privacy_budget,
                    output_measure=self.output_measure,
                    groupby_transformation=groupby,
                    quantile_column=query.output_column,
                )
            else:
                assert_never(query)

            if (
                noisy_aggregation.privacy_function(mid_stability)
                != per_query_privacy_budget
            ):
                raise AssertionError(
                    "Privacy function does not match per-query privacy budget. "
                    "This is probably a bug; please let us know so we can "
                    "fix it!"
                )

            query_measurement = transformation | noisy_aggregation
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
        # This also verifies that the query is compatible with the catalog
        for source_id, dataframe in public_sources.items():
            if any(
                dataframe.schema[column_name].nullable
                for column_name in dataframe.columns
            ):
                raise ValueError(
                    f"Public source ({source_id}) contains nullable columns."
                )

        expected_schema = query.accept(OutputSchemaVisitor(catalog))

        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )

        # Inner metric used for IfGroupedBy
        inner_metric: Union[SumOf, RootSumOfSquared]
        if self.mechanism in (CoreNoiseMechanism.LAPLACE, CoreNoiseMechanism.GEOMETRIC):
            # TODO(#1044 and #1547): Update condition to when issue is resolved.
            # isinstance(self._output_measure, RhoZCDP)
            inner_metric = SumOf(SymmetricDifference())
        else:
            if self.mechanism != CoreNoiseMechanism.DISCRETE_GAUSSIAN:
                raise RuntimeError(
                    f"Unsupported mechanism {self.mechanism}. "
                    "Supported mechanisms are "
                    f"{CoreNoiseMechanism.DISCRETE_GAUSSIAN}, "
                    f"{CoreNoiseMechanism.LAPLACE}, and"
                    f"{CoreNoiseMechanism.GEOMETRIC}."
                )
            inner_metric = RootSumOfSquared(SymmetricDifference())
        expected_output_metric = (
            SymmetricDifference()
            if expected_schema.grouping_column is None
            else IfGroupedBy(expected_schema.grouping_column, inner_metric)
        )

        transformation: Transformation
        if isinstance(query, PrivateSource):
            transformation = GetValue(input_domain, input_metric, query.source_id)
            if transformation.output_metric != expected_output_metric:
                raise AssertionError(
                    "Transformation output metric does not "
                    "match expected output metric. This is probably a bug; "
                    "please tell us about it so we can fix it!"
                )
            if transformation.output_domain != expected_output_domain:
                raise AssertionError(
                    "Transformation output domain does not match "
                    "expected output domain. This is probably a bug; "
                    "please tell us about it so we can fix it!"
                )
            return transformation

        if not isinstance(
            query,
            (
                FilterExpr,
                FlatMapExpr,
                JoinPrivateExpr,
                JoinPublicExpr,
                MapExpr,
                RenameExpr,
                SelectExpr,
            ),
        ):  # TODO(#746): Check for UnaryExpr
            raise NotImplementedError(query)

        previous_transformation = self.build_transformation(
            query.child, input_domain, input_metric, public_sources, catalog
        )
        previous_domain = previous_transformation.output_domain
        if not isinstance(previous_domain, SparkDataFrameDomain):
            raise AssertionError(
                "Child query does not have a recognized output domain. "
                "This is probably a bug; please let us know about it so we can "
                "fix it!"
            )
        previous_metric = previous_transformation.output_metric
        if not isinstance(
            previous_metric, (IfGroupedBy, SymmetricDifference, HammingDistance)
        ):
            raise AssertionError(
                "Child query does not have a recognized output "
                "metric. This is probably a bug; please let us know about "
                "it so we can fix it!"
            )

        transformer_input_domain: Domain
        output_domain: Domain
        if isinstance(query, FlatMapExpr):
            transformer_input_domain = SparkRowDomain(previous_domain.schema)
            spark_columns_descriptor = analytics_to_spark_columns_descriptor(
                query.schema_new_columns
            )
            if query.augment:
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
                trusted_f=getattr(query, "f"),
                augment=query.augment,
            )
            if query.schema_new_columns.grouping_column is not None:
                transformation = GroupingFlatMap(
                    output_metric=inner_metric,  # (sqrt) sum of (squared) symm diff
                    row_transformer=row_transformer,
                    max_num_rows=query.max_num_rows,
                )
            else:
                if isinstance(previous_metric, HammingDistance):
                    previous_metric = SymmetricDifference()
                    previous_transformation = (
                        previous_transformation
                        | HammingDistanceToSymmetricDifference(previous_domain)
                    )
                transformation = FlatMapTransformation(
                    metric=previous_metric,
                    row_transformer=row_transformer,
                    max_num_rows=query.max_num_rows,
                )
        elif isinstance(query, MapExpr):
            transformer_input_domain = SparkRowDomain(previous_domain.schema)
            spark_columns_descriptor = analytics_to_spark_columns_descriptor(
                query.schema_new_columns
            )
            if query.augment:
                output_schema = {
                    **transformer_input_domain.schema,
                    **spark_columns_descriptor,
                }
            else:
                output_schema = spark_columns_descriptor
            output_domain = SparkRowDomain(output_schema)

            if isinstance(previous_metric, HammingDistance):
                previous_metric = SymmetricDifference()
                previous_transformation = (
                    previous_transformation
                    | HammingDistanceToSymmetricDifference(previous_domain)
                )
            transformation = MapTransformation(
                metric=previous_metric,
                row_transformer=RowToRowTransformation(
                    input_domain=transformer_input_domain,
                    output_domain=output_domain,
                    trusted_f=getattr(query, "f"),
                    augment=query.augment,
                ),
            )
        elif isinstance(query, FilterExpr):
            domain = SparkDataFrameDomain(previous_domain.schema)
            if isinstance(previous_metric, HammingDistance):
                previous_metric = SymmetricDifference()
                previous_transformation = (
                    previous_transformation
                    | HammingDistanceToSymmetricDifference(previous_domain)
                )
            transformation = FilterTransformation(
                domain=domain, metric=previous_metric, filter_expr=query.predicate
            )
        elif isinstance(query, RenameExpr):
            if not isinstance(query.column_mapper, dict):
                # since the query was already verified, this shouldn't happen
                raise AssertionError(
                    "A query to rename dataframes must have a column_mapper "
                    "dictionary mapping old column names to new column names. "
                    "This is probably a bug; let us know so we can fix it!"
                )
            transformation = RenameTransformation(
                input_domain=previous_domain,
                metric=previous_metric,
                rename_mapping=query.column_mapper,
            )
        elif isinstance(query, SelectExpr):
            transformation = SelectTransformation(
                input_domain=previous_domain,
                metric=previous_metric,
                columns=list(query.columns),
            )
        elif isinstance(query, JoinPublicExpr):
            if isinstance(query.public_table, str):
                public_df = public_sources[query.public_table]
            else:
                public_df = query.public_table
            if isinstance(previous_metric, HammingDistance):
                previous_metric = SymmetricDifference()
                previous_transformation = (
                    previous_transformation
                    | HammingDistanceToSymmetricDifference(previous_domain)
                )

            public_df_schema = Schema(
                spark_schema_to_analytics_columns(public_df.schema)
            )
            transformation = PublicJoinTransformation(
                input_domain=SparkDataFrameDomain(previous_domain.schema),
                public_df=public_df,
                public_df_domain=SparkDataFrameDomain(
                    analytics_to_spark_columns_descriptor(public_df_schema)
                ),
                join_cols=list(query.join_columns) if query.join_columns else None,
                metric=previous_metric,
            )
        elif isinstance(query, JoinPrivateExpr):
            left_metric, left_transformation = (
                previous_metric,
                previous_transformation,
            )

            # Check that left metrics are correct
            if isinstance(left_metric, IfGroupedBy):
                raise ValueError(
                    "Left operand used a grouping transformation. "
                    "This is not yet supported for private joins."
                )
            if isinstance(left_metric, HammingDistance):
                if not isinstance(
                    left_transformation.output_domain, SparkDataFrameDomain
                ):
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
                    "The only supported output metric is "
                    f"{SymmetricDifference()}"
                )

            add_left_transformation = AugmentDictTransformation(
                left_transformation
                | CreateDictFromValue(
                    input_domain=left_transformation.output_domain,
                    input_metric=left_transformation.output_metric,
                    key="left_output",
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
            right_transformation = self.build_transformation(
                query.right_operand_expr,
                add_left_transformation.output_domain,
                add_left_transformation.output_metric,
                public_sources,
                catalog,
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
                    key="right_output",
                )
            )
            # input = {left_input, right_input, left_output},
            # output = {left_input, right_input, left_output, right_output}

            combined_transformations = (
                add_left_transformation | add_right_transformation
            )
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
                keys=["left_output", "right_output"],
            )
            # input = {left_input, right_input}, output = {left_output, right_output}

            # Create the PrivateJoin transformation
            previous_domain = previous_transformation.output_domain
            if not isinstance(previous_domain, DictDomain):
                raise AssertionError(
                    "This is a bug. Please let us know so we can fix it!"
                )
            left_domain = previous_domain.key_to_domain["left_output"]
            right_domain = previous_domain.key_to_domain["right_output"]
            if not isinstance(left_domain, SparkDataFrameDomain):
                raise ValueError(
                    "Left operand has an output domain that is not a "
                    "SparkDataFrameDomain."
                )
            if not isinstance(right_domain, SparkDataFrameDomain):
                raise ValueError(
                    "Right operand has an output domain that is not a "
                    "SparkDataFrameDomain."
                )

            join_keys = (
                query.join_columns
                if query.join_columns is not None
                else sorted(
                    set(left_domain.schema) & set(right_domain.schema),
                    key=list(left_domain.schema).index,
                )
            )

            def truncation_strategy_to_truncator(strategy, domain) -> Truncation:
                if isinstance(strategy, TruncationStrategy.DropExcess):
                    return HashTopKTruncation(
                        domain=domain, keys=join_keys, threshold=strategy.max_records
                    )
                elif isinstance(strategy, TruncationStrategy.DropNonUnique):
                    return DropAllTruncation(domain=domain, keys=join_keys, threshold=1)
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

            left_truncator = truncation_strategy_to_truncator(
                query.truncation_strategy_left, left_domain
            )
            right_truncator = truncation_strategy_to_truncator(
                query.truncation_strategy_right, right_domain
            )

            transformation = PrivateJoinTransformation(
                input_domain=previous_domain,
                left="left_output",
                right="right_output",
                left_truncator=left_truncator,
                right_truncator=right_truncator,
                join_cols=query.join_columns,
            )

        else:
            raise AssertionError("This should be unreachable")

        transformation = previous_transformation | transformation
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
        return transformation
