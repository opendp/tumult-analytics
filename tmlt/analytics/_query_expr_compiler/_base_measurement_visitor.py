"""Defines a base class for building measurement visitors."""
import dataclasses
import math
import warnings
from abc import abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import sympy as sp
from pyspark.sql import DataFrame, SparkSession

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructType,
    TimestampType,
)
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_average_measurement,
    create_bound_selection_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_partition_selection_measurement,
    create_quantile_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.interactive_measurements import (
    MeasurementQuery,
    Queryable,
    SequentialComposition,
    create_adaptive_composition,
)
from tmlt.core.measurements.postprocess import NonInteractivePostProcess, PostProcess
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import (
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.converters import UnwrapIfGroupedBy
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._noise_info import NoiseInfo, _noise_from_measurement
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._schema import ColumnType, Schema
from tmlt.analytics._table_identifier import Identifier
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics._transformation_utils import get_table_from_ref
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)
from tmlt.analytics.query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
)
from tmlt.analytics.query_expr import DropInfinity as DropInfExpr
from tmlt.analytics.query_expr import DropNullAndNan, EnforceConstraint
from tmlt.analytics.query_expr import Filter as FilterExpr
from tmlt.analytics.query_expr import FlatMap as FlatMapExpr
from tmlt.analytics.query_expr import (
    GetBounds,
    GetGroups,
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
from tmlt.analytics.query_expr import PrivateSource as PrivateSourceExpr
from tmlt.analytics.query_expr import QueryExpr, QueryExprVisitor
from tmlt.analytics.query_expr import Rename as RenameExpr
from tmlt.analytics.query_expr import ReplaceInfinity, ReplaceNullAndNan
from tmlt.analytics.query_expr import Select as SelectExpr
from tmlt.analytics.query_expr import (
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)


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


def _get_truncatable_constraints(
    constraints: List[Constraint],
) -> List[Tuple[Constraint, ...]]:
    """Get sets of constraints that produce a finite aggregation stability."""
    # Because of constraint simplification, there should be at most one
    # MaxRowsPerID constraint, and at most one MaxGroupsPerID and
    # MaxRowsPerGroupPerID each per column.
    max_rows_per_id = next(
        (c for c in constraints if isinstance(c, MaxRowsPerID)), None
    )
    max_groups_per_id = {
        c.grouping_column: c for c in constraints if isinstance(c, MaxGroupsPerID)
    }
    max_rows_per_group_per_id = {
        c.grouping_column: c for c in constraints if isinstance(c, MaxRowsPerGroupPerID)
    }

    ret: List[Tuple[Constraint, ...]] = [
        (max_groups_per_id[col], max_rows_per_group_per_id[col])
        for col in set(max_groups_per_id) & set(max_rows_per_group_per_id)
    ]
    if max_rows_per_id:
        ret.append((max_rows_per_id,))
    return ret


def _constraint_stability(
    constraints: Tuple[Constraint, ...],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    grouping_columns: List[str],
) -> float:
    """Compute the transformation stability of applying the given constraints.

    The values produced by this method are not intended for use doing actual
    stability calculations, they are just to provide an easy way to evaluate the
    relative stabilities of different possible truncations.
    """
    if len(constraints) == 1 and isinstance(constraints[0], MaxRowsPerID):
        return constraints[0].max
    elif (
        len(constraints) == 2
        and isinstance(constraints[0], MaxGroupsPerID)
        and isinstance(constraints[1], MaxRowsPerGroupPerID)
    ):
        if (
            output_measure == PureDP()
            or output_measure == ApproxDP()
            or constraints[0].grouping_column not in grouping_columns
        ):
            return constraints[0].max * constraints[1].max
        elif output_measure == RhoZCDP():
            return math.sqrt(constraints[0].max) * constraints[1].max
        else:
            raise AssertionError(
                f"Unknown output measure {output_measure}. "
                "This is probably a bug; please let us know about it so we can fix it!"
            )
    else:
        raise AssertionError(
            f"Constraints {constraints} are not a combination for which a stability "
            "can be computed. This is probably a bug; please let us know about it "
            "so we can fix it!"
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
    if isinstance(query.groupby_keys, KeySet):
        groupby_columns = query.groupby_keys.dataframe().columns
    else:
        groupby_columns = query.groupby_keys

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
        # MaxRowsPerGroupPerID(grouping_column, 1) constraint.
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


def _create_single_row_df_with_spark_schema(schema: StructType) -> DataFrame:
    """Create a single-row DataFrame with the given schema."""
    spark = SparkSession.builder.getOrCreate()
    default_values = []
    for field in schema.fields:
        default_value: Any
        if isinstance(field.dataType, (ByteType, ShortType, IntegerType, LongType)):
            default_value = 0
        elif isinstance(field.dataType, (FloatType, DoubleType)):
            default_value = 0.0
        elif isinstance(field.dataType, StringType):
            default_value = ""
        elif isinstance(field.dataType, BooleanType):
            default_value = False
        elif isinstance(field.dataType, DateType):
            default_value = datetime.strptime("1970-01-01", "%Y-%m-%d").date()
        elif isinstance(field.dataType, TimestampType):
            default_value = datetime.strptime(
                "1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
            )
        elif isinstance(field.dataType, DecimalType):
            default_value = Decimal("0.0")
        else:
            raise ValueError(f"Unsupported data type {field.dataType}")
        default_values.append(default_value)

    # Create a DataFrame with a single row using the default values
    df = spark.createDataFrame([tuple(default_values)], schema=schema)
    assert df.schema == schema
    return df


def _split_auto_partition_budget(
    budget: PrivacyBudget,
) -> Tuple[ApproxDPBudget, ApproxDPBudget]:
    """Split a budget for use in a query with automatic partition selection.

    The entire delta is consumed by partition selection since the subsequent queries
    do not consume delta.
    """
    try:
        assert isinstance(budget, ApproxDPBudget)
    except AssertionError as exp:
        raise ValueError(
            "An ApproxDPBudget is required for a query with partition selection."
        ) from exp

    if budget.is_infinite:
        # Need to set delta to zero in the second query which cannot consume delta.
        return ApproxDPBudget(float("inf"), 1), ApproxDPBudget(float("inf"), 0)

    epsilon, delta = budget.value
    half_epsilon = epsilon * ExactNumber("0.5")
    return ApproxDPBudget(half_epsilon, delta), ApproxDPBudget(half_epsilon, 0)


class BaseMeasurementVisitor(QueryExprVisitor):
    """A visitor to create a measurement from a query expression."""

    def __init__(
        self,
        privacy_budget: PrivacyBudget,
        stability: Any,
        input_domain: DictDomain,
        input_metric: DictMetric,
        output_measure: Union[PureDP, ApproxDP, RhoZCDP],
        default_mechanism: NoiseMechanism,
        public_sources: Dict[str, DataFrame],
        catalog: Catalog,
        table_constraints: Dict[Identifier, List[Constraint]],
    ):
        """Constructor for MeasurementVisitor."""
        self.budget = privacy_budget
        self.adjusted_budget = privacy_budget
        self.stability = stability
        self.input_domain = input_domain
        self.input_metric = input_metric
        self.default_mechanism = default_mechanism
        self.public_sources = public_sources
        self.output_measure = output_measure
        self.catalog = catalog
        self.table_constraints = table_constraints

    def _get_zero_budget(self) -> PrivacyBudget:
        """Return a budget with zero epsilon and zero delta."""
        if isinstance(self.budget, PureDPBudget):
            return PureDPBudget(0)
        if isinstance(self.budget, ApproxDPBudget):
            return ApproxDPBudget(0, 0)
        if isinstance(self.budget, RhoZCDPBudget):
            return RhoZCDPBudget(0)
        raise AssertionError(
            f"Unknown budget type {type(self.budget)}. This is probably a bug; "
            "please let us know about it so we can fix it!"
        )

    @staticmethod
    def _build_groupby(
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        mechanism: NoiseMechanism,
        keyset: KeySet,
    ) -> GroupBy:
        """Build a groupby transformation."""
        # TODO(#1044 and #1547): Update condition to when issue is resolved.
        # isinstance(self._output_measure, RhoZCDP)
        use_l2 = mechanism in (
            NoiseMechanism.DISCRETE_GAUSSIAN,
            NoiseMechanism.GAUSSIAN,
        )
        return GroupBy(
            input_domain=input_domain,
            input_metric=input_metric,
            use_l2=use_l2,
            group_keys=keyset.dataframe(),
        )

    def _build_adaptive_groupby_agg_and_noise_info(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        stability: Any,
        mechanism: NoiseMechanism,
        columns: List[str],
        keyset: Union[KeySet, List[str]],
        build_groupby_agg_from_groupby: Callable[[GroupBy], Measurement],
        keyset_budget: PrivacyBudget,
    ) -> Tuple[Measurement, NoiseInfo]:
        """Builds a measurement which gets a keyset and performs a groupby aggregation.

        Args:
            input_domain: The domain of the input data.
            input_metric: The metric of the input data.
            stability: The stability of the transformation.
            mechanism: The noise mechanism to use for the aggregation. This is used
                to determine whether to use L1/L2 for the groupby transformation.
            columns: The columns to get the keyset for.
            keyset: The KeySet to use, if using a public keyset. Must match
                `columns`.
            build_groupby_agg_from_groupby: A function that builds a groupby aggregation
                from a groupby. The resulting measurement should satisfy
                `measurement.privacy_relation(stability, agg_budget.value)`.
            keyset_budget: The budget to use for the keyset measurement.

        Returns:
            A tuple of the groupby aggregation measurement and the noise info.
        """
        assert keyset is not None
        if isinstance(keyset, KeySet):
            assert keyset.dataframe().columns == columns

        def perform_groupby_agg(
            queryable: Queryable,
        ):
            measurement_keyset = queryable(
                MeasurementQuery(
                    self._build_get_keyset_measurement(
                        input_domain=input_domain,
                        input_metric=input_metric,
                        stability=stability,
                        budget=keyset_budget,
                        keyset_or_columns=keyset,
                    )
                )
            )
            groupby = self._build_groupby(
                input_domain, input_metric, mechanism, measurement_keyset
            )
            agg = build_groupby_agg_from_groupby(groupby)
            return queryable(MeasurementQuery(agg))
            # agg will be a groupby sum, avg, ect. measurement.

        groupby_agg = NonInteractivePostProcess(
            create_adaptive_composition(
                input_domain=input_domain,
                input_metric=input_metric,
                d_in=stability,
                privacy_budget=self.adjusted_budget.value,
                output_measure=self.output_measure,
            ),
            perform_groupby_agg,
        )
        # Now calculate noise info, it needs to have the same privacy analysis as the
        # groupby_agg measurement without being adaptive.
        # The key assumption is that a keyset with 1 arbitrary row will have the same
        # privacy analysis as the adaptively selected keyset.
        groupby_schema = StructType(
            [
                struct_field
                for struct_field in input_domain.spark_schema
                if struct_field.name in columns
            ]
        )
        one_row_keyset = KeySet.from_dataframe(
            _create_single_row_df_with_spark_schema(groupby_schema)
        )
        groupby_one_row_keyset = self._build_groupby(
            input_domain, input_metric, mechanism, one_row_keyset
        )
        groupby_agg_one_row_keyset = build_groupby_agg_from_groupby(
            groupby_one_row_keyset
        )
        noise_info = _noise_from_measurement(groupby_agg_one_row_keyset)
        return groupby_agg, noise_info

    @abstractmethod
    def _visit_child_transformation(
        self, expr: QueryExpr, mechanism: NoiseMechanism
    ) -> Tuple[Transformation, TableReference, List[Constraint]]:
        pass

    @abstractmethod
    def _handle_enforce(
        self,
        constraint: Constraint,
        child_transformation: Transformation,
        child_ref: TableReference,
        **kwargs,
    ) -> Tuple[Transformation, TableReference]:
        """Append the constraint to the end of the transformation.

        This is a helper method for :meth:`~._truncate_table`.

        It is pulled out to make it easier to override in subclasses which change the
        behavior of constraints, not for code maintainability.
        """

    def _truncate_table(
        self,
        transformation: Transformation,
        reference: TableReference,
        constraints: List[Constraint],
        grouping_columns: List[str],
    ) -> Tuple[Transformation, TableReference]:
        table_transformation = get_table_from_ref(transformation, reference)
        table_metric = table_transformation.output_metric
        if (
            isinstance(table_metric, IfGroupedBy)
            and table_metric.inner_metric == SymmetricDifference()
        ):
            truncatable_constraints = _get_truncatable_constraints(constraints)
            truncatable_constraints.sort(
                key=lambda cs: _constraint_stability(
                    cs, self.output_measure, grouping_columns
                )
            )
            if not truncatable_constraints:
                raise RuntimeError(
                    "A constraint on the number of rows contributed by each ID "
                    "is needed to perform this query (e.g. MaxRowsPerID)."
                )

            for c in truncatable_constraints[0]:
                assert isinstance(
                    c, (MaxRowsPerID, MaxGroupsPerID, MaxRowsPerGroupPerID)
                )
                if isinstance(c, MaxGroupsPerID):
                    # Taking advantage of the L2 noise behavior only works for
                    # Sessions initialized with a RhoZCDP privacy budget,
                    # and then only when the grouping column
                    # of the constraints is being grouped on.
                    use_l2 = (
                        isinstance(self.output_measure, RhoZCDP)
                        and c.grouping_column in grouping_columns
                    )
                    transformation, reference = self._handle_enforce(
                        c, transformation, reference, update_metric=True, use_l2=use_l2
                    )
                else:
                    (
                        transformation,
                        reference,
                    ) = self._handle_enforce(
                        c, transformation, reference, update_metric=True
                    )
            return transformation, reference

        else:
            # Tables without IDs don't need truncation
            return transformation, reference

    def _validate_approxDP_and_adjust_budget(
        self,
        expr: Union[
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
            GroupByCount,
            GroupByCountDistinct,
            GroupByQuantile,
        ],
    ) -> None:
        """Validate and set adjusted_budget for ApproxDP queries.

        First, validate that the user is not using a Gaussian noise mechanism with
        ApproxDP. Then, for queries that use noise addition mechanisms replace non-zero
        deltas with zero in self.adjusted_budget. If the user chose this mechanism
        (i.e. didn't use the DEFAULT mechanism) we warn them of this replacement.
        """
        if not isinstance(self.budget, ApproxDPBudget):
            return

        mechanism: Any = getattr(expr, "mechanism", None)
        if mechanism in (
            AverageMechanism.GAUSSIAN,
            CountDistinctMechanism.GAUSSIAN,
            CountMechanism.GAUSSIAN,
            StdevMechanism.GAUSSIAN,
            SumMechanism.GAUSSIAN,
            VarianceMechanism.GAUSSIAN,
        ):
            raise NotImplementedError(
                "Gaussian noise is only supported with a RhoZCDPBudget. Please use "
                "CountMechanism.LAPLACE instead."
            )

        epsilon, delta = self.budget.value
        if isinstance(expr.groupby_keys, list) and not self.budget.is_infinite:
            # Automatic Partition Selection is being implemented. Validate the budget.
            if epsilon <= 0:
                raise ValueError(
                    "Automatic partition selection requires a positive epsilon. "
                    f"The budget provided was {self.budget}."
                )
            if delta <= 0:
                raise ValueError(
                    "Automatic partition selection requires a positive delta. "
                    f"The budget provided was {self.budget}."
                )
            return
        else:
            if mechanism in (
                AverageMechanism.LAPLACE,
                CountDistinctMechanism.LAPLACE,
                CountMechanism.LAPLACE,
                StdevMechanism.LAPLACE,
                SumMechanism.LAPLACE,
                VarianceMechanism.LAPLACE,
            ):
                warnings.warn(
                    "When using LAPLACE with an ApproxDPBudget, the delta value of "
                    "the budget will be replaced with zero."
                )
                self.adjusted_budget = ApproxDPBudget(epsilon, 0)
            elif mechanism in (
                AverageMechanism.DEFAULT,
                CountDistinctMechanism.DEFAULT,
                CountMechanism.DEFAULT,
                StdevMechanism.DEFAULT,
                SumMechanism.DEFAULT,
                VarianceMechanism.DEFAULT,
            ):
                self.adjusted_budget = ApproxDPBudget(epsilon, 0)
            elif mechanism is None:
                # Quantile has no mechanism
                self.adjusted_budget = ApproxDPBudget(epsilon, 0)
            else:
                raise AssertionError(
                    f"Unknown mechanism {mechanism}. This is probably a bug; "
                    "please let us know so we can fix it!"
                )

    def _pick_noise_for_count(
        self, query: Union[GroupByCount, GroupByCountDistinct]
    ) -> NoiseMechanism:
        """Pick the noise mechanism to use for a count or count-distinct query."""
        requested_mechanism: NoiseMechanism
        if query.mechanism in (CountMechanism.DEFAULT, CountDistinctMechanism.DEFAULT):
            if isinstance(self.output_measure, (PureDP, ApproxDP)):
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
    ) -> NoiseMechanism:
        """Pick the noise mechanism for non-count queries.

        GroupByQuantile only supports one noise mechanism, so it is not
        included here.
        """
        measure_column_type = query.child.accept(OutputSchemaVisitor(self.catalog))[
            query.measure_column
        ].column_type
        requested_mechanism: NoiseMechanism
        if query.mechanism in (
            SumMechanism.DEFAULT,
            AverageMechanism.DEFAULT,
            VarianceMechanism.DEFAULT,
            StdevMechanism.DEFAULT,
        ):
            requested_mechanism = (
                NoiseMechanism.LAPLACE
                if isinstance(self.output_measure, (PureDP, ApproxDP))
                else NoiseMechanism.GAUSSIAN
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
            requested_mechanism = NoiseMechanism.GAUSSIAN
        else:
            raise ValueError(
                f"Did not recognize requested mechanism {query.mechanism}."
                " Supported mechanisms are DEFAULT, LAPLACE,  and GAUSSIAN."
            )

        # If the query requested a Laplace measure ...
        if requested_mechanism == NoiseMechanism.LAPLACE:
            if measure_column_type == ColumnType.INTEGER:
                return NoiseMechanism.GEOMETRIC
            elif measure_column_type == ColumnType.DECIMAL:
                return NoiseMechanism.LAPLACE
            else:
                raise AssertionError(
                    "Query's measure column should be numeric. This should"
                    " not happen and is probably a bug;  please let us know"
                    " so we can fix it!"
                )

        # If the query requested a Gaussian measure...
        elif requested_mechanism == NoiseMechanism.GAUSSIAN:
            if isinstance(self.output_measure, PureDP):
                raise ValueError(
                    "Gaussian noise is not supported under PureDP. "
                    "Please use RhoZCDP or another measure."
                )
            if measure_column_type == ColumnType.DECIMAL:
                return NoiseMechanism.GAUSSIAN
            elif measure_column_type == ColumnType.INTEGER:
                return NoiseMechanism.DISCRETE_GAUSSIAN
            else:
                raise AssertionError(
                    "Query's measure column should be numeric. This should"
                    " not happen and is probably a bug;  please let us know"
                    " so we can fix it!"
                )

        # The requested_mechanism should be either LAPLACE or
        # GAUSSIAN, so something has gone awry
        else:
            raise AssertionError(
                f"Did not recognize requested mechanism {requested_mechanism}."
                " This is probably a bug; please let us know about it so we can fix it!"
            )

    def _add_special_value_handling_to_query(
        self,
        query: Union[
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
            GroupByQuantile,
        ],
    ):
        """Returns a new query that handles nulls, NaNs and infinite values.

        If the measure column allows nulls or NaNs, the new query
        will drop those values.

        If the measure column allows infinite values, the new query will replace those
        values with the low and high values specified in the query.

        These changes are added immediately before the groupby aggregation in the query.
        """
        expected_schema = query.child.accept(OutputSchemaVisitor(self.catalog))

        # You can't perform these queries on nulls, NaNs, or infinite values
        # so check for those
        try:
            measure_desc = expected_schema[query.measure_column]
        except KeyError as e:
            raise KeyError(
                f"Measure column {query.measure_column} is not in the input schema."
            ) from e

        new_child: QueryExpr
        # If null or NaN values are allowed ...
        if measure_desc.allow_null or (
            measure_desc.column_type == ColumnType.DECIMAL and measure_desc.allow_nan
        ):
            # then drop those values
            # (but don't mutate the original query)
            new_child = DropNullAndNan(
                child=query.child, columns=[query.measure_column]
            )
            query = dataclasses.replace(query, child=new_child)
        # If infinite values are allowed...
        if measure_desc.column_type == ColumnType.DECIMAL and measure_desc.allow_inf:
            # then clamp them (to low/high values)
            new_child = ReplaceInfinity(
                child=query.child,
                replace_with={query.measure_column: (query.low, query.high)},
            )
            query = dataclasses.replace(query, child=new_child)
        return query

    def _validate_measurement(self, measurement: Measurement, mid_stability: sp.Expr):
        """Validate a measurement."""
        if isinstance(self.adjusted_budget.value, tuple):
            # TODO(#2754): add a log message.
            privacy_function_budget_mismatch = any(
                x > y
                for x, y in zip(
                    measurement.privacy_function(mid_stability),
                    self.adjusted_budget.value,
                )
            )
        else:
            assert isinstance(self.adjusted_budget.value, ExactNumber)
            privacy_function_budget_mismatch = (
                measurement.privacy_function(mid_stability)
                != self.adjusted_budget.value
            )

        if privacy_function_budget_mismatch:
            raise AssertionError(
                "Privacy function does not match per-query privacy budget. "
                "This is probably a bug; please let us know so we can "
                "fix it!"
            )

    # these don't produce measurements, so they return an error
    def visit_private_source(self, expr: PrivateSourceExpr) -> Any:
        """Visit a PrivateSource query expression (raises an error)."""
        raise NotImplementedError

    def visit_rename(self, expr: RenameExpr) -> Any:
        """Visit a Rename query expression (raises an error)."""
        raise NotImplementedError

    def visit_filter(self, expr: FilterExpr) -> Any:
        """Visit a Filter query expression (raises an error)."""
        raise NotImplementedError

    def visit_select(self, expr: SelectExpr) -> Any:
        """Visit a Select query expression (raises an error)."""
        raise NotImplementedError

    def visit_map(self, expr: MapExpr) -> Any:
        """Visit a Map query expression (raises an error)."""
        raise NotImplementedError

    def visit_flat_map(self, expr: FlatMapExpr) -> Any:
        """Visit a FlatMap query expression (raises an error)."""
        raise NotImplementedError

    def visit_join_private(self, expr: JoinPrivateExpr) -> Any:
        """Visit a JoinPrivate query expression (raises an error)."""
        raise NotImplementedError

    def visit_join_public(self, expr: JoinPublicExpr) -> Any:
        """Visit a JoinPublic query expression (raises an error)."""
        raise NotImplementedError

    def visit_replace_null_and_nan(self, expr: ReplaceNullAndNan) -> Any:
        """Visit a ReplaceNullAndNan query expression (raises an error)."""
        raise NotImplementedError

    def visit_replace_infinity(self, expr: ReplaceInfinity) -> Any:
        """Visit a ReplaceInfinity query expression (raises an error)."""
        raise NotImplementedError

    def visit_drop_null_and_nan(self, expr: DropNullAndNan) -> Any:
        """Visit a DropNullAndNan query expression (raises an error)."""
        raise NotImplementedError

    def visit_drop_infinity(self, expr: DropInfExpr) -> Any:
        """Visit a DropInfinity query expression (raises an error)."""
        raise NotImplementedError

    def visit_enforce_constraint(self, expr: EnforceConstraint) -> Any:
        """Visit a EnforceConstraint query expression (raises an error)."""
        raise NotImplementedError

    def visit_get_groups(self, expr: GetGroups) -> Any:
        """Visit a GetGroups query expression (raises an error)."""
        raise NotImplementedError

    def _build_get_keyset_measurement(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        stability: ExactNumber,
        budget: PrivacyBudget,
        keyset_or_columns: Union[KeySet, List[str]],
    ) -> Measurement:
        """Build a Measurement that returns a KeySet.

        Args:
            input_domain: The domain of the input data.
            input_metric: The metric of the input data.
            stability: The input stability the measurement will be applied on.
            budget: The privacy budget to use. Should be zero if using a public keyset.
            keyset_or_columns: Either a KeySet object to return, if using a public
                keyset, or the columns to get the keyset for.
        """
        if isinstance(keyset_or_columns, list):
            columns: List[str] = keyset_or_columns
            # Check that the budget is ApproxDP and is nonzero.
            if not isinstance(budget, ApproxDPBudget):
                raise AssertionError(
                    "Automatic partition selection requires an ApproxDPBudget. "
                    f"The budget provided was {budget}. This is probably a bug; "
                    "please let us know about it so we can fix it!"
                )
            assert isinstance(budget, ApproxDPBudget)  # Assertion to help mypy later.

            if budget.epsilon <= 0 or budget.delta <= 0:
                raise AssertionError(
                    "Automatic partition selection requires an ApproxDPBudget with "
                    "epsilon and delta greater than 0. The budget provided was "
                    f"{budget}. This is probably a bug; please let us know about it "
                    "so we can fix it!"
                )
            select = SelectTransformation(input_domain, input_metric, columns)
            keyset_domain = SparkDataFrameDomain(
                schema={col: input_domain[col] for col in columns}
            )

            epsilon, delta = budget.value
            keyset_measurement = select | create_partition_selection_measurement(
                input_domain=keyset_domain,
                d_in=stability,
                epsilon=epsilon,
                delta=delta,
            )

            # Use the resulting KeySet if it is nonempty, otherwise use an empty KeySet.
            def process_function(new_df):
                if new_df.count() == 0:
                    warnings.warn(
                        "This query tried to automatically determine a keyset, but "
                        "a null dataframe was returned from the partition selection."
                        "This may be because the dataset is empty or because the "
                        " ApproxDPBudget used was too small."
                    )
                return KeySet.from_dataframe(new_df.select(columns))

            return PostProcess(keyset_measurement, process_function)

        else:
            columns = keyset_or_columns.dataframe().columns

            if isinstance(budget, PureDPBudget):
                assert budget.epsilon == 0
            elif isinstance(budget, ApproxDPBudget):
                assert budget.epsilon == 0
                assert budget.delta == 0
            elif isinstance(budget, RhoZCDPBudget):
                assert budget.rho == 0
            else:
                raise AssertionError(f"Unrecognized budget type {type(budget)}")
            keyset_measurement = lambda _: keyset_or_columns  # type: ignore
            no_op = SequentialComposition(
                input_domain=input_domain,
                input_metric=input_metric,
                output_measure=self.output_measure,
                d_in=stability,
                privacy_budget=budget.value,
            )
            return NonInteractivePostProcess(no_op, keyset_measurement)

    def build_groupby_count(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        stability: Any,
        mechanism: NoiseMechanism,
        budget: PrivacyBudget,
        groupby: GroupBy,
        output_column: str,
    ) -> Measurement:
        """Build a Measurement for a GroupByCount query."""
        return create_count_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            noise_mechanism=mechanism,
            d_in=stability,
            d_out=budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            count_column=output_column,
        )

    def visit_groupby_count(self, expr: GroupByCount) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GroupByCount query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))

        if isinstance(expr.groupby_keys, KeySet):
            groupby_cols = expr.groupby_keys.dataframe().columns
            keyset_budget = self._get_zero_budget()
            query_budget = self.adjusted_budget
        else:
            groupby_cols = expr.groupby_keys
            keyset_budget, query_budget = _split_auto_partition_budget(
                self.adjusted_budget
            )

        mechanism = self._pick_noise_for_count(expr)
        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, mechanism),
            grouping_columns=groupby_cols,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)
        mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
        mid_metric = cast(
            Union[IfGroupedBy, HammingDistance, SymmetricDifference],
            transformation.output_metric,
        )
        mid_stability = transformation.stability_function(self.stability)

        def _build_groupby_agg_from_groupby(
            groupby: GroupBy,
        ) -> Measurement:
            """Build a Measurement for a GroupByCount query."""
            return self.build_groupby_count(
                input_domain=mid_domain,
                input_metric=mid_metric,
                stability=mid_stability,
                mechanism=mechanism,
                budget=query_budget,
                groupby=groupby,
                output_column=expr.output_column,
            )

        (
            adaptive_groupby_agg,
            noise_info,
        ) = self._build_adaptive_groupby_agg_and_noise_info(
            input_domain=mid_domain,
            input_metric=mid_metric,
            stability=transformation.stability_function(self.stability),
            mechanism=mechanism,
            columns=groupby_cols,
            keyset=expr.groupby_keys,
            build_groupby_agg_from_groupby=_build_groupby_agg_from_groupby,
            keyset_budget=keyset_budget,
        )
        self._validate_measurement(adaptive_groupby_agg, mid_stability)
        return transformation | adaptive_groupby_agg, noise_info

    def build_count_distinct_measurement(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        mechanism: NoiseMechanism,
        stability: Any,
        budget: PrivacyBudget,
        groupby: GroupBy,
        output_column: str,
    ) -> Measurement:
        """Build a Measurement for a GroupByCountDistinct query."""
        return create_count_distinct_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            noise_mechanism=mechanism,
            d_in=stability,
            d_out=budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            count_column=output_column,
        )

    def visit_groupby_count_distinct(
        self, expr: GroupByCountDistinct
    ) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GroupByCountDistinct query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))

        if isinstance(expr.groupby_keys, KeySet):
            groupby_cols = expr.groupby_keys.dataframe().columns
            keyset_budget = self._get_zero_budget()
            query_budget = self.adjusted_budget
        else:
            groupby_cols = expr.groupby_keys
            keyset_budget, query_budget = _split_auto_partition_budget(
                self.adjusted_budget
            )

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

        child_transformation, child_ref = self._truncate_table(
            child_transformation,
            child_ref,
            child_constraints,
            grouping_columns=groupby_cols,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)

        mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
        mid_metric = cast(
            Union[IfGroupedBy, HammingDistance, SymmetricDifference],
            transformation.output_metric,
        )
        # If not counting all columns, drop the ones that are neither counted
        # nor grouped on.
        if expr.columns_to_count:
            groupby_columns = list(expr.groupby_keys.schema().keys())  # type: ignore
            transformation |= SelectTransformation(
                mid_domain,
                mid_metric,
                list(set(expr.columns_to_count + groupby_columns)),
            )
            mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
            mid_metric = cast(
                Union[IfGroupedBy, HammingDistance, SymmetricDifference],
                transformation.output_metric,
            )

        mid_stability = transformation.stability_function(self.stability)

        def _build_groupby_agg_from_groupby(
            groupby: GroupBy,
        ) -> Measurement:
            """Build a Measurement for a GroupByCountDistinct query."""
            return self.build_count_distinct_measurement(
                input_domain=mid_domain,
                input_metric=mid_metric,
                mechanism=mechanism,
                stability=mid_stability,
                budget=query_budget,
                groupby=groupby,
                output_column=expr.output_column,
            )

        (
            adaptive_groupby_agg,
            noise_info,
        ) = self._build_adaptive_groupby_agg_and_noise_info(
            input_domain=mid_domain,
            input_metric=mid_metric,
            stability=transformation.stability_function(self.stability),
            mechanism=mechanism,
            columns=groupby_cols,
            keyset=expr.groupby_keys,
            build_groupby_agg_from_groupby=_build_groupby_agg_from_groupby,
            keyset_budget=keyset_budget,
        )
        self._validate_measurement(adaptive_groupby_agg, mid_stability)
        return transformation | adaptive_groupby_agg, noise_info

    def build_groupby_quantile(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        measure_column: str,
        quantile: float,
        lower: Union[int, float],
        upper: Union[int, float],
        stability: Any,
        budget: PrivacyBudget,
        groupby: GroupBy,
        output_column: str,
    ) -> Measurement:
        """Build a Measurement for a GroupByQuantile query."""
        return create_quantile_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            quantile=quantile,
            lower=lower,
            upper=upper,
            d_in=stability,
            d_out=budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            quantile_column=output_column,
        )

    def visit_groupby_quantile(
        self, expr: GroupByQuantile
    ) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GroupByQuantile query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))
        expr = self._add_special_value_handling_to_query(expr)

        if isinstance(expr.groupby_keys, KeySet):
            groupby_cols = expr.groupby_keys.dataframe().columns
            keyset_budget = self._get_zero_budget()
            query_budget = self.adjusted_budget
        else:
            groupby_cols = expr.groupby_keys
            keyset_budget, query_budget = _split_auto_partition_budget(
                self.adjusted_budget
            )

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, self.default_mechanism),
            grouping_columns=groupby_cols,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)
        mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
        mid_metric = cast(
            Union[IfGroupedBy, HammingDistance, SymmetricDifference],
            transformation.output_metric,
        )
        mid_stability = transformation.stability_function(self.stability)

        def _build_groupby_agg_from_groupby(
            groupby: GroupBy,
        ) -> Measurement:
            """Build a Measurement for a GroupByQuantile query."""
            return self.build_groupby_quantile(
                input_domain=mid_domain,
                input_metric=mid_metric,
                measure_column=expr.measure_column,
                quantile=expr.quantile,
                lower=expr.low,  # Uses floats, so doesn't use _get_query_bounds
                upper=expr.high,
                stability=mid_stability,
                budget=query_budget,
                groupby=groupby,
                output_column=expr.output_column,
            )

        (
            adaptive_groupby_agg,
            noise_info,
        ) = self._build_adaptive_groupby_agg_and_noise_info(
            input_domain=mid_domain,
            input_metric=mid_metric,
            stability=transformation.stability_function(self.stability),
            mechanism=self.default_mechanism,
            columns=groupby_cols,
            keyset=expr.groupby_keys,
            build_groupby_agg_from_groupby=_build_groupby_agg_from_groupby,
            keyset_budget=keyset_budget,
        )
        self._validate_measurement(adaptive_groupby_agg, mid_stability)
        return transformation | adaptive_groupby_agg, noise_info

    def build_groupby_bounded_sum(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        measure_column: str,
        lower: ExactNumber,
        upper: ExactNumber,
        stability: Any,
        mechanism: NoiseMechanism,
        budget: PrivacyBudget,
        groupby: GroupBy,
        output_column: str,
    ) -> Measurement:
        """Build a Measurement for a GroupByBoundedSum query."""
        return create_sum_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            lower=lower,
            upper=upper,
            noise_mechanism=mechanism,
            d_in=stability,
            d_out=budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            sum_column=output_column,
        )

    def visit_groupby_bounded_sum(
        self, expr: GroupByBoundedSum
    ) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GroupByBoundedSum query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))
        expr = self._add_special_value_handling_to_query(expr)

        if isinstance(expr.groupby_keys, KeySet):
            groupby_cols = expr.groupby_keys.dataframe().columns
            keyset_budget = self._get_zero_budget()
            query_budget = self.adjusted_budget
        else:
            groupby_cols = expr.groupby_keys
            keyset_budget, query_budget = _split_auto_partition_budget(
                self.adjusted_budget
            )

        mechanism = self._pick_noise_for_non_count(expr)
        lower, upper = _get_query_bounds(expr)

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, mechanism),
            grouping_columns=groupby_cols,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)
        mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
        mid_metric = cast(
            Union[IfGroupedBy, HammingDistance, SymmetricDifference],
            transformation.output_metric,
        )
        mid_stability = transformation.stability_function(self.stability)

        def _build_groupby_agg_from_groupby(
            groupby: GroupBy,
        ) -> Measurement:
            """Build a Measurement for a GroupByBoundedSum query."""
            return self.build_groupby_bounded_sum(
                input_domain=mid_domain,
                input_metric=mid_metric,
                measure_column=expr.measure_column,
                lower=lower,
                upper=upper,
                stability=mid_stability,
                mechanism=mechanism,
                budget=query_budget,
                groupby=groupby,
                output_column=expr.output_column,
            )

        (
            adaptive_groupby_agg,
            noise_info,
        ) = self._build_adaptive_groupby_agg_and_noise_info(
            input_domain=mid_domain,
            input_metric=mid_metric,
            stability=transformation.stability_function(self.stability),
            mechanism=mechanism,
            columns=groupby_cols,
            keyset=expr.groupby_keys,
            build_groupby_agg_from_groupby=_build_groupby_agg_from_groupby,
            keyset_budget=keyset_budget,
        )
        self._validate_measurement(adaptive_groupby_agg, mid_stability)
        return transformation | adaptive_groupby_agg, noise_info

    def build_groupby_bounded_average(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        measure_column: str,
        lower: ExactNumber,
        upper: ExactNumber,
        stability: Any,
        mechanism: NoiseMechanism,
        budget: PrivacyBudget,
        groupby: GroupBy,
        output_column: str,
    ) -> Measurement:
        """Build a Measurement for a GroupByBoundedAverage query."""
        return create_average_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            lower=lower,
            upper=upper,
            noise_mechanism=mechanism,
            d_in=stability,
            d_out=budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            average_column=output_column,
        )

    def visit_groupby_bounded_average(
        self, expr: GroupByBoundedAverage
    ) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GroupByBoundedAverage query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))
        expr = self._add_special_value_handling_to_query(expr)

        if isinstance(expr.groupby_keys, KeySet):
            groupby_cols = expr.groupby_keys.dataframe().columns
            keyset_budget = self._get_zero_budget()
            query_budget = self.adjusted_budget
        else:
            groupby_cols = expr.groupby_keys
            keyset_budget, query_budget = _split_auto_partition_budget(
                self.adjusted_budget
            )

        lower, upper = _get_query_bounds(expr)
        mechanism = self._pick_noise_for_non_count(expr)

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, self.default_mechanism),
            grouping_columns=groupby_cols,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)
        mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
        mid_metric = cast(
            Union[IfGroupedBy, HammingDistance, SymmetricDifference],
            transformation.output_metric,
        )
        mid_stability = transformation.stability_function(self.stability)

        def _build_groupby_agg_from_groupby(
            groupby: GroupBy,
        ) -> Measurement:
            """Build a Measurement for a GroupByBoundedAverage query."""
            return self.build_groupby_bounded_average(
                input_domain=mid_domain,
                input_metric=mid_metric,
                measure_column=expr.measure_column,
                lower=lower,
                upper=upper,
                stability=mid_stability,
                mechanism=mechanism,
                budget=query_budget,
                groupby=groupby,
                output_column=expr.output_column,
            )

        (
            adaptive_groupby_agg,
            noise_info,
        ) = self._build_adaptive_groupby_agg_and_noise_info(
            input_domain=mid_domain,
            input_metric=mid_metric,
            stability=transformation.stability_function(self.stability),
            mechanism=mechanism,
            columns=groupby_cols,
            keyset=expr.groupby_keys,
            build_groupby_agg_from_groupby=_build_groupby_agg_from_groupby,
            keyset_budget=keyset_budget,
        )
        self._validate_measurement(adaptive_groupby_agg, mid_stability)
        return transformation | adaptive_groupby_agg, noise_info

    def build_groupby_bounded_variance(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        measure_column: str,
        lower: ExactNumber,
        upper: ExactNumber,
        stability: Any,
        mechanism: NoiseMechanism,
        budget: PrivacyBudget,
        groupby: GroupBy,
        output_column: str,
    ) -> Measurement:
        """Build a Measurement for a GroupByBoundedVariance query."""
        return create_variance_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            lower=lower,
            upper=upper,
            noise_mechanism=mechanism,
            d_in=stability,
            d_out=budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            variance_column=output_column,
        )

    def visit_groupby_bounded_variance(
        self, expr: GroupByBoundedVariance
    ) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GroupByBoundedVariance query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))
        expr = self._add_special_value_handling_to_query(expr)

        if isinstance(expr.groupby_keys, KeySet):
            groupby_cols = expr.groupby_keys.dataframe().columns
            keyset_budget = self._get_zero_budget()
            query_budget = self.adjusted_budget
        else:
            groupby_cols = expr.groupby_keys
            keyset_budget, query_budget = _split_auto_partition_budget(
                self.adjusted_budget
            )

        lower, upper = _get_query_bounds(expr)
        mechanism = self._pick_noise_for_non_count(expr)

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, mechanism),
            grouping_columns=groupby_cols,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)
        mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
        mid_metric = cast(
            Union[IfGroupedBy, HammingDistance, SymmetricDifference],
            transformation.output_metric,
        )
        mid_stability = transformation.stability_function(self.stability)

        def _build_groupby_agg_from_groupby(
            groupby: GroupBy,
        ) -> Measurement:
            """Build a Measurement for a GroupByBoundedVariance query."""
            return self.build_groupby_bounded_variance(
                input_domain=mid_domain,
                input_metric=mid_metric,
                measure_column=expr.measure_column,
                lower=lower,
                upper=upper,
                stability=mid_stability,
                mechanism=mechanism,
                budget=query_budget,
                groupby=groupby,
                output_column=expr.output_column,
            )

        (
            adaptive_groupby_agg,
            noise_info,
        ) = self._build_adaptive_groupby_agg_and_noise_info(
            input_domain=mid_domain,
            input_metric=mid_metric,
            stability=transformation.stability_function(self.stability),
            mechanism=mechanism,
            columns=groupby_cols,
            keyset=expr.groupby_keys,
            build_groupby_agg_from_groupby=_build_groupby_agg_from_groupby,
            keyset_budget=keyset_budget,
        )
        self._validate_measurement(adaptive_groupby_agg, mid_stability)
        return transformation | adaptive_groupby_agg, noise_info

    def build_groupby_bounded_stdev(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference, HammingDistance],
        measure_column: str,
        lower: ExactNumber,
        upper: ExactNumber,
        stability: Any,
        mechanism: NoiseMechanism,
        budget: PrivacyBudget,
        groupby: GroupBy,
        output_column: str,
    ) -> Measurement:
        """Build a Measurement for a GroupByBoundedStdev query."""
        return create_standard_deviation_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            lower=lower,
            upper=upper,
            noise_mechanism=mechanism,
            d_in=stability,
            d_out=budget.value,
            output_measure=self.output_measure,
            groupby_transformation=groupby,
            standard_deviation_column=output_column,
        )

    def visit_groupby_bounded_stdev(
        self, expr: GroupByBoundedSTDEV
    ) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GroupByBoundedStdev query expression."""
        self._validate_approxDP_and_adjust_budget(expr)

        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))
        expr = self._add_special_value_handling_to_query(expr)

        if isinstance(expr.groupby_keys, KeySet):
            groupby_cols = expr.groupby_keys.dataframe().columns
            keyset_budget = self._get_zero_budget()
            query_budget = self.adjusted_budget
        else:
            groupby_cols = expr.groupby_keys
            keyset_budget, query_budget = _split_auto_partition_budget(
                self.adjusted_budget
            )

        lower, upper = _get_query_bounds(expr)
        mechanism = self._pick_noise_for_non_count(expr)

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, mechanism),
            grouping_columns=groupby_cols,
        )
        transformation = get_table_from_ref(child_transformation, child_ref)
        mid_domain = cast(SparkDataFrameDomain, transformation.output_domain)
        mid_metric = cast(
            Union[IfGroupedBy, HammingDistance, SymmetricDifference],
            transformation.output_metric,
        )
        mid_stability = transformation.stability_function(self.stability)

        def _build_groupby_agg_from_groupby(
            groupby: GroupBy,
        ) -> Measurement:
            """Build a Measurement for a GroupByBoundedStdev query."""
            return self.build_groupby_bounded_stdev(
                input_domain=mid_domain,
                input_metric=mid_metric,
                measure_column=expr.measure_column,
                lower=lower,
                upper=upper,
                stability=mid_stability,
                mechanism=mechanism,
                budget=query_budget,
                groupby=groupby,
                output_column=expr.output_column,
            )

        (
            adaptive_groupby_agg,
            noise_info,
        ) = self._build_adaptive_groupby_agg_and_noise_info(
            input_domain=mid_domain,
            input_metric=mid_metric,
            stability=transformation.stability_function(self.stability),
            mechanism=mechanism,
            columns=groupby_cols,
            keyset=expr.groupby_keys,
            build_groupby_agg_from_groupby=_build_groupby_agg_from_groupby,
            keyset_budget=keyset_budget,
        )
        self._validate_measurement(adaptive_groupby_agg, mid_stability)
        return transformation | adaptive_groupby_agg, noise_info

    # pylint: disable=no-self-use
    def build_bound_selection_measurement(
        self,
        input_domain,
        output_measure,
        d_out,
        bound_column,
        threshold,
        d_in,
    ) -> Measurement:
        """Helper method to build the appropriate bound selection Measurement."""
        return create_bound_selection_measurement(
            input_domain=input_domain,
            output_measure=output_measure,
            d_out=d_out,
            bound_column=bound_column,
            threshold=threshold,  # TODO: Make threshold optional.
            d_in=d_in,
        )

    # pylint: enable=no-self-use

    def visit_get_bounds(self, expr: GetBounds) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GetBounds query expression."""
        # Peek at the schema, to see if there are errors there
        expr.accept(OutputSchemaVisitor(self.catalog))

        schema = expr.child.accept(OutputSchemaVisitor(self.catalog))

        # Check if we're trying to get the bounds of the ID column.
        if schema.id_column and (schema.id_column == expr.column):
            raise RuntimeError(
                "get_bounds cannot be used on the privacy ID column"
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
            transformation.output_domain, transformation.output_metric, [expr.column]
        )

        mid_stability = transformation.stability_function(self.stability)
        assert isinstance(transformation.output_domain, SparkDataFrameDomain)

        agg = self.build_bound_selection_measurement(
            input_domain=transformation.output_domain,
            output_measure=self.output_measure,
            d_out=self.budget.value,
            bound_column=expr.column,
            threshold=0.95,  # TODO: Make threshold optional.
            d_in=mid_stability,
        )

        self._validate_measurement(agg, mid_stability)

        measurement = transformation | agg
        noise_info = _noise_from_measurement(measurement)
        return measurement, noise_info

    def visit_suppress_aggregates(
        self, expr: SuppressAggregates
    ) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a SuppressAggregates query expression."""
        expr.accept(OutputSchemaVisitor(self.catalog))

        child_measurement, noise_info = expr.child.accept(self)
        assert isinstance(child_measurement, Measurement)

        def suppression_function(df: DataFrame) -> DataFrame:
            """Suppress rows where the column is less than the desired threshold."""
            return df.filter(df[expr.column] >= expr.threshold)

        return (PostProcess(child_measurement, suppression_function), noise_info)
