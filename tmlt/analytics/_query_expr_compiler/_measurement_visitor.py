"""Defines a visitor for creating noisy measurements from query expressions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import dataclasses
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy as sp
from pyspark.sql import DataFrame
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
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import (
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._schema import (
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
)
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
from tmlt.analytics.privacy_budget import ApproxDPBudget, PrivacyBudget
from tmlt.analytics.query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
    DropNullAndNan,
    EnforceConstraint,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    QueryExpr,
    QueryExprVisitor,
    ReplaceInfinity,
    StdevMechanism,
    SumMechanism,
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


class MeasurementVisitor(QueryExprVisitor):
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
                    # RhoZCDP Sessions, and then only when the grouping column
                    # of the constraints is being grouped on.
                    use_l2 = (
                        isinstance(self.output_measure, RhoZCDP)
                        and c.grouping_column in grouping_columns
                    )
                    # pylint: disable=protected-access
                    transformation, reference = c._enforce(
                        transformation, reference, update_metric=True, use_l2=use_l2
                    )
                    # pylint: enable=protected-access
                else:
                    (
                        transformation,
                        reference,
                    ) = c._enforce(  # pylint: disable=protected-access
                        transformation, reference, update_metric=True
                    )
            return transformation, reference

        else:
            # Tables without IDs don't need truncation
            return transformation, reference

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

    def _validate_approxDP_and_adjust_budget(
        self,
        expr: Union[
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
            GroupByCount,
            GroupByCountDistinct,
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

        if expr.mechanism in (
            AverageMechanism.GAUSSIAN,
            CountDistinctMechanism.GAUSSIAN,
            CountMechanism.GAUSSIAN,
            StdevMechanism.GAUSSIAN,
            SumMechanism.GAUSSIAN,
            VarianceMechanism.GAUSSIAN,
        ):
            raise NotImplementedError(
                "Gaussian noise is only supported with RhoZCDP. Please use "
                "CountMechanism.LAPLACE instead."
            )

        epsilon, delta = self.budget.value
        if delta != 0:
            if expr.mechanism in (
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
            elif expr.mechanism in (
                AverageMechanism.DEFAULT,
                CountDistinctMechanism.DEFAULT,
                CountMechanism.DEFAULT,
                StdevMechanism.DEFAULT,
                SumMechanism.DEFAULT,
                VarianceMechanism.DEFAULT,
            ):
                self.adjusted_budget = ApproxDPBudget(epsilon, 0)
            else:
                raise AssertionError(
                    f"Unknown mechanism {expr.mechanism}. This is probably a bug; "
                    "please let us know so we can fix it!"
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
        """Pick the noise mechanism for non-count queries.

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
        elif requested_mechanism == NoiseMechanism.GAUSSIAN:
            if isinstance(self.output_measure, PureDP):
                raise ValueError(
                    "Gaussian noise is not supported under PureDP. "
                    "Please use RhoZCDP or another measure."
                )
            if isinstance(measure_column_type, SparkFloatColumnDescriptor):
                return NoiseMechanism.GAUSSIAN
            elif isinstance(measure_column_type, SparkIntegerColumnDescriptor):
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
        use_l2 = mechanism in (
            NoiseMechanism.DISCRETE_GAUSSIAN,
            NoiseMechanism.GAUSSIAN,
        )

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
        """Everything you need to build a measurement for these query types.

        This function also checks to see if the measure_column allows
        invalid values (nulls, NaNs, and infinite values), and adds
        DropNullAndNan and/or DropInfinity queries to remove them if they are present.
        """
        lower_bound, upper_bound = _get_query_bounds(query)

        expected_schema = query.child.accept(OutputSchemaVisitor(self.catalog))

        # You can't perform these queries on nulls, NaNs, or infinite values
        # so check for those
        try:
            measure_desc = expected_schema[query.measure_column]
        except KeyError as e:
            raise KeyError(
                f"Measure column {query.measure_column} is not in the input schema."
            ) from e

        new_child: Optional[QueryExpr] = None
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
            expected_schema = query.child.accept(OutputSchemaVisitor(self.catalog))

        # If infinite values are allowed...
        if measure_desc.column_type == ColumnType.DECIMAL and measure_desc.allow_inf:
            # then clamp them (to low/high values)
            new_child = ReplaceInfinity(
                child=query.child,
                replace_with={query.measure_column: (query.low, query.high)},
            )
            query = dataclasses.replace(query, child=new_child)
            expected_schema = query.child.accept(OutputSchemaVisitor(self.catalog))

        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        measure_column_type = expected_output_domain[query.measure_column]

        mechanism = self._pick_noise_for_non_count(query, measure_column_type)
        child_transformation, table_ref = self._truncate_table(
            *self._visit_child_transformation(query.child, mechanism),
            grouping_columns=query.groupby_keys.dataframe().columns,
        )
        transformation = get_table_from_ref(child_transformation, table_ref)
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
        if measurement.privacy_function(mid_stability) != self.adjusted_budget.value:
            raise AssertionError(
                "Privacy function does not match per-query privacy budget. "
                "This is probably a bug; please let us know so we can "
                "fix it!"
            )

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
        the query has DropNullAndNan and/or DropInfinity queries
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

    # None of these produce measurements, so they all return a NotImplementedError
    def visit_private_source(self, expr) -> Any:
        """Visit a PrivateSource query expression (raises an error)."""
        raise NotImplementedError

    def visit_rename(self, expr) -> Any:
        """Visit a Rename query expression (raises an error)."""
        raise NotImplementedError

    def visit_filter(self, expr) -> Any:
        """Visit a Filter query expression (raises an error)."""
        raise NotImplementedError

    def visit_select(self, expr) -> Any:
        """Visit a Select query expression (raises an error)."""
        raise NotImplementedError

    def visit_map(self, expr) -> Any:
        """Visit a Map query expression (raises an error)."""
        raise NotImplementedError

    def visit_flat_map(self, expr) -> Any:
        """Visit a FlatMap query expression (raises an error)."""
        raise NotImplementedError

    def visit_join_private(self, expr) -> Any:
        """Visit a JoinPrivate query expression (raises an error)."""
        raise NotImplementedError

    def visit_join_public(self, expr) -> Any:
        """Visit a JoinPublic query expression (raises an error)."""
        raise NotImplementedError

    def visit_replace_null_and_nan(self, expr) -> Any:
        """Visit a ReplaceNullAndNan query expression (raises an error)."""
        raise NotImplementedError

    def visit_replace_infinity(self, expr) -> Any:
        """Visit a ReplaceInfinity query expression (raises an error)."""
        raise NotImplementedError

    def visit_drop_infinity(self, expr) -> Any:
        """Visit a DropInfinity query expression (raises an error)."""
        raise NotImplementedError

    def visit_drop_null_and_nan(self, expr) -> Any:
        """Visit a DropNullAndNan query expression (raises an error)."""
        raise NotImplementedError

    def visit_enforce_constraint(self, expr) -> Any:
        """Visit an EnforceConstraint query expression (raises an error)."""
        raise NotImplementedError
