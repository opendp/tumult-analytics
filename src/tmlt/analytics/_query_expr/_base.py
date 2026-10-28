"""Base types and shared helpers for query expressions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

import datetime
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple, Union

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.utils.join import domain_after_join
from typeguard import check_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_dataframe_domain_to_analytics_columns,
)
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.keyset import KeySet

if TYPE_CHECKING:
    from ._drop_infinity import DropInfinity
    from ._drop_null_and_nan import DropNullAndNan
    from ._enforce_constraint import EnforceConstraint
    from ._filter import Filter
    from ._flat_map import FlatMap
    from ._flat_map_by_id import FlatMapByID
    from ._get_bounds import GetBounds
    from ._get_groups import GetGroups
    from ._group_by_bounded_average import GroupByBoundedAverage
    from ._group_by_bounded_stdev import GroupByBoundedStdev
    from ._group_by_bounded_sum import GroupByBoundedSum
    from ._group_by_bounded_variance import GroupByBoundedVariance
    from ._group_by_count import GroupByCount
    from ._group_by_count_distinct import GroupByCountDistinct
    from ._group_by_quantile import GroupByQuantile
    from ._join_private import JoinPrivate
    from ._join_public import JoinPublic
    from ._map import Map
    from ._private_source import PrivateSource
    from ._rename import Rename
    from ._replace_infinity import ReplaceInfinity
    from ._replace_null_and_nan import ReplaceNullAndNan
    from ._select import Select
    from ._suppress_aggregates import SuppressAggregates


Row = Dict[str, Any]
"""Type alias for dictionary with string keys."""


class QueryExpr(ABC):
    """A query expression, base class for relational operators.

    QueryExpr are organized in a tree, where each node is an operator that returns a
    table. They are built using the :class:`tmlt.analytics.QueryBuilder`, then rewritten
    during the compilation process. They should not be created directly, except in
    tests.
    """

    @abstractmethod
    def schema(self, catalog: Catalog) -> Any:
        """Returns the schema resulting from evaluating this QueryExpr."""
        raise NotImplementedError()

    @abstractmethod
    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Constraints on the output of this query when applied to the given catalog."""
        raise NotImplementedError()

    def is_measurement(self) -> bool:
        """Returns True if evaluating this expression produces a publishable table."""
        return False

    @abstractmethod
    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Dispatch methods on a visitor based on the QueryExpr type."""
        raise NotImplementedError()


def _validate_child_is_transformation(child: QueryExpr):
    """Validate that an object is a non-measurement query expression."""
    check_type(child, QueryExpr)
    if child.is_measurement():
        raise AnalyticsInternalError(
            "Measurement used as query expression child where only transformations are allowed"
        )


@dataclass(frozen=True)
class SingleChildQueryExpr(QueryExpr):
    """A QueryExpr that has a single child.

    This is used in the compilation step, to make it easier for rewrite rules to
    automatically recurse along the QueryExpr tree.
    """

    child: QueryExpr
    """The QueryExpr used to generate the input table to this QueryExpr."""


class MeasurementQueryExpr(SingleChildQueryExpr):
    """A QueryExpr that can be evaluated to produce a publishable table."""

    def is_measurement(self) -> bool:
        """Returns True if evaluating this expression produces a publishable table."""
        return True

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        raise AnalyticsInternalError(
            "Measurement queries do not have constraints on their outputs"
        )


class GroupedMeasurementQueryExpr(MeasurementQueryExpr):
    """A measurement query expression with group-by keys."""

    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""

    @property
    def groupby_columns(self) -> Tuple[str, ...]:
        """The names of the columns to group by, in output order."""
        if isinstance(self.groupby_keys, KeySet):
            return tuple(self.groupby_keys.schema().keys())
        return self.groupby_keys


def _validate_join(
    left_schema: Schema,
    right_schema: Schema,
    join_columns: Optional[Tuple[str, ...]],
):
    """Validates that both tables can be joined by comparing their schemas.

    This is used for both public and private joins; therefore, this does not check
    any properties related to ID columns & ID spaces.
    """
    if (
        left_schema.grouping_column is not None
        and right_schema.grouping_column is not None
        and left_schema.grouping_column != right_schema.grouping_column
    ):
        raise ValueError(
            "Joining tables which both have grouping columns is only supported "
            "if they have the same grouping column"
        )

    if join_columns is not None and not join_columns:
        # This error case should be caught when constructing the query
        # expression, so it should never get here.
        raise AnalyticsInternalError("Empty list of join columns provided.")

    common_columns = set(left_schema) & set(right_schema)
    if join_columns is None and not common_columns:
        raise ValueError("Tables have no common columns to join on")
    join_columns = join_columns or tuple(common_columns)
    if not set(join_columns) <= common_columns:
        raise ValueError("Join columns must be common to both tables")

    for column in join_columns:
        if left_schema[column].column_type != right_schema[column].column_type:
            raise ValueError(
                "Join columns must have identical types on both tables, "
                f"but column '{column}' does not: {left_schema[column]} and "
                f"{right_schema[column]} are incompatible"
            )


def _schema_for_join(
    left_schema: Schema,
    right_schema: Schema,
    join_columns: Optional[Tuple[str, ...]],
    join_id_space: Optional[str],
    how: str,
) -> Schema:
    """Return the schema resulting from joining two tables.

    It is assumed that if either schema has an ID column, the one from left_schema
    should be used, because this is true for both public and private joins. With private
    joins, the ID columns must be compatible; this check must happen outside this
    function.

    Args:
        left_schema: Schema for the left table.
        right_schema: Schema for the right table.
        join_columns: The set of columns to join on.
        join_id_space: The ID space of the resulting join.
        how: The type of join to perform.
    """
    grouping_column = left_schema.grouping_column or right_schema.grouping_column
    common_columns = set(left_schema) & set(right_schema)
    join_columns = join_columns or tuple(
        sorted(common_columns, key=list(left_schema).index)
    )

    # Get the join schema from the Core convenience method
    output_domain = domain_after_join(
        left_domain=SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(left_schema)
        ),
        right_domain=SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(right_schema)
        ),
        on=list(join_columns),
        how=how,
        nulls_are_equal=True,
    )
    return Schema(
        column_descs=spark_dataframe_domain_to_analytics_columns(output_domain),
        grouping_column=grouping_column,
        id_column=left_schema.id_column,
        id_space=join_id_space,
    )


def _propagate_join_by_stability(
    c: Constraint,
    join_stability: Optional[int],
    overlapping_cols: Set[str],
    suffix: str,
) -> Optional[Constraint]:
    """Propagate a constraint through a join based on the maximum join stability.

    This function returns a constraint that results from propagating the given
    constraint through a join with the given join stability. If that stability
    is None, the join is assumed to be unbounded. Column renaming because of
    overlapping columns is handled, appending the given suffix to any relevant
    columns mentioned in the constraint.
    """

    def col_name(base: str):
        return base + suffix if base in overlapping_cols else base

    if isinstance(c, MaxRowsPerID):
        if join_stability:
            return MaxRowsPerID(c.max * join_stability)
        else:
            return None
    elif isinstance(c, MaxGroupsPerID):
        return MaxGroupsPerID(col_name(c.grouping_column), c.max)
    elif isinstance(c, MaxRowsPerGroupPerID):
        if join_stability:
            return MaxRowsPerGroupPerID(
                col_name(c.grouping_column), c.max * join_stability
            )
        else:
            return None
    return None


class AnalyticsDefault:
    """Default values for each type of column in Tumult Analytics."""

    INTEGER = 0
    """The default value used for integers (0)."""
    DECIMAL = 0.0
    """The default value used for floats (0)."""
    VARCHAR = ""
    """The default value used for VARCHARs (the empty string)."""
    DATE = datetime.date.fromtimestamp(0)
    """The default value used for dates (``datetime.date.fromtimestamp(0)``).

    See :meth:`~.datetime.date.fromtimestamp`.
    """
    TIMESTAMP = datetime.datetime.fromtimestamp(0)
    """The default value used for timestamps (``datetime.datetime.fromtimestamp(0)``).

    See :meth:`~.datetime.datetime.fromtimestamp`.
    """


def _valid_replacement_value(val: Any):
    """Determine if a value is a valid replacement, independent of schema matching."""
    return val is not None and not (
        isinstance(val, float) and (math.isnan(val) or math.isinf(val))
    )


def _validate_groupby(
    query: GroupedMeasurementQueryExpr,
    input_schema: Schema,
):
    """Validates the arguments of a group-by QueryExpr."""
    # Validating group-by columns
    if isinstance(query.groupby_keys, KeySet):
        # Checks that the KeySet is valid
        schema = query.groupby_keys.schema()
        for column_name, column_desc in schema.items():
            try:
                input_column_desc = input_schema[column_name]
            except KeyError as e:
                raise KeyError(
                    f"Groupby column '{column_name}' is not in the input schema."
                ) from e
            if column_desc.column_type != input_column_desc.column_type:
                raise ValueError(
                    f"Groupby column '{column_name}' has type"
                    f" '{column_desc.column_type.name}', but the column with the same "
                    f"name in the input data has type "
                    f"'{input_column_desc.column_type.name}' instead."
                )
    elif isinstance(query.groupby_keys, tuple):
        # Checks that the listed groupby columns exist in the schema
        for col in query.groupby_columns:
            if col not in input_schema:
                raise ValueError(f"Groupby column '{col}' is not in the input schema.")
    else:
        raise AnalyticsInternalError(
            f"Unexpected groupby_keys type: {type(query.groupby_keys)}."
        )

    # Validating compatibility between grouping columns and group-by columns
    groupby_columns = query.groupby_columns
    grouping_column = input_schema.grouping_column
    if grouping_column is not None and grouping_column not in groupby_columns:
        raise ValueError(
            f"Column '{grouping_column}' produced by grouping transformation "
            f"is not in groupby columns {list(groupby_columns)}."
        )


def _validate_measure_column(
    query: Union[
        "GetBounds",
        "GroupByBoundedAverage",
        "GroupByBoundedStdev",
        "GroupByBoundedSum",
        "GroupByBoundedVariance",
        "GroupByQuantile",
    ],
    input_schema: Schema,
):
    """Validates the measure column of a group-by QueryExpr."""
    if query.measure_column in query.groupby_columns:
        raise ValueError(
            "Column to aggregate must be a non-grouped column, not "
            f"'{query.measure_column}'."
        )

    if query.measure_column not in input_schema:
        raise ValueError(
            f"{type(query).__name__} query's measure column "
            f"'{query.measure_column}' does not exist in the table."
        )
    if input_schema[query.measure_column].column_type not in [
        ColumnType.INTEGER,
        ColumnType.DECIMAL,
    ]:
        raise ValueError(
            f"{type(query).__name__} query's measure column "
            f"'{query.measure_column}' has invalid type "
            f"'{input_schema[query.measure_column].column_type.name}'. "
            "Expected types: 'INTEGER' or 'DECIMAL'."
        )
    if input_schema.id_column and (input_schema.id_column == query.measure_column):
        raise ValueError(
            f"{type(query).__name__} query's measure column is the same as the "
            f"privacy ID column({input_schema.id_column}) on a table with the "
            "AddRowsWithID protected change."
        )


def _schema_for_groupby(
    query: GroupedMeasurementQueryExpr,
    input_schema: Schema,
    output_columns: Dict[str, ColumnDescriptor],
) -> Schema:
    """Combines group-by columns with the given aggregate output columns."""
    return Schema(
        {
            **{column: input_schema[column] for column in query.groupby_columns},
            **output_columns,
        },
        grouping_column=None,
        id_column=None,
    )


class QueryExprVisitor(ABC):
    """A base class for implementing visitors for :class:`QueryExpr`."""

    @abstractmethod
    def visit_private_source(self, expr: "PrivateSource") -> Any:
        """Visit a :class:`PrivateSource`."""
        raise NotImplementedError

    @abstractmethod
    def visit_rename(self, expr: "Rename") -> Any:
        """Visit a :class:`Rename`."""
        raise NotImplementedError

    @abstractmethod
    def visit_filter(self, expr: "Filter") -> Any:
        """Visit a :class:`Filter`."""
        raise NotImplementedError

    @abstractmethod
    def visit_select(self, expr: "Select") -> Any:
        """Visit a :class:`Select`."""
        raise NotImplementedError

    @abstractmethod
    def visit_map(self, expr: "Map") -> Any:
        """Visit a :class:`Map`."""
        raise NotImplementedError

    @abstractmethod
    def visit_flat_map(self, expr: "FlatMap") -> Any:
        """Visit a :class:`FlatMap`."""
        raise NotImplementedError

    @abstractmethod
    def visit_flat_map_by_id(self, expr: "FlatMapByID") -> Any:
        """Visit a :class:`FlatMapByID`."""
        raise NotImplementedError

    @abstractmethod
    def visit_join_private(self, expr: "JoinPrivate") -> Any:
        """Visit a :class:`JoinPrivate`."""
        raise NotImplementedError

    @abstractmethod
    def visit_join_public(self, expr: "JoinPublic") -> Any:
        """Visit a :class:`JoinPublic`."""
        raise NotImplementedError

    @abstractmethod
    def visit_replace_null_and_nan(self, expr: "ReplaceNullAndNan") -> Any:
        """Visit a :class:`ReplaceNullAndNan`."""
        raise NotImplementedError

    @abstractmethod
    def visit_replace_infinity(self, expr: "ReplaceInfinity") -> Any:
        """Visit a :class:`ReplaceInfinity`."""
        raise NotImplementedError

    @abstractmethod
    def visit_drop_null_and_nan(self, expr: "DropNullAndNan") -> Any:
        """Visit a :class:`DropNullAndNan`."""
        raise NotImplementedError

    @abstractmethod
    def visit_drop_infinity(self, expr: "DropInfinity") -> Any:
        """Visit a :class:`DropInfinity`."""
        raise NotImplementedError

    @abstractmethod
    def visit_enforce_constraint(self, expr: "EnforceConstraint") -> Any:
        """Visit a :class:`EnforceConstraint`."""
        raise NotImplementedError

    @abstractmethod
    def visit_get_groups(self, expr: "GetGroups") -> Any:
        """Visit a :class:`GetGroups`."""
        raise NotImplementedError

    @abstractmethod
    def visit_get_bounds(self, expr: "GetBounds") -> Any:
        """Visit a :class:`GetBounds`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_count(self, expr: "GroupByCount") -> Any:
        """Visit a :class:`GroupByCount`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_count_distinct(self, expr: "GroupByCountDistinct") -> Any:
        """Visit a :class:`GroupByCountDistinct`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_quantile(self, expr: "GroupByQuantile") -> Any:
        """Visit a :class:`GroupByQuantile`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_sum(self, expr: "GroupByBoundedSum") -> Any:
        """Visit a :class:`GroupByBoundedSum`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_average(self, expr: "GroupByBoundedAverage") -> Any:
        """Visit a :class:`GroupByBoundedAverage`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_variance(self, expr: "GroupByBoundedVariance") -> Any:
        """Visit a :class:`GroupByBoundedVariance`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_stdev(self, expr: "GroupByBoundedStdev") -> Any:
        """Visit a :class:`GroupByBoundedStdev`."""
        raise NotImplementedError

    @abstractmethod
    def visit_suppress_aggregates(self, expr: "SuppressAggregates") -> Any:
        """Visit a :class:`SuppressAggregates`."""
        raise NotImplementedError
