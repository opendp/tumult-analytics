# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

"""Query expressions."""

import datetime
import math
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from typeguard import check_type, typechecked

from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    FrozenDict,
    Schema,
    analytics_to_py_types,
)
from tmlt.analytics._table_identifier import TableCollection, TableIdentifier
from tmlt.analytics.binning_spec import BinningSpec
from tmlt.analytics.constraints import Constraint
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.truncation_strategy import TruncationStrategy

T = TypeVar("T", bound="QueryExpr")


class QueryExprVisitor(object):
    """Abstract base class for QueryExpr visitors."""

    def visit(self, expr: "QueryExpr"):
        """Call the appropriate method for this QueryExpr subclass."""
        if isinstance(expr, PrivateSource):
            return self.visit_private_source(expr)
        if isinstance(expr, Rename):
            return self.visit_rename(expr)
        if isinstance(expr, Filter):
            return self.visit_filter(expr)
        if isinstance(expr, Select):
            return self.visit_select(expr)
        if isinstance(expr, Map):
            return self.visit_map(expr)
        if isinstance(expr, FlatMap):
            return self.visit_flat_map(expr)
        if isinstance(expr, FlatMapByID):
            return self.visit_flat_map_by_id(expr)
        if isinstance(expr, JoinPrivate):
            return self.visit_join_private(expr)
        if isinstance(expr, JoinPublic):
            return self.visit_join_public(expr)
        if isinstance(expr, ReplaceNullAndNan):
            return self.visit_replace_null_and_nan(expr)
        if isinstance(expr, ReplaceInfinity):
            return self.visit_replace_infinity(expr)
        if isinstance(expr, DropNullAndNan):
            return self.visit_drop_null_and_nan(expr)
        if isinstance(expr, DropInfinity):
            return self.visit_drop_infinity(expr)
        if isinstance(expr, EnforceConstraint):
            return self.visit_enforce_constraint(expr)
        if isinstance(expr, GetGroups):
            return self.visit_get_groups(expr)
        if isinstance(expr, GetBounds):
            return self.visit_get_bounds(expr)
        if isinstance(expr, GroupByCount):
            return self.visit_groupby_count(expr)
        if isinstance(expr, GroupByCountDistinct):
            return self.visit_groupby_count_distinct(expr)
        if isinstance(expr, GroupByQuantile):
            return self.visit_groupby_quantile(expr)
        if isinstance(expr, GroupByBoundedSum):
            return self.visit_groupby_bounded_sum(expr)
        if isinstance(expr, GroupByBoundedAverage):
            return self.visit_groupby_bounded_average(expr)
        if isinstance(expr, GroupByBoundedVariance):
            return self.visit_groupby_bounded_variance(expr)
        if isinstance(expr, GroupByBoundedSTDEV):
            return self.visit_groupby_bounded_stdev(expr)
        if isinstance(expr, SuppressAggregates):
            return self.visit_suppress_aggregates(expr)
        raise NotImplementedError(f"Unsupported QueryExpr type: {type(expr)}")

    def visit_private_source(self, expr: "PrivateSource"):
        """Visit a :class:`PrivateSource` object."""
        raise NotImplementedError

    def visit_rename(self, expr: "Rename"):
        """Visit a :class:`Rename` object."""
        raise NotImplementedError

    def visit_filter(self, expr: "Filter"):
        """Visit a :class:`Filter` object."""
        raise NotImplementedError

    def visit_select(self, expr: "Select"):
        """Visit a :class:`Select` object."""
        raise NotImplementedError

    def visit_map(self, expr: "Map"):
        """Visit a :class:`Map` object."""
        raise NotImplementedError

    def visit_flat_map(self, expr: "FlatMap"):
        """Visit a :class:`FlatMap` object."""
        raise NotImplementedError

    def visit_flat_map_by_id(self, expr: "FlatMapByID"):
        """Visit a :class:`FlatMapByID` object."""
        raise NotImplementedError

    def visit_join_private(self, expr: "JoinPrivate"):
        """Visit a :class:`JoinPrivate` object."""
        raise NotImplementedError

    def visit_join_public(self, expr: "JoinPublic"):
        """Visit a :class:`JoinPublic` object."""
        raise NotImplementedError

    def visit_replace_null_and_nan(self, expr: "ReplaceNullAndNan"):
        """Visit a :class:`ReplaceNullAndNan` object."""
        raise NotImplementedError

    def visit_replace_infinity(self, expr: "ReplaceInfinity"):
        """Visit a :class:`ReplaceInfinity` object."""
        raise NotImplementedError

    def visit_drop_null_and_nan(self, expr: "DropNullAndNan"):
        """Visit a :class:`DropNullAndNan` object."""
        raise NotImplementedError

    def visit_drop_infinity(self, expr: "DropInfinity"):
        """Visit a :class:`DropInfinity` object."""
        raise NotImplementedError

    def visit_enforce_constraint(self, expr: "EnforceConstraint"):
        """Visit an :class:`EnforceConstraint` object."""
        raise NotImplementedError

    def visit_get_groups(self, expr: "GetGroups"):
        """Visit a :class:`GetGroups` object."""
        raise NotImplementedError

    def visit_get_bounds(self, expr: "GetBounds"):
        """Visit a :class:`GetBounds` object."""
        raise NotImplementedError

    def visit_groupby_count(self, expr: "GroupByCount"):
        """Visit a :class:`GroupByCount` object."""
        raise NotImplementedError

    def visit_groupby_count_distinct(self, expr: "GroupByCountDistinct"):
        """Visit a :class:`GroupByCountDistinct` object."""
        raise NotImplementedError

    def visit_groupby_quantile(self, expr: "GroupByQuantile"):
        """Visit a :class:`GroupByQuantile` object."""
        raise NotImplementedError

    def visit_groupby_bounded_sum(self, expr: "GroupByBoundedSum"):
        """Visit a :class:`GroupByBoundedSum` object."""
        raise NotImplementedError

    def visit_groupby_bounded_average(self, expr: "GroupByBoundedAverage"):
        """Visit a :class:`GroupByBoundedAverage` object."""
        raise NotImplementedError

    def visit_groupby_bounded_variance(self, expr: "GroupByBoundedVariance"):
        """Visit a :class:`GroupByBoundedVariance` object."""
        raise NotImplementedError

    def visit_groupby_bounded_stdev(self, expr: "GroupByBoundedSTDEV"):
        """Visit a :class:`GroupByBoundedSTDEV` object."""
        raise NotImplementedError

    def visit_suppress_aggregates(self, expr: "SuppressAggregates"):
        """Visit a :class:`SuppressAggregates` object."""
        raise NotImplementedError


@dataclass(frozen=True)
class QueryExpr:
    """The base class for query expressions.

    Subclasses of :class:`QueryExpr` represent nodes in a QueryExpr tree. They
    should generally be constructed using the :class:`QueryBuilder` API.
    """

    def accept(self, visitor: QueryExprVisitor):
        """Invoke ``visitor.visit_xxx(self)``."""
        return visitor.visit(self)


@dataclass(frozen=True)
class PrivateSource(QueryExpr):
    """Represents a private data source.

    Attributes:
        source_id: The table ID of the private data source.
    """

    source_id: str

    @typechecked
    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", self.source_id):
            raise ValueError("source_id must be a valid Python identifier.")


@dataclass(frozen=True)
class Rename(QueryExpr):
    """Renames one or more columns in the DataFrame.

    Attributes:
        child: The input query.
        column_mapper: A mapping of old column names to new column names.
    """

    child: QueryExpr
    column_mapper: FrozenDict[str, str]

    @typechecked
    def __post_init__(self) -> None:
        if any(not new_name for new_name in self.column_mapper.values()):
            bad_col = next(n for n in self.column_mapper.values() if not n)
            old_col = next(k for k, v in self.column_mapper.items() if v == bad_col)
            raise ValueError(
                f'Cannot rename column {old_col} to "{bad_col}" (the empty string):'
                " columns named \"\" are not allowed"
            )


@dataclass(frozen=True)
class Filter(QueryExpr):
    """Filters the DataFrame using a SQL expression.

    Attributes:
        child: The input query.
        condition: A SQL expression (e.g. "A < B AND C > 0") representing the
            filter condition.
    """

    child: QueryExpr
    condition: str


@dataclass(frozen=True)
class Select(QueryExpr):
    """Selects a subset of columns from the DataFrame.

    Attributes:
        child: The input query.
        columns: The names of columns to select.
    """

    child: QueryExpr
    columns: Tuple[str, ...]

    @typechecked
    def __post_init__(self) -> None:
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Column names must be distinct.")


@dataclass(frozen=True)
class Map(QueryExpr):
    """Applies a Python function to each row of the DataFrame.

    Attributes:
        child: The input query.
        f: A Python function that takes a row (as a dictionary from column
            name to value) and returns a dictionary of new column name to new
            value.
        schema_new_columns: The :class:`~tmlt.analytics._schema.Schema`
            of the columns returned by `f`.
        augment: If True, the original columns are kept and the new columns
            returned by `f` are added to the DataFrame. If False, only the
            new columns returned by `f` are kept.
    """

    child: QueryExpr
    f: Callable[[Dict[str, Any]], Dict[str, Any]]
    schema_new_columns: Schema
    augment: bool

    @typechecked
    def __post_init__(self) -> None:
        if self.schema_new_columns.grouping_column:
            raise ValueError("Map cannot be be used to create grouping columns")
        if any(not new_name for new_name in self.schema_new_columns.keys()):
            bad_col = next(n for n in self.schema_new_columns.keys() if not n)
            raise ValueError(
                f'"" (the empty string) is not a supported column name for the '
                f"new column {bad_col!r}"
            )


@dataclass(frozen=True)
class FlatMap(QueryExpr):
    """Applies a Python function to each row of the DataFrame that may return
    multiple rows.

    Attributes:
        child: The input query.
        f: A Python function that takes a row (as a dictionary from column
            name to value) and returns an iterable of dictionaries from new
            column name to new value.
        schema_new_columns: The :class:`~tmlt.analytics._schema.Schema`
            of the columns returned by `f`.
        augment: If True, the original columns are kept and the new columns
            returned by `f` are added to the DataFrame. If False, only the
            new columns returned by `f` are kept.
        max_rows: The maximum number of rows that can be returned by `f` for
            any given input row.
    """

    child: QueryExpr
    f: Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]
    schema_new_columns: Schema
    augment: bool
    max_rows: int

    @typechecked
    def __post_init__(self) -> None:
        if self.max_rows < 0:
            raise ValueError(
                f"Limit on number of rows '{self.max_rows}' must be non-negative."
            )
        if any(not new_name for new_name in self.schema_new_columns.keys()):
            bad_col = next(n for n in self.schema_new_columns.keys() if not n)
            raise ValueError(
                f'"" (the empty string) is not a supported column name for the '
                f"new column {bad_col!r}"
            )
        if self.schema_new_columns.grouping_column and len(self.schema_new_columns) > 1:
            raise ValueError(
                f"schema_new_columns contains {len(self.schema_new_columns)} columns,"
                " grouping flat map can only result in 1 new column"
            )


@dataclass(frozen=True)
class FlatMapByID(QueryExpr):
    """Applies a Python function to each ID-grouped collection of rows in the
    DataFrame that may return multiple rows.

    Attributes:
        child: The input query.
        f: A Python function that takes an iterable of rows (as a dictionary
            from column name to value) and returns an iterable of dictionaries
            from new column name to new value. The input iterable represents
            all rows belonging to a single privacy ID.
        schema_new_columns: The :class:`~tmlt.analytics._schema.Schema`
            of the columns returned by `f`. Note that this schema cannot contain
            a grouping column or an ID column.
        max_rows: The maximum number of rows that can be returned by `f` for
            any given input iterable of rows. By default, there is no limit.
            This value, if provided, directly impacts the output stability of the
            transformation in the DP analysis.
    """

    child: QueryExpr
    f: Callable[[Iterable[Dict[str, Any]]], Iterable[Dict[str, Any]]]
    schema_new_columns: Schema
    max_rows: Optional[int] = None

    @typechecked
    def __post_init__(self) -> None:
        if self.schema_new_columns.grouping_column:
            # We don't support `FlatMapByID` as a way to convert to a grouped
            # table, as that is hard to do efficiently and cleanly.
            from tmlt.analytics import AnalyticsInternalError

            raise AnalyticsInternalError(
                "FlatMapByID cannot be used to create grouping columns"
            )
        if self.schema_new_columns.id_column:
            # We also don't support `FlatMapByID` to create ID columns.
            from tmlt.analytics import AnalyticsInternalError

            raise AnalyticsInternalError(
                "FlatMapByID cannot be used to create ID columns"
            )
        if any(not new_name for new_name in self.schema_new_columns.keys()):
            bad_col = next(n for n in self.schema_new_columns.keys() if not n)
            raise ValueError(
                f'"" (the empty string) is not a supported column name for the '
                f"new column {bad_col!r}"
            )
        if self.max_rows is not None and self.max_rows < 0:
            raise ValueError(
                f"Limit on number of rows '{self.max_rows}' must be non-negative."
            )


@dataclass(frozen=True)
class JoinPrivate(QueryExpr):
    """Joins two private tables.

    Attributes:
        child: The left-hand side of the join.
        right_operand_expr: The right-hand side of the join.
        truncation_strategy_left: The truncation strategy to apply to the
            left-hand side of the join.
        truncation_strategy_right: The truncation strategy to apply to the
            right-hand side of the join.
        join_columns: The columns to join on. If None, joins on all common
            columns.
        how: The join type. Must be 'inner' or 'left'.
    """

    child: QueryExpr
    right_operand_expr: QueryExpr
    truncation_strategy_left: TruncationStrategy.Type
    truncation_strategy_right: TruncationStrategy.Type
    join_columns: Optional[Tuple[str, ...]] = None
    how: str = "inner"

    @typechecked
    def __post_init__(self) -> None:
        if self.join_columns is not None:
            if not self.join_columns:
                raise ValueError("Provided join columns must not be empty")
            if len(self.join_columns) != len(set(self.join_columns)):
                raise ValueError("Join columns must be distinct")
        if self.how not in ["inner", "left"]:
            raise ValueError(
                f"Invalid join type '{self.how}': must be 'inner' or 'left'"
            )


@dataclass(frozen=True)
class JoinPublic(QueryExpr):
    """Joins a private table with a public table.

    Attributes:
        child: The private table.
        public_table: The public table to join with. This can be the name of
            a public table added to the session, or a Spark DataFrame.
        join_columns: The columns to join on. If None, joins on all common
            columns.
        how: The join type. Must be 'inner' or 'left'.
    """

    child: QueryExpr
    public_table: Union[str, Any]
    join_columns: Optional[Tuple[str, ...]] = None
    how: str = "inner"

    @typechecked
    def __post_init__(self) -> None:
        if self.join_columns is not None:
            if not self.join_columns:
                raise ValueError("Provided join columns must not be empty")
            if len(self.join_columns) != len(set(self.join_columns)):
                raise ValueError("Join columns must be distinct")
        if self.how not in ["inner", "left"]:
            raise ValueError(
                f"Invalid join type '{self.how}': must be 'inner' or 'left'"
            )


@dataclass(frozen=True)
class ReplaceNullAndNan(QueryExpr):
    """Replaces null and NaN values in the DataFrame.

    Attributes:
        child: The input query.
        replace_with: A mapping from column name to replacement value. If no
            columns are specified, replaces all null and NaN values in numeric
            columns with 0, and all null values in string columns with "".
            Replacement values cannot be NaN or infinity.
    """

    child: QueryExpr
    replace_with: FrozenDict[
        str, Union[int, float, str, datetime.date, datetime.datetime]
    ]

    @typechecked
    def __post_init__(self) -> None:
        for column, value in self.replace_with.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ValueError(
                    f"Replacement value for column '{column}' cannot be NaN or infinity."
                )


@dataclass(frozen=True)
class ReplaceInfinity(QueryExpr):
    """Replaces infinite values in the DataFrame.

    Attributes:
        child: The input query.
        replace_with: A mapping from column name to (lower bound, upper bound)
            tuple. Values outside these bounds (including infinity) are clamped
            to the corresponding bound. If no columns are specified, replaces
            all infinite values in numeric columns with 0.
            Replacement bounds cannot be NaN or infinity.
    """

    child: QueryExpr
    replace_with: FrozenDict[str, Tuple[float, float]]

    @typechecked
    def __post_init__(self) -> None:
        for column, (lower_bound, upper_bound) in self.replace_with.items():
            # The @typechecked decorator ensures lower_bound and upper_bound are floats.
            # Explicit isinstance check is technically redundant but harmless.
            if math.isnan(lower_bound) or math.isinf(lower_bound):
                raise ValueError(
                    f"Lower bound for column '{column}' cannot be NaN or infinity."
                )
            if math.isnan(upper_bound) or math.isinf(upper_bound):
                raise ValueError(
                    f"Upper bound for column '{column}' cannot be NaN or infinity."
                )


@dataclass(frozen=True)
class DropNullAndNan(QueryExpr):
    """Drops rows from the DataFrame containing null or NaN values.

    Attributes:
        child: The input query.
        columns: The subset of columns for which null/NaN values are used for
            filtering. If no columns are specified, uses all columns.
    """

    child: QueryExpr
    columns: Tuple[str, ...]


@dataclass(frozen=True)
class DropInfinity(QueryExpr):
    """Drops rows from the DataFrame containing infinite values.

    Attributes:
        child: The input query.
        columns: The subset of columns for which infinite values are used for
            filtering. If no columns are specified, uses all columns.
    """

    child: QueryExpr
    columns: Tuple[str, ...]


@dataclass(frozen=True)
class EnforceConstraint(QueryExpr):
    """Enforces a privacy constraint on the DataFrame.

    Attributes:
        child: The input query.
        constraint: The constraint to enforce.
        per_id_columns_map: Dictionary mapping the ID column to the columns (if any)
            that are unique for a given ID. This is typically used internally
            for computing per-ID bounds, and is passed in from `Session.from_dataframe()`.
    """

    child: QueryExpr
    constraint: Constraint
    per_id_columns_map: FrozenDict[TableIdentifier, Tuple[str, ...]]


@dataclass(frozen=True)
class GetGroups(QueryExpr):
    """Gets the groups in the DataFrame.

    Attributes:
        child: The input query.
        columns: The columns to group on. If no columns are specified, uses all
            columns except the privacy ID column if the table is identified by a
            privacy ID.
    """

    child: QueryExpr
    columns: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class GetBounds(QueryExpr):
    """Gets the lower and upper bounds of a column's values after per-ID clamping.

    The lower and upper bound are computed on a per-ID basis. For each privacy ID,
    the minimum and maximum values of the specified `measure_column` are computed.
    The lower bound column then contains the per-ID minimums, and the upper bound
    column contains the per-ID maximums.

    Attributes:
        child: The input query.
        groupby_keys: The grouping columns. If empty, computes the minimum and maximum
            values of the `measure_column` over the entire dataset.
        measure_column: The column for which to compute the per-ID bounds.
        lower_bound_column: The name of the column that contains the per-ID lower
            bounds of `measure_column`.
        upper_bound_column: The name of the column that contains the per-ID upper
            bounds of `measure_column`.
    """

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    lower_bound_column: str
    upper_bound_column: str


@dataclass(frozen=True)
class SuppressAggregates(QueryExpr):
    """Suppresses aggregated values below a threshold.

    Attributes:
        child: The aggregate to suppress.
        column: The name of the column to suppress.
        threshold: The threshold below which aggregated values are suppressed.
    """

    child: "GroupByCount"
    column: str
    threshold: Union[int, float]

    @typechecked
    def __post_init__(self) -> None:
        if not isinstance(self.child, GroupByCount):
            # SuppressAggregates is only supported on CountAggregates
            raise TypeError(
                "SuppressAggregates is only supported on aggregates that are"
                f" GroupByCounts, but received an aggregate of type {type(self.child)}"
            )


@dataclass(frozen=True)
class GroupByCount(QueryExpr):
    """Performs a grouped count.

    Attributes:
        child: The input query.
        groupby_keys: The grouping keys. If the :class:`KeySet` is empty,
            performs a total count of all records.
        output_column: The name of the column to store the count.
        mechanism: The noise mechanism to use for counting.
    """

    child: QueryExpr
    groupby_keys: KeySet
    output_column: str = "count"
    mechanism: "CountMechanism" = "GEOMETRIC"

    @typechecked
    def __post_init__(self) -> None:
        if not self.output_column:
            raise ValueError("Output column name cannot be an empty string")


@dataclass(frozen=True)
class GroupByCountDistinct(QueryExpr):
    """Performs a grouped distinct count.

    Attributes:
        child: The input query.
        groupby_keys: The grouping keys. If the :class:`KeySet` is empty,
            performs a total count of all records.
        columns_to_count: The columns on which to count distinct values.
            If None, counts distinct values over all columns.
        output_column: The name of the column to store the count.
    """

    child: QueryExpr
    groupby_keys: KeySet
    columns_to_count: Optional[Tuple[str, ...]] = None
    output_column: str = "count_distinct"

    @typechecked
    def __post_init__(self) -> None:
        if not self.output_column:
            raise ValueError("Output column name cannot be an empty string")
        if self.columns_to_count is not None and len(self.columns_to_count) != len(
            set(self.columns_to_count)
        ):
            raise ValueError("Columns to count must be distinct")


@dataclass(frozen=True)
class GroupByQuantile(QueryExpr):
    """Computes a grouped quantile of a specified column, with values clamped to
    a specified range.

    Attributes:
        child: The input query.
        groupby_keys: The grouping keys. If the :class:`KeySet` is empty,
            computes a total quantile over all records.
        measure_column: The column for which to compute the quantile.
        quantile: The quantile to compute. Must be between 0 and 1, inclusive.
        low: The lower bound for values in `measure_column`. Values in
            `measure_column` less than `low` are clamped to `low`. Must be
            less than `high`.
        high: The upper bound for values in `measure_column`. Values in
            `measure_column` greater than `high` are clamped to `high`. Must be
            greater than `low`.
        output_column: The name of the column to store the quantile.
    """

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    quantile: float
    low: float
    high: float
    output_column: str = "quantile"

    @typechecked
    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError(
                f"Quantile must be between 0 and 1, and not {self.quantile}."
            )
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than the upper bound"
                f" '{self.high}'."
            )
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        if not self.output_column:
            raise ValueError("Output column name cannot be an empty string")


@dataclass(frozen=True)
class GroupByBoundedSum(QueryExpr):
    """Computes a grouped sum of a specified column, with values clamped to a
    specified range.

    Attributes:
        child: The input query.
        groupby_keys: The grouping keys. If the :class:`KeySet` is empty,
            computes a total sum over all records.
        measure_column: The column for which to compute the sum.
        low: The lower bound for values in `measure_column`. Values in
            `measure_column` less than `low` are clamped to `low`. Must be
            less than `high`.
        high: The upper bound for values in `measure_column`. Values in
            `measure_column` greater than `high` are clamped to `high`. Must be
            greater than `low`.
        output_column: The name of the column to store the sum.
        mechanism: The noise mechanism to use for summing.
    """

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str = "sum"
    mechanism: "SumMechanism" = "LAPLACE"

    @typechecked
    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than the upper bound"
                f" '{self.high}'."
            )
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        if not self.output_column:
            raise ValueError("Output column name cannot be an empty string")


@dataclass(frozen=True)
class GroupByBoundedAverage(QueryExpr):
    """Computes a grouped average of a specified column, with values clamped to
    a specified range.

    Attributes:
        child: The input query.
        groupby_keys: The grouping keys. If the :class:`KeySet` is empty,
            computes a total average over all records.
        measure_column: The column for which to compute the average.
        low: The lower bound for values in `measure_column`. Values in
            `measure_column` less than `low` are clamped to `low`. Must be
            less than `high`.
        high: The upper bound for values in `measure_column`. Values in
            `measure_column` greater than `high` are clamped to `high`. Must be
            greater than `low`.
        output_column: The name of the column to store the average.
        mechanism: The noise mechanism to use for computing the average.
    """

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str = "average"
    mechanism: "AverageMechanism" = "LAPLACE"

    @typechecked
    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than the upper bound"
                f" '{self.high}'."
            )
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        if not self.output_column:
            raise ValueError("Output column name cannot be an empty string")


@dataclass(frozen=True)
class GroupByBoundedVariance(QueryExpr):
    """Computes a grouped variance of a specified column, with values clamped to
    a specified range.

    Attributes:
        child: The input query.
        groupby_keys: The grouping keys. If the :class:`KeySet` is empty,
            computes a total variance over all records.
        measure_column: The column for which to compute the variance.
        low: The lower bound for values in `measure_column`. Values in
            `measure_column` less than `low` are clamped to `low`. Must be
            less than `high`.
        high: The upper bound for values in `measure_column`. Values in
            `measure_column` greater than `high` are clamped to `high`. Must be
            greater than `low`.
        output_column: The name of the column to store the variance.
        mechanism: The noise mechanism to use for computing the variance.
    """

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str = "variance"
    mechanism: "VarianceMechanism" = "LAPLACE"

    @typechecked
    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than the upper bound"
                f" '{self.high}'."
            )
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        if not self.output_column:
            raise ValueError("Output column name cannot be an empty string")


@dataclass(frozen=True)
class GroupByBoundedSTDEV(QueryExpr):
    """Computes a grouped standard deviation of a specified column, with values
    clamped to a specified range.

    Attributes:
        child: The input query.
        groupby_keys: The grouping keys. If the :class:`KeySet` is empty,
            computes a total standard deviation over all records.
        measure_column: The column for which to compute the standard deviation.
        low: The lower bound for values in `measure_column`. Values in
            `measure_column` less than `low` are clamped to `low`. Must be
            less than `high`.
        high: The upper bound for values in `measure_column`. Values in
            `measure_column` greater than `high` are clamped to `high`. Must be
            greater than `low`.
        output_column: The name of the column to store the standard deviation.
        mechanism: The noise mechanism to use for computing the standard
            deviation.
    """

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str = "stdev"
    mechanism: "StdevMechanism" = "LAPLACE"

    @typechecked
    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than the upper bound"
                f" '{self.high}'."
            )
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        if not self.output_column:
            raise ValueError("Output column name cannot be an empty string")


class CountMechanism(str):
    """The noise mechanism used for counting."""

    LAPLACE = "LAPLACE"
    GAUSSIAN = "GAUSSIAN"
    GEOMETRIC = "GEOMETRIC"


class SumMechanism(str):
    """The noise mechanism used for summing."""

    LAPLACE = "LAPLACE"
    GAUSSIAN = "GAUSSIAN"


class AverageMechanism(str):
    """The noise mechanism used for averaging."""

    LAPLACE = "LAPLACE"
    GAUSSIAN = "GAUSSIAN"


class VarianceMechanism(str):
    """The noise mechanism used for computing variance."""

    LAPLACE = "LAPLACE"
    GAUSSIAN = "GAUSSIAN"


class StdevMechanism(str):
    """The noise mechanism used for computing standard deviation."""

    LAPLACE = "LAPLACE"
    GAUSSIAN = "GAUSSIAN"


@dataclass(frozen=True)
class Query:
    """Represents a differentially private query.

    Attributes:
        _query_expr: The query expression.
    """

    _query_expr: QueryExpr

    @typechecked
    def _is_equivalent(self, other: "Query") -> bool:
        """Returns True if the underlying QueryExpr objects are equivalent."""
        return self._query_expr == other._query_expr

    def _get_transformation_chain(self) -> List[QueryExpr]:
        """Returns the chain of QueryExpr objects as a list."""
        chain = []
        current_expr: Optional[QueryExpr] = self._query_expr
        while current_expr is not None:
            chain.append(current_expr)
            current_expr = getattr(current_expr, "child", None)
            if isinstance(current_expr, PrivateSource):
                # The PrivateSource is the beginning of the chain, but its
                # attribute name for child is `source_id`, which is of type str
                # and not a QueryExpr.
                break
        chain.reverse()
        return chain


class QueryBuilder(object):
    """Helper class for creating differentially private queries.

    A :class:`QueryBuilder` object allows you to build a query by chaining
    transformation and aggregation methods. Each method returns a new
    :class:`QueryBuilder` object, allowing for a fluent API. The terminal
    method in the chain (an aggregation) returns a :class:`Query` object.

    Args:
        source_id: The ID of the private table that is the initial input for
            this query.
    """

    @typechecked
    def __init__(self, source_id: str):
        self._query_expr = PrivateSource(source_id)

    @typechecked
    def _from_query_expr(self, query_expr: QueryExpr) -> "QueryBuilder":
        """Creates a :class:`QueryBuilder` from a :class:`QueryExpr`."""
        builder = QueryBuilder.__new__(QueryBuilder)
        builder._query_expr = query_expr
        return builder

    @property
    def source_id(self) -> str:
        """The source ID that this QueryBuilder refers to."""
        current_expr = self._query_expr
        while not isinstance(current_expr, PrivateSource):
            current_expr = getattr(current_expr, "child")
        return current_expr.source_id

    @typechecked
    def clone(self) -> "QueryBuilder":
        """Returns a deep copy of the :class:`QueryBuilder`.

        Returns:
            A deep copy of the :class:`QueryBuilder`.
        """
        return self._from_query_expr(self._query_expr)

    @typechecked
    def rename(self, column_mapper: Mapping[str, str]) -> "QueryBuilder":
        """Renames one or more columns in the DataFrame.

        Args:
            column_mapper: A dictionary mapping old column names to new column
                names.

        Returns:
            A new :class:`QueryBuilder` object with the `rename` transformation
            applied.
        """
        return self._from_query_expr(Rename(self._query_expr, FrozenDict.from_dict(column_mapper)))

    @typechecked
    def filter(self, condition: str) -> "QueryBuilder":
        """Filters the DataFrame using a SQL expression.

        Args:
            condition: A SQL expression (e.g. "A < B AND C > 0") representing the
                filter condition.

        Returns:
            A new :class:`QueryBuilder` object with the `filter` transformation
            applied.
        """
        return self._from_query_expr(Filter(self._query_expr, condition))

    @typechecked
    def select(self, columns: Iterable[str]) -> "QueryBuilder":
        """Selects a subset of columns from the DataFrame.

        Args:
            columns: The names of columns to select.

        Returns:
            A new :class:`QueryBuilder` object with the `select` transformation
            applied.
        """
        return self._from_query_expr(Select(self._query_expr, tuple(columns)))

    @typechecked
    def map(
        self,
        f: Callable[[Dict[str, Any]], Dict[str, Any]],
        new_column_types: Mapping[str, Union[str, ColumnDescriptor]],
        augment: bool = False,
    ) -> "QueryBuilder":
        """Applies a Python function to each row of the DataFrame.

        Args:
            f: A Python function that takes a row (as a dictionary from column
                name to value) and returns a dictionary of new column name to new
                value.
            new_column_types: A mapping from new column name to its type.
                For convenience, string representations of
                :class:`~tmlt.analytics.ColumnType` (e.g. "INTEGER", "VARCHAR")
                are accepted.
            augment: If True, the original columns are kept and the new columns
                returned by `f` are added to the DataFrame. If False, only the
                new columns returned by `f` are kept.

        Returns:
            A new :class:`QueryBuilder` object with the `map` transformation
            applied.
        """
        return self._from_query_expr(
            Map(self._query_expr, f, Schema(new_column_types), augment)
        )

    @typechecked
    def flat_map(
        self,
        f: Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]],
        new_column_types: Mapping[str, Union[str, ColumnDescriptor]],
        augment: bool = False,
        grouping: bool = False,
        max_rows: int = 1,
    ) -> "QueryBuilder":
        """Applies a Python function to each row of the DataFrame that may return
        multiple rows.

        Args:
            f: A Python function that takes a row (as a dictionary from column
                name to value) and returns an iterable of dictionaries from new
                column name to new value.
            new_column_types: A mapping from new column name to its type.
                For convenience, string representations of
                :class:`~tmlt.analytics.ColumnType` (e.g. "INTEGER", "VARCHAR")
                are accepted.
            augment: If True, the original columns are kept and the new columns
                returned by `f` are added to the DataFrame. If False, only the
                new columns returned by `f` are kept.
            grouping: If True, the new columns returned by `f` form a new
                grouping column. Note that if `grouping` is True, `f` must
                return exactly one new column.
            max_rows: The maximum number of rows that can be returned by `f` for
                any given input row. This value directly impacts the output
                stability of the transformation in the DP analysis.

        Returns:
            A new :class:`QueryBuilder` object with the `flat_map` transformation
            applied.
        """
        schema_new_columns = Schema(new_column_types)
        if grouping:
            if len(schema_new_columns) != 1:
                raise ValueError(
                    f"If `grouping` is True, `new_column_types` must contain "
                    f"exactly one column, but found {len(schema_new_columns)}."
                )
            schema_new_columns = Schema(
                new_column_types, grouping_column=list(new_column_types.keys())[0]
            )
        return self._from_query_expr(
            FlatMap(self._query_expr, f, schema_new_columns, augment, max_rows)
        )

    @typechecked
    def flat_map_by_id(
        self,
        f: Callable[[Iterable[Dict[str, Any]]], Iterable[Dict[str, Any]]],
        new_column_types: Mapping[str, Union[str, ColumnDescriptor]],
        max_rows: Optional[int] = None,
    ) -> "QueryBuilder":
        """Applies a Python function to each ID-grouped collection of rows in the
        DataFrame that may return multiple rows.

        Args:
            f: A Python function that takes an iterable of rows (as a dictionary
                from column name to value) and returns an iterable of
                dictionaries from new column name to new value. The input
                iterable represents all rows belonging to a single privacy ID.
            new_column_types: A mapping from new column name to its type.
                For convenience, string representations of
                :class:`~tmlt.analytics.ColumnType` (e.g. "INTEGER", "VARCHAR")
                are accepted. Note that this schema cannot contain a grouping
                column or an ID column.
            max_rows: The maximum number of rows that can be returned by `f` for
                any given input iterable of rows. By default, there is no limit.
                This value, if provided, directly impacts the output stability of
                the transformation in the DP analysis.

        Returns:
            A new :class:`QueryBuilder` object with the `flat_map_by_id`
            transformation applied.
        """
        return self._from_query_expr(
            FlatMapByID(self._query_expr, f, Schema(new_column_types), max_rows)
        )

    @typechecked
    def join_private(
        self,
        right_operand: Union[str, "QueryBuilder"],
        truncation_strategy_left: TruncationStrategy.Type,
        truncation_strategy_right: TruncationStrategy.Type,
        join_columns: Optional[Iterable[str]] = None,
        how: str = "inner",
    ) -> "QueryBuilder":
        """Joins two private tables.

        Args:
            right_operand: The right-hand side of the join. This can be the
                name of a private table or a :class:`QueryBuilder` object.
            truncation_strategy_left: The truncation strategy to apply to the
                left-hand side of the join.
            truncation_strategy_right: The truncation strategy to apply to the
                right-hand side of the join.
            join_columns: The columns to join on. If None, joins on all common
                columns.
            how: The join type. Must be 'inner' or 'left'.

        Returns:
            A new :class:`QueryBuilder` object with the `join_private`
            transformation applied.
        """
        right_operand_expr: QueryExpr
        if isinstance(right_operand, str):
            right_operand_expr = PrivateSource(right_operand)
        elif isinstance(right_operand, QueryBuilder):
            right_operand_expr = right_operand._query_expr
        else:
            raise TypeError(
                f"right_operand must be of type str or QueryBuilder, but received"
                f" {type(right_operand)}"
            )
        return self._from_query_expr(
            JoinPrivate(
                self._query_expr,
                right_operand_expr,
                truncation_strategy_left,
                truncation_strategy_right,
                tuple(join_columns) if join_columns is not None else None,
                how,
            )
        )

    @typechecked
    def join_public(
        self,
        public_table: Any,
        join_columns: Optional[Iterable[str]] = None,
        how: str = "inner",
    ) -> "QueryBuilder":
        """Joins a private table with a public table.

        Args:
            public_table: The public table to join with. This can be the name of
                a public table added to the session, or a Spark DataFrame.
            join_columns: The columns to join on. If None, joins on all common
                columns.
            how: The join type. Must be 'inner' or 'left'.

        Returns:
            A new :class:`QueryBuilder` object with the `join_public`
            transformation applied.
        """
        return self._from_query_expr(
            JoinPublic(
                self._query_expr,
                public_table,
                tuple(join_columns) if join_columns is not None else None,
                how,
            )
        )

    @typechecked
    def replace_null_and_nan(
        self,
        replace_with: Optional[
            Mapping[str, Union[int, float, str, datetime.date, datetime.datetime]]
        ] = None,
    ) -> "QueryBuilder":
        """Replaces null and NaN values in the DataFrame.

        Args:
            replace_with: A mapping from column name to replacement value. If no
                columns are specified, replaces all null and NaN values in
                numeric columns with 0, and all null values in string columns
                with "". Replacement values cannot be NaN or infinity.

        Returns:
            A new :class:`QueryBuilder` object with the `replace_null_and_nan`
            transformation applied.
        """
        return self._from_query_expr(
            ReplaceNullAndNan(
                self._query_expr,
                FrozenDict.from_dict(replace_with) if replace_with is not None else FrozenDict(),
            )
        )

    @typechecked
    def replace_infinity(
        self, replace_with: Optional[Mapping[str, Tuple[float, float]]] = None
    ) -> "QueryBuilder":
        """Replaces infinite values in the DataFrame.

        Args:
            replace_with: A mapping from column name to (lower bound, upper
                bound) tuple. Values outside these bounds (including infinity)
                are clamped to the corresponding bound. If no columns are
                specified, replaces all infinite values in numeric columns with
                0. Replacement bounds cannot be NaN or infinity.

        Returns:
            A new :class:`QueryBuilder` object with the `replace_infinity`
            transformation applied.
        """
        return self._from_query_expr(
            ReplaceInfinity(
                self._query_expr,
                FrozenDict.from_dict(replace_with) if replace_with is not None else FrozenDict(),
            )
        )

    @typechecked
    def drop_null_and_nan(self, columns: Optional[Iterable[str]] = None) -> "QueryBuilder":
        """Drops rows from the DataFrame containing null or NaN values.

        Args:
            columns: The subset of columns for which null/NaN values are used for
                filtering. If no columns are specified, uses all columns.

        Returns:
            A new :class:`QueryBuilder` object with the `drop_null_and_nan`
            transformation applied.
        """
        return self._from_query_expr(
            DropNullAndNan(
                self._query_expr, tuple(columns) if columns is not None else tuple()
            )
        )

    @typechecked
    def drop_infinity(self, columns: Optional[Iterable[str]] = None) -> "QueryBuilder":
        """Drops rows from the DataFrame containing infinite values.

        Args:
            columns: The subset of columns for which infinite values are used for
                filtering. If no columns are specified, uses all columns.

        Returns:
            A new :class:`QueryBuilder` object with the `drop_infinity`
            transformation applied.
        """
        return self._from_query_expr(
            DropInfinity(
                self._query_expr, tuple(columns) if columns is not None else tuple()
            )
        )

    @typechecked
    def enforce(self, constraint: Constraint) -> "QueryBuilder":
        """Enforces a privacy constraint on the DataFrame.

        Args:
            constraint: The constraint to enforce.

        Returns:
            A new :class:`QueryBuilder` object with the `enforce` transformation
            applied.
        """
        return self._from_query_expr(
            EnforceConstraint(self._query_expr, constraint, FrozenDict.from_dict({}))
        )

    @typechecked
    def bin_column(
        self, column: str, spec: Union[BinningSpec, Sequence[Any]], name: Optional[str] = None
    ) -> "QueryBuilder":
        """Replaces a numeric column with a string column of its corresponding
        bins.

        Args:
            column: The name of the column to bin.
            spec: The binning specification to use. Can either be a
                :class:`BinningSpec` object or a list of bin edges.
            name: The name of the new binned column. If no name is provided,
                the new column will be named `f"{column}_binned"`.

        Returns:
            A new :class:`QueryBuilder` object with the `bin_column`
            transformation applied.
        """
        if name is None:
            name = f"{column}_binned"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        bin_spec = spec if isinstance(spec, BinningSpec) else BinningSpec(spec)

        def bin_function(row: Dict[str, Any]) -> Dict[str, Any]:
            return {name: bin_spec(row.get(column))}

        new_column_types = {name: bin_spec.column_descriptor}
        return self.map(bin_function, new_column_types, augment=True)

    @typechecked
    def get_groups(self, columns: Optional[Iterable[str]] = None) -> Query:
        """Gets the groups in the DataFrame.

        Args:
            columns: The columns to group on. If no columns are specified, uses
                all columns except the privacy ID column if the table is
                identified by a privacy ID.

        Returns:
            A :class:`Query` object that represents the groups.
        """
        return Query(
            GetGroups(
                self._query_expr, tuple(columns) if columns is not None else None
            )
        )

    @typechecked
    def get_bounds(
        self, column: str, lower_bound_column: str, upper_bound_column: str
    ) -> Query:
        """Gets the lower and upper bounds of a column's values after per-ID clamping.

        The lower and upper bound are computed on a per-ID basis. For each privacy ID,
        the minimum and maximum values of the specified `measure_column` are computed.
        The lower bound column then contains the per-ID minimums, and the upper bound
        column contains the per-ID maximums.

        Args:
            column: The column for which to compute the per-ID bounds.
            lower_bound_column: The name of the column that contains the per-ID
                lower bounds.
            upper_bound_column: The name of the column that contains the per-ID
                upper bounds.

        Returns:
            A :class:`Query` object that represents the lower and upper bounds.
        """
        return Query(
            GetBounds(self._query_expr, KeySet.from_dict({}), column, lower_bound_column, upper_bound_column)  # type: ignore
        )

    @typechecked
    def histogram(
        self, column: str, bin_edges: Union[BinningSpec, Sequence[Any]], name: Optional[str] = None
    ) -> Query:
        """Computes a histogram of a column.

        This is a convenience method that is equivalent to:

        .. code-block:: python

            query.bin_column(column, bin_edges, name=name).count()

        Args:
            column: The name of the column to bin.
            bin_edges: The binning specification to use. Can either be a
                :class:`BinningSpec` object or a list of bin edges.
            name: The name of the new binned column. If no name is provided,
                the new column will be named `f"{column}_binned"`.

        Returns:
            A :class:`Query` object that represents the histogram.
        """
        return self.bin_column(column, bin_edges, name=name).count()

    @typechecked
    def count(self, name: Optional[str] = None, mechanism: CountMechanism = "GEOMETRIC") -> Query:
        """Computes the count of records in the DataFrame.

        Args:
            name: The name of the column to store the count. If no name is
                provided, the column will be named "count".
            mechanism: The noise mechanism to use for counting.

        Returns:
            A :class:`Query` object that represents the count.
        """
        return Query(
            GroupByCount(
                self._query_expr, KeySet.from_dict({}), name or "count", mechanism
            )
        )

    @typechecked
    def count_distinct(
        self,
        columns: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        cols: Optional[Iterable[str]] = None,  # TODO(#3564): Remove `cols` argument
    ) -> Query:
        """Computes the distinct count of records.

        Args:
            columns: The columns on which to count distinct values. If None,
                counts distinct values over all columns.
            name: The name of the column to store the count. If no name is
                provided, the column will be named "count_distinct(C1, C2, ...)"
                if `columns` are provided, or "count_distinct" otherwise.
            cols: Deprecated. Use `columns` instead.

        Returns:
            A :class:`Query` object that represents the distinct count.
        """
        if cols is not None:
            import warnings

            warnings.warn(
                "`cols` argument is deprecated, use `columns` instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if columns is not None:
                raise ValueError("cannot provide both `cols` and `columns` arguments")
            columns = cols

        if name is None:
            name = (
                f"count_distinct({', '.join(columns)})"
                if columns is not None
                else "count_distinct"
            )
        if not name:
            raise ValueError("Output column name cannot be an empty string")

        return Query(
            GroupByCountDistinct(
                self._query_expr,
                KeySet.from_dict({}),
                tuple(columns) if columns is not None else None,
                name,
            )
        )

    @typechecked
    def quantile(
        self,
        column: str,
        quantile: float,
        low: float,
        high: float,
        name: Optional[str] = None,
    ) -> Query:
        """Computes a quantile of a specified column, with values clamped to a
        specified range.

        Args:
            column: The column for which to compute the quantile.
            quantile: The quantile to compute. Must be between 0 and 1,
                inclusive.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the quantile. If no name is
                provided, the column will be named "f{column}_quantile({quantile})".

        Returns:
            A :class:`Query` object that represents the quantile.
        """
        if name is None:
            name = f"{column}_quantile({quantile})"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, KeySet.from_dict({}), column, quantile, low, high, name
            )
        )

    @typechecked
    def min(self, column: str, low: float, high: float, name: Optional[str] = None) -> Query:
        """Computes the minimum of a specified column, with values clamped to a
        specified range.

        Args:
            column: The column for which to compute the minimum.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the minimum. If no name is
                provided, the column will be named "f{column}_min".

        Returns:
            A :class:`Query` object that represents the minimum.
        """
        if name is None:
            name = f"{column}_min"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, KeySet.from_dict({}), column, 0.0, low, high, name
            )
        )

    @typechecked
    def max(self, column: str, low: float, high: float, name: Optional[str] = None) -> Query:
        """Computes the maximum of a specified column, with values clamped to a
        specified range.

        Args:
            column: The column for which to compute the maximum.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the maximum. If no name is
                provided, the column will be named "f{column}_max".

        Returns:
            A :class:`Query` object that represents the maximum.
        """
        if name is None:
            name = f"{column}_max"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, KeySet.from_dict({}), column, 1.0, low, high, name
            )
        )

    @typechecked
    def median(self, column: str, low: float, high: float, name: Optional[str] = None) -> Query:
        """Computes the median of a specified column, with values clamped to a
        specified range.

        Args:
            column: The column for which to compute the median.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the median. If no name is
                provided, the column will be named "f{column}_median".

        Returns:
            A :class:`Query` object that represents the median.
        """
        if name is None:
            name = f"{column}_median"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, KeySet.from_dict({}), column, 0.5, low, high, name
            )
        )

    @typechecked
    def sum(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: SumMechanism = "LAPLACE",
    ) -> Query:
        """Computes a sum of a specified column, with values clamped to a
        specified range.

        Args:
            column: The column for which to compute the sum.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the sum. If no name is
                provided, the column will be named "f{column}_sum".
            mechanism: The noise mechanism to use for summing.

        Returns:
            A :class:`Query` object that represents the sum.
        """
        if name is None:
            name = f"{column}_sum"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedSum(
                self._query_expr, KeySet.from_dict({}), column, low, high, name, mechanism
            )
        )

    @typechecked
    def average(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: AverageMechanism = "LAPLACE",
    ) -> Query:
        """Computes an average of a specified column, with values clamped to a
        specified range.

        Args:
            column: The column for which to compute the average.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the average. If no name is
                provided, the column will be named "f{column}_average".
            mechanism: The noise mechanism to use for computing the average.

        Returns:
            A :class:`Query` object that represents the average.
        """
        if name is None:
            name = f"{column}_average"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedAverage(
                self._query_expr, KeySet.from_dict({}), column, low, high, name, mechanism
            )
        )

    @typechecked
    def variance(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: VarianceMechanism = "LAPLACE",
    ) -> Query:
        """Computes a variance of a specified column, with values clamped to a
        specified range.

        Args:
            column: The column for which to compute the variance.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the variance. If no name is
                provided, the column will be named "f{column}_variance".
            mechanism: The noise mechanism to use for computing the variance.

        Returns:
            A :class:`Query` object that represents the variance.
        """
        if name is None:
            name = f"{column}_variance"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedVariance(
                self._query_expr, KeySet.from_dict({}), column, low, high, name, mechanism
            )
        )

    @typechecked
    def stdev(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: StdevMechanism = "LAPLACE",
    ) -> Query:
        """Computes a standard deviation of a specified column, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the standard deviation.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the standard deviation. If no
                name is provided, the column will be named "f{column}_stdev".
            mechanism: The noise mechanism to use for computing the standard
                deviation.

        Returns:
            A :class:`Query` object that represents the standard deviation.
        """
        if name is None:
            name = f"{column}_stdev"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedSTDEV(
                self._query_expr, KeySet.from_dict({}), column, low, high, name, mechanism
            )
        )

    @typechecked
    def groupby(self, keys: Union[KeySet, Iterable[str]]) -> "GroupedQueryBuilder":
        """Groups the DataFrame by the specified keys.

        Args:
            keys: The columns by which to group the DataFrame. This can be either
                a :class:`KeySet` object or an iterable of strings representing
                column names.

        Returns:
            A new :class:`GroupedQueryBuilder` object, on which aggregation
            methods can be called.
        """
        if isinstance(keys, KeySet):
            return GroupedQueryBuilder(self._query_expr, keys)
        if isinstance(keys, Iterable):
            return GroupedQueryBuilder(
                self._query_expr, KeySet.from_dict({k: [] for k in keys})
            )
        raise TypeError(
            f"Expected keys of type KeySet or Iterable[str], got {type(keys)}."
        )


class GroupedQueryBuilder(object):
    """Helper class for creating differentially private grouped queries.

    A :class:`GroupedQueryBuilder` object allows you to build a grouped query
    by chaining transformation and aggregation methods. Each transformation
    method returns a new :class:`GroupedQueryBuilder` object, allowing for a
    fluent API. The terminal method in the chain (an aggregation) returns a
    :class:`Query` object.

    Args:
        query_expr: The query expression representing the grouped DataFrame.
        groupby_keys: The grouping keys.
    """

    @typechecked
    def __init__(self, query_expr: QueryExpr, groupby_keys: KeySet):
        self._query_expr = query_expr
        self._groupby_keys = groupby_keys

    @typechecked
    def _from_query_expr(
        self, query_expr: QueryExpr, groupby_keys: KeySet
    ) -> "GroupedQueryBuilder":
        """Creates a :class:`GroupedQueryBuilder` from a :class:`QueryExpr`."""
        builder = GroupedQueryBuilder.__new__(GroupedQueryBuilder)
        builder._query_expr = query_expr
        builder._groupby_keys = groupby_keys
        return builder

    @property
    def source_id(self) -> str:
        """The source ID that this GroupedQueryBuilder refers to."""
        current_expr = self._query_expr
        while not isinstance(current_expr, PrivateSource):
            current_expr = getattr(current_expr, "child")
        return current_expr.source_id

    @typechecked
    def clone(self) -> "GroupedQueryBuilder":
        """Returns a deep copy of the :class:`GroupedQueryBuilder`.

        Returns:
            A deep copy of the :class:`GroupedQueryBuilder`.
        """
        return self._from_query_expr(self._query_expr, self._groupby_keys)

    @typechecked
    def rename(self, column_mapper: Mapping[str, str]) -> "GroupedQueryBuilder":
        """Renames one or more columns in the DataFrame.

        Args:
            column_mapper: A dictionary mapping old column names to new column
                names.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `rename`
            transformation applied.
        """
        # If any of the grouping keys are renamed, create a new KeySet
        new_groupby_keys = self._groupby_keys
        if any(c in self._groupby_keys for c in column_mapper):
            updated_keys_dict: Dict[str, List[Any]] = {}
            for k, v in self._groupby_keys.key_to_values.items():
                if k in column_mapper:
                    updated_keys_dict[column_mapper[k]] = v
                else:
                    updated_keys_dict[k] = v
            new_groupby_keys = KeySet.from_dict(updated_keys_dict)

        return self._from_query_expr(
            Rename(self._query_expr, FrozenDict.from_dict(column_mapper)),
            new_groupby_keys,
        )

    @typechecked
    def filter(self, condition: str) -> "GroupedQueryBuilder":
        """Filters the DataFrame using a SQL expression.

        Args:
            condition: A SQL expression (e.g. "A < B AND C > 0") representing the
                filter condition.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `filter`
            transformation applied.
        """
        return self._from_query_expr(Filter(self._query_expr, condition), self._groupby_keys)

    @typechecked
    def select(self, columns: Iterable[str]) -> "GroupedQueryBuilder":
        """Selects a subset of columns from the DataFrame.

        Args:
            columns: The names of columns to select.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `select`
            transformation applied.
        """
        # If any of the grouping keys are dropped, update the KeySet
        new_groupby_keys = self._groupby_keys.intersection(columns)

        return self._from_query_expr(
            Select(self._query_expr, tuple(columns)), new_groupby_keys
        )

    @typechecked
    def map(
        self,
        f: Callable[[Dict[str, Any]], Dict[str, Any]],
        new_column_types: Mapping[str, Union[str, ColumnDescriptor]],
        augment: bool = False,
    ) -> "GroupedQueryBuilder":
        """Applies a Python function to each row of the DataFrame.

        Args:
            f: A Python function that takes a row (as a dictionary from column
                name to value) and returns a dictionary of new column name to new
                value.
            new_column_types: A mapping from new column name to its type. For
                convenience, string representations of
                :class:`~tmlt.analytics.ColumnType` (e.g. "INTEGER", "VARCHAR") are
                accepted.
            augment: If True, the original columns are kept and the new columns
                returned by `f` are added to the DataFrame. If False, only the new
                columns returned by `f` are kept.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `map`
            transformation applied.
        """
        return self._from_query_expr(
            Map(self._query_expr, f, Schema(new_column_types), augment),
            self._groupby_keys,
        )

    @typechecked
    def flat_map(
        self,
        f: Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]],
        new_column_types: Mapping[str, Union[str, ColumnDescriptor]],
        augment: bool = False,
        grouping: bool = False,
        max_rows: int = 1,
    ) -> "GroupedQueryBuilder":
        """Applies a Python function to each row of the DataFrame that may return
        multiple rows.

        Args:
            f: A Python function that takes a row (as a dictionary from column
                name to value) and returns an iterable of dictionaries from new
                column name to new value.
            new_column_types: A mapping from new column name to its type. For
                convenience, string representations of
                :class:`~tmlt.analytics.ColumnType` (e.g. "INTEGER", "VARCHAR") are
                accepted.
            augment: If True, the original columns are kept and the new columns
                returned by `f` are added to the DataFrame. If False, only the new
                columns returned by `f` are kept.
            grouping: If True, the new columns returned by `f` form a new
                grouping column. Note that if `grouping` is True, `f` must
                return exactly one new column.
            max_rows: The maximum number of rows that can be returned by `f` for
                any given input row. This value directly impacts the output
                stability of the transformation in the DP analysis.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `flat_map`
            transformation applied.
        """
        schema_new_columns = Schema(new_column_types)
        if grouping:
            if len(schema_new_columns) != 1:
                raise ValueError(
                    f"If `grouping` is True, `new_column_types` must contain "
                    f"exactly one column, but found {len(schema_new_columns)}."
                )
            schema_new_columns = Schema(
                new_column_types, grouping_column=list(new_column_types.keys())[0]
            )
        return self._from_query_expr(
            FlatMap(self._query_expr, f, schema_new_columns, augment, max_rows),
            self._groupby_keys,
        )

    @typechecked
    def join_private(
        self,
        right_operand: Union[str, QueryBuilder],
        truncation_strategy_left: TruncationStrategy.Type,
        truncation_strategy_right: TruncationStrategy.Type,
        join_columns: Optional[Iterable[str]] = None,
        how: str = "inner",
    ) -> "GroupedQueryBuilder":
        """Joins two private tables.

        Args:
            right_operand: The right-hand side of the join. This can be the
                name of a private table or a :class:`QueryBuilder` object.
            truncation_strategy_left: The truncation strategy to apply to the
                left-hand side of the join.
            truncation_strategy_right: The truncation strategy to apply to the
                right-hand side of the join.
            join_columns: The columns to join on. If None, joins on all common
                columns.
            how: The join type. Must be 'inner' or 'left'.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `join_private`
            transformation applied.
        """
        right_operand_expr: QueryExpr
        if isinstance(right_operand, str):
            right_operand_expr = PrivateSource(right_operand)
        elif isinstance(right_operand, QueryBuilder):
            right_operand_expr = right_operand._query_expr
        else:
            raise TypeError(
                f"right_operand must be of type str or QueryBuilder, but received"
                f" {type(right_operand)}"
            )
        return self._from_query_expr(
            JoinPrivate(
                self._query_expr,
                right_operand_expr,
                truncation_strategy_left,
                truncation_strategy_right,
                tuple(join_columns) if join_columns is not None else None,
                how,
            ),
            self._groupby_keys,
        )

    @typechecked
    def join_public(
        self,
        public_table: Any,
        join_columns: Optional[Iterable[str]] = None,
        how: str = "inner",
    ) -> "GroupedQueryBuilder":
        """Joins a private table with a public table.

        Args:
            public_table: The public table to join with. This can be the name of
                a public table added to the session, or a Spark DataFrame.
            join_columns: The columns to join on. If None, joins on all common
                columns.
            how: The join type. Must be 'inner' or 'left'.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `join_public`
            transformation applied.
        """
        return self._from_query_expr(
            JoinPublic(
                self._query_expr,
                public_table,
                tuple(join_columns) if join_columns is not None else None,
                how,
            ),
            self._groupby_keys,
        )

    @typechecked
    def replace_null_and_nan(
        self,
        replace_with: Optional[
            Mapping[str, Union[int, float, str, datetime.date, datetime.datetime]]
        ] = None,
    ) -> "GroupedQueryBuilder":
        """Replaces null and NaN values in the DataFrame.

        Args:
            replace_with: A mapping from column name to replacement value. If no
                columns are specified, replaces all null and NaN values in
                numeric columns with 0, and all null values in string columns
                with "". Replacement values cannot be NaN or infinity.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the
            `replace_null_and_nan` transformation applied.
        """
        return self._from_query_expr(
            ReplaceNullAndNan(
                self._query_expr,
                FrozenDict.from_dict(replace_with) if replace_with is not None else FrozenDict(),
            ),
            self._groupby_keys,
        )

    @typechecked
    def replace_infinity(
        self, replace_with: Optional[Mapping[str, Tuple[float, float]]] = None
    ) -> "GroupedQueryBuilder":
        """Replaces infinite values in the DataFrame.

        Args:
            replace_with: A mapping from column name to (lower bound, upper
                bound) tuple. Values outside these bounds (including infinity)
                are clamped to the corresponding bound. If no columns are
                specified, replaces all infinite values in numeric columns with
                0. Replacement bounds cannot be NaN or infinity.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the
            `replace_infinity` transformation applied.
        """
        return self._from_query_expr(
            ReplaceInfinity(
                self._query_expr,
                FrozenDict.from_dict(replace_with) if replace_with is not None else FrozenDict(),
            ),
            self._groupby_keys,
        )

    @typechecked
    def drop_null_and_nan(self, columns: Optional[Iterable[str]] = None) -> "GroupedQueryBuilder":
        """Drops rows from the DataFrame containing null or NaN values.

        Args:
            columns: The subset of columns for which null/NaN values are used for
                filtering. If no columns are specified, uses all columns.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the
            `drop_null_and_nan` transformation applied.
        """
        return self._from_query_expr(
            DropNullAndNan(
                self._query_expr, tuple(columns) if columns is not None else tuple()
            ),
            self._groupby_keys,
        )

    @typechecked
    def drop_infinity(self, columns: Optional[Iterable[str]] = None) -> "GroupedQueryBuilder":
        """Drops rows from the DataFrame containing infinite values.

        Args:
            columns: The subset of columns for which infinite values are used for
                filtering. If no columns are specified, uses all columns.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the
            `drop_infinity` transformation applied.
        """
        return self._from_query_expr(
            DropInfinity(
                self._query_expr, tuple(columns) if columns is not None else tuple()
            ),
            self._groupby_keys,
        )

    @typechecked
    def enforce(self, constraint: Constraint) -> "GroupedQueryBuilder":
        """Enforces a privacy constraint on the DataFrame.

        Args:
            constraint: The constraint to enforce.

        Returns:
            A new :class:`GroupedQueryBuilder` object with the `enforce`
            transformation applied.
        """
        return self._from_query_expr(
            EnforceConstraint(self._query_expr, constraint, FrozenDict.from_dict({})),
            self._groupby_keys,
        )

    @typechecked
    def bin_column(
        self, column: str, spec: Union[BinningSpec, Sequence[Any]], name: Optional[str] = None
    ) -> "GroupedQueryBuilder":
        """Replaces a numeric column with a string column of its corresponding
        bins.

        Args:
            column: The name of the column to bin.
            spec: The binning specification to use. Can either be a
                :class:`BinningSpec` object or a list of bin edges.
            name: The name of the new binned column. If no name is provided,
                the new column will be named `f"{column}_binned"`.

        Returns:
            A new :class:`QueryBuilder` object with the `bin_column`
            transformation applied.
        """
        if name is None:
            name = f"{column}_binned"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        bin_spec = spec if isinstance(spec, BinningSpec) else BinningSpec(spec)

        def bin_function(row: Dict[str, Any]) -> Dict[str, Any]:
            return {name: bin_spec(row.get(column))}

        new_column_types = {name: bin_spec.column_descriptor}
        return self._from_query_expr(
            Map(self._query_expr, bin_function, Schema(new_column_types), augment=True),
            self._groupby_keys,
        )

    @typechecked
    def count(self, name: Optional[str] = None, mechanism: CountMechanism = "GEOMETRIC") -> Query:
        """Computes the count of records for each group in the DataFrame.

        Args:
            name: The name of the column to store the count. If no name is
                provided, the column will be named "count".
            mechanism: The noise mechanism to use for counting.

        Returns:
            A :class:`Query` object that represents the grouped count.
        """
        return Query(
            GroupByCount(self._query_expr, self._groupby_keys, name or "count", mechanism)
        )

    @typechecked
    def count_distinct(
        self,
        columns: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        cols: Optional[Iterable[str]] = None,  # TODO(#3564): Remove `cols` argument
    ) -> Query:
        """Computes the distinct count of records for each group.

        Args:
            columns: The columns on which to count distinct values. If None,
                counts distinct values over all columns.
            name: The name of the column to store the count. If no name is
                provided, the column will be named "count_distinct(C1, C2, ...)"
                if `columns` are provided, or "count_distinct" otherwise.
            cols: Deprecated. Use `columns` instead.

        Returns:
            A :class:`Query` object that represents the grouped distinct count.
        """
        if cols is not None:
            import warnings

            warnings.warn(
                "`cols` argument is deprecated, use `columns` instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if columns is not None:
                raise ValueError("cannot provide both `cols` and `columns` arguments")
            columns = cols

        if name is None:
            name = (
                f"count_distinct({', '.join(columns)})"
                if columns is not None
                else "count_distinct"
            )
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByCountDistinct(
                self._query_expr,
                self._groupby_keys,
                tuple(columns) if columns is not None else None,
                name,
            )
        )

    @typechecked
    def quantile(
        self,
        column: str,
        quantile: float,
        low: float,
        high: float,
        name: Optional[str] = None,
    ) -> Query:
        """Computes a quantile of a specified column for each group, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the quantile.
            quantile: The quantile to compute. Must be between 0 and 1,
                inclusive.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the quantile. If no name is
                provided, the column will be named "f{column}_quantile({quantile})".

        Returns:
            A :class:`Query` object that represents the grouped quantile.
        """
        if name is None:
            name = f"{column}_quantile({quantile})"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, self._groupby_keys, column, quantile, low, high, name
            )
        )

    @typechecked
    def min(self, column: str, low: float, high: float, name: Optional[str] = None) -> Query:
        """Computes the minimum of a specified column for each group, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the minimum.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the minimum. If no name is
                provided, the column will be named "f{column}_min".

        Returns:
            A :class:`Query` object that represents the grouped minimum.
        """
        if name is None:
            name = f"{column}_min"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, self._groupby_keys, column, 0.0, low, high, name
            )
        )

    @typechecked
    def max(self, column: str, low: float, high: float, name: Optional[str] = None) -> Query:
        """Computes the maximum of a specified column for each group, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the maximum.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the maximum. If no name is
                provided, the column will be named "f{column}_max".

        Returns:
            A :class:`Query` object that represents the grouped maximum.
        """
        if name is None:
            name = f"{column}_max"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, self._groupby_keys, column, 1.0, low, high, name
            )
        )

    @typechecked
    def median(self, column: str, low: float, high: float, name: Optional[str] = None) -> Query:
        """Computes the median of a specified column for each group, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the median.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the median. If no name is
                provided, the column will be named "f{column}_median".

        Returns:
            A :class:`Query` object that represents the grouped median.
        """
        if name is None:
            name = f"{column}_median"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByQuantile(
                self._query_expr, self._groupby_keys, column, 0.5, low, high, name
            )
        )

    @typechecked
    def sum(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: SumMechanism = "LAPLACE",
    ) -> Query:
        """Computes a sum of a specified column for each group, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the sum.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the sum. If no name is
                provided, the column will be named "f{column}_sum".
            mechanism: The noise mechanism to use for summing.

        Returns:
            A :class:`Query` object that represents the grouped sum.
        """
        if name is None:
            name = f"{column}_sum"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedSum(
                self._query_expr, self._groupby_keys, column, low, high, name, mechanism
            )
        )

    @typechecked
    def average(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: AverageMechanism = "LAPLACE",
    ) -> Query:
        """Computes an average of a specified column for each group, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the average.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the average. If no name is
                provided, the column will be named "f{column}_average".
            mechanism: The noise mechanism to use for computing the average.

        Returns:
            A :class:`Query` object that represents the grouped average.
        """
        if name is None:
            name = f"{column}_average"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedAverage(
                self._query_expr, self._groupby_keys, column, low, high, name, mechanism
            )
        )

    @typechecked
    def variance(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: VarianceMechanism = "LAPLACE",
    ) -> Query:
        """Computes a variance of a specified column for each group, with values
        clamped to a specified range.

        Args:
            column: The column for which to compute the variance.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the variance. If no name is
                provided, the column will be named "f{column}_variance".
            mechanism: The noise mechanism to use for computing the variance.

        Returns:
            A :class:`Query` object that represents the grouped variance.
        """
        if name is None:
            name = f"{column}_variance"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedVariance(
                self._query_expr, self._groupby_keys, column, low, high, name, mechanism
            )
        )

    @typechecked
    def stdev(
        self,
        column: str,
        low: float,
        high: float,
        name: Optional[str] = None,
        mechanism: StdevMechanism = "LAPLACE",
    ) -> Query:
        """Computes a standard deviation of a specified column for each group,
        with values clamped to a specified range.

        Args:
            column: The column for which to compute the standard deviation.
            low: The lower bound for values in `column`. Values in `column`
                less than `low` are clamped to `low`. Must be less than `high`.
            high: The upper bound for values in `column`. Values in `column`
                greater than `high` are clamped to `high`. Must be greater than
                `low`.
            name: The name of the column to store the standard deviation. If no
                name is provided, the column will be named "f{column}_stdev".
            mechanism: The noise mechanism to use for computing the standard
                deviation.

        Returns:
            A :class:`Query` object that represents the grouped standard
            deviation.
        """
        if name is None:
            name = f"{column}_stdev"
        if not name:
            raise ValueError("Output column name cannot be an empty string")
        return Query(
            GroupByBoundedSTDEV(
                self._query_expr, self._groupby_keys, column, low, high, name, mechanism
            )
        )

    @typechecked
    def suppress(self, threshold: Union[int, float]) -> Query:
        """Suppresses aggregated values below a threshold.

        This method should be called on a count aggregate (e.g., the output
        of :meth:`count`).

        Args:
            threshold: The threshold below which aggregated values are suppressed.

        Returns:
            A :class:`Query` object that represents the suppressed aggregates.
        """
        query_expr = self._query_expr
        check_type("query_expr", query_expr, GroupByCount)
        return Query(
            SuppressAggregates(
                cast(GroupByCount, query_expr),
                query_expr.output_column,
                threshold,
            )
        )