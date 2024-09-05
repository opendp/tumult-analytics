"""Building blocks of the Tumult Analytics query language. Not for direct use.

Defines the :class:`QueryExpr` class, which represents expressions in the
Tumult Analytics query language. QueryExpr and its subclasses should not be
directly constructed or deconstructed by most users; interfaces such as
:class:`tmlt.analytics.query_builder.QueryBuilder` to create them and
:class:`tmlt.analytics.session.Session` to consume them provide more
user-friendly features.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pyspark.sql import DataFrame
from typeguard import check_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import FrozenDict, Schema
from tmlt.analytics.config import config
from tmlt.analytics.constraints import Constraint
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.truncation_strategy import TruncationStrategy

Row = Dict[str, Any]
"""Type alias for dictionary with string keys."""


class CountMechanism(Enum):
    """Possible mechanisms for the count() aggregation.

    Currently, the
    :meth:`~tmlt.analytics.query_builder.GroupedQueryBuilder.count` aggregation
    uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Double-sided geometric noise is used."""
    GAUSSIAN = auto()
    """The discrete Gaussian mechanism is used. Not compatible with pure DP."""


class CountDistinctMechanism(Enum):
    """Enumerating the possible mechanisms used for the count_distinct aggregation.

    Currently, the
    :meth:`~tmlt.analytics.query_builder.GroupedQueryBuilder.count_distinct`
    aggregation uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Double-sided geometric noise is used."""
    GAUSSIAN = auto()
    """The discrete Gaussian mechanism is used. Not compatible with pure DP."""


class SumMechanism(Enum):
    """Possible mechanisms for the sum() aggregation.

    Currently, the
    :meth:`~.tmlt.analytics.query_builder.GroupedQueryBuilder.sum`
    aggregation uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Laplace and/or double-sided geometric noise is used, depending on the
    column type.
    """
    GAUSSIAN = auto()
    """Discrete and/or continuous Gaussian noise is used, depending on the column type.
    Not compatible with pure DP.
    """


class AverageMechanism(Enum):
    """Possible mechanisms for the average() aggregation.

    Currently, the
    :meth:`~tmlt.analytics.query_builder.GroupedQueryBuilder.average`
    aggregation uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Laplace and/or double-sided geometric noise is used, depending on the
    column type.
    """
    GAUSSIAN = auto()
    """Discrete and/or continuous Gaussian noise is used, depending on the column type.
    Not compatible with pure DP.
    """


class VarianceMechanism(Enum):
    """Possible mechanisms for the variance() aggregation.

    Currently, the
    :meth:`~tmlt.analytics.query_builder.GroupedQueryBuilder.variance`
    aggregation uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Laplace and/or double-sided geometric noise is used, depending on the
    column type.
    """
    GAUSSIAN = auto()
    """Discrete and/or continuous Gaussian noise is used, depending on the column type.
    Not compatible with pure DP.
    """


class StdevMechanism(Enum):
    """Possible mechanisms for the stdev() aggregation.

    Currently, the
    :meth:`~tmlt.analytics.query_builder.GroupedQueryBuilder.stdev`
    aggregation uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Laplace and/or double-sided geometric noise is used, depending on the
    column type.
    """
    GAUSSIAN = auto()
    """Discrete and/or continuous Gaussian noise is used, depending on the column type.
    Not compatible with pure DP.
    """


class QueryExpr(ABC):
    """A query expression, base class for relational operators.

    In most cases, QueryExpr should not be manipulated directly, but rather
    created using :class:`tmlt.analytics.query_builder.QueryBuilder` and then
    consumed by :class:`tmlt.analytics.session.Session`. While they can be
    created and modified directly, this is an advanced usage and is not
    recommended for typical users.

    QueryExpr are organized in a tree, where each node is an operator which
    returns a relation.
    """

    @abstractmethod
    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Dispatch methods on a visitor based on the QueryExpr type."""
        raise NotImplementedError()


@dataclass(frozen=True)
class PrivateSource(QueryExpr):
    """Loads the private source."""

    source_id: str
    """The ID for the private source to load."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.source_id, str)

        if not self.source_id.isidentifier():
            raise ValueError(
                "The string passed as source_id must be a valid Python identifier: it"
                " can only contain alphanumeric letters (a-z) and (0-9), or underscores"
                " (_), and it cannot start with a number, or contain any spaces."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_private_source(self)


@dataclass(frozen=True)
class GetGroups(QueryExpr):
    """Returns groups based on the geometric partition selection for these columns."""

    child: QueryExpr
    """The QueryExpr to get groups for."""

    columns: Optional[Tuple[str, ...]] = None
    """The columns used for geometric partition selection.

    If empty or none are provided, will use all of the columns in the table
    for partition selection.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Optional[Tuple[str, ...]])

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_get_groups(self)


@dataclass(frozen=True)
class GetBounds(QueryExpr):
    """Returns approximate upper and lower bounds of a column."""

    child: QueryExpr
    """The QueryExpr to get groups for."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to get bounds of."""
    lower_bound_column: str
    """The name of the column to store the lower bound in."""
    upper_bound_column: str
    """The name of the column to store the upper bound in."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.lower_bound_column, str)
        check_type(self.upper_bound_column, str)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_get_bounds(self)


@dataclass(frozen=True)
class Rename(QueryExpr):
    """Returns the dataframe with columns renamed."""

    child: QueryExpr
    """The QueryExpr to apply Rename to."""
    column_mapper: FrozenDict

    """The mapping of old column names to new column names.

    This mapping can contain all column names or just a subset. If it
    contains a subset of columns, it will only rename those columns
    and keep the other column names the same.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.column_mapper, FrozenDict)
        check_type(dict(self.column_mapper), Dict[str, str])
        for k, v in self.column_mapper.items():
            if v == "":
                raise ValueError(
                    f'Cannot rename column {k} to "" (the empty string): columns named'
                    ' "" are not allowed'
                )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_rename(self)


@dataclass(frozen=True)
class Filter(QueryExpr):
    """Returns the subset of the rows that satisfy the condition."""

    child: QueryExpr
    """The QueryExpr to filter."""
    condition: str
    """A string of SQL expression specifying the filter to apply to the data.

    For example, the string "A > B" matches rows where column A is greater than
    column B.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.condition, str)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_filter(self)


@dataclass(frozen=True)
class Select(QueryExpr):
    """Returns a subset of the columns."""

    child: QueryExpr
    """The QueryExpr to apply the select on."""
    columns: Tuple[str, ...]
    """The columns to select."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Tuple[str, ...])
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(f"Column name appears more than once in {self.columns}")

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_select(self)


@dataclass(frozen=True)
class Map(QueryExpr):
    """Applies a map function to each row of a relation."""

    child: QueryExpr
    """The QueryExpr to apply the map on."""
    f: Callable[[Row], Row]
    """The map function."""
    schema_new_columns: Schema
    """The expected schema for new columns produced by ``f``."""
    augment: bool
    """Whether to keep the existing columns.

    If True, schema = old schema + schema_new_columns, otherwise only keeps the new
    columns (schema = schema_new_columns).
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.f, Callable[[Row], Row])
        check_type(self.schema_new_columns, Schema)
        check_type(self.augment, bool)
        if self.schema_new_columns.grouping_column is not None:
            raise ValueError("Map cannot be be used to create grouping columns")

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_map(self)

    def __eq__(self, other: object) -> bool:
        """Returns true iff self == other.

        This uses the bytecode of self.f and other.f to determine if the two
        functions are equal.
        """
        if not isinstance(other, Map):
            return False
        if self.f != other.f and self.f.__code__.co_code != other.f.__code__.co_code:
            return False
        return (
            self.schema_new_columns == other.schema_new_columns
            and self.augment == other.augment
            and self.child == other.child
        )


@dataclass(frozen=True)
class FlatMap(QueryExpr):
    """Applies a flat map function to each row of a relation."""

    child: QueryExpr
    """The QueryExpr to apply the flat map on."""
    f: Callable[[Row], List[Row]]
    """The flat map function."""
    schema_new_columns: Schema
    """The expected schema for new columns produced by ``f``.

    If the ``schema_new_columns`` has a ``grouping_column``, that means this FlatMap
    produces a column that must be grouped by eventually. It also must be the only
    column in the schema.
    """
    augment: bool
    """Whether to keep the existing columns.

    If True, schema = old schema + schema_new_columns, otherwise only keeps the new
    columns (schema = schema_new_columns)."""

    max_rows: Optional[int] = None
    """The enforced limit on number of rows from each f(row)."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.f, Callable[[Row], List[Row]])
        check_type(self.max_rows, Optional[int])
        check_type(self.schema_new_columns, Schema)
        check_type(self.augment, bool)
        if self.max_rows and self.max_rows < 0:
            raise ValueError(
                f"Limit on number of rows '{self.max_rows}' must be non-negative."
            )
        if (
            self.schema_new_columns.grouping_column
            and len(self.schema_new_columns) != 1
        ):
            raise ValueError(
                "schema_new_columns contains "
                f"{len(self.schema_new_columns)} "
                "columns, grouping flat map can only result in 1 new column"
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_flat_map(self)

    def __eq__(self, other: object) -> bool:
        """Returns true iff self == other.

        This uses the bytecode of self.f and other.f to determine if the two
        functions are equal.
        """
        if not isinstance(other, FlatMap):
            return False
        if self.f != other.f and self.f.__code__.co_code != other.f.__code__.co_code:
            return False
        return (
            self.max_rows == other.max_rows
            and self.schema_new_columns == other.schema_new_columns
            and self.augment == other.augment
            and self.child == other.child
        )


@dataclass(frozen=True)
class FlatMapByID(QueryExpr):
    """Applies a flat map function to each group of rows with a common ID."""

    child: QueryExpr
    """The QueryExpr to apply the flat map on."""
    f: Callable[[List[Row]], List[Row]]
    """The flat map function."""
    schema_new_columns: Schema
    """The expected schema for new columns produced by ``f``.

    ``schema_new_column`` must not have a grouping or ID column.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.f, Callable[[List[Row]], List[Row]])
        check_type(self.schema_new_columns, Schema)
        if self.schema_new_columns.grouping_column or self.schema_new_columns.id_column:
            raise AnalyticsInternalError(
                "FlatMapByID new column schema must not have a grouping or ID column."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_flat_map_by_id(self)

    def __eq__(self, other: object) -> bool:
        """Returns true iff self == other.

        This uses the bytecode of self.f and other.f to determine if the two
        functions are equal.
        """
        if not isinstance(other, FlatMapByID):
            return False
        if self.f != other.f and self.f.__code__.co_code != other.f.__code__.co_code:
            return False
        return (
            self.schema_new_columns == other.schema_new_columns
            and self.child == other.child
        )


@dataclass(frozen=True)
class JoinPrivate(QueryExpr):
    """Returns the join of two private tables.

    Before performing the join, each table is truncated based on the corresponding
    :class:`~tmlt.analytics.truncation_strategy.TruncationStrategy`.  For a more
    detailed overview of ``JoinPrivate``'s behavior, see
    :meth:`~tmlt.analytics.query_builder.QueryBuilder.join_private`.
    """

    child: QueryExpr
    """The QueryExpr to join with right operand."""
    right_operand_expr: QueryExpr
    """The QueryExpr for private source to join with."""
    truncation_strategy_left: Optional[TruncationStrategy.Type] = None
    """Truncation strategy to be used for the left table."""
    truncation_strategy_right: Optional[TruncationStrategy.Type] = None
    """Truncation strategy to be used for the right table."""
    join_columns: Optional[Tuple[str, ...]] = None
    """The columns used for joining the tables, or None to use all common columns."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.right_operand_expr, QueryExpr)
        check_type(
            self.truncation_strategy_left,
            Optional[TruncationStrategy.Type],
        )
        check_type(
            self.truncation_strategy_right,
            Optional[TruncationStrategy.Type],
        )
        check_type(self.join_columns, Optional[Tuple[str, ...]])

        if self.join_columns is not None:
            if len(self.join_columns) == 0:
                raise ValueError("Provided join columns must not be empty")
            if len(self.join_columns) != len(set(self.join_columns)):
                raise ValueError("Join columns must be distinct")

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_join_private(self)


@dataclass(frozen=True)
class JoinPublic(QueryExpr):
    """Returns the join of a private and public table."""

    child: QueryExpr
    """The QueryExpr to join with public_df."""
    public_table: Union[DataFrame, str]
    """A DataFrame or public source to join with."""
    join_columns: Optional[Tuple[str, ...]] = None
    """The columns used for joining the tables, or None to use all common columns."""
    how: str = "inner"
    """The type of join to perform. Must be either "inner" or "left"."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.public_table, Union[DataFrame, str])
        check_type(self.join_columns, Optional[Tuple[str, ...]])

        if self.join_columns is not None:
            if len(self.join_columns) == 0:
                raise ValueError("Provided join columns must not be empty")
            if len(self.join_columns) != len(set(self.join_columns)):
                raise ValueError("Join columns must be distinct")

        if isinstance(self.public_table, DataFrame):
            # because this is a frozen dataclass, we have to use object.__setattr__
            # instead of just using self.public_table = <new value>
            object.__setattr__(
                self, "public_table", coerce_spark_schema_or_fail(self.public_table)
            )
        if self.how not in ["inner", "left"]:
            raise ValueError(
                f"Invalid join type '{self.how}': must be 'inner' or 'left'"
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_join_public(self)

    def __eq__(self, other: object) -> bool:
        """Returns true iff self == other.

        For the purposes of this equality operation, two dataframes are equal
        if they contain the same data, in any order.

        Calling this on a JoinPublic that includes a very large dataframe
        could take a long time or consume a lot of resources, and is not
        recommended.
        """
        if not isinstance(other, JoinPublic):
            return False
        if isinstance(self.public_table, str):
            if self.public_table != other.public_table:
                return False
        else:
            # public_table is a dataframe
            if not isinstance(other.public_table, DataFrame):
                return False
            # Make sure both dataframes contain the same data, in any order
            # TODO(#2107): Fix typing once Pandas has working type stubs
            self_table = self.public_table.toPandas()
            other_table = other.public_table.toPandas()
            if sorted(self_table.columns) != sorted(  # type: ignore
                other_table.columns  # type: ignore
            ):
                return False
            if not self_table.empty and not other_table.empty:  # type: ignore
                sort_columns = list(self_table.columns)  # type: ignore
                self_table = (
                    self_table.set_index(sort_columns)  # type: ignore
                    .sort_index()
                    .reset_index()
                )
                other_table = (
                    other_table.set_index(sort_columns)  # type: ignore
                    .sort_index()
                    .reset_index()
                )
                if not self_table.equals(other_table):  # type: ignore
                    return False
        return (
            self.join_columns == other.join_columns
            and self.child == other.child
            and self.how == other.how
        )


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


@dataclass(frozen=True)
class ReplaceNullAndNan(QueryExpr):
    """Returns data with null and NaN expressions replaced by a default.

    .. warning::
        after a ``ReplaceNullAndNan`` query has been performed for a column,
        Tumult Analytics will raise an error if you use a
        :class:`~.tmlt.analytics.keyset.KeySet` for that column
        that contains null values.
    """

    child: QueryExpr
    """The QueryExpr to replace null/NaN values in."""

    replace_with: FrozenDict = FrozenDict.from_dict({})
    """New values to replace with, by column.

    If this dictionary is empty, *all* columns will be changed, with values
    replaced by a default value for each column's type (see the
    :class:`~.AnalyticsDefault` class variables).
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(
            self.replace_with,
            FrozenDict,
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_replace_null_and_nan(self)


@dataclass(frozen=True, init=False, eq=True)
class ReplaceInfinity(QueryExpr):
    """Returns data with +inf and -inf expressions replaced by defaults."""

    child: QueryExpr
    """The QueryExpr to replace +inf and -inf values in."""

    replace_with: FrozenDict = FrozenDict.from_dict({})
    """New values to replace with, by column. The first value for each column
    will be used to replace -infinity, and the second value will be used to
    replace +infinity.

    If this dictionary is empty, *all* columns of type DECIMAL will be changed,
    with infinite values replaced with a default value (see the
    :class:`~.AnalyticsDefault` class variables).
    """

    def __init__(
        self, child: QueryExpr, replace_with: FrozenDict = FrozenDict.from_dict({})
    ) -> None:
        """Checks arguments to constructor."""
        check_type(child, QueryExpr)
        check_type(replace_with, FrozenDict)
        check_type(dict(replace_with), Dict[str, Tuple[float, float]])

        # Ensure that the values in replace_with are tuples of floats
        updated_dict = {}
        for col, val in replace_with.items():
            updated_dict[col] = (float(val[0]), float(val[1]))

        # Subverting the frozen dataclass to update the replace_with attribute
        object.__setattr__(self, "replace_with", FrozenDict.from_dict(updated_dict))
        object.__setattr__(self, "child", child)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_replace_infinity(self)


@dataclass(frozen=True)
class DropNullAndNan(QueryExpr):
    """Returns data with rows that contain null or NaN value dropped.

    .. warning::
        After a ``DropNullAndNan`` query has been performed for a column,
        Tumult Analytics will raise an error if you use a
        :class:`~.tmlt.analytics.keyset.KeySet` for that column
        that contains null values.
    """

    child: QueryExpr
    """The QueryExpr in which to drop nulls/NaNs."""

    columns: Tuple[str, ...] = tuple()
    """Columns in which to look for nulls and NaNs.

    If this tuple is empty, *all* columns will be looked at - so if *any* column
    contains a null or NaN value that row will be dropped."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Tuple[str, ...])

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_drop_null_and_nan(self)


@dataclass(frozen=True)
class DropInfinity(QueryExpr):
    """Returns data with rows that contain +inf/-inf dropped."""

    child: QueryExpr
    """The QueryExpr in which to drop +inf/-inf."""

    columns: Tuple[str, ...] = tuple()
    """Columns in which to look for and infinite values.

    If this tuple is empty, *all* columns will be looked at - so if *any* column
    contains an infinite value, that row will be dropped."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Tuple[str, ...])

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_drop_infinity(self)


@dataclass(frozen=True)
class EnforceConstraint(QueryExpr):
    """Enforces a constraint on the data."""

    child: QueryExpr
    """The QueryExpr to which the constraint will be applied."""
    constraint: Constraint
    """A constraint to be enforced."""
    options: FrozenDict = FrozenDict.from_dict({})
    """Options to be used when enforcing the constraint.

    Appropriate values here vary depending on the constraint. These options are
    to support advanced use cases, and generally should not be used."""

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_enforce_constraint(self)


@dataclass(frozen=True)
class GroupByCount(QueryExpr):
    """Returns the count of each combination of the groupby domains."""

    child: QueryExpr
    """The QueryExpr to measure."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    output_column: str = "count"
    """The name of the column to store the counts in."""
    mechanism: CountMechanism = CountMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an
    appropriate mechanism.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.output_column, str)
        check_type(self.mechanism, CountMechanism)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_count(self)


@dataclass(frozen=True)
class GroupByCountDistinct(QueryExpr):
    """Returns the count of distinct rows in each groupby domain value."""

    child: QueryExpr
    """The QueryExpr to measure."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    columns_to_count: Optional[Tuple[str, ...]] = None
    """The columns that are compared when determining if two rows are distinct.

    If empty, will count all distinct rows.
    """
    output_column: str = "count_distinct"
    """The name of the column to store the distinct counts in."""
    mechanism: CountDistinctMechanism = CountDistinctMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an appropriate mechanism.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.columns_to_count, Optional[Tuple[str, ...]])
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.output_column, str)
        check_type(self.mechanism, CountDistinctMechanism)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_count_distinct(self)


@dataclass(frozen=True)
class GroupByQuantile(QueryExpr):
    """Returns the quantile of a column for each combination of the groupby domains.

    If the column to be measured contains null, NaN, or positive or negative infinity,
    those values will be dropped (as if dropped explicitly via
    :class:`DropNullAndNan` and :class:`DropInfinity`) before the quantile is
    calculated.
    """

    child: QueryExpr
    """The QueryExpr to measure."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to compute the quantile over."""
    quantile: float
    """The quantile to compute (between 0 and 1)."""
    low: float
    """The lower bound for clamping the ``measure_column``.
    Should be less than ``high``.
    """
    high: float
    """The upper bound for clamping the ``measure_column``.
    Should be greater than ``low``.
    """
    output_column: str = "quantile"
    """The name of the column to store the quantiles in."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.quantile, float)
        check_type(self.low, float)
        check_type(self.high, float)
        check_type(self.output_column, str)

        if not 0 <= self.quantile <= 1:
            raise ValueError(
                f"Quantile must be between 0 and 1, and not '{self.quantile}'."
            )
        if type(self.low) != type(self.high):  # pylint: disable=unidiomatic-typecheck
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_quantile(self)


@dataclass(frozen=True)
class GroupByBoundedSum(QueryExpr):
    """Returns the bounded sum of a column for each combination of groupby domains.

    If the column to be measured contains null, NaN, or positive or negative infinity,
    those values will be dropped (as if dropped explicitly via
    :class:`DropNullAndNan` and :class:`DropInfinity`) before the sum is
    calculated.
    """

    child: QueryExpr
    """The QueryExpr to measure."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to compute the sum over."""
    low: float
    """The lower bound for clamping the ``measure_column``.
    Should be less than ``high``.
    """
    high: float
    """The upper bound for clamping the ``measure_column``.
    Should be greater than ``low``.
    """
    output_column: str = "sum"
    """The name of the column to store the sums in."""
    mechanism: SumMechanism = SumMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an
    appropriate mechanism.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.low, float)
        check_type(self.high, float)
        check_type(self.output_column, str)
        check_type(self.mechanism, SumMechanism)

        if type(self.low) != type(self.high):  # pylint: disable=unidiomatic-typecheck
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_sum(self)


@dataclass(frozen=True)
class GroupByBoundedAverage(QueryExpr):
    """Returns bounded average of a column for each combination of groupby domains.

    If the column to be measured contains null, NaN, or positive or negative infinity,
    those values will be dropped (as if dropped explicitly via
    :class:`DropNullAndNan` and :class:`DropInfinity`) before the average is
    calculated.
    """

    child: QueryExpr
    """The QueryExpr to measure."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to compute the average over."""
    low: float
    """The lower bound for clamping the ``measure_column``.
    Should be less than ``high``.
    """
    high: float
    """The upper bound for clamping the ``measure_column``.
    Should be greater than ``low``.
    """
    output_column: str = "average"
    """The name of the column to store the averages in."""
    mechanism: AverageMechanism = AverageMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an
    appropriate mechanism.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.low, float)
        check_type(self.high, float)
        check_type(self.output_column, str)
        check_type(self.mechanism, AverageMechanism)

        if type(self.low) != type(self.high):  # pylint: disable=unidiomatic-typecheck
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_average(self)


@dataclass(frozen=True)
class GroupByBoundedVariance(QueryExpr):
    """Returns bounded variance of a column for each combination of groupby domains.

    If the column to be measured contains null, NaN, or positive or negative infinity,
    those values will be dropped (as if dropped explicitly via
    :class:`DropNullAndNan` and :class:`DropInfinity`) before the variance is
    calculated.
    """

    child: QueryExpr
    """The QueryExpr to measure."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to compute the variance over."""
    low: float
    """The lower bound for clamping the ``measure_column``.
    Should be less than ``high``.
    """
    high: float
    """The upper bound for clamping the ``measure_column``.
    Should be greater than ``low``.
    """
    output_column: str = "variance"
    """The name of the column to store the variances in."""
    mechanism: VarianceMechanism = VarianceMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an
    appropriate mechanism.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.low, float)
        check_type(self.high, float)
        check_type(self.output_column, str)
        check_type(self.mechanism, VarianceMechanism)

        if type(self.low) != type(self.high):  # pylint: disable=unidiomatic-typecheck
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_variance(self)


@dataclass(frozen=True)
class GroupByBoundedSTDEV(QueryExpr):
    """Returns bounded stdev of a column for each combination of groupby domains.

    If the column to be measured contains null, NaN, or positive or negative infinity,
    those values will be dropped (as if dropped explicitly via
    :class:`DropNullAndNan` and :class:`DropInfinity`) before the
    standard deviation is calculated.
    """

    child: QueryExpr
    """The QueryExpr to measure."""
    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to compute the standard deviation over."""
    low: float
    """The lower bound for clamping the ``measure_column``.
    Should be less than ``high``.
    """
    high: float
    """The upper bound for clamping the ``measure_column``.
    Should be greater than ``low``.
    """
    output_column: str = "stdev"
    """The name of the column to store the stdev in."""
    mechanism: StdevMechanism = StdevMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an
    appropriate mechanism.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.low, float)
        check_type(self.high, float)
        check_type(self.output_column, str)
        check_type(self.mechanism, StdevMechanism)

        if type(self.low) != type(self.high):  # pylint: disable=unidiomatic-typecheck
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))

        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_stdev(self)


@dataclass(frozen=True)
class SuppressAggregates(QueryExpr):
    """Remove all counts that are less than the threshold."""

    child: GroupByCount
    """The aggregate on which to suppress small counts.

    Currently, only GroupByCount is supported.
    """

    column: str
    """The name of the column to suppress."""

    threshold: float
    """Threshold. All counts less than this will be suppressed."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        if not isinstance(self.child, GroupByCount):
            raise TypeError(
                "SuppressAggregates is only supported on aggregates that are "
                "GroupByCounts"
            )
        check_type(self.column, str)
        check_type(self.threshold, float)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_suppress_aggregates(self)


class QueryExprVisitor(ABC):
    """A base class for implementing visitors for :class:`QueryExpr`."""

    @abstractmethod
    def visit_private_source(self, expr: PrivateSource) -> Any:
        """Visit a :class:`PrivateSource`."""
        raise NotImplementedError

    @abstractmethod
    def visit_rename(self, expr: Rename) -> Any:
        """Visit a :class:`Rename`."""
        raise NotImplementedError

    @abstractmethod
    def visit_filter(self, expr: Filter) -> Any:
        """Visit a :class:`Filter`."""
        raise NotImplementedError

    @abstractmethod
    def visit_select(self, expr: Select) -> Any:
        """Visit a :class:`Select`."""
        raise NotImplementedError

    @abstractmethod
    def visit_map(self, expr: Map) -> Any:
        """Visit a :class:`Map`."""
        raise NotImplementedError

    @abstractmethod
    def visit_flat_map(self, expr: FlatMap) -> Any:
        """Visit a :class:`FlatMap`."""
        raise NotImplementedError

    @abstractmethod
    def visit_flat_map_by_id(self, expr: FlatMapByID) -> Any:
        """Visit a :class:`FlatMapByID`."""
        raise NotImplementedError

    @abstractmethod
    def visit_join_private(self, expr: JoinPrivate) -> Any:
        """Visit a :class:`JoinPrivate`."""
        raise NotImplementedError

    @abstractmethod
    def visit_join_public(self, expr: JoinPublic) -> Any:
        """Visit a :class:`JoinPublic`."""
        raise NotImplementedError

    @abstractmethod
    def visit_replace_null_and_nan(self, expr: ReplaceNullAndNan) -> Any:
        """Visit a :class:`ReplaceNullAndNan`."""
        raise NotImplementedError

    @abstractmethod
    def visit_replace_infinity(self, expr: ReplaceInfinity) -> Any:
        """Visit a :class:`ReplaceInfinity`."""
        raise NotImplementedError

    @abstractmethod
    def visit_drop_null_and_nan(self, expr: DropNullAndNan) -> Any:
        """Visit a :class:`DropNullAndNan`."""
        raise NotImplementedError

    @abstractmethod
    def visit_drop_infinity(self, expr: DropInfinity) -> Any:
        """Visit a :class:`DropInfinity`."""
        raise NotImplementedError

    @abstractmethod
    def visit_enforce_constraint(self, expr: EnforceConstraint) -> Any:
        """Visit a :class:`EnforceConstraint`."""
        raise NotImplementedError

    @abstractmethod
    def visit_get_groups(self, expr: GetGroups) -> Any:
        """Visit a :class:`GetGroups`."""
        raise NotImplementedError

    @abstractmethod
    def visit_get_bounds(self, expr: GetBounds) -> Any:
        """Visit a :class:`GetBounds`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_count(self, expr: GroupByCount) -> Any:
        """Visit a :class:`GroupByCount`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_count_distinct(self, expr: GroupByCountDistinct) -> Any:
        """Visit a :class:`GroupByCountDistinct`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_quantile(self, expr: GroupByQuantile) -> Any:
        """Visit a :class:`GroupByQuantile`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_sum(self, expr: GroupByBoundedSum) -> Any:
        """Visit a :class:`GroupByBoundedSum`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_average(self, expr: GroupByBoundedAverage) -> Any:
        """Visit a :class:`GroupByBoundedAverage`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_variance(self, expr: GroupByBoundedVariance) -> Any:
        """Visit a :class:`GroupByBoundedVariance`."""
        raise NotImplementedError

    @abstractmethod
    def visit_groupby_bounded_stdev(self, expr: GroupByBoundedSTDEV) -> Any:
        """Visit a :class:`GroupByBoundedSTDEV`."""
        raise NotImplementedError

    @abstractmethod
    def visit_suppress_aggregates(self, expr: SuppressAggregates) -> Any:
        """Visit a :class:`SuppressAggregates`."""
        raise NotImplementedError
