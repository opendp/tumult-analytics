# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

"""Abstract syntax tree for Tumult Analytics queries.

This module defines classes that represent the nodes of an abstract syntax tree (AST)
for Tumult Analytics queries. These classes are immutable.
"""
import datetime
import inspect
from dataclasses import FrozenInstanceError, dataclass, field
from functools import reduce
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import pyspark.sql.dataframe
import sympy as sp
from pyspark.sql import DataFrame
from tmlt.core.domains.base import Domain
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import Metric, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber
from typeguard import typechecked

from tmlt.analytics import KeySet
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    FrozenDict,
    Schema,
)
from tmlt.analytics.constraints import Constraint
from tmlt.analytics.truncation_strategy import TruncationStrategy

# Set type checking to `True` for dataclasses as `False` can introduce subtle bugs.
# See https://github.com/agronholm/typeguard/issues/242#issuecomment-746736283
typechecked_dataclass = typechecked(dataclass(frozen=True, eq=True))


class AnalyticsInternalError(Exception):
    """Raised for errors in Tumult Analytics' internal logic."""


class QueryExprVisitor:
    """A visitor pattern for QueryExpr objects.

    Subclasses must implement methods named ``visit_X`` for each concrete
    subclass ``X`` of :class:`QueryExpr` for which they are defined. For
    example, to visit a :class:`Rename` node, one must implement the
    method ``visit_rename()``.
    """

    def visit(self, expr: "QueryExpr", *args: Any, **kwargs: Any) -> Any:
        """Dispatches the visit to the correct method."""
        method = "visit_" + expr.__class__.__name__.lower()
        visitor = getattr(self, method)
        return visitor(expr, *args, **kwargs)

    # pylint: disable=missing-function-docstring
    def visit_private_source(self, expr: "PrivateSource") -> Any:
        raise NotImplementedError

    def visit_rename(self, expr: "Rename") -> Any:
        raise NotImplementedError

    def visit_filter(self, expr: "Filter") -> Any:
        raise NotImplementedError

    def visit_select(self, expr: "Select") -> Any:
        raise NotImplementedError

    def visit_map(self, expr: "Map") -> Any:
        raise NotImplementedError

    def visit_flat_map(self, expr: "FlatMap") -> Any:
        raise NotImplementedError

    def visit_flat_map_by_id(self, expr: "FlatMapByID") -> Any:
        raise NotImplementedError

    def visit_join_private(self, expr: "JoinPrivate") -> Any:
        raise NotImplementedError

    def visit_join_public(self, expr: "JoinPublic") -> Any:
        raise NotImplementedError

    def visit_replace_null_and_nan(self, expr: "ReplaceNullAndNan") -> Any:
        raise NotImplementedError

    def visit_replace_infinity(self, expr: "ReplaceInfinity") -> Any:
        raise NotImplementedError

    def visit_drop_null_and_nan(self, expr: "DropNullAndNan") -> Any:
        raise NotImplementedError

    def visit_drop_infinity(self, expr: "DropInfinity") -> Any:
        raise NotImplementedError

    def visit_enforce_constraint(self, expr: "EnforceConstraint") -> Any:
        raise NotImplementedError

    def visit_get_groups(self, expr: "GetGroups") -> Any:
        raise NotImplementedError

    def visit_get_bounds(self, expr: "GetBounds") -> Any:
        raise NotImplementedError

    def visit_groupby_count(self, expr: "GroupByCount") -> Any:
        raise NotImplementedError

    def visit_groupby_count_distinct(self, expr: "GroupByCountDistinct") -> Any:
        raise NotImplementedError

    def visit_groupby_quantile(self, expr: "GroupByQuantile") -> Any:
        raise NotImplementedError

    def visit_groupby_bounded_sum(self, expr: "GroupByBoundedSum") -> Any:
        raise NotImplementedError

    def visit_groupby_bounded_average(self, expr: "GroupByBoundedAverage") -> Any:
        raise NotImplementedError

    def visit_groupby_bounded_variance(self, expr: "GroupByBoundedVariance") -> Any:
        raise NotImplementedError

    def visit_groupby_bounded_stdev(self, expr: "GroupByBoundedSTDEV") -> Any:
        raise NotImplementedError

    def visit_suppress_aggregates(self, expr: "SuppressAggregates") -> Any:
        raise NotImplementedError

    # pylint: enable=missing-function-docstring


@typechecked_dataclass
class QueryExpr:
    """Base class for all query expressions."""

    def accept(self, visitor: QueryExprVisitor, *args: Any, **kwargs: Any) -> Any:
        """Accepts a QueryExprVisitor and dispatches it to the correct method."""
        return visitor.visit(self, *args, **kwargs)


@typechecked_dataclass
class PrivateSource(QueryExpr):
    """The root node of any query for a private source.

    Attributes:
        source_id: ID of the private source.
    """

    source_id: str

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if not self.source_id.isidentifier():
            raise ValueError(
                f"The `PrivateSource` expression received an invalid source ID "
                f"`{self.source_id}`. Source IDs must be valid Python identifiers: "
                "they can only contain alphanumeric characters and underscores, and "
                "cannot begin with a number. Please provide a valid source ID."
            )


@typechecked_dataclass
class Rename(QueryExpr):
    """Renames columns in the input DataFrame."""

    child: QueryExpr
    column_mapper: FrozenDict[str, str]

    def __post_init__(self) -> None:
        """Validate input types and values."""
        for old_name, new_name in self.column_mapper.items():
            if not new_name.isidentifier():
                if new_name == "":
                    raise ValueError(
                        f"The `rename` method failed: Cannot rename column "
                        f"`{old_name}` to `''` (the empty string). Column names "
                        "cannot be empty strings. Please provide a non-empty, "
                        "valid Python identifier as the new column name."
                    )
                raise ValueError(
                    f"The `rename` method failed: Cannot rename column "
                    f"`{old_name}` to `{new_name}`. New column names must be "
                    "valid Python identifiers: they can only contain alphanumeric "
                    "characters and underscores, and cannot begin with a number. "
                    "Please provide a valid Python identifier."
                )


@typechecked_dataclass
class Filter(QueryExpr):
    """Filters the input DataFrame."""

    child: QueryExpr
    condition: str


@typechecked_dataclass
class Select(QueryExpr):
    """Selects a subset of columns from the input DataFrame."""

    child: QueryExpr
    columns: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if len(set(self.columns)) != len(self.columns):
            duplicate_columns = [
                col for col in self.columns if self.columns.count(col) > 1
            ]
            # Use sorted to make error message deterministic in case of multiple duplicates
            duplicate_columns = sorted(list(set(duplicate_columns)))
            raise ValueError(
                f"The `select` method failed: The specified columns "
                f"`{duplicate_columns}` contain duplicates. Each column name in "
                "`columns` must be unique. Please provide a list of distinct column names."
            )
        for col_name in self.columns:
            if not col_name.isidentifier():
                if col_name == "":
                    raise ValueError(
                        f"The `select` method failed: Column name `''` (empty string) "
                        "is not allowed. Please provide a non-empty, valid Python "
                        "identifier as a column name."
                    )
                raise ValueError(
                    f"The `select` method failed: Column name `{col_name}` is not "
                    "a valid Python identifier. Column names can only contain "
                    "alphanumeric characters and underscores, and cannot begin "
                    "with a number. Please provide a valid Python identifier."
                )


@typechecked_dataclass
class Map(QueryExpr):
    """Maps each row of the input DataFrame to a new row."""

    child: QueryExpr
    f: Callable[[Dict[str, Any]], Dict[str, Any]]
    schema_new_columns: Schema
    augment: bool

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.schema_new_columns.grouping_column is not None:
            raise ValueError(
                f"The `map` method failed: The `new_column_types` parameter cannot "
                f"specify a grouping column (`'{self.schema_new_columns.grouping_column}'`). "
                "Grouping columns are managed at the session level. Please provide a "
                "`new_column_types` schema that does not include a grouping column."
            )
        if self.schema_new_columns.id_column is not None:
            raise ValueError(
                f"The `map` method failed: The `new_column_types` parameter cannot "
                f"specify an ID column (`'{self.schema_new_columns.id_column}'`). "
                "ID columns are managed at the session level. Please provide a "
                "`new_column_types` schema that does not include an ID column."
            )
        for new_col_name in self.schema_new_columns.column_types.keys():
            if not new_col_name.isidentifier():
                if new_col_name == "":
                    raise ValueError(
                        f"The `map` method failed: New column name `''` (empty string) "
                        "is not allowed. Please provide a non-empty, valid Python "
                        "identifier as a new column name."
                    )
                raise ValueError(
                    f"The `map` method failed: New column name `{new_col_name}` is not "
                    "a valid Python identifier. Column names can only contain "
                    "alphanumeric characters and underscores, and cannot begin "
                    "with a number. Please provide a valid Python identifier."
                )


@typechecked_dataclass
class FlatMap(QueryExpr):
    """Maps each row of the input DataFrame to multiple new rows."""

    child: QueryExpr
    f: Callable[[Dict[str, Any]], List[Dict[str, Any]]]
    schema_new_columns: Schema
    augment: bool
    max_rows: int

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.max_rows < 0:
            raise ValueError(
                f"The `flat_map` method failed: The `max_rows` parameter "
                f"must be non-negative, but received `{self.max_rows}`. "
                "Please provide a non-negative integer for `max_rows`."
            )
        if self.schema_new_columns.id_column is not None:
            raise ValueError(
                f"The `flat_map` method failed: The `new_column_types` parameter cannot "
                f"specify an ID column (`'{self.schema_new_columns.id_column}'`). "
                "ID columns are managed at the session level. Please provide a "
                "`new_column_types` schema that does not include an ID column."
            )
        if self.schema_new_columns.grouping_column is not None:
            if self.augment:
                raise ValueError(
                    "The `flat_map` method failed: Cannot set `augment=True` "
                    "when specifying a grouping column in `new_column_types`. "
                    "When creating new groups, `augment` must be `False`. "
                    "Please set `augment=False`."
                )
            if len(self.schema_new_columns.column_types) != 1:
                raise ValueError(
                    f"The `flat_map` method failed: When `new_column_types` "
                    "specifies a grouping column, it must contain exactly one new "
                    "column. However, `new_column_types` contains "
                    f"{len(self.schema_new_columns.column_types)} columns. "
                    "Please ensure `new_column_types` only includes the grouping column."
                )
        for new_col_name in self.schema_new_columns.column_types.keys():
            if not new_col_name.isidentifier():
                if new_col_name == "":
                    raise ValueError(
                        f"The `flat_map` method failed: New column name `''` (empty string) "
                        "is not allowed. Please provide a non-empty, valid Python "
                        "identifier as a new column name."
                    )
                raise ValueError(
                    f"The `flat_map` method failed: New column name `{new_col_name}` is not "
                    "a valid Python identifier. Column names can only contain "
                    "alphanumeric characters and underscores, and cannot begin "
                    "with a number. Please provide a valid Python identifier."
                )


@typechecked_dataclass
class FlatMapByID(QueryExpr):
    """Maps a list of rows for each ID to multiple new rows."""

    child: QueryExpr
    f: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
    schema_new_columns: Schema

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.schema_new_columns.grouping_column is not None:
            raise ValueError(
                f"The `flat_map_by_id` method failed: The `new_column_types` parameter "
                f"cannot specify a grouping column (`'{self.schema_new_columns.grouping_column}'`). "
                "Grouping columns are managed at the session level. Please provide a "
                "`new_column_types` schema that does not include a grouping column."
            )
        if self.schema_new_columns.id_column is not None:
            raise ValueError(
                f"The `flat_map_by_id` method failed: The `new_column_types` parameter "
                f"cannot specify an ID column (`'{self.schema_new_columns.id_column}'`). "
                "ID columns are managed at the session level. Please provide a "
                "`new_column_types` schema that does not include an ID column."
            )
        for new_col_name in self.schema_new_columns.column_types.keys():
            if not new_col_name.isidentifier():
                if new_col_name == "":
                    raise ValueError(
                        f"The `flat_map_by_id` method failed: New column name `''` "
                        "(empty string) is not allowed. Please provide a non-empty, "
                        "valid Python identifier as a new column name."
                    )
                raise ValueError(
                    f"The `flat_map_by_id` method failed: New column name "
                    f"`{new_col_name}` is not a valid Python identifier. Column names "
                    "can only contain alphanumeric characters and underscores, and "
                    "cannot begin with a number. Please provide a valid Python identifier."
                )


@typechecked_dataclass
class JoinPrivate(QueryExpr):
    """Joins the input DataFrame with another private DataFrame."""

    child: QueryExpr
    right_operand_expr: QueryExpr
    truncation_strategy_left: TruncationStrategy.Type
    truncation_strategy_right: TruncationStrategy.Type
    join_columns: Optional[Tuple[str, ...]]

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.join_columns is not None:
            if not self.join_columns:
                raise ValueError(
                    "The `join_private` method failed: The `join_columns` parameter "
                    "cannot be an empty tuple if specified. Please provide at least "
                    "one column name for joining, or set `join_columns=None` to use "
                    "all common columns."
                )
            if len(set(self.join_columns)) != len(self.join_columns):
                duplicate_columns = [
                    col for col in self.join_columns if self.join_columns.count(col) > 1
                ]
                duplicate_columns = sorted(list(set(duplicate_columns)))
                raise ValueError(
                    f"The `join_private` method failed: The specified join columns "
                    f"`{duplicate_columns}` contain duplicates. Each join column must be "
                    "unique. Please provide a list of distinct join columns."
                )
            for col_name in self.join_columns:
                if not col_name.isidentifier():
                    if col_name == "":
                        raise ValueError(
                            f"The `join_private` method failed: Join column name `''` "
                            "(empty string) is not allowed. Please provide a non-empty, "
                            "valid Python identifier as a join column name."
                        )
                    raise ValueError(
                        f"The `join_private` method failed: Join column name "
                        f"`{col_name}` is not a valid Python identifier. Column names "
                        "can only contain alphanumeric characters and underscores, "
                        "and cannot begin with a number. Please provide a valid "
                        "Python identifier."
                    )


@typechecked_dataclass
class JoinPublic(QueryExpr):
    """Joins the input DataFrame with a public DataFrame."""

    child: QueryExpr
    public_table: Union[str, DataFrame]
    join_columns: Optional[Tuple[str, ...]]
    how: Literal["inner", "left"] = "inner"

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.join_columns is not None:
            if not self.join_columns:
                raise ValueError(
                    "The `join_public` method failed: The `join_columns` parameter "
                    "cannot be an empty tuple if specified. Please provide at least "
                    "one column name for joining, or set `join_columns=None` to use "
                    "all common columns."
                )
            if len(set(self.join_columns)) != len(self.join_columns):
                duplicate_columns = [
                    col for col in self.join_columns if self.join_columns.count(col) > 1
                ]
                duplicate_columns = sorted(list(set(duplicate_columns)))
                raise ValueError(
                    f"The `join_public` method failed: The specified join columns "
                    f"`{duplicate_columns}` contain duplicates. Each join column must be "
                    "unique. Please provide a list of distinct join columns."
                )
            for col_name in self.join_columns:
                if not col_name.isidentifier():
                    if col_name == "":
                        raise ValueError(
                            f"The `join_public` method failed: Join column name `''` "
                            "(empty string) is not allowed. Please provide a non-empty, "
                            "valid Python identifier as a join column name."
                        )
                    raise ValueError(
                        f"The `join_public` method failed: Join column name "
                        f"`{col_name}` is not a valid Python identifier. Column names "
                        "can only contain alphanumeric characters and underscores, "
                        "and cannot begin with a number. Please provide a valid "
                        "Python identifier."
                    )

        if self.how not in ["inner", "left"]:
            raise ValueError(
                f"The `join_public` method failed: Invalid join type `'{self.how}'`. "
                "The `how` parameter must be either `'inner'` or `'left'`. "
                "Please choose one of these valid join types."
            )


@typechecked_dataclass
class ReplaceNullAndNan(QueryExpr):
    """Replaces null and NaN values in the input DataFrame."""

    child: QueryExpr
    replace_with: FrozenDict[
        str, Union[int, float, str, datetime.date, datetime.datetime]
    ]


@typechecked_dataclass
class ReplaceInfinity(QueryExpr):
    """Replaces positive and negative infinity values in the input DataFrame."""

    child: QueryExpr
    replace_with: FrozenDict[str, Tuple[float, float]]

    def __post_init__(self) -> None:
        """Validate input types and values."""
        for column, bounds in self.replace_with.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    f"The `replace_infinity` method failed: The bounds for column "
                    f"`{column}` must be a tuple of two numbers `(low, high)`. "
                    f"Received `{bounds}`. Please provide a valid tuple of two numbers."
                )
            if not isinstance(bounds[0], (int, float)) or not isinstance(
                bounds[1], (int, float)
            ):
                raise ValueError(
                    f"The `replace_infinity` method failed: The bounds for column "
                    f"`{column}` must be a tuple of two numbers. "
                    f"Received `({bounds[0]}, {bounds[1]})`. Please ensure both "
                    "lower and upper bounds are numeric."
                )
            if bounds[0] >= bounds[1]:
                raise ValueError(
                    f"The `replace_infinity` method failed: The lower bound "
                    f"`{bounds[0]}` for column `{column}` must be strictly less than "
                    f"the upper bound `{bounds[1]}`. Please ensure `low < high`."
                )


@typechecked_dataclass
class DropNullAndNan(QueryExpr):
    """Drops rows containing null or NaN values in specified columns."""

    child: QueryExpr
    columns: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate input types and values."""
        for col_name in self.columns:
            if not col_name.isidentifier():
                if col_name == "":
                    raise ValueError(
                        f"The `drop_null_and_nan` method failed: Column name `''` "
                        "(empty string) is not allowed. Please provide a non-empty, "
                        "valid Python identifier as a column name."
                    )
                raise ValueError(
                    f"The `drop_null_and_nan` method failed: Column name "
                    f"`{col_name}` is not a valid Python identifier. Column names "
                    "can only contain alphanumeric characters and underscores, "
                    "and cannot begin with a number. Please provide a valid "
                    "Python identifier."
                )


@typechecked_dataclass
class DropInfinity(QueryExpr):
    """Drops rows containing infinity values in specified columns."""

    child: QueryExpr
    columns: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate input types and values."""
        for col_name in self.columns:
            if not col_name.isidentifier():
                if col_name == "":
                    raise ValueError(
                        f"The `drop_infinity` method failed: Column name `''` (empty string) "
                        "is not allowed. Please provide a non-empty, valid Python "
                        "identifier as a column name."
                    )
                raise ValueError(
                    f"The `drop_infinity` method failed: Column name `{col_name}` is not "
                    "a valid Python identifier. Column names can only contain "
                    "alphanumeric characters and underscores, and cannot begin "
                    "with a number. Please provide a valid Python identifier."
                )


@typechecked_dataclass
class EnforceConstraint(QueryExpr):
    """Enforces a stability constraint."""

    child: QueryExpr
    constraint: Constraint
    # This stores the mapping from dataframe column to Core ID columns
    id_map: FrozenDict[str, str]


@typechecked_dataclass
class GetGroups(QueryExpr):
    """Returns the unique groups of specified columns after privacy transformations."""

    child: QueryExpr
    columns: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.columns is not None:
            if not self.columns:
                raise ValueError(
                    "The `get_groups` method failed: The `columns` parameter "
                    "cannot be an empty tuple if specified. To return all "
                    "non-ID columns, set `columns=None`."
                )
            for col_name in self.columns:
                if not col_name.isidentifier():
                    if col_name == "":
                        raise ValueError(
                            f"The `get_groups` method failed: Column name `''` "
                            "(empty string) is not allowed. Please provide a non-empty, "
                            "valid Python identifier as a column name."
                        )
                    raise ValueError(
                        f"The `get_groups` method failed: Column name "
                        f"`{col_name}` is not a valid Python identifier. Column names "
                        "can only contain alphanumeric characters and underscores, "
                        "and cannot begin with a number. Please provide a valid "
                        "Python identifier."
                    )


@typechecked_dataclass
class GetBounds(QueryExpr):
    """Computes differentially private lower and upper bounds for a numeric column."""

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    lower_bound_column: str
    upper_bound_column: str

    def __post_init__(self) -> None:
        """Validate input types and values."""
        for param_name, col_name in [
            ("measure_column", self.measure_column),
            ("lower_bound_column", self.lower_bound_column),
            ("upper_bound_column", self.upper_bound_column),
        ]:
            if not col_name.isidentifier():
                if col_name == "":
                    raise ValueError(
                        f"The `get_bounds` method failed: The `{param_name}` parameter "
                        f"received an empty string `''`. Column names cannot be empty "
                        "strings. Please provide a non-empty, valid Python identifier."
                    )
                raise ValueError(
                    f"The `get_bounds` method failed: The column name `{col_name}` "
                    f"provided for `{param_name}` is not a valid Python identifier. "
                    "Column names can only contain alphanumeric characters and "
                    "underscores, and cannot begin with a number. Please provide a "
                    "valid Python identifier."
                )
        if self.lower_bound_column == self.upper_bound_column:
            raise ValueError(
                "The `get_bounds` method failed: The `lower_bound_column` and "
                "`upper_bound_column` parameters cannot have the same name "
                f"(`'{self.lower_bound_column}'`). Please provide distinct column names."
            )


class CountMechanism:
    """Class holding supported count mechanisms."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


@typechecked_dataclass
class GroupByCount(QueryExpr):
    """Counts the number of rows in each group."""

    child: QueryExpr
    groupby_keys: KeySet
    output_column: str = "count"
    mechanism: str = CountMechanism.LAPLACE

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if not self.output_column.isidentifier():
            if self.output_column == "":
                raise ValueError(
                    f"The `count` method failed: The `name` parameter received an empty "
                    "string `''`. Output column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `count` method failed: The output column name "
                f"`{self.output_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if self.mechanism not in [CountMechanism.LAPLACE, CountMechanism.GAUSSIAN]:
            raise ValueError(
                f"The `count` method failed: Invalid mechanism `'{self.mechanism}'`. "
                f"Valid mechanisms are `'{CountMechanism.LAPLACE}'` or "
                f"`'{CountMechanism.GAUSSIAN}'`. Please choose a valid mechanism."
            )


@typechecked_dataclass
class GroupByCountDistinct(QueryExpr):
    """Counts the number of distinct values in columns for each group."""

    child: QueryExpr
    groupby_keys: KeySet
    columns_to_count: Optional[Tuple[str, ...]] = None
    output_column: str = "count_distinct"
    mechanism: str = CountMechanism.LAPLACE

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if not self.output_column.isidentifier():
            if self.output_column == "":
                raise ValueError(
                    f"The `count_distinct` method failed: The `name` parameter "
                    "received an empty string `''`. Output column names cannot be "
                    "empty strings. Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `count_distinct` method failed: The output column name "
                f"`{self.output_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if self.columns_to_count is not None:
            if not self.columns_to_count:
                raise ValueError(
                    "The `count_distinct` method failed: The `columns` parameter "
                    "cannot be an empty tuple if specified. To count distinct "
                    "rows, set `columns=None`."
                )
            if len(set(self.columns_to_count)) != len(self.columns_to_count):
                duplicate_columns = [
                    col
                    for col in self.columns_to_count
                    if self.columns_to_count.count(col) > 1
                ]
                duplicate_columns = sorted(list(set(duplicate_columns)))
                raise ValueError(
                    f"The `count_distinct` method failed: The specified columns "
                    f"`{duplicate_columns}` contain duplicates. Each column name in "
                    "`columns` must be unique. Please provide a list of distinct "
                    "column names."
                )
            for col_name in self.columns_to_count:
                if not col_name.isidentifier():
                    if col_name == "":
                        raise ValueError(
                            f"The `count_distinct` method failed: Column name `''` "
                            "(empty string) is not allowed. Please provide a non-empty, "
                            "valid Python identifier as a column name."
                        )
                    raise ValueError(
                        f"The `count_distinct` method failed: Column name "
                        f"`{col_name}` is not a valid Python identifier. Column names "
                        "can only contain alphanumeric characters and underscores, "
                        "and cannot begin with a number. Please provide a valid "
                        "Python identifier."
                    )
        if self.mechanism not in [CountMechanism.LAPLACE, CountMechanism.GAUSSIAN]:
            raise ValueError(
                f"The `count_distinct` method failed: Invalid mechanism `'{self.mechanism}'`. "
                f"Valid mechanisms are `'{CountMechanism.LAPLACE}'` or "
                f"`'{CountMechanism.GAUSSIAN}'`. Please choose a valid mechanism."
            )


class QuantileMechanism:
    """Class holding supported quantile mechanisms."""

    EXPONENTIAL = "exponential"


@typechecked_dataclass
class GroupByQuantile(QueryExpr):
    """Computes the differentially private quantile for a numeric column within each group."""

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    quantile: float
    low: float
    high: float
    output_column: str
    mechanism: str = QuantileMechanism.EXPONENTIAL

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.low >= self.high:
            raise ValueError(
                f"The `quantile` method failed: The lower bound `{self.low}` "
                f"must be strictly less than the upper bound `{self.high}`. "
                "Please ensure `low < high`."
            )
        if not (0 <= self.quantile <= 1):
            raise ValueError(
                f"The `quantile` method failed: The `quantile` parameter must be "
                f"between `0` and `1` (inclusive), but received `{self.quantile}`. "
                "Please provide a `quantile` value within this range."
            )
        if not self.output_column.isidentifier():
            if self.output_column == "":
                raise ValueError(
                    f"The `quantile` method failed: The `name` parameter received an empty "
                    "string `''`. Output column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `quantile` method failed: The output column name "
                f"`{self.output_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if not self.measure_column.isidentifier():
            if self.measure_column == "":
                raise ValueError(
                    f"The `quantile` method failed: The `column` parameter received an empty "
                    "string `''`. Measure column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `quantile` method failed: The measure column name "
                f"`{self.measure_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if self.mechanism != QuantileMechanism.EXPONENTIAL:
            raise ValueError(
                f"The `quantile` method failed: Invalid mechanism `'{self.mechanism}'`. "
                f"The only valid mechanism for quantile is `'{QuantileMechanism.EXPONENTIAL}'`. "
                "Please choose a valid mechanism."
            )


class SumMechanism:
    """Class holding supported sum mechanisms."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


@typechecked_dataclass
class GroupByBoundedSum(QueryExpr):
    """Computes the differentially private sum of a numeric column within each group."""

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str
    mechanism: str = SumMechanism.LAPLACE

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.low >= self.high:
            raise ValueError(
                f"The `sum` method failed: The lower bound `{self.low}` must be "
                f"strictly less than the upper bound `{self.high}`. "
                "Please ensure `low < high`."
            )
        if not self.output_column.isidentifier():
            if self.output_column == "":
                raise ValueError(
                    f"The `sum` method failed: The `name` parameter received an empty "
                    "string `''`. Output column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `sum` method failed: The output column name "
                f"`{self.output_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if not self.measure_column.isidentifier():
            if self.measure_column == "":
                raise ValueError(
                    f"The `sum` method failed: The `column` parameter received an empty "
                    "string `''`. Measure column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `sum` method failed: The measure column name "
                f"`{self.measure_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if self.mechanism not in [SumMechanism.LAPLACE, SumMechanism.GAUSSIAN]:
            raise ValueError(
                f"The `sum` method failed: Invalid mechanism `'{self.mechanism}'`. "
                f"Valid mechanisms are `'{SumMechanism.LAPLACE}'` or "
                f"`'{SumMechanism.GAUSSIAN}'`. Please choose a valid mechanism."
            )


class AverageMechanism:
    """Class holding supported average mechanisms."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


@typechecked_dataclass
class GroupByBoundedAverage(QueryExpr):
    """Computes the differentially private average of a numeric column within each group."""

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str
    mechanism: str = AverageMechanism.LAPLACE

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.low >= self.high:
            raise ValueError(
                f"The `average` method failed: The lower bound `{self.low}` must be "
                f"strictly less than the upper bound `{self.high}`. "
                "Please ensure `low < high`."
            )
        if not self.output_column.isidentifier():
            if self.output_column == "":
                raise ValueError(
                    f"The `average` method failed: The `name` parameter received an empty "
                    "string `''`. Output column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `average` method failed: The output column name "
                f"`{self.output_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if not self.measure_column.isidentifier():
            if self.measure_column == "":
                raise ValueError(
                    f"The `average` method failed: The `column` parameter received an empty "
                    "string `''`. Measure column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `average` method failed: The measure column name "
                f"`{self.measure_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if self.mechanism not in [AverageMechanism.LAPLACE, AverageMechanism.GAUSSIAN]:
            raise ValueError(
                f"The `average` method failed: Invalid mechanism `'{self.mechanism}'`. "
                f"Valid mechanisms are `'{AverageMechanism.LAPLACE}'` or "
                f"`'{AverageMechanism.GAUSSIAN}'`. Please choose a valid mechanism."
            )


class VarianceMechanism:
    """Class holding supported variance mechanisms."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


@typechecked_dataclass
class GroupByBoundedVariance(QueryExpr):
    """Computes the differentially private variance of a numeric column within each group."""

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str
    mechanism: str = VarianceMechanism.LAPLACE

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.low >= self.high:
            raise ValueError(
                f"The `variance` method failed: The lower bound `{self.low}` must be "
                f"strictly less than the upper bound `{self.high}`. "
                "Please ensure `low < high`."
            )
        if not self.output_column.isidentifier():
            if self.output_column == "":
                raise ValueError(
                    f"The `variance` method failed: The `name` parameter received an empty "
                    "string `''`. Output column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `variance` method failed: The output column name "
                f"`{self.output_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if not self.measure_column.isidentifier():
            if self.measure_column == "":
                raise ValueError(
                    f"The `variance` method failed: The `column` parameter received an empty "
                    "string `''`. Measure column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `variance` method failed: The measure column name "
                f"`{self.measure_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if self.mechanism not in [VarianceMechanism.LAPLACE, VarianceMechanism.GAUSSIAN]:
            raise ValueError(
                f"The `variance` method failed: Invalid mechanism `'{self.mechanism}'`. "
                f"Valid mechanisms are `'{VarianceMechanism.LAPLACE}'` or "
                f"`'{VarianceMechanism.GAUSSIAN}'`. Please choose a valid mechanism."
            )


class StdevMechanism:
    """Class holding supported stdev mechanisms."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


@typechecked_dataclass
class GroupByBoundedSTDEV(QueryExpr):
    """Computes the differentially private standard deviation of a numeric column within each group."""

    child: QueryExpr
    groupby_keys: KeySet
    measure_column: str
    low: float
    high: float
    output_column: str
    mechanism: str = StdevMechanism.LAPLACE

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if self.low >= self.high:
            raise ValueError(
                f"The `stdev` method failed: The lower bound `{self.low}` must be "
                f"strictly less than the upper bound `{self.high}`. "
                "Please ensure `low < high`."
            )
        if not self.output_column.isidentifier():
            if self.output_column == "":
                raise ValueError(
                    f"The `stdev` method failed: The `name` parameter received an empty "
                    "string `''`. Output column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `stdev` method failed: The output column name "
                f"`{self.output_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if not self.measure_column.isidentifier():
            if self.measure_column == "":
                raise ValueError(
                    f"The `stdev` method failed: The `column` parameter received an empty "
                    "string `''`. Measure column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `stdev` method failed: The measure column name "
                f"`{self.measure_column}` is not a valid Python identifier. "
                "Column names can only contain alphanumeric characters and "
                "underscores, and cannot begin with a number. Please provide a "
                "valid Python identifier."
            )
        if self.mechanism not in [StdevMechanism.LAPLACE, StdevMechanism.GAUSSIAN]:
            raise ValueError(
                f"The `stdev` method failed: Invalid mechanism `'{self.mechanism}'`. "
                f"Valid mechanisms are `'{StdevMechanism.LAPLACE}'` or "
                f"`'{StdevMechanism.GAUSSIAN}'`. Please choose a valid mechanism."
            )


@typechecked_dataclass
class SuppressAggregates(QueryExpr):
    """Suppresses aggregated rows that are below a certain threshold."""

    child: QueryExpr
    column: str
    threshold: int

    def __post_init__(self) -> None:
        """Validate input types and values."""
        if not isinstance(self.child, (GroupByCount, GroupByCountDistinct)):
            raise TypeError(
                f"The `suppress` method failed: `SuppressAggregates` is only supported "
                "on results from `count()` or `count_distinct()` aggregations. "
                f"Received a query expression of type `{type(self.child).__name__}`. "
                "Please apply `suppress()` to a `count()` or `count_distinct()` query."
            )
        if not self.column.isidentifier():
            if self.column == "":
                raise ValueError(
                    f"The `suppress` method failed: The `column` parameter received an "
                    "empty string `''`. Column names cannot be empty strings. "
                    "Please provide a non-empty, valid Python identifier."
                )
            raise ValueError(
                f"The `suppress` method failed: The `column` name `{self.column}` "
                "is not a valid Python identifier. Column names can only contain "
                "alphanumeric characters and underscores, and cannot begin with a "
                "number. Please provide a valid Python identifier."
            )
        if self.threshold < 0:
            raise ValueError(
                f"The `suppress` method failed: The `threshold` parameter must be "
                f"non-negative, but received `{self.threshold}`. "
                "Please provide a non-negative integer for `threshold`."
            )


# Aliases for convenience
QueryExpr.PrivateSource = PrivateSource
QueryExpr.Rename = Rename
QueryExpr.Filter = Filter
QueryExpr.Select = Select
QueryExpr.Map = Map
QueryExpr.FlatMap = FlatMap
QueryExpr.FlatMapByID = FlatMapByID
QueryExpr.JoinPrivate = JoinPrivate
QueryExpr.JoinPublic = JoinPublic
QueryExpr.ReplaceNullAndNan = ReplaceNullAndNan
QueryExpr.ReplaceInfinity = ReplaceInfinity
QueryExpr.DropNullAndNan = DropNullAndNan
QueryExpr.DropInfinity = DropInfinity
QueryExpr.EnforceConstraint = EnforceConstraint
QueryExpr.GetGroups = GetGroups
QueryExpr.GetBounds = GetBounds
QueryExpr.GroupByCount = GroupByCount
QueryExpr.GroupByCountDistinct = GroupByCountDistinct
QueryExpr.GroupByQuantile = GroupByQuantile
QueryExpr.GroupByBoundedSum = GroupByBoundedSum
QueryExpr.GroupByBoundedAverage = GroupByBoundedAverage
QueryExpr.GroupByBoundedVariance = GroupByBoundedVariance
QueryExpr.GroupByBoundedSTDEV = GroupByBoundedSTDEV
QueryExpr.SuppressAggregates = SuppressAggregates