"""Building blocks of the Tumult Analytics query language. Not for direct use.

Defines the :class:`QueryExpr` class, which represents expressions in the Tumult
Analytics query language. QueryExpr and its subclasses should not be directly
constructed; but instead built using a :class:`tmlt.analytics.QueryBuilder`. The
documentation of the :class:`tmlt.analytics.QueryBuilder` provides more information
about the intended semantics of :class:`QueryExpr` objects.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pyspark.sql import DataFrame, SparkSession
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.utils.join import domain_after_join
from typeguard import check_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog, PrivateTable, PublicTable
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    FrozenDict,
    Schema,
    analytics_to_py_types,
    analytics_to_spark_columns_descriptor,
    analytics_to_spark_schema,
    spark_dataframe_domain_to_analytics_columns,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics.config import config
from tmlt.analytics.constraints import Constraint, MaxGroupsPerID, MaxRowsPerGroupPerID
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.truncation_strategy import TruncationStrategy

Row = Dict[str, Any]
"""Type alias for dictionary with string keys."""


class CountMechanism(Enum):
    """Possible mechanisms for the count() aggregation.

    Currently, the
    :meth:`~tmlt.analytics.GroupedQueryBuilder.count` aggregation
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
    :meth:`~tmlt.analytics.GroupedQueryBuilder.count_distinct`
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
    :meth:`~.tmlt.analytics.GroupedQueryBuilder.sum`
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
    :meth:`~tmlt.analytics.GroupedQueryBuilder.average`
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
    :meth:`~tmlt.analytics.GroupedQueryBuilder.variance`
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
    :meth:`~tmlt.analytics.GroupedQueryBuilder.stdev`
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
    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Dispatch methods on a visitor based on the QueryExpr type."""
        raise NotImplementedError()


@dataclass(frozen=True)
class SingleChildQueryExpr(QueryExpr):
    """A QueryExpr that has a single child.

    This is used in the compilation step, to make it easier for rewrite rules to
    automatically recurse along the QueryExpr tree.
    """

    child: QueryExpr
    """The QueryExpr used to generate the input table to this QueryExpr."""


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

    def _validate(self, catalog: Catalog):
        """Validation checks for this QueryExpr."""
        if self.source_id not in catalog.tables:
            raise ValueError(f"Query references nonexistent table '{self.source_id}'")
        if not isinstance(catalog.tables[self.source_id], PrivateTable):
            raise ValueError(
                f"Attempted query on table '{self.source_id}', which is "
                "not a private table."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        self._validate(catalog)
        return catalog.tables[self.source_id].schema

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visits this QueryExpr with visitor."""
        return visitor.visit_private_source(self)


@dataclass(frozen=True)
class GetGroups(SingleChildQueryExpr):
    """Returns groups based on the geometric partition selection for these columns."""

    columns: Tuple[str, ...] = tuple()
    """The columns used for geometric partition selection.

    If empty, will use all of the columns in the table for partition selection.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Tuple[str, ...])

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if self.columns:
            nonexistent_columns = set(self.columns) - set(input_schema)
            if nonexistent_columns:
                raise ValueError(
                    f"Nonexistent columns in get_groups query: {nonexistent_columns}"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        if self.columns:
            return Schema({column: input_schema[column] for column in self.columns})
        return Schema(
            {
                column: input_schema[column]
                for column in input_schema
                if column != input_schema.id_column
            }
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_get_groups(self)


@dataclass(frozen=True)
class GetBounds(SingleChildQueryExpr):
    """Returns approximate upper and lower bounds of a column."""

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

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_get_bounds(self)


@dataclass(frozen=True)
class Rename(SingleChildQueryExpr):
    """Returns the dataframe with columns renamed."""

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

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        nonexistent_columns = set(self.column_mapper) - set(input_schema)
        if nonexistent_columns:
            raise ValueError(
                f"Nonexistent columns in rename query: {nonexistent_columns}"
            )
        for old, new in self.column_mapper.items():
            if new in input_schema and new != old:
                raise ValueError(
                    f"Cannot rename '{old}' to '{new}': column '{new}' already exists"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        grouping_column = input_schema.grouping_column
        if grouping_column in self.column_mapper:
            grouping_column = self.column_mapper[grouping_column]

        id_column = input_schema.id_column
        if id_column in self.column_mapper:
            id_column = self.column_mapper[id_column]

        return Schema(
            {
                self.column_mapper.get(column, column): input_schema[column]
                for column in input_schema
            },
            grouping_column=grouping_column,
            id_column=id_column,
            id_space=input_schema.id_space,
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_rename(self)


@dataclass(frozen=True)
class Filter(SingleChildQueryExpr):
    """Returns the subset of the rows that satisfy the condition."""

    condition: str
    """A string of SQL expression specifying the filter to apply to the data.

    For example, the string "A > B" matches rows where column A is greater than
    column B.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.condition, str)

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame(
            [], schema=analytics_to_spark_schema(input_schema)
        )
        try:
            test_df.filter(self.condition)
        except Exception as e:
            raise ValueError(f"Invalid filter condition '{self.condition}': {e}") from e

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)
        return input_schema

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_filter(self)


@dataclass(frozen=True)
class Select(SingleChildQueryExpr):
    """Returns a subset of the columns."""

    columns: Tuple[str, ...]
    """The columns to select."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Tuple[str, ...])
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(f"Column name appears more than once in {self.columns}")

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        grouping_column = input_schema.grouping_column
        id_column = input_schema.id_column
        if grouping_column is not None and grouping_column not in self.columns:
            raise ValueError(
                f"Grouping column '{grouping_column}' may not "
                "be dropped by select query"
            )
        if id_column is not None and id_column not in self.columns:
            raise ValueError(
                f"ID column '{id_column}' may not be dropped by select query"
            )
        nonexistent_columns = set(self.columns) - set(input_schema)
        if nonexistent_columns:
            raise ValueError(
                f"Nonexistent columns in select query: {nonexistent_columns}"
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)
        return Schema(
            {column: input_schema[column] for column in self.columns},
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_select(self)


@dataclass(frozen=True)
class Map(SingleChildQueryExpr):
    """Applies a map function to each row of a relation."""

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

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        new_columns = self.schema_new_columns.column_descs
        if self.augment:
            overlapping_columns = set(input_schema.keys()) & set(new_columns.keys())
            if overlapping_columns:
                raise ValueError(
                    "New columns in augmenting map must not overwrite "
                    "existing columns, but found new columns that "
                    f"already exist: {', '.join(overlapping_columns)}"
                )
            return
        if input_schema.grouping_column:
            raise ValueError(
                "Map must set augment=True to ensure that "
                f"grouping column '{input_schema.grouping_column}' is not lost."
            )
        if input_schema.id_column:
            raise ValueError(
                "Map must set augment=True to ensure that "
                f"ID column '{input_schema.id_column}' is not lost."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)
        new_columns = self.schema_new_columns.column_descs
        # Any column created by Map could contain a null value
        for name in list(new_columns.keys()):
            new_columns[name] = replace(new_columns[name], allow_null=True)

        if self.augment:
            return Schema(
                {**input_schema, **new_columns},
                grouping_column=input_schema.grouping_column,
                id_column=input_schema.id_column,
                id_space=input_schema.id_space,
            )
        # If augment=False, there is no grouping column nor ID column
        return Schema(new_columns)

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
class FlatMap(SingleChildQueryExpr):
    """Applies a flat map function to each row of a relation."""

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

    def _validate(self, input_schema):
        """Validation checks for this QueryExpr."""
        if self.schema_new_columns.grouping_column is not None:
            if input_schema.grouping_column:
                raise ValueError(
                    "Multiple grouping transformations are used in this query. "
                    "Only one grouping transformation is allowed."
                )
            if input_schema.id_column:
                raise ValueError(
                    "Grouping flat map cannot be used on tables with "
                    "the AddRowsWithID protected change."
                )

        new_columns = self.schema_new_columns.column_descs
        if self.augment:
            overlapping_columns = set(input_schema.keys()) & set(new_columns.keys())
            if overlapping_columns:
                raise ValueError(
                    "New columns in augmenting map must not overwrite "
                    "existing columns, but found new columns that "
                    f"already exist: {', '.join(overlapping_columns)}"
                )
            return
        if input_schema.grouping_column:
            raise ValueError(
                "Flat map must set augment=True to ensure that "
                f"grouping column '{input_schema.grouping_column}' is not lost."
            )
        if input_schema.id_column:
            raise ValueError(
                "Flat map must set augment=True to ensure that "
                f"ID column '{input_schema.id_column}' is not lost."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        grouping_column = (
            self.schema_new_columns.grouping_column
            if self.schema_new_columns.grouping_column is not None
            else input_schema.grouping_column
        )
        new_columns = self.schema_new_columns.column_descs
        # Any column created by the FlatMap could contain a null value
        for name in list(new_columns.keys()):
            new_columns[name] = replace(new_columns[name], allow_null=True)

        if self.augment:
            return Schema(
                {**input_schema, **new_columns},
                grouping_column=grouping_column,
                id_column=input_schema.id_column,
                id_space=input_schema.id_space,
            )
        # If augment=False, there is no grouping column nor ID column
        return Schema(new_columns)

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
class FlatMapByID(SingleChildQueryExpr):
    """Applies a flat map function to each group of rows with a common ID."""

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

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if not input_schema.id_column:
            raise ValueError(
                "Flat-map-by-ID may only be used on tables with ID columns."
            )
        if input_schema.grouping_column:
            raise AnalyticsInternalError(
                "Encountered table with both an ID column and a grouping column."
            )
        if input_schema.id_column in self.schema_new_columns.column_descs:
            raise ValueError(
                "Flat-map-by-ID mapping function output cannot include ID column."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        id_column = input_schema.id_column
        new_columns = self.schema_new_columns.column_descs

        for name in list(new_columns.keys()):
            new_columns[name] = replace(new_columns[name], allow_null=True)
        return Schema(
            {id_column: input_schema[id_column], **new_columns},
            id_column=id_column,
            id_space=input_schema.id_space,
        )

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


@dataclass(frozen=True)
class JoinPrivate(QueryExpr):
    """Returns the join of two private tables.

    Before performing the join, each table is truncated based on the corresponding
    :class:`~tmlt.analytics.TruncationStrategy`.  For a more
    detailed overview of ``JoinPrivate``'s behavior, see
    :meth:`~tmlt.analytics.QueryBuilder.join_private`.
    """

    left_child: QueryExpr
    """The QueryExpr to join with right operand."""
    right_child: QueryExpr
    """The QueryExpr for private source to join with."""
    truncation_strategy_left: Optional[TruncationStrategy.Type] = None
    """Truncation strategy to be used for the left table."""
    truncation_strategy_right: Optional[TruncationStrategy.Type] = None
    """Truncation strategy to be used for the right table."""
    join_columns: Optional[Tuple[str, ...]] = None
    """The columns used for joining the tables, or None to use all common columns."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.left_child, QueryExpr)
        check_type(self.right_child, QueryExpr)
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

    def _validate(self, left_schema: Schema, right_schema: Schema):
        """Validation checks for this QueryExpr."""
        if left_schema.id_column != right_schema.id_column:
            if left_schema.id_column is None or right_schema.id_column is None:
                raise ValueError(
                    "Private joins can only be performed between two tables "
                    "with the same type of protected change"
                )
            raise ValueError(
                "Private joins between tables with the AddRowsWithID "
                "protected change are only possible when the ID columns of "
                "the two tables have the same name"
            )
        if left_schema.id_space != right_schema.id_space:
            raise ValueError(
                "Private joins between tables with the AddRowsWithID protected change"
                " are only possible when both tables are in the same ID space"
            )
        _validate_join(left_schema, right_schema, self.join_columns)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr.

        The ordering of output columns are:

        1. The join columns
        2. Columns that are only in the left table
        3. Columns that are only in the right table
        4. Columns that are in both tables, but not included in the join columns. These
           columns are included with _left and _right suffixes.
        """
        left_schema = self.left_child.schema(catalog)
        right_schema = self.right_child.schema(catalog)
        self._validate(left_schema, right_schema)
        return _schema_for_join(
            left_schema=left_schema,
            right_schema=right_schema,
            join_columns=self.join_columns,
            join_id_space=left_schema.id_space,
            how="inner",
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_join_private(self)


@dataclass(frozen=True)
class JoinPublic(SingleChildQueryExpr):
    """Returns the join of a private and public table."""

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

    def _validate(self, catalog: Catalog, left_schema: Schema, right_schema: Schema):
        """Validation checks for this QueryExpr."""
        if isinstance(self.public_table, str):
            if not isinstance(catalog.tables[self.public_table], PublicTable):
                raise ValueError(
                    f"Attempted public join on table '{self.public_table}', "
                    "which is not a public table"
                )
        _validate_join(left_schema, right_schema, self.join_columns)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr.

        Has analogous behavior to :meth:`JoinPrivate.schema`, where the private
        table is the left table.
        """
        input_schema = self.child.schema(catalog)
        if isinstance(self.public_table, str):
            right_schema = catalog.tables[self.public_table].schema
        else:
            right_schema = Schema(
                spark_schema_to_analytics_columns(self.public_table.schema)
            )
        self._validate(catalog, input_schema, right_schema)
        return _schema_for_join(
            left_schema=input_schema,
            right_schema=right_schema,
            join_columns=self.join_columns,
            join_id_space=input_schema.id_space,
            how=self.how,
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
                if not self_table.equals(other_table):
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
class ReplaceNullAndNan(SingleChildQueryExpr):
    """Returns data with null and NaN expressions replaced by a default.

    .. warning::
        after a ``ReplaceNullAndNan`` query has been performed for a column,
        Tumult Analytics will raise an error if you use a
        :class:`~.tmlt.analytics.KeySet` for that column
        that contains null values.
    """

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

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if (
            input_schema.grouping_column
            and input_schema.grouping_column in self.replace_with
        ):
            raise ValueError(
                "Cannot replace null values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column."
            )
        if input_schema.id_column and input_schema.id_column in self.replace_with:
            raise ValueError(
                f"Cannot replace null values in column '{input_schema.id_column}', "
                "as it is an ID column."
            )
        if input_schema.id_column and (len(self.replace_with) == 0):
            warnings.warn(
                f"Replacing null values in the ID column '{input_schema.id_column}' "
                "is not allowed, so the ID column may still contain null values.",
                RuntimeWarning,
            )

        pytypes = analytics_to_py_types(input_schema)
        for col, val in self.replace_with.items():
            if col not in input_schema.keys():
                raise ValueError(
                    f"Column '{col}' does not exist in this table, "
                    f"available columns are {list(input_schema.keys())}"
                )
            if not isinstance(val, pytypes[col]):
                # Using an int as a float is OK
                if not (isinstance(val, int) and pytypes[col] == float):
                    raise ValueError(
                        f"Column '{col}' cannot have nulls replaced with "
                        f"{repr(val)}, as that value's type does not match the "
                        f"column type {input_schema[col].column_type.name}"
                    )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        columns_to_change = list(dict(self.replace_with).keys())
        if len(columns_to_change) == 0:
            columns_to_change = [
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan)
                and name not in [input_schema.grouping_column, input_schema.id_column]
            ]
        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=(cd.allow_null and name not in columns_to_change),
                    allow_nan=(cd.allow_nan and name not in columns_to_change),
                    allow_inf=cd.allow_inf,
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_replace_null_and_nan(self)


@dataclass(frozen=True)
class ReplaceInfinity(SingleChildQueryExpr):
    """Returns data with +inf and -inf expressions replaced by defaults."""

    replace_with: FrozenDict = FrozenDict.from_dict({})
    """New values to replace with, by column. The first value for each column
    will be used to replace -infinity, and the second value will be used to
    replace +infinity.

    If this dictionary is empty, *all* columns of type DECIMAL will be changed,
    with infinite values replaced with a default value (see the
    :class:`~.AnalyticsDefault` class variables).
    """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.replace_with, FrozenDict)
        check_type(dict(self.replace_with), Dict[str, Tuple[float, float]])

        # Ensure that the values in replace_with are tuples of floats
        updated_dict = {}
        for col, val in self.replace_with.items():
            updated_dict[col] = (float(val[0]), float(val[1]))

        # Subverting the frozen dataclass to update the replace_with attribute
        object.__setattr__(self, "replace_with", FrozenDict.from_dict(updated_dict))

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if (
            input_schema.grouping_column
            and input_schema.grouping_column in self.replace_with
        ):
            raise ValueError(
                "Cannot replace infinite values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column"
            )
        if input_schema.id_column and input_schema.id_column in self.replace_with:
            raise ValueError(
                f"Cannot replace infinite values in column '{input_schema.id_column}', "
                "as it is an ID column"
            )

        for name in self.replace_with:
            if name not in input_schema.keys():
                raise ValueError(
                    f"Column '{name}' does not exist in this table, "
                    f"available columns are {list(input_schema.keys())}"
                )
            if input_schema[name].column_type != ColumnType.DECIMAL:
                raise ValueError(
                    f"Column '{name}' has a replacement value provided, but it is "
                    f"of type {input_schema[name].column_type.name} (not "
                    f"{ColumnType.DECIMAL.name}) and so cannot "
                    "contain infinite values"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        columns_to_change = list(self.replace_with.keys())
        if len(columns_to_change) == 0:
            columns_to_change = [
                name
                for name, cd in input_schema.column_descs.items()
                if cd.column_type == ColumnType.DECIMAL
                and cd.allow_inf
                and name not in [input_schema.grouping_column, input_schema.id_column]
            ]
        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=cd.allow_null,
                    allow_nan=cd.allow_nan,
                    allow_inf=(cd.allow_inf and name not in columns_to_change),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_replace_infinity(self)


@dataclass(frozen=True)
class DropNullAndNan(SingleChildQueryExpr):
    """Returns data with rows that contain null or NaN value dropped.

    .. warning::
        After a ``DropNullAndNan`` query has been performed for a column,
        Tumult Analytics will raise an error if you use a
        :class:`~.tmlt.analytics.KeySet` for that column
        that contains null values.
    """

    columns: Tuple[str, ...] = tuple()
    """Columns in which to look for nulls and NaNs.

    If this tuple is empty, *all* columns will be looked at - so if *any* column
    contains a null or NaN value that row will be dropped."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Tuple[str, ...])

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if (
            input_schema.grouping_column
            and input_schema.grouping_column in self.columns
        ):
            raise ValueError(
                f"Cannot drop null values in column '{input_schema.grouping_column}', "
                "as it is a grouping column"
            )
        if input_schema.id_column and input_schema.id_column in self.columns:
            raise ValueError(
                f"Cannot drop null values in column '{input_schema.id_column}', "
                "as it is an ID column."
            )
        if input_schema.id_column and len(self.columns) == 0:
            warnings.warn(
                f"Replacing null values in the ID column '{input_schema.id_column}' "
                "is not allowed, so the ID column may still contain null values.",
                RuntimeWarning,
            )
        for name in self.columns:
            if name not in input_schema.keys():
                raise ValueError(
                    f"Column '{name}' does not exist in this table, "
                    f"available columns are {list(input_schema.keys())}"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        columns = self.columns
        if len(columns) == 0:
            columns = tuple(
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan)
                and name not in [input_schema.grouping_column, input_schema.id_column]
            )

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=(cd.allow_null and name not in columns),
                    allow_nan=(cd.allow_nan and name not in columns),
                    allow_inf=(cd.allow_inf),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_drop_null_and_nan(self)


@dataclass(frozen=True)
class DropInfinity(SingleChildQueryExpr):
    """Returns data with rows that contain +inf/-inf dropped."""

    columns: Tuple[str, ...] = tuple()
    """Columns in which to look for and infinite values.

    If this tuple is empty, *all* columns will be looked at - so if *any* column
    contains an infinite value, that row will be dropped."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        check_type(self.columns, Tuple[str, ...])

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if (
            input_schema.grouping_column
            and input_schema.grouping_column in self.columns
        ):
            raise ValueError(
                "Cannot drop infinite values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column"
            )
        # Float-valued columns cannot be ID columns, but include this to be safe.
        if input_schema.id_column and input_schema.id_column in self.columns:
            raise ValueError(
                f"Cannot drop infinite values in column '{input_schema.id_column}', "
                "as it is an ID column"
            )
        for name in self.columns:
            if name not in input_schema.keys():
                raise ValueError(
                    f"Column '{name}' does not exist in this table, "
                    f"available columns are {list(input_schema.keys())}"
                )
            if input_schema[name].column_type != ColumnType.DECIMAL:
                raise ValueError(
                    f"Column '{name}' was given as a column to drop "
                    "infinite values from, but it is of type"
                    f"{input_schema[name].column_type.name} (not "
                    f"{ColumnType.DECIMAL.name}) and so cannot "
                    "contain infinite values"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExp."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        columns = self.columns
        if len(columns) == 0:
            columns = tuple(
                name
                for name, cd in input_schema.column_descs.items()
                if cd.column_type == ColumnType.DECIMAL
                and cd.allow_inf
                and name not in (input_schema.grouping_column, input_schema.id_column)
            )

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=cd.allow_null,
                    allow_nan=cd.allow_nan,
                    allow_inf=(cd.allow_inf and name not in columns),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_drop_infinity(self)


@dataclass(frozen=True)
class EnforceConstraint(SingleChildQueryExpr):
    """Enforces a constraint on the data."""

    constraint: Constraint
    """A constraint to be enforced."""

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if not input_schema.id_column:
            raise ValueError(
                f"Constraint {self.constraint} can only be applied to tables"
                " with the AddRowsWithID protected change"
            )
        if isinstance(self.constraint, (MaxGroupsPerID, MaxRowsPerGroupPerID)):
            grouping_column = self.constraint.grouping_column
            if grouping_column not in input_schema:
                raise ValueError(
                    f"The grouping column of constraint {self.constraint}"
                    " does not exist in this table; available columns"
                    f" are: {', '.join(input_schema.keys())}"
                )
            if grouping_column == input_schema.id_column:
                raise ValueError(
                    f"The grouping column of constraint {self.constraint} cannot be"
                    " the ID column of the table it is applied to"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)
        return input_schema

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_enforce_constraint(self)


def _validate_groupby(
    query: Union[
        "GetBounds",
        "GroupByBoundedAverage",
        "GroupByBoundedStdev",
        "GroupByBoundedSum",
        "GroupByBoundedVariance",
        "GroupByCount",
        "GroupByCountDistinct",
        "GroupByQuantile",
    ],
    input_schema: Schema,
):
    """Validates the arguments of a group-by QueryExpr."""
    # Validating group-by columns
    if isinstance(query.groupby_keys, KeySet):
        # Checks that the KeySet is valid
        schema = query.groupby_keys.schema()
        groupby_columns: Collection[str] = schema.keys()

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
        for col in query.groupby_keys:
            if col not in input_schema:
                raise ValueError(f"Groupby column '{col}' is not in the input schema.")
        groupby_columns = query.groupby_keys
    else:
        raise AnalyticsInternalError(
            f"Unexpected groupby_keys type: {type(query.groupby_keys)}."
        )

    # Validating compatibility between grouping columns and group-by columns
    grouping_column = input_schema.grouping_column
    if grouping_column is not None and grouping_column not in groupby_columns:
        raise ValueError(
            f"Column '{grouping_column}' produced by grouping transformation "
            f"is not in groupby columns {list(groupby_columns)}."
        )
    if (
        not isinstance(query, (GroupByCount, GroupByCountDistinct))
        and query.measure_column in groupby_columns
    ):
        raise ValueError(
            "Column to aggregate must be a non-grouped column, not "
            f"'{query.measure_column}'."
        )

    # Validating the measure column
    if isinstance(
        query,
        (
            GetBounds,
            GroupByQuantile,
            GroupByBoundedSum,
            GroupByBoundedStdev,
            GroupByBoundedAverage,
            GroupByBoundedVariance,
        ),
    ):
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
    query: Union[
        "GetBounds",
        "GroupByBoundedAverage",
        "GroupByBoundedStdev",
        "GroupByBoundedSum",
        "GroupByBoundedVariance",
        "GroupByCount",
        "GroupByCountDistinct",
        "GroupByQuantile",
    ],
    input_schema: Schema,
) -> Schema:
    """Returns the schema of a group-by QueryExpr."""
    groupby_columns = (
        query.groupby_keys.schema().keys()
        if isinstance(query.groupby_keys, KeySet)
        else query.groupby_keys
    )

    # Determining the output column types & names
    if isinstance(query, (GroupByCount, GroupByCountDistinct)):
        output_column_type = ColumnType.INTEGER
    elif isinstance(query, (GetBounds, GroupByBoundedSum)):
        output_column_type = input_schema[query.measure_column].column_type
    elif isinstance(
        query,
        (
            GroupByQuantile,
            GroupByBoundedSum,
            GroupByBoundedStdev,
            GroupByBoundedAverage,
            GroupByBoundedVariance,
        ),
    ):
        output_column_type = ColumnType.DECIMAL
    else:
        raise AnalyticsInternalError(f"Unexpected QueryExpr type: {type(query)}.")
    if isinstance(query, GetBounds):
        output_columns = {
            query.lower_bound_column: ColumnDescriptor(
                output_column_type, allow_null=False
            ),
            query.upper_bound_column: ColumnDescriptor(
                output_column_type, allow_null=False
            ),
        }
    else:
        output_columns = {
            query.output_column: ColumnDescriptor(output_column_type, allow_null=False),
        }

    return Schema(
        {
            **{column: input_schema[column] for column in groupby_columns},
            **output_columns,
        },
        grouping_column=None,
        id_column=None,
    )


@dataclass(frozen=True)
class GroupByCount(SingleChildQueryExpr):
    """Returns the count of each combination of the groupby domains."""

    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    output_column: str = "count"
    """The name of the column to store the counts in."""
    mechanism: CountMechanism = CountMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an
    appropriate mechanism.
    """
    core_mechanism: Optional[NoiseMechanism] = None
    """The Core mechanism used for this aggregation. Specified during compilation."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.output_column, str)
        check_type(self.mechanism, CountMechanism)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_count(self)


@dataclass(frozen=True)
class GroupByCountDistinct(SingleChildQueryExpr):
    """Returns the count of distinct rows in each groupby domain value."""

    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    columns_to_count: Tuple[str, ...] = tuple()
    """The columns that are compared when determining if two rows are distinct.

    If empty, will count all distinct rows.
    """
    output_column: str = "count_distinct"
    """The name of the column to store the distinct counts in."""
    mechanism: CountDistinctMechanism = CountDistinctMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an appropriate mechanism.
    """
    core_mechanism: Optional[NoiseMechanism] = None
    """The Core mechanism used for this aggregation. Specified during compilation."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.child, QueryExpr)
        check_type(self.columns_to_count, Tuple[str, ...])
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.output_column, str)
        check_type(self.mechanism, CountDistinctMechanism)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_count_distinct(self)


@dataclass(frozen=True)
class GroupByQuantile(SingleChildQueryExpr):
    """Returns the quantile of a column for each combination of the groupby domains.

    If the column to be measured contains null, NaN, or positive or negative infinity,
    those values will be dropped (as if dropped explicitly via
    :class:`DropNullAndNan` and :class:`DropInfinity`) before the quantile is
    calculated.
    """

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
        if type(self.low) != type(self.high):
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_quantile(self)


@dataclass(frozen=True)
class GroupByBoundedSum(SingleChildQueryExpr):
    """Returns the bounded sum of a column for each combination of groupby domains."""

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
    core_mechanism: Optional[NoiseMechanism] = None
    """The Core mechanism used for this aggregation. Specified during compilation."""

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

        if type(self.low) != type(self.high):
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_sum(self)


@dataclass(frozen=True)
class GroupByBoundedAverage(SingleChildQueryExpr):
    """Returns bounded average of a column for each combination of groupby domains."""

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
    core_mechanism: Optional[NoiseMechanism] = None
    """The Core mechanism used for this aggregation. Specified during compilation."""

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

        if type(self.low) != type(self.high):
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_average(self)


@dataclass(frozen=True)
class GroupByBoundedVariance(SingleChildQueryExpr):
    """Returns bounded variance of a column for each combination of groupby domains."""

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
    core_mechanism: Optional[NoiseMechanism] = None
    """The Core mechanism used for this aggregation. Specified during compilation."""

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

        if type(self.low) != type(self.high):
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_variance(self)


@dataclass(frozen=True)
class GroupByBoundedStdev(SingleChildQueryExpr):
    """Returns bounded stdev of a column for each combination of groupby domains."""

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
    core_mechanism: Optional[NoiseMechanism] = None
    """The Core mechanism used for this aggregation. Specified during compilation."""

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

        if type(self.low) != type(self.high):
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))

        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        return _schema_for_groupby(self, input_schema)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_stdev(self)


@dataclass(frozen=True)
class SuppressAggregates(SingleChildQueryExpr):
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

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        return self.child.schema(catalog)

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
    def visit_groupby_bounded_stdev(self, expr: GroupByBoundedStdev) -> Any:
        """Visit a :class:`GroupByBoundedStdev`."""
        raise NotImplementedError

    @abstractmethod
    def visit_suppress_aggregates(self, expr: SuppressAggregates) -> Any:
        """Visit a :class:`SuppressAggregates`."""
        raise NotImplementedError
