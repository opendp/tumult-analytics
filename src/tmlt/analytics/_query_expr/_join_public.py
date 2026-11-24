"""Query expression for JoinPublic."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass, field
from typing import Any, Tuple, Union, cast

import pandas as pd
from pyspark.sql import DataFrame
from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import Schema, spark_schema_to_analytics_columns
from tmlt.analytics.constraints import Constraint, simplify_constraints

from ._base import (
    AutoColumnQueryExpr,
    QueryExprVisitor,
    SingleChildQueryExpr,
    common_join_columns,
    propagate_join_by_stability,
    schema_for_join,
    validate_child_is_transformation,
    validate_join,
)


@dataclass(frozen=True, eq=False)
class _JoinPublicBase(SingleChildQueryExpr):
    """Shared arguments for public joins."""

    public_table: Union[DataFrame, str]
    """A DataFrame or public source to join with."""
    how: str = field(default="inner", kw_only=True)
    """The type of join to perform. Must be either "inner" or "left"."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        validate_child_is_transformation(self.child)
        check_type(self.public_table, Union[DataFrame, str])
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

    def _public_schema(self, catalog: Catalog) -> Schema:
        """Return the schema of the table on the public side of the join."""
        if isinstance(self.public_table, str):
            if self.public_table not in catalog.public_tables:
                raise ValueError(
                    f"Attempted public join on table '{self.public_table}', "
                    "which is not a public table"
                )
            return catalog.public_tables[self.public_table].schema
        return Schema(spark_schema_to_analytics_columns(self.public_table.schema))

    def __eq__(self, other: object) -> bool:
        """Compare public tables by contents, independent of row order."""
        if not isinstance(other, _JoinPublicBase) or type(self) is not type(other):
            return False
        if self.child != other.child or self.how != other.how:
            return False
        if isinstance(self.public_table, str):
            return self.public_table == other.public_table
        if not isinstance(other.public_table, DataFrame):
            return False
        # TODO(#2107): Fix typing once Pandas has working type stubs
        self_table = self.public_table.toPandas()
        other_table = other.public_table.toPandas()
        if len(self_table) != len(other_table):
            return False
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
            return bool(self_table.equals(other_table))
        return True

    def __hash__(self) -> int:
        """Hash without collecting dataframe contents for comparison."""
        public_name = self.public_table if isinstance(self.public_table, str) else None
        return hash((type(self), self.child, public_name, self.how))


@dataclass(frozen=True, eq=False)
class JoinPublic(_JoinPublicBase):
    """Join a private and public table using explicit columns."""

    join_columns: Tuple[str, ...]
    """The columns used for joining the tables."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        super().__post_init__()
        check_type(self.join_columns, Tuple[str, ...])
        if not self.join_columns:
            raise ValueError("Provided join columns must not be empty")
        if len(self.join_columns) != len(set(self.join_columns)):
            raise ValueError("Join columns must be distinct")

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr.

        Has analogous behavior to :meth:`JoinPrivate.schema`, where the private
        table is the left table.
        """
        input_schema = self.child.schema(catalog)
        right_schema = self._public_schema(catalog)
        validate_join(input_schema, right_schema, self.join_columns)
        return schema_for_join(
            left_schema=input_schema,
            right_schema=right_schema,
            join_columns=self.join_columns,
            join_id_space=input_schema.id_space,
            how=self.how,
        )

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)

        right_schema = self._public_schema(catalog)
        public_df = (
            catalog.public_tables[self.public_table].dataframe
            if isinstance(self.public_table, str)
            else self.public_table
        )

        input_schema = self.child.schema(catalog)
        common_cols = set(input_schema) & set(right_schema)
        overlapping_cols = common_cols - set(self.join_columns)

        count_df = cast(
            pd.DataFrame,
            public_df.select(*self.join_columns)
            .groupby(*self.join_columns)
            .count()
            .select("count")
            .toPandas(),
        )
        # The maximum number of matching public rows for any join key bounds the
        # number of times each private row can appear in the join output.
        join_stability = max(count_df["count"].to_list(), default=0)

        propagated = [
            propagate_join_by_stability(
                c, join_stability, overlapping_cols, suffix="_left"
            )
            for c in child_constraints
        ]
        return simplify_constraints(c for c in propagated if c is not None)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_join_public(self)

    def __eq__(self, other: object) -> bool:
        """Compare join columns as well as the input tables and join type."""
        return (
            isinstance(other, JoinPublic)
            and self.join_columns == other.join_columns
            and super().__eq__(other)
        )

    def __hash__(self) -> int:
        """Include the explicit columns in the hash."""
        return hash((super().__hash__(), self.join_columns))


@dataclass(frozen=True, eq=False)
class AutoJoinPublic(AutoColumnQueryExpr, _JoinPublicBase):
    """Join a private and public table on their common columns."""

    def resolve_columns(self, catalog: Catalog) -> JoinPublic:
        """Select all common join columns."""
        return JoinPublic(
            self.child,
            self.public_table,
            common_join_columns(
                self.child.schema(catalog), self._public_schema(catalog)
            ),
            how=self.how,
        )
