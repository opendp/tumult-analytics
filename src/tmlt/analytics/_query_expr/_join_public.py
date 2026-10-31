"""Query expression for JoinPublic."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast

import pandas as pd
from pyspark.sql import DataFrame
from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import Schema, spark_schema_to_analytics_columns
from tmlt.analytics.constraints import Constraint, simplify_constraints

from ._base import (
    QueryExprVisitor,
    SingleChildQueryExpr,
    _propagate_join_by_stability,
    _schema_for_join,
    _validate_child_is_transformation,
    _validate_join,
)


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
        _validate_child_is_transformation(self.child)
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
        _validate_join(left_schema, right_schema, self.join_columns)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr.

        Has analogous behavior to :meth:`JoinPrivate.schema`, where the private
        table is the left table.
        """
        input_schema = self.child.schema(catalog)
        if isinstance(self.public_table, str):
            if self.public_table not in catalog.public_tables:
                raise ValueError(
                    f"Attempted public join on table '{self.public_table}', "
                    "which is not a public table"
                )
            right_schema = catalog.public_tables[self.public_table].schema
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

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)

        if isinstance(self.public_table, str):
            if self.public_table not in catalog.public_tables:
                raise ValueError(
                    f"Attempted public join on table '{self.public_table}', "
                    "which is not a public table"
                )
            public_df = catalog.public_tables[self.public_table].dataframe
            right_schema = catalog.public_tables[self.public_table].schema
        else:
            public_df = self.public_table
            right_schema = Schema(
                spark_schema_to_analytics_columns(self.public_table.schema)
            )

        input_schema = self.child.schema(catalog)
        common_cols = set(input_schema) & set(right_schema)
        join_cols = set(self.join_columns or tuple(common_cols))
        overlapping_cols = common_cols - join_cols

        count_df = cast(
            pd.DataFrame,
            public_df.select(*join_cols)
            .groupby(*join_cols)
            .count()
            .select("count")
            .toPandas(),
        )
        # The maximum number of matching public rows for any join key bounds the
        # number of times each private row can appear in the join output.
        join_stability = max(count_df["count"].to_list(), default=0)

        propagated = [
            _propagate_join_by_stability(
                c, join_stability, overlapping_cols, suffix="_left"
            )
            for c in child_constraints
        ]
        return simplify_constraints(c for c in propagated if c is not None)

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
