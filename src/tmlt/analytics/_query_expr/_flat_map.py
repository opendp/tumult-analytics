"""Query expression for FlatMap."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass, replace
from typing import Any, Callable, List, Optional

from typeguard import check_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema
from tmlt.analytics.constraints import Constraint, MaxGroupsPerID, simplify_constraints

from ._base import (
    QueryExprVisitor,
    Row,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
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
        _validate_child_is_transformation(self.child)
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

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)
        if not self.augment and child_constraints:
            raise AnalyticsInternalError(
                "Non-augmenting flat map applied to table with constraints"
            )
        # Because rows can be duplicated arbitrarily many times by a flat map,
        # MaxRowsPerID and MaxRowsPerGroupPerID constraints cannot be propagated
        # through them; MaxGroupsPerID remains valid when existing rows are
        # duplicated, because this can't add any new groups.
        return simplify_constraints(
            c for c in child_constraints if isinstance(c, MaxGroupsPerID)
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
