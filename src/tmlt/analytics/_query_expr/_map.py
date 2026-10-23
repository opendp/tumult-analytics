"""Query expression for Map."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass, replace
from typing import Any, Callable

from typeguard import check_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema
from tmlt.analytics.constraints import Constraint

from ._base import (
    QueryExprVisitor,
    Row,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
)


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
        _validate_child_is_transformation(self.child)
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

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)
        if not self.augment and child_constraints:
            raise AnalyticsInternalError(
                "Non-augmenting map applied to table with constraints"
            )
        # Map can only add columns to the existing rows, so existing constraints
        # remain valid.
        return child_constraints

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
