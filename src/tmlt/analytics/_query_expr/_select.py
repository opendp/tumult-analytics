"""Query expression for Select."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Tuple

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    simplify_constraints,
)

from ._base import (
    QueryExprVisitor,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
)


@dataclass(frozen=True)
class Select(SingleChildQueryExpr):
    """Returns a subset of the columns."""

    columns: Tuple[str, ...]
    """The columns to select."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
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

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)
        # Select does not change the rows in a table, so row constraints remain
        # valid. Constraints that refer to grouping columns not present in the
        # output schema are dropped.
        return simplify_constraints(
            c
            for c in child_constraints
            if not (
                isinstance(c, (MaxGroupsPerID, MaxRowsPerGroupPerID))
                and c.grouping_column not in self.columns
            )
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_select(self)
