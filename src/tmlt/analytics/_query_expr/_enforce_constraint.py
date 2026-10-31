"""Query expression for EnforceConstraint."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any

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
class EnforceConstraint(SingleChildQueryExpr):
    """Enforces a constraint on the data."""

    constraint: Constraint
    """A constraint to be enforced."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
        check_type(self.constraint, Constraint)

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

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)
        return simplify_constraints(child_constraints | frozenset([self.constraint]))

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_enforce_constraint(self)
