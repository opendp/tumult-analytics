"""Query expression for GetGroups."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Tuple

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema

from ._base import (
    MeasurementQueryExpr,
    QueryExprVisitor,
    _validate_child_is_transformation,
)


@dataclass(frozen=True)
class GetGroups(MeasurementQueryExpr):
    """Returns groups based on the geometric partition selection for these columns."""

    columns: Tuple[str, ...] = tuple()
    """The columns used for geometric partition selection.

    If empty, will use all of the columns in the table for partition selection.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
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
