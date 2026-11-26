"""Query expression for GetGroups."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Tuple

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema

from ._base import (
    AutoColumnQueryExpr,
    MeasurementQueryExpr,
    QueryExprVisitor,
    validate_child_is_transformation,
)


@dataclass(frozen=True)
class GetGroups(MeasurementQueryExpr):
    """Returns groups based on the geometric partition selection for these columns."""

    columns: Tuple[str, ...]
    """The columns used for geometric partition selection."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        validate_child_is_transformation(self.child)
        check_type(self.columns, Tuple[str, ...])

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        missing_columns = [c for c in self.columns if c not in input_schema]
        if missing_columns:
            raise ValueError(
                f"Columns {missing_columns} passed to get_groups() are not in "
                f"the table. Available columns are {list(input_schema.keys())}."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        return Schema({column: input_schema[column] for column in self.columns})

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_get_groups(self)


@dataclass(frozen=True)
class AutoGetGroups(AutoColumnQueryExpr, MeasurementQueryExpr):
    """Discover groups over all columns other than the privacy ID."""

    def __post_init__(self):
        """Check the child expression."""
        validate_child_is_transformation(self.child)

    def resolve_columns(self, catalog: Catalog) -> GetGroups:
        """Select the non-ID columns for partition selection."""
        schema = self.child.schema(catalog)
        return GetGroups(self.child, tuple(c for c in schema if c != schema.id_column))
