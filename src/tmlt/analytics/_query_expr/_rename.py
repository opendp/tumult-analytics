"""Query expression for Rename."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Dict, List

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import FrozenDict, Schema
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    simplify_constraints,
)

from ._base import (
    QueryExprVisitor,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
)


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
        _validate_child_is_transformation(self.child)
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

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)
        # Renames do not change the rows in a table, so all constraints remain
        # valid after updating references to renamed grouping columns.
        result: List[Constraint] = []
        for c in child_constraints:
            if isinstance(c, MaxRowsPerID):
                result.append(c)
            elif isinstance(c, MaxGroupsPerID):
                result.append(
                    MaxGroupsPerID(
                        self.column_mapper.get(c.grouping_column, c.grouping_column),
                        c.max,
                    )
                )
            elif isinstance(c, MaxRowsPerGroupPerID):
                result.append(
                    MaxRowsPerGroupPerID(
                        self.column_mapper.get(c.grouping_column, c.grouping_column),
                        c.max,
                    )
                )
        return simplify_constraints(result)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_rename(self)
