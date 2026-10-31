"""Query expression for DropInfinity."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Tuple

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.constraints import Constraint

from ._base import (
    QueryExprVisitor,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
)


@dataclass(frozen=True)
class DropInfinity(SingleChildQueryExpr):
    """Returns data with rows that contain +inf/-inf dropped."""

    columns: Tuple[str, ...] = tuple()
    """Columns in which to look for and infinite values.

    If this tuple is empty, *all* columns will be looked at - so if *any* column
    contains an infinite value, that row will be dropped."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
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

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        # DropInfinities can only modify a table by dropping rows, and all current
        # constraints remain valid when rows are dropped, so just return the input
        # constraints unmodified.
        return self.child.constraints(catalog)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_drop_infinity(self)
