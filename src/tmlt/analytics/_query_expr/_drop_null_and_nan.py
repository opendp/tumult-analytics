"""Query expression for DropNullAndNan."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

import warnings
from dataclasses import dataclass
from typing import Any, Tuple

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import ColumnDescriptor, Schema
from tmlt.analytics.constraints import Constraint

from ._base import (
    AutoColumnQueryExpr,
    QueryExprVisitor,
    SingleChildQueryExpr,
    validate_child_is_transformation,
)


@dataclass(frozen=True)
class DropNullAndNan(SingleChildQueryExpr):
    """Returns data with rows that contain null or NaN value dropped.

    .. warning::
        After a ``DropNullAndNan`` query has been performed for a column,
        Tumult Analytics will raise an error if you use a
        :class:`~.tmlt.analytics.KeySet` for that column
        that contains null values.
    """

    columns: Tuple[str, ...]
    """Columns in which to look for nulls and NaNs. An empty tuple is a no-op."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        validate_child_is_transformation(self.child)
        check_type(self.columns, Tuple[str, ...])

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if (
            input_schema.grouping_column
            and input_schema.grouping_column in self.columns
        ):
            raise ValueError(
                f"Cannot drop null values in column '{input_schema.grouping_column}', "
                "as it is a grouping column"
            )
        if input_schema.id_column and input_schema.id_column in self.columns:
            raise ValueError(
                f"Cannot drop null values in column '{input_schema.id_column}', "
                "as it is an ID column."
            )
        for name in self.columns:
            if name not in input_schema.keys():
                raise ValueError(
                    f"Column '{name}' does not exist in this table, "
                    f"available columns are {list(input_schema.keys())}"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=(cd.allow_null and name not in self.columns),
                    allow_nan=(cd.allow_nan and name not in self.columns),
                    allow_inf=(cd.allow_inf),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        # DropNullAndNans can only modify a table by dropping rows, and all
        # current constraints remain valid when rows are dropped, so just return
        # the input constraints unmodified.
        return self.child.constraints(catalog)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_drop_null_and_nan(self)


@dataclass(frozen=True)
class AutoDropNullAndNan(AutoColumnQueryExpr, SingleChildQueryExpr):
    """Drop rows with nulls or NaNs in eligible columns."""

    def __post_init__(self):
        """Check the child expression."""
        validate_child_is_transformation(self.child)

    def resolve_columns(self, catalog: Catalog) -> DropNullAndNan:
        """Select columns that may contain nulls or NaNs."""
        input_schema = self.child.schema(catalog)
        if input_schema.id_column:
            warnings.warn(
                f"Dropping null values in the ID column '{input_schema.id_column}' "
                "is not allowed, so that column may still contain null values.",
                RuntimeWarning,
            )
        if input_schema.grouping_column:
            warnings.warn(
                "Dropping null values in the grouping column "
                f"'{input_schema.grouping_column}' is not allowed, so that column "
                "may still contain null values.",
                RuntimeWarning,
            )
        return DropNullAndNan(
            self.child,
            tuple(
                name
                for name, desc in input_schema.column_descs.items()
                if (desc.allow_null or desc.allow_nan)
                and name not in (input_schema.grouping_column, input_schema.id_column)
            ),
        )
