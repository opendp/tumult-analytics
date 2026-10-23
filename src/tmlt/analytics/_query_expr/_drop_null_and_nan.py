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
    QueryExprVisitor,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
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

    columns: Tuple[str, ...] = tuple()
    """Columns in which to look for nulls and NaNs.

    If this tuple is empty, *all* columns will be looked at - so if *any* column
    contains a null or NaN value that row will be dropped."""

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
                f"Cannot drop null values in column '{input_schema.grouping_column}', "
                "as it is a grouping column"
            )
        if input_schema.id_column and input_schema.id_column in self.columns:
            raise ValueError(
                f"Cannot drop null values in column '{input_schema.id_column}', "
                "as it is an ID column."
            )
        if input_schema.id_column and len(self.columns) == 0:
            warnings.warn(
                f"Replacing null values in the ID column '{input_schema.id_column}' "
                "is not allowed, so the ID column may still contain null values.",
                RuntimeWarning,
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

        columns = self.columns
        if len(columns) == 0:
            columns = tuple(
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan)
                and name not in [input_schema.grouping_column, input_schema.id_column]
            )

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=(cd.allow_null and name not in columns),
                    allow_nan=(cd.allow_nan and name not in columns),
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
