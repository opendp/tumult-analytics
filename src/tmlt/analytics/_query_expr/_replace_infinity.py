"""Query expression for ReplaceInfinity."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, FrozenDict, Schema
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    simplify_constraints,
)

from ._base import (
    AnalyticsDefault,
    AutoColumnQueryExpr,
    QueryExprVisitor,
    SingleChildQueryExpr,
    valid_replacement_value,
    validate_child_is_transformation,
)
from ._drop_infinity import _infinity_columns


@dataclass(frozen=True)
class ReplaceInfinity(SingleChildQueryExpr):
    """Returns data with +inf and -inf expressions replaced by defaults."""

    replace_with: FrozenDict
    """New values to replace with, by column. The first value for each column
    will be used to replace -infinity, and the second value will be used to
    replace +infinity.

    An empty dictionary leaves the input unchanged.
    """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        validate_child_is_transformation(self.child)
        check_type(self.replace_with, FrozenDict)
        # Allow passing None as a replacement for the purposes of this check,
        # even though we disallow it later -- better consistency of error
        # messages between the invalid values.
        check_type(
            dict(self.replace_with), Dict[str, Tuple[float | None, float | None]]
        )

        for col, (lower, upper) in self.replace_with.items():
            if not valid_replacement_value(lower):
                raise ValueError(
                    f"Invalid lower replacement value {lower} for column '{col}'"
                )
            if not valid_replacement_value(upper):
                raise ValueError(
                    f"Invalid upper replacement value {upper} for column '{col}'"
                )

        # Ensure that the values in replace_with are tuples of floats
        updated_dict = {}
        for col, val in self.replace_with.items():
            updated_dict[col] = (float(val[0]), float(val[1]))

        # Subverting the frozen dataclass to update the replace_with attribute
        object.__setattr__(self, "replace_with", FrozenDict.from_dict(updated_dict))

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if (
            input_schema.grouping_column
            and input_schema.grouping_column in self.replace_with
        ):
            raise ValueError(
                "Cannot replace infinite values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column"
            )
        if input_schema.id_column and input_schema.id_column in self.replace_with:
            raise ValueError(
                f"Cannot replace infinite values in column '{input_schema.id_column}', "
                "as it is an ID column"
            )

        for name in self.replace_with:
            if name not in input_schema.keys():
                raise ValueError(
                    f"Column '{name}' does not exist in this table, "
                    f"available columns are {list(input_schema.keys())}"
                )
            if input_schema[name].column_type != ColumnType.DECIMAL:
                raise ValueError(
                    f"Column '{name}' has a replacement value provided, but it is "
                    f"of type {input_schema[name].column_type.name} (not "
                    f"{ColumnType.DECIMAL.name}) and so cannot "
                    "contain infinite values"
                )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=cd.allow_null,
                    allow_nan=cd.allow_nan,
                    allow_inf=(cd.allow_inf and name not in self.replace_with),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        child_constraints = self.child.constraints(catalog)
        # Replacing grouping values can combine groups. This cannot increase the
        # number of rows or groups per ID, but it can increase the rows in one
        # group, so only MaxRowsPerGroupPerID constraints on replaced columns
        # are dropped.
        return simplify_constraints(
            c
            for c in child_constraints
            if isinstance(c, (MaxRowsPerID, MaxGroupsPerID))
            or (
                isinstance(c, MaxRowsPerGroupPerID)
                and not any(g in self.replace_with for g in c.grouping_columns)
            )
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_replace_infinity(self)


@dataclass(frozen=True)
class AutoReplaceInfinity(AutoColumnQueryExpr, SingleChildQueryExpr):
    """Replace infinities in eligible columns with the decimal default."""

    def __post_init__(self):
        """Check the child expression."""
        validate_child_is_transformation(self.child)

    def resolve_columns(self, catalog: Catalog) -> ReplaceInfinity:
        """Select eligible columns and their default replacement values.

        Any DECIMAL-valued columns that allow infinite values in the input
        schema are included, and both positive and negative infinities are
        replaced with ``AnalyticsDefault.DECIMAL``.
        """
        columns = _infinity_columns(self.child.schema(catalog))
        if not columns:
            warnings.warn(
                "replace_infinity() was called with no columns specified, and the"
                " table has no eligible columns that can contain infinite values,"
                " so there is nothing to replace."
            )
        return ReplaceInfinity(
            self.child,
            FrozenDict.from_dict(
                {
                    name: (AnalyticsDefault.DECIMAL, AnalyticsDefault.DECIMAL)
                    for name in columns
                }
            ),
        )
