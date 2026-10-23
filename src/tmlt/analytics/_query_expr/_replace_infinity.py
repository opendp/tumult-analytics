"""Query expression for ReplaceInfinity."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

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
    QueryExprVisitor,
    SingleChildQueryExpr,
    _valid_replacement_value,
    _validate_child_is_transformation,
)


@dataclass(frozen=True)
class ReplaceInfinity(SingleChildQueryExpr):
    """Returns data with +inf and -inf expressions replaced by defaults."""

    replace_with: FrozenDict = FrozenDict.from_dict({})
    """New values to replace with, by column. The first value for each column
    will be used to replace -infinity, and the second value will be used to
    replace +infinity.

    If this dictionary is empty, *all* columns of type DECIMAL will be changed,
    with infinite values replaced with a default value (see the
    :class:`~.AnalyticsDefault` class variables).
    """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
        check_type(self.replace_with, FrozenDict)
        # Allow passing None as a replacement for the purposes of this check,
        # even though we disallow it later -- better consistency of error
        # messages between the invalid values.
        check_type(
            dict(self.replace_with), Dict[str, Tuple[float | None, float | None]]
        )

        for col, (lower, upper) in self.replace_with.items():
            if not _valid_replacement_value(lower):
                raise ValueError(
                    f"Invalid lower replacement value {lower} for column '{col}'"
                )
            if not _valid_replacement_value(upper):
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

        columns_to_change = list(self.replace_with.keys())
        if len(columns_to_change) == 0:
            columns_to_change = [
                name
                for name, cd in input_schema.column_descs.items()
                if cd.column_type == ColumnType.DECIMAL
                and cd.allow_inf
                and name not in [input_schema.grouping_column, input_schema.id_column]
            ]
        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=cd.allow_null,
                    allow_nan=cd.allow_nan,
                    allow_inf=(cd.allow_inf and name not in columns_to_change),
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
                and c.grouping_column not in self.replace_with
            )
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_replace_infinity(self)
