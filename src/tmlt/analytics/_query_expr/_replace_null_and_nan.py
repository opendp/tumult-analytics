"""Query expression for ReplaceNullAndNan."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

import warnings
from dataclasses import dataclass
from typing import Any

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import (
    ColumnDescriptor,
    FrozenDict,
    Schema,
    analytics_to_py_types,
)
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
class ReplaceNullAndNan(SingleChildQueryExpr):
    """Returns data with null and NaN expressions replaced by a default.

    .. warning::
        after a ``ReplaceNullAndNan`` query has been performed for a column,
        Tumult Analytics will raise an error if you use a
        :class:`~.tmlt.analytics.KeySet` for that column
        that contains null values.
    """

    replace_with: FrozenDict = FrozenDict.from_dict({})
    """New values to replace with, by column.

    If this dictionary is empty, *all* columns will be changed, with values
    replaced by a default value for each column's type (see the
    :class:`~.AnalyticsDefault` class variables).
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
        check_type(
            self.replace_with,
            FrozenDict,
        )
        for col, val in self.replace_with.items():
            if not _valid_replacement_value(val):
                raise ValueError(f"Invalid replacement value {val} for column '{col}'")

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if (
            input_schema.grouping_column
            and input_schema.grouping_column in self.replace_with
        ):
            raise ValueError(
                "Cannot replace null values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column."
            )
        if input_schema.id_column and input_schema.id_column in self.replace_with:
            raise ValueError(
                f"Cannot replace null values in column '{input_schema.id_column}', "
                "as it is an ID column."
            )
        if input_schema.id_column and (len(self.replace_with) == 0):
            warnings.warn(
                f"Replacing null values in the ID column '{input_schema.id_column}' "
                "is not allowed, so the ID column may still contain null values.",
                RuntimeWarning,
            )

        pytypes = analytics_to_py_types(input_schema)
        for col, val in self.replace_with.items():
            if col not in input_schema.keys():
                raise ValueError(
                    f"Column '{col}' does not exist in this table, "
                    f"available columns are {list(input_schema.keys())}"
                )
            if not isinstance(val, pytypes[col]):
                # Using an int as a float is OK
                if not (isinstance(val, int) and pytypes[col] == float):
                    raise ValueError(
                        f"Column '{col}' cannot have nulls replaced with "
                        f"{repr(val)}, as that value's type does not match the "
                        f"column type {input_schema[col].column_type.name}"
                    )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        columns_to_change = list(dict(self.replace_with).keys())
        if len(columns_to_change) == 0:
            columns_to_change = [
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan)
                and name not in [input_schema.grouping_column, input_schema.id_column]
            ]
        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=(cd.allow_null and name not in columns_to_change),
                    allow_nan=(cd.allow_nan and name not in columns_to_change),
                    allow_inf=cd.allow_inf,
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
        return visitor.visit_replace_null_and_nan(self)
