"""Query expression for FlatMapByID."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass, replace
from typing import Any, Callable, List

from typeguard import check_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema
from tmlt.analytics.constraints import Constraint

from ._base import (
    QueryExprVisitor,
    Row,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
)


@dataclass(frozen=True)
class FlatMapByID(SingleChildQueryExpr):
    """Applies a flat map function to each group of rows with a common ID."""

    f: Callable[[List[Row]], List[Row]]
    """The flat map function."""
    schema_new_columns: Schema
    """The expected schema for new columns produced by ``f``.

    ``schema_new_column`` must not have a grouping or ID column.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
        check_type(self.f, Callable[[List[Row]], List[Row]])
        check_type(self.schema_new_columns, Schema)
        if self.schema_new_columns.grouping_column or self.schema_new_columns.id_column:
            raise AnalyticsInternalError(
                "FlatMapByID new column schema must not have a grouping or ID column."
            )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_flat_map_by_id(self)

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        if not input_schema.id_column:
            raise ValueError(
                "Flat-map-by-ID may only be used on tables with ID columns."
            )
        if input_schema.grouping_column:
            raise AnalyticsInternalError(
                "Encountered table with both an ID column and a grouping column."
            )
        if input_schema.id_column in self.schema_new_columns.column_descs:
            raise ValueError(
                "Flat-map-by-ID mapping function output cannot include ID column."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)

        id_column = input_schema.id_column
        new_columns = self.schema_new_columns.column_descs

        for name in list(new_columns.keys()):
            new_columns[name] = replace(new_columns[name], allow_null=True)
        return Schema(
            {id_column: input_schema[id_column], **new_columns},
            id_column=id_column,
            id_space=input_schema.id_space,
        )

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        # FlatMapByID does not preserve anything except the ID column from the
        # original table, and each ID is not guaranteed to have the same number of
        # records afterwards as it did before, so no constraints can be propagated.
        return frozenset()

    def __eq__(self, other: object) -> bool:
        """Returns true iff self == other.

        This uses the bytecode of self.f and other.f to determine if the two
        functions are equal.
        """
        if not isinstance(other, FlatMapByID):
            return False
        if self.f != other.f and self.f.__code__.co_code != other.f.__code__.co_code:
            return False
        return (
            self.schema_new_columns == other.schema_new_columns
            and self.child == other.child
        )
