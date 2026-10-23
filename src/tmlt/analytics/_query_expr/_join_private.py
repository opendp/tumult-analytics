"""Query expression for JoinPrivate."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema
from tmlt.analytics.constraints import (
    Constraint,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    simplify_constraints,
)
from tmlt.analytics.truncation_strategy import TruncationStrategy

from ._base import (
    QueryExpr,
    QueryExprVisitor,
    _propagate_join_by_stability,
    _schema_for_join,
    _validate_child_is_transformation,
    _validate_join,
)


@dataclass(frozen=True)
class JoinPrivate(QueryExpr):
    """Returns the join of two private tables.

    Before performing the join, each table is truncated based on the corresponding
    :class:`~tmlt.analytics.TruncationStrategy`.  For a more
    detailed overview of ``JoinPrivate``'s behavior, see
    :meth:`~tmlt.analytics.QueryBuilder.join_private`.
    """

    left_child: QueryExpr
    """The QueryExpr to join with right operand."""
    right_child: QueryExpr
    """The QueryExpr for private source to join with."""
    truncation_strategy_left: Optional[TruncationStrategy.Type] = None
    """Truncation strategy to be used for the left table."""
    truncation_strategy_right: Optional[TruncationStrategy.Type] = None
    """Truncation strategy to be used for the right table."""
    join_columns: Optional[Tuple[str, ...]] = None
    """The columns used for joining the tables, or None to use all common columns."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.left_child)
        _validate_child_is_transformation(self.right_child)
        check_type(
            self.truncation_strategy_left,
            Optional[TruncationStrategy.Type],
        )
        check_type(
            self.truncation_strategy_right,
            Optional[TruncationStrategy.Type],
        )
        check_type(self.join_columns, Optional[Tuple[str, ...]])

        if self.join_columns is not None:
            if len(self.join_columns) == 0:
                raise ValueError("Provided join columns must not be empty")
            if len(self.join_columns) != len(set(self.join_columns)):
                raise ValueError("Join columns must be distinct")

    def _validate(self, left_schema: Schema, right_schema: Schema):
        """Validation checks for this QueryExpr."""
        if left_schema.id_column != right_schema.id_column:
            if left_schema.id_column is None or right_schema.id_column is None:
                raise ValueError(
                    "Private joins can only be performed between two tables "
                    "with the same type of protected change"
                )
            raise ValueError(
                "Private joins between tables with the AddRowsWithID "
                "protected change are only possible when the ID columns of "
                "the two tables have the same name"
            )
        if left_schema.id_space != right_schema.id_space:
            raise ValueError(
                "Private joins between tables with the AddRowsWithID protected change"
                " are only possible when both tables are in the same ID space"
            )
        _validate_join(left_schema, right_schema, self.join_columns)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr.

        The ordering of output columns are:

        1. The join columns
        2. Columns that are only in the left table
        3. Columns that are only in the right table
        4. Columns that are in both tables, but not included in the join columns. These
           columns are included with _left and _right suffixes.
        """
        left_schema = self.left_child.schema(catalog)
        right_schema = self.right_child.schema(catalog)
        self._validate(left_schema, right_schema)
        return _schema_for_join(
            left_schema=left_schema,
            right_schema=right_schema,
            join_columns=self.join_columns,
            join_id_space=left_schema.id_space,
            how="inner",
        )

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        left_constraints = self.left_child.constraints(catalog)
        right_constraints = self.right_child.constraints(catalog)
        left_schema = self.left_child.schema(catalog)
        right_schema = self.right_child.schema(catalog)
        common_cols = set(left_schema) & set(right_schema)
        join_cols = set(self.join_columns or common_cols)
        overlapping_cols = common_cols - join_cols

        def max_join_stability(cs: frozenset[Constraint]) -> Optional[int]:
            """Compute the join stability for one side of a join from its constraints.

            Using a set of constraints on one table in the join, this determines
            the maximum number of times each row in the other table can be
            duplicated by the join. If there is no limit, None is returned.
            """
            cs = simplify_constraints(cs)
            stabilities = []
            # A MaxRowsPerID constraint limits the duplication factor from a table,
            # as it limits the number of times each value in the ID column can
            # appear.
            max_rows_per_id = next((c for c in cs if isinstance(c, MaxRowsPerID)), None)
            if max_rows_per_id:
                stabilities.append(max_rows_per_id.max)
            # When a MaxRowsPerGroupPerID constraint has a grouping column in
            # the join columns, that also limits the duplication factor because
            # each (ID, grouping column) value pair can only appear a limited
            # number of times.
            for c in (c for c in cs if isinstance(c, MaxRowsPerGroupPerID)):
                if c.grouping_column in join_cols:
                    stabilities.append(c.max)
            if not stabilities:
                return None
            return min(stabilities)

        left_stability = max_join_stability(left_constraints)
        right_stability = max_join_stability(right_constraints)

        left_propagated = [
            _propagate_join_by_stability(
                c, right_stability, overlapping_cols, suffix="_left"
            )
            for c in left_constraints
        ]
        right_propagated = [
            _propagate_join_by_stability(
                c, left_stability, overlapping_cols, suffix="_right"
            )
            for c in right_constraints
        ]
        return simplify_constraints(
            c for c in left_propagated + right_propagated if c is not None
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_join_private(self)
