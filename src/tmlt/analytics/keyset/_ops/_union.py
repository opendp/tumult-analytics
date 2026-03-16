"""Operation for computing the union of two KeySets or Plans."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass
from typing import Collection, Literal, Optional, overload

from pyspark.sql import DataFrame

from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp


@dataclass(frozen=True)
class Union(KeySetOp):
    """Compute the union of two KeySetOps.

    The schemas of ``left`` and ``right`` must match exactly aside from
    nullability; if a column is nullable in either operand, it will be nullable
    in the union. The result of this operation is concrete iff both operands are
    concrete.
    """

    left: KeySetOp
    right: KeySetOp

    def __post_init__(self):
        """Validation."""
        if self.left.columns() != self.right.columns():
            raise ValueError(
                "KeySet union operands must have the same columns:\n"
                f"Left:  {' '.join(sorted(self.left.columns()))}\n"
                f"Right: {' '.join(sorted(self.right.columns()))}"
            )

        if not (self.left.is_plan() or self.right.is_plan()):
            mismatched_columns = {}
            for c in self.columns():
                if (
                    self.left.schema()[c].column_type
                    != self.right.schema()[c].column_type
                ):
                    mismatched_columns[c] = (
                        self.left.schema()[c].column_type,
                        self.right.schema()[c].column_type,
                    )
            if mismatched_columns:
                raise ValueError(
                    "KeySet union operands have mismatched column types:\n"
                    + "\n".join(
                        f"{c}: {left} / {right}"
                        for c, (left, right) in mismatched_columns.items()
                    )
                )

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.left.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        left_schema = self.left.schema()
        right_schema = self.right.schema()
        return {
            c: ColumnDescriptor(
                left_schema[c].column_type,
                allow_null=left_schema[c].allow_null or right_schema[c].allow_null,
                allow_nan=left_schema[c].allow_nan or right_schema[c].allow_nan,
                allow_inf=left_schema[c].allow_inf or right_schema[c].allow_inf,
            )
            for c in left_schema
        }

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        left_df = self.left.dataframe()
        right_df = self.right.dataframe()
        return left_df.unionByName(right_df).distinct()

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        This operation may be expensive.
        """
        return self.left.is_empty() and self.right.is_empty()

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.left.is_plan() or self.right.is_plan()

    @overload
    def size(self, fast: Literal[True]) -> Optional[int]: ...

    @overload
    def size(self, fast: Literal[False]) -> int: ...

    @overload
    def size(self, fast: bool) -> Optional[int]: ...

    def size(self, fast):
        """Determine the size of the KeySet resulting from this operation."""
        # There's no shortcut to get this count due to deduplication
        if fast:
            return None
        return self.dataframe().count()

    def __str__(self):
        """Human-readable string representation."""
        return (
            "Union\n"
            + textwrap.indent(str(self.left), "  ")
            + "\n"
            + textwrap.indent(str(self.right), "  ")
        )

    def decompose(
        self, split_columns: Collection[str]
    ) -> tuple[list[KeySetOp], list[KeySetOp]]:
        """Decompose this KeySetOp into a collection of factors and subtracted values.

        See :meth:`KeySet._decompose` for details.
        """
        # Union doesn't naturally decompose into factors, so treat it as atomic
        return [self], []
