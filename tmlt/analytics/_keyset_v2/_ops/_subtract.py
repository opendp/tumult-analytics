"""Operation for subtracting a KeySet from a KeySet or Plan."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass
from typing import Literal, Optional, overload

from pyspark.sql import DataFrame
from tmlt.core.utils.join import join

from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp


@dataclass(frozen=True)
class Subtract(KeySetOp):
    """Subtract a (non-plan) KeySetOp from another KeySetOp.

    The columns of ``right`` must be a subset of the columns of ``left``.
    The result is a plan iff ``left`` is a plan.
    """

    left: KeySetOp
    right: KeySetOp

    def __post_init__(self):
        """Validation."""
        if self.right.is_plan():
            raise ValueError(
                "Cannot subtract a KeySetPlan from a KeySet or KeySetPlan."
            )

        non_subset_columns = self.right.columns() - self.left.columns()
        if non_subset_columns:
            raise ValueError(
                "Unable to subtract KeySets, right hand side has columns that "
                f"do not exist in the left hand side: {', '.join(non_subset_columns)}"
            )

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.left.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return self.left.schema()

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return join(
            self.left.dataframe(),
            self.right.dataframe(),
            on=list(self.right.columns()),
            how="left_anti",
            nulls_are_equal=True,
        )

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        This operation may be expensive.
        """
        return self.left.is_empty() or self.dataframe().isEmpty()

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.left.is_plan()

    @overload
    def size(self, fast: Literal[True]) -> Optional[int]:
        ...

    @overload
    def size(self, fast: Literal[False]) -> int:
        ...

    @overload
    def size(self, fast: bool) -> Optional[int]:
        ...

    def size(self, fast):
        """Determine the size of the KeySet resulting from this operation.

        Subtract cannot determine its size fast.
        """
        if fast:
            return None
        return self.dataframe().count()

    def __str__(self):
        """Human-readable string representation."""
        return (
            "Subtract\n"
            + textwrap.indent(str(self.left), "  ")
            + "\n"
            + textwrap.indent(str(self.right), "  ")
        )
