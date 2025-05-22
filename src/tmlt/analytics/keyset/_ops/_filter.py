"""Operation for filtering the rows in a KeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass
from typing import Literal, Optional, Union, overload

from pyspark.sql import Column, DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp


@dataclass(frozen=True)
class Filter(KeySetOp):
    """Filter the rows of a KeySet."""

    child: KeySetOp
    condition: Union[Column, str]

    def __post_init__(self):
        """Validation."""
        if not isinstance(self.child, KeySetOp):
            raise AnalyticsInternalError(
                "Child of Project KeySetOp must be a KeySetOp, "
                f"not {type(self.child).__qualname__}."
            )

        if isinstance(self.condition, str) and self.condition == "":
            raise ValueError("A KeySet cannot be filtered by an empty condition.")

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.child.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return self.child.schema()

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return self.child.dataframe().filter(self.condition)

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        This operation may be expensive.
        """
        return self.dataframe().isEmpty()

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.child.is_plan()

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
        """Determine the size of the KeySet resulting from this operation."""
        if fast:
            return None
        return self.dataframe().count()

    def __str__(self):
        """Human-readable string representation."""
        return f"Filter {self.condition}\n" + textwrap.indent(str(self.child), "  ")
