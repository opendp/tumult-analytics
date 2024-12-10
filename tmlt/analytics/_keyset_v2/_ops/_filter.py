"""Operation for filtering the rows in a KeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from dataclasses import dataclass
from typing import Optional, Union

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

    def columns(self) -> list[str]:
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

    def size(self) -> Optional[int]:
        """Determine the size of the KeySet resulting from this operation.

        Filter cannot compute its size in a guaranteed-computationally-cheap
        way, so this method always returns None for it.
        """
        return None
