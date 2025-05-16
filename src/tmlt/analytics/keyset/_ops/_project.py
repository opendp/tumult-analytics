"""Operation for projecting columns out of a KeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass
from typing import Literal, Optional, overload

from pyspark.sql import DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp
from ._utils import validate_column_names


@dataclass(frozen=True)
class Project(KeySetOp):
    """Project a set of columns out of a KeySet."""

    child: KeySetOp
    projected_columns: frozenset[str]

    def __post_init__(self):
        """Validation."""
        if not isinstance(self.child, KeySetOp):
            raise AnalyticsInternalError(
                "Child of Project KeySetOp must be a KeySetOp, "
                f"not {type(self.child).__qualname__}."
            )
        if not isinstance(self.projected_columns, frozenset):
            raise AnalyticsInternalError(
                "Project KeySetOp's columns must be a frozenset, "
                f"not {type(self.projected_columns).__qualname__}."
            )
        validate_column_names(self.projected_columns)

        if len(self.projected_columns) == 0:
            raise ValueError(
                "At least one column must be kept when subscripting a KeySet."
            )

        missing_columns = self.projected_columns - set(self.child.columns())
        if len(missing_columns) == 1:
            raise ValueError(
                f"Column {list(missing_columns)[0]} is not present in KeySet, "
                f"available columns are: {', '.join(self.child.columns())}"
            )
        if len(missing_columns) > 1:
            raise ValueError(
                f"Columns {', '.join(sorted(missing_columns))} are not present in "
                f"KeySet, available columns are: {', '.join(self.child.columns())}"
            )

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return set(self.projected_columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        child_schema = self.child.schema()
        return {c: child_schema[c] for c in self.projected_columns}

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return self.child.dataframe().select(*self.projected_columns).dropDuplicates()

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return self.child.is_empty()

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
        return f"Project {', '.join(self.projected_columns)}\n" + textwrap.indent(
            str(self.child), "  "
        )
