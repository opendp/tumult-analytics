"""Operation for detecting the KeySet for a group of columns."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from dataclasses import dataclass
from typing import Optional

from pyspark.sql import DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp
from ._utils import validate_column_names


@dataclass(frozen=True)
class Detect(KeySetOp):
    """Detect a KeySet from a group of columns on a Session."""

    detect_columns: frozenset[str]

    def __post_init__(self):
        """Validation."""
        if len(self.detect_columns) == 0:
            raise ValueError(
                "Detect must be used on a non-empty collection of columns."
            )
        validate_column_names(self.detect_columns)

    def columns(self) -> list[str]:
        """Get a list of the columns included in the output of this operation."""
        return list(self.detect_columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation.

        Raises ``AnalyticsInternalError``, as the schema is not known until the
        column values are supplied.
        """
        raise AnalyticsInternalError("KeySetPlan does not have a fixed schema.")

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        Raises ``AnalyticsInternalError``, as the dataframe is not known until
        the fixed values are supplied.
        """
        raise AnalyticsInternalError(
            "KeySetPlan does not have a fixed dataframe representation."
        )

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        Raises ``AnalyticsInternalError``, as whether the operation's output
        dataframe is empty is not known until the fixed values are supplied.
        """
        raise AnalyticsInternalError("KeySetPlan does not have a fixed size.")

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return True

    def size(self) -> Optional[int]:
        """Determine the size of the KeySet resulting from this operation.

        Raises ``AnalyticsInternalError``, as the operation's output dataframe
        size is not known until the fixed values are supplied.
        """
        raise AnalyticsInternalError("KeySetPlan does not have a fixed size.")
