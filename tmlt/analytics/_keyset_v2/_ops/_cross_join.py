"""Operation for constructing a KeySet by cross-joining two factors."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass
from typing import Literal, Optional, overload

from pyspark.sql import DataFrame, SparkSession

from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp


@dataclass(frozen=True)
class CrossJoin(KeySetOp):
    """Construct a KeySet by cross-joining two existing KeySetOps.

    The ``left`` and ``right`` factors must have disjoint sets of
    columns. ``CrossJoin`` is concrete if both of the factors are concrete.
    """

    left: KeySetOp
    right: KeySetOp

    def __post_init__(self):
        """Validation."""
        overlapping_columns = set(self.left.columns()) & set(self.right.columns())
        if overlapping_columns:
            raise ValueError(
                "Unable to cross-join KeySets, they have "
                f"overlapping columns: {' '.join(overlapping_columns)}"
            )

    def columns(self) -> list[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.left.columns() + self.right.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return self.left.schema() | self.right.schema()

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        # If either factor corresponds to a total aggregation (no columns),
        # crossing it with another factor should just produce the other
        # factor. A Spark crossjoin produces an empty dataframe in this case
        # because the total aggregation has no rows, so skip the cross-join in
        # those cases.
        if self.left.columns() == []:
            return self.right.dataframe()
        if self.right.columns() == []:
            return self.left.dataframe()

        # Repeated Spark crossjoins can have terrible performance if the number
        # of partitions involved isn't managed correctly, either developing far
        # too many partitions if using many small factors or not having enough
        # partitions to effectively make use of the available executors when
        # crossing dataframes with few partitions. This aims to keep the number
        # of partitions between 2x and 4x Spark's default parallelism, though
        # the number of partitions may be lower than this on small inputs.
        spark = SparkSession.builder.getOrCreate()
        partition_target = 2 * spark.sparkContext.defaultParallelism

        left = self.left.dataframe()
        right = self.right.dataframe()
        if left.rdd.getNumPartitions() > 2 * partition_target:
            left = left.coalesce(partition_target)
        if right.rdd.getNumPartitions() > 2 * partition_target:
            right = left.coalesce(partition_target)

        return left.crossJoin(right)

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return self.left.is_empty() or self.right.is_empty()

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.left.is_plan() or self.right.is_plan()

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
        left = self.left.size(fast=fast)
        right = self.right.size(fast=fast)
        if left is not None and right is not None:
            return left * right
        return None

    def __str__(self):
        """Human-readable string representation."""
        return (
            "CrossJoin\n"
            + textwrap.indent(str(self.left), "  ")
            + "\n"
            + textwrap.indent(str(self.right), "  ")
        )
