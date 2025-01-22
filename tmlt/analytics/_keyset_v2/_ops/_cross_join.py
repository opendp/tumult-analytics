"""Operation for constructing a KeySet by cross-joining two factors."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import operator
import textwrap
from dataclasses import dataclass
from functools import reduce
from typing import Literal, Optional, overload

from pyspark.sql import DataFrame, SparkSession

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp


@dataclass(frozen=True)
class CrossJoin(KeySetOp):
    """Construct a KeySet by cross-joining two existing KeySetOps.

    The ``left`` and ``right`` factors must have disjoint sets of
    columns. ``CrossJoin`` is concrete if both of the factors are concrete.
    """

    factors: tuple[KeySetOp, ...]

    def __post_init__(self):
        """Validation."""
        if len(self.factors) == 0:
            raise AnalyticsInternalError("CrossJoin must have at least one factor.")

        column_counts: dict[str, int] = {}
        for f in self.factors:
            for c in f.columns():
                column_counts[c] = column_counts.get(c, 0) + 1

        overlapping_columns = sorted(
            c for c, count in column_counts.items() if count > 1
        )
        if overlapping_columns:
            raise ValueError(
                "Unable to cross-join KeySets, they have "
                f"overlapping columns: {' '.join(overlapping_columns)}"
            )

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return reduce(operator.or_, (f.columns() for f in self.factors))

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return reduce(operator.or_, (f.schema() for f in self.factors))

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        # Repeated Spark crossjoins can have terrible performance if the number
        # of partitions involved isn't managed correctly, either developing far
        # too many partitions if using many small factors or not having enough
        # partitions to effectively make use of the available executors when
        # crossing dataframes with few partitions. This aims to keep the number
        # of partitions between 2x and 4x Spark's default parallelism, though
        # the number of partitions may be lower than this on small inputs.
        spark = SparkSession.builder.getOrCreate()
        partition_target = 2 * spark.sparkContext.defaultParallelism

        def cross_join(left: DataFrame, right: DataFrame):
            left_partitions = left.rdd.getNumPartitions()
            right_partitions = right.rdd.getNumPartitions()
            if left_partitions == 1:
                left = left.repartition(2)
            elif left_partitions > 2 * partition_target:
                left = left.coalesce(partition_target)
            if right_partitions == 1:
                right = right.repartition(2)
            if right_partitions > 2 * partition_target:
                right = right.coalesce(partition_target)

            return left.crossJoin(right)

        # Cross-joining with a factor that contains no rows and no columns
        # produces an empty dataframe, but such factors in a KeySet correspond
        # to total aggregations. If *only* factors like that are present, just
        # return the dataframe for one of them; otherwise, ignore them and do
        # the cross-join on the remaining factors.
        nonempty_dfs = [f.dataframe() for f in self.factors if len(f.columns()) > 0]
        if len(nonempty_dfs) == 0:
            return self.factors[0].dataframe()

        return reduce(cross_join, nonempty_dfs)

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return any(f.is_empty() for f in self.factors)

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return any(f.is_plan() for f in self.factors)

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
        sizes = [f.size(fast=fast) for f in self.factors]
        if fast and None in sizes:
            return None
        return reduce(operator.mul, sizes)

    def __str__(self):
        """Human-readable string representation."""
        return "CrossJoin\n" + "\n".join(
            textwrap.indent(str(f), "  ") for f in self.factors
        )
