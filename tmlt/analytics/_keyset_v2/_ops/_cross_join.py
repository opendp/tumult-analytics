"""Operation for constructing a KeySet by cross-joining two factors."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from dataclasses import dataclass
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp


def _adjust_partitioning(
    df: DataFrame, size: Optional[int], partition_target: int
) -> DataFrame:
    partitions = df.rdd.getNumPartitions()

    if size is not None and size > 64 and partitions == 1:
        return df.repartition(2)
    if partitions > 2 * partition_target:
        return df.coalesce(partition_target)
    return df


@dataclass(frozen=True)
class CrossJoin(KeySetOp):
    """Construct a KeySet by cross-joining two existing KeySetOps.

    The ``left`` and ``right`` factors must have disjoint sets of
    columns. ``CrossJoin`` is concrete if both of the factors are concrete.
    """

    left: KeySetOp
    right: KeySetOp

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
        # crossing dataframes with few partitions. Coupled with partition
        # control logic in FromTuples, this code aims to do the following:
        #   * In all cases, limit the number of partitions for each factor going
        #     into the crossjoin to about 4x Spark's default parallelism.
        #   * If either factor going into the crossjoin has more than 64 rows
        #     and has only one partition, repartition it to have two. This will
        #     allow the partition count for the product to grow along with the
        #     resulting dataset size (up to the limit from the previous point),
        #     ensuring that Spark doesn't get stuck with a huge dataset that
        #     only has one partition.

        spark = SparkSession.builder.getOrCreate()
        partition_target = 2 * spark.sparkContext.defaultParallelism

        left = _adjust_partitioning(
            self.left.dataframe(), self.left.size(), partition_target
        )
        right = _adjust_partitioning(
            self.right.dataframe(), self.right.size(), partition_target
        )
        # Because of the way Spark crossjoins handle partitions, both factors
        # need to be repartitioned to have at least two partitions in order to
        # guarantee that Spark doesn't combine them back together and still end
        # up with one partition in the output.
        if left.rdd.getNumPartitions() == 1 and right.rdd.getNumPartitions() > 1:
            left = left.repartition(2)
        if left.rdd.getNumPartitions() > 1 and right.rdd.getNumPartitions() == 1:
            right = right.repartition(2)

        return left.crossJoin(right)

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return self.left.is_empty() or self.right.is_empty()

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.left.is_plan() or self.right.is_plan()

    def size(self) -> Optional[int]:
        """Determine the size of the KeySet resulting from this operation."""
        left = self.left.size()
        right = self.right.size()
        if left is not None and right is not None:
            return left * right
        return None
