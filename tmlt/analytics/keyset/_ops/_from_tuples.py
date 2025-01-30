"""Operation for constructing a KeySet from tuples."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from dataclasses import dataclass
from typing import Literal, Optional, Union, overload

from pyspark.sql import DataFrame, SparkSession

from tmlt.analytics._schema import (
    ColumnDescriptor,
    FrozenDict,
    Schema,
    analytics_to_spark_schema,
)

from ._base import KeySetOp
from ._utils import validate_schema


@dataclass(frozen=True)
class FromTuples(KeySetOp):
    """Construct a KeySet from a collection of tuples."""

    tuples: frozenset[tuple[Union[str, int, datetime.date, None], ...]]
    column_descriptors: FrozenDict

    def __post_init__(self):
        """Validation."""
        # Validation of the tuples themselves occurs in the KeySet.from_tuples
        # method, as constructing column_descriptors already requires scanning
        # the input data. The checking there guarantees that all of the tuples
        # match column_descriptors and that every column has a known type.
        validate_schema(self.column_descriptors)
        if len(self.column_descriptors) == 0 and len(self.tuples) > 0:
            raise ValueError("A KeySet with no columns must not have any rows.")

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return set(self.column_descriptors.keys())

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return dict(self.column_descriptors)

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        schema = analytics_to_spark_schema(Schema(self.schema()))
        spark = SparkSession.builder.getOrCreate()
        # For small collections of tuples, Spark's default parallelism results
        # in far too many partitions, leading to poor performance when
        # cross-joining due to the overhead of managing all of the small
        # tasks. Instead, use just a couple of partitions and slowly add more
        # only for large keysets.
        return spark.createDataFrame(
            spark.sparkContext.parallelize(
                self.tuples, numSlices=2 + len(self.tuples) // 1024
            ),
            schema=schema,
        )

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return len(self.column_descriptors) > 0 and len(self.tuples) == 0

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return False

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
        if len(self.column_descriptors) == 0:
            return 1
        return len(self.tuples)

    def __str__(self):
        """Human-readable string representation."""
        cols = "\n  ".join(
            f"{col}: {desc.column_type}{' not NULL' if not desc.allow_null else ''}"
            for col, desc in self.column_descriptors.items()
        )
        return f"FromTuples ({len(self.tuples)} rows)\n  {cols}"

    def __iter__(self):
        """Return an iterator of tuples corresponding to keys."""
        return iter(self.tuples)
