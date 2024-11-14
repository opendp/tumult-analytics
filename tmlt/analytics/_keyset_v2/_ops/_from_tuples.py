"""Operation for constructing a KeySet from tuples."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import datetime
from dataclasses import dataclass
from typing import Union

from pyspark.sql import DataFrame, SparkSession

from tmlt.analytics._schema import (
    ColumnDescriptor,
    FrozenDict,
    Schema,
    analytics_to_spark_schema,
)

from ._base import KeySetOp


@dataclass(frozen=True)
class FromTuples(KeySetOp):
    """Construct a KeySet from a collection of tuples."""

    tuples: frozenset[tuple[Union[str, int, datetime.date, None], ...]]
    column_descriptors: FrozenDict

    def columns(self) -> list[str]:
        """Get a list of the columns included in the output of this operation."""
        return list(self.column_descriptors.keys())

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
        return spark.createDataFrame(self.tuples, schema=schema)

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return len(self.tuples) == 0

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return False
