"""Operation for constructing a KeySet from a Spark DataFrame."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from dataclasses import dataclass
from typing import Optional

from pyspark.sql import DataFrame

from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import ColumnDescriptor, spark_schema_to_analytics_columns

from ._base import KeySetOp
from ._utils import validate_schema


@dataclass(frozen=True)
class FromSparkDataFrame(KeySetOp):
    """Construct a KeySet from a Spark DataFrame."""

    df: DataFrame

    def __post_init__(self):
        """Validation."""
        validate_schema(self.schema())
        if len(self.columns()) == 0 and not self.df.isEmpty():
            raise ValueError("A KeySet with no columns must not have any rows.")

    def columns(self) -> list[str]:
        """Get a list of the columns included in the output of this operation."""
        return list(self.df.columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return spark_schema_to_analytics_columns(self.df.schema)

    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return coerce_spark_schema_or_fail(self.df.dropDuplicates())

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return self.df.isEmpty()

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return False

    def size(self) -> Optional[int]:
        """Determine the size of the KeySet resulting from this operation.

        FromSparkDataFrame cannot compute its size in a guaranteed-computationally-cheap
        way, so this method always returns None for it.
        """
        return None
