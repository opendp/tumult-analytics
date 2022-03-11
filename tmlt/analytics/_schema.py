"""Schema management for private and public tables.

The schema represents the column types of the underlying table. This allows
for seamless transitions of the data representation type.
"""

# <placeholder: boilerplate>

from collections.abc import Mapping
from enum import Enum
from typing import Dict, Iterator
from typing import Mapping as MappingType
from typing import Optional, Union, cast

from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from tmlt.core.domains.base import Domain
from tmlt.core.domains.spark_domains import (
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)


class ColumnType(Enum):
    """The supported SQL92 column types for Analytics data."""

    INTEGER = int
    """Integer column type."""
    BIT = bool
    """Boolean column type."""
    DECIMAL = float
    """Floating-point column type."""
    VARCHAR = str
    """String column type."""

    def __str__(self) -> str:
        """Return a printable version of a ColumnType."""
        return str(self.name)

    def __repr__(self) -> str:
        """Return a string representation of a ColumnType."""
        return "ColumnType." + self.name


class Schema(Mapping):
    """Schema class describing the column types of the data.

    Note:
        nulls and nans are disallowed.

    The following SQL92 types are currently supported: INTEGER, BIT, DECIMAL, VARCHAR.
    """

    def __init__(
        self,
        column_types: MappingType[str, Union[str, ColumnType]],
        grouping_column: Optional[str] = None,
    ):
        """Constructor.

        Args:
            column_types: Mapping from column names to supported types.
            grouping_column: Optional column that must be grouped by in this query.
        """
        # TODO(#1539): Refactor Schema internal representation to use ColumnTypes
        # instead of strings, and update its interface to use them everywhere.
        self._column_types = {
            col: (ty.name if isinstance(ty, ColumnType) else ty)
            for col, ty in column_types.items()
        }
        self._grouping_column = grouping_column

        supported_types = [ty.name for ty in list(ColumnType)]
        invalid_types = set(self._column_types.values()) - set(supported_types)
        if invalid_types:
            raise ValueError(
                f"Column types {invalid_types} not supported; "
                f"use supported types {supported_types}."
            )
        if grouping_column is not None and grouping_column not in column_types:
            raise KeyError(
                f"grouping_column ({grouping_column}) is not in column_types"
            )

    @property
    def column_types(self) -> Dict[str, str]:
        """Returns the mapping from column names to supported types."""
        return dict(self._column_types)

    @property
    def grouping_column(self) -> Optional[str]:
        """Returns the optional column that must be grouped by."""
        return self._grouping_column

    def __eq__(self, other: object) -> bool:
        """Returns True if schemas are equal.

        Args:
            other: Schema to check against.
        """
        if isinstance(other, Schema):
            return (
                self.column_types == other.column_types
                and self.grouping_column == other.grouping_column
            )
        return False

    def __getitem__(self, column: str) -> str:
        """Returns the data type for the given column.

        Args:
            column: The column to get the data type for.
        """
        return self.column_types[column]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the columns in the schema."""
        return iter(self.column_types)

    def __len__(self) -> int:
        """Return the number of columns in the schema."""
        return len(self.column_types)

    def __repr__(self) -> str:
        """Return a string representation of self."""
        if self.grouping_column:
            return (
                f"Schema({self.column_types}, grouping_column='{self.grouping_column}')"
            )
        return f"Schema({self.column_types})"


_SPARK_TO_ANALYTICS = {
    IntegerType(): ColumnType.INTEGER,
    LongType(): ColumnType.INTEGER,
    DoubleType(): ColumnType.DECIMAL,
    FloatType(): ColumnType.DECIMAL,
    StringType(): ColumnType.VARCHAR,
}
"""Mapping from Spark type to supported Analytics column types."""

_ANALYTICS_TO_SPARK = {
    "INTEGER": LongType(),
    "DECIMAL": DoubleType(),
    "VARCHAR": StringType(),
}
"""Mapping from Analytics column types to Spark types."""

_ANALYTICS_TYPE_TO_COLUMN_DESCRIPTOR = {
    ColumnType.INTEGER: SparkIntegerColumnDescriptor,
    ColumnType.DECIMAL: SparkFloatColumnDescriptor,
    ColumnType.VARCHAR: SparkStringColumnDescriptor,
}
"""Mapping from Analytics column types to Spark columns descriptor.

More information regarding Spark columns descriptor can be found in
:class:`~tmlt.core.domains.spark_domains.SparkColumnDescriptor`"""


def analytics_to_py_types(analytics_schema: Schema) -> Dict[str, type]:
    """Returns the mapping from column names to supported python types."""
    return {
        column_name: ColumnType[column_type].value
        for column_name, column_type in analytics_schema.column_types.items()
    }


def analytics_to_spark_schema(analytics_schema: Schema) -> StructType:
    """Convert an Analytics schema to a Spark schema."""
    return StructType(
        [
            StructField(column_name, _ANALYTICS_TO_SPARK[column_type], nullable=False)
            for column_name, column_type in analytics_schema.column_types.items()
        ]
    )


def analytics_to_spark_columns_descriptor(
    analytics_schema: Schema,
) -> SparkColumnsDescriptor:
    """Convert a schema in Analytics representation to a Spark columns descriptor."""
    return {
        column_name: _ANALYTICS_TYPE_TO_COLUMN_DESCRIPTOR[ColumnType[column_type]]()
        for column_name, column_type in analytics_schema.column_types.items()
    }


def spark_schema_to_analytics_columns(
    spark_schema: StructType,
) -> Dict[str, ColumnType]:
    """Convert Spark schema to Analytics columns."""
    column_types = {
        field.name: _SPARK_TO_ANALYTICS[field.dataType] for field in spark_schema
    }
    return column_types


def spark_dataframe_domain_to_analytics_columns(
    domain: Domain,
) -> Dict[str, ColumnType]:
    """Convert a Spark dataframe domain to Analytics columns."""
    column_types = {
        column_name: _SPARK_TO_ANALYTICS[descriptor.data_type]
        for column_name, descriptor in cast(SparkDataFrameDomain, domain).schema.items()
    }
    return column_types
