"""Schema management for private and public tables.

The schema represents the column types of the underlying table. This allows
for seamless transitions of the data representation type.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List
from typing import Mapping as MappingType
from typing import Optional, Union, cast

from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tmlt.core.domains.base import Domain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)


class ColumnType(Enum):
    """The supported SQL92 column types for Analytics data."""

    INTEGER = int
    """Integer column type."""
    DECIMAL = float
    """Floating-point column type."""
    VARCHAR = str
    """String column type."""
    DATE = datetime.date
    """Date column type."""
    TIMESTAMP = datetime.datetime
    """Timestamp column type."""

    def __str__(self) -> str:
        """Return a printable version of a ColumnType."""
        return str(self.name)

    def __repr__(self) -> str:
        """Return a string representation of a ColumnType."""
        return "ColumnType." + self.name


@dataclass
class ColumnDescriptor:
    """Information about a column.

    ColumnDescriptors have the following attributes:

    Attributes:
        column_type: A :class:`ColumnType`, specifying what type this column has.
        allow_null: :class:`bool`. If ``True``, this column allows null values.
        allow_nan: :class:`bool`. If ``True``, this column allows NaN values.
        allow_inf: :class:`bool`. If ``True``, this column allows infinite values.
    """

    column_type: ColumnType
    allow_null: bool = False
    allow_nan: bool = False
    allow_inf: bool = False


class Schema(Mapping):
    """Schema class describing the column information of the data.

    The following SQL92 types are currently supported:
      INTEGER, DECIMAL, VARCHAR, DATE, TIMESTAMP
    """

    def __init__(
        self,
        column_descs: MappingType[str, Union[str, ColumnType, ColumnDescriptor]],
        grouping_column: Optional[str] = None,
        id_column: Optional[str] = None,
        id_space: Optional[str] = None,
        default_allow_null: bool = False,
        default_allow_nan: bool = False,
        default_allow_inf: bool = False,
    ):
        """Constructor.

        Args:
            column_descs: Mapping from column names to supported types.
            grouping_column: Optional column that must be grouped by in this query.
            id_column: The ID column on this table, if one exists.
            id_space: The ID space for this table, if one exists.
            default_allow_null: When a ColumnType or string is used as the value
                in the ColumnDescriptors mapping, the column will allow_null if
                default_allow_null is True.
            default_allow_nan: When a ColumnType or string is used as the value
                in the ColumnDescriptors mapping, the column will allow_nan if
                default_allow_nan is True.
            default_allow_inf: When a ColumnType or string is used as the value
                in the ColumnDescriptors mapping, the column will allow_inf if
                default_allow_inf is True.
        """
        # TODO(#1539): update Schema interface to use ColumnDescriptor everywhere.
        if "" in column_descs:
            raise ValueError('"" (the empty string) is not a supported column name')
        if grouping_column is not None and grouping_column not in column_descs:
            raise KeyError(
                f"Grouping column '{grouping_column}' is not one of the provided"
                " columns"
            )
        self._grouping_column = grouping_column
        if id_column is not None and id_column not in column_descs:
            raise KeyError(
                f"ID column '{id_column}' is not one of the provided columns"
            )
        self._id_column = id_column
        self._id_space = id_space

        supported_types: List[str] = [t.name for t in list(ColumnType)]
        column_types: List[str] = []
        for cd in column_descs.values():
            if isinstance(cd, ColumnDescriptor):
                column_types.append(cd.column_type.name)
            elif isinstance(cd, ColumnType):
                column_types.append(cd.name)
            else:
                column_types.append(cd)
        invalid_types = set(column_types) - set(supported_types)
        if invalid_types:
            raise ValueError(
                f"Column types {invalid_types} not supported; "
                f"use supported types {supported_types}."
            )

        self._column_descs: Dict[str, ColumnDescriptor] = {}
        for col, ty in column_descs.items():
            if isinstance(ty, ColumnDescriptor):
                self._column_descs[col] = ty
            elif isinstance(ty, ColumnType):
                self._column_descs[col] = ColumnDescriptor(
                    ty,
                    allow_null=default_allow_null,
                    allow_nan=default_allow_nan,
                    allow_inf=default_allow_inf,
                )
            else:
                self._column_descs[col] = ColumnDescriptor(
                    column_type=ColumnType[ty],
                    allow_null=default_allow_null,
                    allow_nan=default_allow_nan,
                    allow_inf=default_allow_inf,
                )

    @property
    def columns(self):
        """Return the names of the columns in the schema."""
        return self._column_descs.keys()

    @property
    def column_descs(self) -> Dict[str, ColumnDescriptor]:
        """Returns a mapping from column name to column descriptor."""
        return dict(self._column_descs)

    @property
    def column_types(self) -> Dict[str, str]:
        """Returns a mapping from column name to column type."""
        # TODO(#1539): Remove this
        return {col: desc.column_type.name for col, desc in self.column_descs.items()}

    @property
    def grouping_column(self) -> Optional[str]:
        """Returns the optional column that must be grouped by."""
        return self._grouping_column

    @property
    def id_column(self) -> Optional[str]:
        """Return whether the grouping column is an ID column."""
        return self._id_column

    @property
    def id_space(self) -> Optional[str]:
        """Return the ID space for this schema."""
        return self._id_space

    def __eq__(self, other: object) -> bool:
        """Returns True if schemas are equal.

        Args:
            other: Schema to check against.
        """
        if isinstance(other, Schema):
            return (
                self.column_descs == other.column_descs
                and self.grouping_column == other.grouping_column
                and self.id_column == other.id_column
                and self.id_space == other.id_space
            )
        return False

    def __getitem__(self, column: str) -> ColumnDescriptor:
        """Returns the data type for the given column.

        Args:
            column: The column to get the data type for.
        """
        return self.column_descs[column]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the columns in the schema."""
        return iter(self.column_descs)

    def __len__(self) -> int:
        """Return the number of columns in the schema."""
        return len(self.column_descs)

    def __repr__(self) -> str:
        """Return a string representation of self."""
        out = f"Schema({self.column_descs}"
        if self.grouping_column:
            out += f", grouping_column={self.grouping_column}"
        if self.id_column:
            out += f", id_column={self.id_column}"
        if self.id_space:
            out += f", id_space={self.id_space}"
        out += ")"
        return out


_SPARK_TO_ANALYTICS: Dict[DataType, ColumnType] = {
    IntegerType(): ColumnType.INTEGER,
    LongType(): ColumnType.INTEGER,
    DoubleType(): ColumnType.DECIMAL,
    FloatType(): ColumnType.DECIMAL,
    StringType(): ColumnType.VARCHAR,
    DateType(): ColumnType.DATE,
    TimestampType(): ColumnType.TIMESTAMP,
}
"""Mapping from Spark type to supported Analytics column types."""

_ANALYTICS_TO_SPARK = {
    "INTEGER": LongType(),
    "DECIMAL": DoubleType(),
    "VARCHAR": StringType(),
    "DATE": DateType(),
    "TIMESTAMP": TimestampType(),
}
"""Mapping from Analytics column types to Spark types."""

_ANALYTICS_TYPE_TO_COLUMN_DESCRIPTOR = {
    ColumnType.INTEGER: SparkIntegerColumnDescriptor,
    ColumnType.DECIMAL: SparkFloatColumnDescriptor,
    ColumnType.VARCHAR: SparkStringColumnDescriptor,
    ColumnType.DATE: SparkDateColumnDescriptor,
    ColumnType.TIMESTAMP: SparkTimestampColumnDescriptor,
}
"""Mapping from Analytics column types to Spark columns descriptor.

More information regarding Spark columns descriptor can be found in
:class:`~tmlt.core.domains.spark_domains.SparkColumnDescriptor`"""


def column_type_to_py_type(column_type: ColumnType) -> type:
    """Converts a ColumnType to a python type."""
    return column_type.value


def analytics_to_py_types(analytics_schema: Schema) -> Dict[str, type]:
    """Returns the mapping from column names to supported python types."""
    return {
        column_name: ColumnType[column_desc.column_type.name].value
        for column_name, column_desc in analytics_schema.column_descs.items()
    }


def analytics_to_spark_schema(analytics_schema: Schema) -> StructType:
    """Convert an Analytics schema to a Spark schema."""
    return StructType(
        [
            StructField(
                column_name,
                _ANALYTICS_TO_SPARK[column_desc.column_type.name],
                nullable=column_desc.allow_null,
            )
            for column_name, column_desc in analytics_schema.column_descs.items()
        ]
    )


def analytics_to_spark_columns_descriptor(
    analytics_schema: Schema,
) -> SparkColumnsDescriptor:
    """Convert a schema in Analytics representation to a Spark columns descriptor."""
    out: Dict[str, SparkColumnDescriptor] = {}
    for column_name, column_desc in analytics_schema.column_descs.items():
        if column_desc.column_type == ColumnType.DECIMAL:
            out[column_name] = SparkFloatColumnDescriptor(
                allow_nan=column_desc.allow_nan,
                allow_inf=column_desc.allow_inf,
                allow_null=column_desc.allow_null,
            )
        else:
            out[column_name] = _ANALYTICS_TYPE_TO_COLUMN_DESCRIPTOR[
                ColumnType[column_desc.column_type.name]
            ](allow_null=column_desc.allow_null)
    return out


def spark_schema_to_analytics_columns(
    spark_schema: StructType,
) -> Dict[str, ColumnDescriptor]:
    """Convert Spark schema to Analytics columns."""
    column_descs = {
        field.name: ColumnDescriptor(
            column_type=_SPARK_TO_ANALYTICS[field.dataType],
            allow_null=field.nullable,
            # Spark doesn't contain any information on whether a field contains NaNs,
            # so just assume that it does
            allow_nan=(_SPARK_TO_ANALYTICS[field.dataType] == ColumnType.DECIMAL),
            # Same for infinite values
            allow_inf=(_SPARK_TO_ANALYTICS[field.dataType] == ColumnType.DECIMAL),
        )
        for field in spark_schema
    }
    return column_descs


def spark_dataframe_domain_to_analytics_columns(
    domain: Domain,
) -> Dict[str, ColumnDescriptor]:
    """Convert a Spark dataframe domain to Analytics columns."""
    column_descs: Dict[str, ColumnDescriptor] = {}
    for column_name, descriptor in cast(SparkDataFrameDomain, domain).schema.items():
        if isinstance(descriptor, SparkFloatColumnDescriptor):
            column_descs[column_name] = ColumnDescriptor(
                ColumnType.DECIMAL,
                allow_null=descriptor.allow_null,
                allow_nan=descriptor.allow_nan,
                allow_inf=descriptor.allow_inf,
            )
        else:
            column_descs[column_name] = ColumnDescriptor(
                column_type=_SPARK_TO_ANALYTICS[descriptor.data_type],
                allow_null=descriptor.allow_null,
            )
    return column_descs
