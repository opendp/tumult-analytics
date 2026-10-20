"""Unit tests conversion functions in :mod:`~tmlt.analytics._schema`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

import datetime

import pytest
from pyspark.sql import types as spark_types
from tmlt.core.domains.spark_domains import (
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)

from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_py_types,
    analytics_to_spark_columns_descriptor,
    analytics_to_spark_schema,
    spark_dataframe_domain_to_analytics_columns,
    spark_schema_to_analytics_columns,
)


class TestSchemaConversions:
    """Unit tests for schema conversions."""

    def test_analytics_to_py_types(self) -> None:
        """Make sure SQL92 types are mapped to the right python types."""
        columns = {
            "int_col": "INTEGER",
            "float_col": "DECIMAL",
            "str_col": "VARCHAR",
            "date_col": "DATE",
            "timestamp_col": "TIMESTAMP",
        }
        py_columns = analytics_to_py_types(Schema(columns))
        assert py_columns["int_col"] == int
        assert py_columns["float_col"] == float
        assert py_columns["str_col"] == str
        assert py_columns["date_col"] == datetime.date
        assert py_columns["timestamp_col"] == datetime.datetime

    def test_analytics_to_spark_schema(self):
        """Make sure conversion to Spark schema works properly."""
        analytics_schema = Schema(
            {
                "int_col": "INTEGER",
                "float_col": "DECIMAL",
                "str_col": "VARCHAR",
                "date_col": "DATE",
                "timestamp_col": "TIMESTAMP",
            }
        )
        expected_spark_schema = spark_types.StructType(
            [
                spark_types.StructField(
                    "int_col", spark_types.LongType(), nullable=False
                ),
                spark_types.StructField(
                    "float_col", spark_types.DoubleType(), nullable=False
                ),
                spark_types.StructField(
                    "str_col", spark_types.StringType(), nullable=False
                ),
                spark_types.StructField(
                    "date_col", spark_types.DateType(), nullable=False
                ),
                spark_types.StructField(
                    "timestamp_col", spark_types.TimestampType(), nullable=False
                ),
            ]
        )
        actual_spark_schema = analytics_to_spark_schema(analytics_schema)
        assert actual_spark_schema == expected_spark_schema

    def test_analytics_to_spark_schema_with_null(self):
        """Test conversion to Spark schema with allow_null=True."""
        analytics_schema = Schema(
            {
                "int_col": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                "float_col": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
                "str_col": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                "date_col": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                "timestamp_col": ColumnDescriptor(
                    ColumnType.TIMESTAMP, allow_null=True
                ),
            }
        )
        expected_spark_schema = spark_types.StructType(
            [
                spark_types.StructField(
                    "int_col", spark_types.LongType(), nullable=True
                ),
                spark_types.StructField(
                    "float_col", spark_types.DoubleType(), nullable=True
                ),
                spark_types.StructField(
                    "str_col", spark_types.StringType(), nullable=True
                ),
                spark_types.StructField(
                    "date_col", spark_types.DateType(), nullable=True
                ),
                spark_types.StructField(
                    "timestamp_col", spark_types.TimestampType(), nullable=True
                ),
            ]
        )
        actual_spark_schema = analytics_to_spark_schema(analytics_schema)
        assert actual_spark_schema == expected_spark_schema

    @pytest.mark.parametrize(
        "analytics_schema,expected",
        [
            (
                Schema(
                    {
                        "int_col": "INTEGER",
                        "float_col": "DECIMAL",
                        "str_col": "VARCHAR",
                        "date_col": "DATE",
                        "timestamp_col": "TIMESTAMP",
                    }
                ),
                {
                    "int_col": SparkIntegerColumnDescriptor(),
                    "float_col": SparkFloatColumnDescriptor(),
                    "str_col": SparkStringColumnDescriptor(),
                    "date_col": SparkDateColumnDescriptor(),
                    "timestamp_col": SparkTimestampColumnDescriptor(),
                },
            ),
            (
                Schema(
                    {
                        "int_col": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=True
                        ),
                        "float_col": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=True
                        ),
                        "str_col": ColumnDescriptor(
                            ColumnType.VARCHAR, allow_null=True
                        ),
                        "date_col": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "timestamp_col": ColumnDescriptor(
                            ColumnType.TIMESTAMP, allow_null=True
                        ),
                    }
                ),
                {
                    "int_col": SparkIntegerColumnDescriptor(allow_null=True),
                    "float_col": SparkFloatColumnDescriptor(allow_null=True),
                    "str_col": SparkStringColumnDescriptor(allow_null=True),
                    "date_col": SparkDateColumnDescriptor(allow_null=True),
                    "timestamp_col": SparkTimestampColumnDescriptor(allow_null=True),
                },
            ),
            (
                Schema(
                    {
                        "nullable_int_col": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=True
                        ),
                        "int_col": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                        "nullable_float_col": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=True, allow_nan=False
                        ),
                        "nullable_nan_float_col": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=True, allow_nan=True
                        ),
                        "nan_float_col": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=False, allow_nan=True
                        ),
                    }
                ),
                {
                    "nullable_int_col": SparkIntegerColumnDescriptor(allow_null=True),
                    "int_col": SparkIntegerColumnDescriptor(allow_null=False),
                    "nullable_float_col": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=False, allow_inf=False
                    ),
                    "nullable_nan_float_col": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=False
                    ),
                    "nan_float_col": SparkFloatColumnDescriptor(
                        allow_null=False, allow_nan=True, allow_inf=False
                    ),
                },
            ),
        ],
    )
    def test_analytics_to_spark_columns_descriptor_schema(
        self, analytics_schema: Schema, expected: SparkColumnsDescriptor
    ) -> None:
        """Test conversion to Spark columns descriptor works correctly."""
        spark_columns_descriptor = analytics_to_spark_columns_descriptor(
            analytics_schema
        )
        # Since we don't care if the dictionaries have the same keys in the
        # same order, we can't use dictionary equality here
        for k in list(expected.keys()):
            assert k in spark_columns_descriptor
        for k in list(expected.keys()):
            assert expected[k] == spark_columns_descriptor[k]

    def test_spark_conversions(self) -> None:
        """Make sure conversion from Spark schema/domain to Analytics works."""
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField(
                    "str_col", spark_types.StringType(), nullable=False
                ),
                spark_types.StructField(
                    "int_col", spark_types.LongType(), nullable=False
                ),
                spark_types.StructField(
                    "float_col", spark_types.DoubleType(), nullable=False
                ),
                spark_types.StructField(
                    "date_col", spark_types.DateType(), nullable=False
                ),
                spark_types.StructField(
                    "timestamp_col", spark_types.TimestampType(), nullable=False
                ),
            ]
        )
        expected = {
            "str_col": ColumnDescriptor(ColumnType.VARCHAR),
            "int_col": ColumnDescriptor(ColumnType.INTEGER),
            "float_col": ColumnDescriptor(
                ColumnType.DECIMAL, allow_nan=True, allow_inf=True
            ),
            "date_col": ColumnDescriptor(ColumnType.DATE),
            "timestamp_col": ColumnDescriptor(ColumnType.TIMESTAMP),
        }

        # First test the schema --> columns conversion
        analytics_columns_1 = spark_schema_to_analytics_columns(spark_schema)
        assert expected == analytics_columns_1

        # Now test the domain --> columns conversion
        domain = SparkDataFrameDomain.from_spark_schema(spark_schema)
        analytics_columns_2 = spark_dataframe_domain_to_analytics_columns(domain)
        assert expected == analytics_columns_2

    def test_spark_conversions_with_null(self) -> None:
        """Test that conversion from Spark to Analytics works with allow_null=True."""
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField(
                    "str_col", spark_types.StringType(), nullable=True
                ),
                spark_types.StructField(
                    "int_col", spark_types.LongType(), nullable=True
                ),
                spark_types.StructField(
                    "float_col", spark_types.DoubleType(), nullable=True
                ),
                spark_types.StructField(
                    "date_col", spark_types.DateType(), nullable=True
                ),
                spark_types.StructField(
                    "timestamp_col", spark_types.TimestampType(), nullable=True
                ),
            ]
        )
        expected = {
            "str_col": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "int_col": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "float_col": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
            "date_col": ColumnDescriptor(ColumnType.DATE, allow_null=True),
            "timestamp_col": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
        }
        # First test the schema --> columns conversion
        analytics_columns_1 = spark_schema_to_analytics_columns(spark_schema)
        assert expected == analytics_columns_1

        # Now test the domain --> columns conversion
        domain = SparkDataFrameDomain.from_spark_schema(spark_schema)
        analytics_columns_2 = spark_dataframe_domain_to_analytics_columns(domain)
        assert expected == analytics_columns_2

    def test_domain_conversion(self) -> None:
        """Test conversion to and from a SparkDataFrameDomain."""
        analytics_columns_1 = Schema(
            {
                "str_col": ColumnDescriptor(ColumnType.VARCHAR),
                "nullable_str_col": ColumnDescriptor(
                    ColumnType.VARCHAR, allow_null=True
                ),
                "int_col": ColumnDescriptor(ColumnType.INTEGER),
                "nullable_int_col": ColumnDescriptor(
                    ColumnType.INTEGER, allow_null=True
                ),
                "inf_float_col": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
                "nan_float_col": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True),
                "nullable_float_col": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True
                ),
                "nullable_nan_inf_float_col": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
                ),
                "date_col": ColumnDescriptor(ColumnType.DATE),
                "nullable_date_col": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                "timestamp_col": ColumnDescriptor(ColumnType.TIMESTAMP),
                "nullable_timestamp_col": ColumnDescriptor(
                    ColumnType.TIMESTAMP, allow_null=True
                ),
            }
        )
        expected = {
            "str_col": SparkStringColumnDescriptor(),
            "nullable_str_col": SparkStringColumnDescriptor(allow_null=True),
            "int_col": SparkIntegerColumnDescriptor(),
            "nullable_int_col": SparkIntegerColumnDescriptor(allow_null=True),
            "inf_float_col": SparkFloatColumnDescriptor(allow_inf=True),
            "nan_float_col": SparkFloatColumnDescriptor(allow_nan=True),
            "nullable_float_col": SparkFloatColumnDescriptor(allow_null=True),
            "nullable_nan_inf_float_col": SparkFloatColumnDescriptor(
                allow_null=True, allow_nan=True, allow_inf=True
            ),
            "date_col": SparkDateColumnDescriptor(),
            "nullable_date_col": SparkDateColumnDescriptor(allow_null=True),
            "timestamp_col": SparkTimestampColumnDescriptor(),
            "nullable_timestamp_col": SparkTimestampColumnDescriptor(allow_null=True),
        }
        # First test the schema -> SparkColumnsDescriptor conversion
        got = analytics_to_spark_columns_descriptor(analytics_columns_1)
        assert got == expected
        # Now test SparkDataFrameDomain -> Analytics conversion
        analytics_columns_2 = Schema(
            spark_dataframe_domain_to_analytics_columns(SparkDataFrameDomain(got))
        )
        assert analytics_columns_2 == analytics_columns_1
