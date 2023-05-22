"""Unit tests conversion functions in :mod:`~tmlt.analytics._schema`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
# pylint: disable=no-self-use
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
            "1": "INTEGER",
            "2": "DECIMAL",
            "3": "VARCHAR",
            "4": "DATE",
            "5": "TIMESTAMP",
        }
        py_columns = analytics_to_py_types(Schema(columns))
        assert py_columns["1"] == int
        assert py_columns["2"] == float
        assert py_columns["3"] == str
        assert py_columns["4"] == datetime.date
        assert py_columns["5"] == datetime.datetime

    def test_analytics_to_spark_schema(self):
        """Make sure conversion to Spark schema works properly."""
        analytics_schema = Schema(
            {
                "1": "INTEGER",
                "2": "DECIMAL",
                "3": "VARCHAR",
                "4": "DATE",
                "5": "TIMESTAMP",
            }
        )
        expected_spark_schema = spark_types.StructType(
            [
                spark_types.StructField("1", spark_types.LongType(), nullable=False),
                spark_types.StructField("2", spark_types.DoubleType(), nullable=False),
                spark_types.StructField("3", spark_types.StringType(), nullable=False),
                spark_types.StructField("4", spark_types.DateType(), nullable=False),
                spark_types.StructField(
                    "5", spark_types.TimestampType(), nullable=False
                ),
            ]
        )
        actual_spark_schema = analytics_to_spark_schema(analytics_schema)
        assert actual_spark_schema == expected_spark_schema

    def test_analytics_to_spark_schema_with_null(self):
        """Test conversion to Spark schema with allow_null=True."""
        analytics_schema = Schema(
            {
                "1": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                "2": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
                "3": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                "4": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                "5": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
            }
        )
        expected_spark_schema = spark_types.StructType(
            [
                spark_types.StructField("1", spark_types.LongType(), nullable=True),
                spark_types.StructField("2", spark_types.DoubleType(), nullable=True),
                spark_types.StructField("3", spark_types.StringType(), nullable=True),
                spark_types.StructField("4", spark_types.DateType(), nullable=True),
                spark_types.StructField(
                    "5", spark_types.TimestampType(), nullable=True
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
                        "1": "INTEGER",
                        "2": "DECIMAL",
                        "3": "VARCHAR",
                        "4": "DATE",
                        "5": "TIMESTAMP",
                    }
                ),
                {
                    "1": SparkIntegerColumnDescriptor(),
                    "2": SparkFloatColumnDescriptor(),
                    "3": SparkStringColumnDescriptor(),
                    "4": SparkDateColumnDescriptor(),
                    "5": SparkTimestampColumnDescriptor(),
                },
            ),
            (
                Schema(
                    {
                        "1": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "2": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
                        "3": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "4": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "5": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                    }
                ),
                {
                    "1": SparkIntegerColumnDescriptor(allow_null=True),
                    "2": SparkFloatColumnDescriptor(allow_null=True),
                    "3": SparkStringColumnDescriptor(allow_null=True),
                    "4": SparkDateColumnDescriptor(allow_null=True),
                    "5": SparkTimestampColumnDescriptor(allow_null=True),
                },
            ),
            (
                Schema(
                    {
                        "i": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "i2": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                        "d1": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=True, allow_nan=False
                        ),
                        "d2": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=True, allow_nan=True
                        ),
                        "d3": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=False, allow_nan=True
                        ),
                    }
                ),
                {
                    "i": SparkIntegerColumnDescriptor(allow_null=True),
                    "i2": SparkIntegerColumnDescriptor(allow_null=False),
                    "d1": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=False, allow_inf=False
                    ),
                    "d2": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=False
                    ),
                    "d3": SparkFloatColumnDescriptor(
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
                spark_types.StructField("A", spark_types.StringType(), nullable=False),
                spark_types.StructField("B", spark_types.IntegerType(), nullable=False),
                spark_types.StructField("C", spark_types.LongType(), nullable=False),
                spark_types.StructField("D", spark_types.FloatType(), nullable=False),
                spark_types.StructField("E", spark_types.DoubleType(), nullable=False),
                spark_types.StructField("F", spark_types.DateType(), nullable=False),
                spark_types.StructField(
                    "G", spark_types.TimestampType(), nullable=False
                ),
            ]
        )
        expected = {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "C": ColumnDescriptor(ColumnType.INTEGER),
            "D": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True, allow_inf=True),
            "E": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True, allow_inf=True),
            "F": ColumnDescriptor(ColumnType.DATE),
            "G": ColumnDescriptor(ColumnType.TIMESTAMP),
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
                spark_types.StructField("A", spark_types.StringType(), nullable=True),
                spark_types.StructField("B", spark_types.IntegerType(), nullable=True),
                spark_types.StructField("C", spark_types.LongType(), nullable=True),
                spark_types.StructField("D", spark_types.FloatType(), nullable=True),
                spark_types.StructField("E", spark_types.DoubleType(), nullable=True),
                spark_types.StructField("F", spark_types.DateType(), nullable=True),
                spark_types.StructField(
                    "G", spark_types.TimestampType(), nullable=True
                ),
            ]
        )
        expected = {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "C": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "D": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
            "E": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
            "F": ColumnDescriptor(ColumnType.DATE, allow_null=True),
            "G": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
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
                "A": ColumnDescriptor(ColumnType.VARCHAR),
                "A2": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                "B": ColumnDescriptor(ColumnType.INTEGER),
                "B2": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                "C": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
                "D": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True),
                "E": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
                "F": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
                ),
                "G": ColumnDescriptor(ColumnType.DATE),
                "G2": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                "H": ColumnDescriptor(ColumnType.TIMESTAMP),
                "H2": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
            }
        )
        expected = {
            "A": SparkStringColumnDescriptor(),
            "A2": SparkStringColumnDescriptor(allow_null=True),
            "B": SparkIntegerColumnDescriptor(),
            "B2": SparkIntegerColumnDescriptor(allow_null=True),
            "C": SparkFloatColumnDescriptor(allow_inf=True),
            "D": SparkFloatColumnDescriptor(allow_nan=True),
            "E": SparkFloatColumnDescriptor(allow_null=True),
            "F": SparkFloatColumnDescriptor(
                allow_null=True, allow_nan=True, allow_inf=True
            ),
            "G": SparkDateColumnDescriptor(),
            "G2": SparkDateColumnDescriptor(allow_null=True),
            "H": SparkTimestampColumnDescriptor(),
            "H2": SparkTimestampColumnDescriptor(allow_null=True),
        }
        # First test the schema -> SparkColumnsDescriptor conversion
        got = analytics_to_spark_columns_descriptor(analytics_columns_1)
        assert got == expected
        # Now test SparkDataFrameDomain -> Analytics conversion
        analytics_columns_2 = Schema(
            spark_dataframe_domain_to_analytics_columns(SparkDataFrameDomain(got))
        )
        assert analytics_columns_2 == analytics_columns_1
