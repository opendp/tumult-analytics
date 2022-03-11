"""Unit tests conversion functions in :mod:`~tmlt.analytics._schema`."""

# <placeholder: boilerplate>

from pyspark.sql import types as spark_types

from tmlt.analytics._schema import (
    ColumnType,
    Schema,
    analytics_to_py_types,
    analytics_to_spark_columns_descriptor,
    analytics_to_spark_schema,
    spark_dataframe_domain_to_analytics_columns,
    spark_schema_to_analytics_columns,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.utils.testing import PySparkTest


class TestSchemaConversions(PySparkTest):
    """Unit tests for schema conversions."""

    def test_analytics_to_py_types(self) -> None:
        """Make sure SQL92 types are mapped to the right python types."""
        columns = {"1": "INTEGER", "2": "BIT", "3": "DECIMAL", "4": "VARCHAR"}
        py_columns = analytics_to_py_types(Schema(columns))
        self.assertEqual(py_columns["1"], int)
        self.assertEqual(py_columns["2"], bool)
        self.assertEqual(py_columns["3"], float)
        self.assertEqual(py_columns["4"], str)

    def test_analytics_to_spark_schema(self):
        """Make sure conversion to Spark schema works properly."""
        analytics_schema = Schema({"1": "INTEGER", "2": "DECIMAL", "3": "VARCHAR"})
        expected_spark_schema = spark_types.StructType(
            [
                spark_types.StructField("1", spark_types.LongType(), nullable=False),
                spark_types.StructField("2", spark_types.DoubleType(), nullable=False),
                spark_types.StructField("3", spark_types.StringType(), nullable=False),
            ]
        )
        actual_spark_schema = analytics_to_spark_schema(analytics_schema)
        self.assertEqual(actual_spark_schema, expected_spark_schema)

    def test_analytics_to_spark_columns_descriptor_schema(self) -> None:
        """Make sure conversion to Spark columns descriptor works properly."""

        # boolean and string aren't yet supported by domains.py -> convert_spark_schema
        columns = {"1": "INTEGER", "2": "DECIMAL"}
        analytics_schema = Schema(columns)
        spark_columns_descriptor = analytics_to_spark_columns_descriptor(
            analytics_schema
        )
        self.assertEqual(2, len(spark_columns_descriptor))
        self.assertTrue("1" in spark_columns_descriptor)
        self.assertTrue("2" in spark_columns_descriptor)
        self.assertIsInstance(
            spark_columns_descriptor["1"], SparkIntegerColumnDescriptor
        )
        self.assertIsInstance(spark_columns_descriptor["2"], SparkFloatColumnDescriptor)

    def test_spark_conversions(self) -> None:
        """Make sure conversion from Spark schema/domain to Analytics works."""
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.StringType()),
                spark_types.StructField("B", spark_types.IntegerType()),
                spark_types.StructField("C", spark_types.LongType()),
                spark_types.StructField("D", spark_types.FloatType()),
                spark_types.StructField("E", spark_types.DoubleType()),
            ]
        )
        expected = {
            "A": ColumnType.VARCHAR,
            "B": ColumnType.INTEGER,
            "C": ColumnType.INTEGER,
            "D": ColumnType.DECIMAL,
            "E": ColumnType.DECIMAL,
        }

        # First test the schema --> columns conversion
        analytics_columns_1 = spark_schema_to_analytics_columns(spark_schema)
        self.assertEqual(expected, analytics_columns_1)

        # Now test the domain --> columns conversion
        domain = SparkDataFrameDomain.from_spark_schema(spark_schema)
        analytics_columns_2 = spark_dataframe_domain_to_analytics_columns(domain)
        self.assertEqual(expected, analytics_columns_2)
