"""Logic for coercing Spark dataframes into forms usable by Tumult Analytics."""

# <placeholder: boilerplate>

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import (
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

SUPPORTED_SPARK_TYPES = {
    IntegerType(),
    LongType(),
    FloatType(),
    DoubleType(),
    StringType(),
    DateType(),
    TimestampType(),
}
"""Set of Spark data types supported by Tumult Analytics.

Support for Spark data types in Analytics is currently as follows:

.. list-table::
   :header-rows: 1

   * - Type
     - Supported
   * - :class:`~pyspark.sql.types.LongType`
     - yes
   * - :class:`~pyspark.sql.types.IntegerType`
     - yes, by coercion to :class:`~pyspark.sql.types.LongType`
   * - :class:`~pyspark.sql.types.DoubleType`
     - yes
   * - :class:`~pyspark.sql.types.FloatType`
     - yes, by coercion to :class:`~pyspark.sql.types.DoubleType`
   * - :class:`~pyspark.sql.types.StringType`
     - yes
   * - :class:`~pyspark.sql.types.DateType`
     - yes
   * - :class:`~pyspark.sql.types.TimestampType`
     - yes
   * - Other Spark types
     - no

Columns with unsupported types must be dropped or converted to supported ones
before loading the data into Analytics.
"""

TYPE_COERCION_MAP = {IntegerType(): LongType(), FloatType(): DoubleType()}
"""Mapping describing how Spark's data types are coerced by Tumult Analytics."""


def _fail_if_dataframe_contains_unsupported_types(dataframe: DataFrame):
    """Raises an error if DataFrame contains unsupported Spark column types."""
    unsupported_types = [
        (field.name, field.dataType)
        for field in dataframe.schema
        if field.dataType not in SUPPORTED_SPARK_TYPES
    ]

    if unsupported_types:
        raise ValueError(
            "Unsupported Spark data type: Tumult Analytics does not yet support the"
            f" Spark data types for the following columns: {unsupported_types}."
            " Consider casting these columns into one of the supported Spark data"
            f" types: {SUPPORTED_SPARK_TYPES}."
        )


def _fail_if_dataframe_contains_nulls_or_nans(dataframe: DataFrame):
    """Raises an error if DataFrame contains nulls or nans."""
    for (i, column) in enumerate(dataframe.columns):
        if (
            dataframe.schema[i].dataType == FloatType()
            or dataframe.schema[i].dataType == DoubleType()
        ):
            if dataframe.where(
                sf.isnull(sf.col(column))  # pylint: disable=no-member
                | sf.isnan(sf.col(column))  # pylint: disable=no-member
            ).first():
                raise ValueError("This DataFrame contains a null or nan value")
        else:
            if dataframe.where(
                sf.isnull(sf.col(column))  # pylint: disable=no-member
            ).first():
                raise ValueError("This DataFrame contains a null or nan value")


def coerce_spark_schema_or_fail(
    dataframe: DataFrame, allow_nan_and_null: bool = False
) -> DataFrame:
    """Returns a new DataFrame where all column data types are supported.

    In particular, this function raises an error:
        * if `dataframe` contains a column type not listed in
            SUPPORTED_SPARK_TYPES
        * if `dataframe` contains nulls or nans in any column
          (if allow_nan_and_null is not True)
        * if `dataframe` contains a column named "" (the empty string)

    This function returns a DataFrame where all column types
        * are coerced according to TYPE_COERCION_MAP if necessary
        * are marked as non-nullable
    """
    if "" in dataframe.columns:
        raise ValueError('This DataFrame contains a column named "" (the empty string)')

    _fail_if_dataframe_contains_unsupported_types(dataframe)

    if not allow_nan_and_null:
        _fail_if_dataframe_contains_nulls_or_nans(dataframe)

    contains_nullable_fields = any(field.nullable for field in dataframe.schema)
    requires_coercion = any(
        field.dataType in TYPE_COERCION_MAP for field in dataframe.schema
    )
    if not (requires_coercion or contains_nullable_fields):
        return dataframe
    if not (requires_coercion) and allow_nan_and_null:
        return dataframe
    spark = SparkSession.builder.getOrCreate()
    coerced_fields = []
    for field in dataframe.schema:
        coerced_fields.append(
            StructField(
                field.name,
                TYPE_COERCION_MAP.get(field.dataType, field.dataType),
                nullable=allow_nan_and_null,
            )
        )

    return spark.createDataFrame(dataframe.rdd, schema=StructType(coerced_fields))
