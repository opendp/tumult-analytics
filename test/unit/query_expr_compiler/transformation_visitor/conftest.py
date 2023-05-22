"""Fixtures and data for TransformationVisitor tests."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Dict, List, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DateType,
    DoubleType,
    FloatType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.metrics import AddRemoveKeys, DictMetric, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType
from tmlt.analytics._table_identifier import Identifier, NamedTable, TableCollection
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics._transformation_utils import get_table_from_ref

from ....conftest import assert_frame_equal_with_sort

# Example date and timestamp
DATE1 = datetime.date.fromisoformat("2022-01-01")
TIMESTAMP1 = datetime.datetime.fromisoformat("2022-01-01T12:30:00")


def chain_to_list(t: ChainTT) -> List[Transformation]:
    """Turns a ChainTT's tree into a list, in order from left to right."""
    left: List[Transformation]
    if not isinstance(t.transformation1, ChainTT):
        left = [t.transformation1]
    else:
        left = chain_to_list(t.transformation1)
    right: List[Transformation]
    if not isinstance(t.transformation2, ChainTT):
        right = [t.transformation2]
    else:
        right = chain_to_list(t.transformation2)
    return left + right


@pytest.fixture(scope="class")
def _dataframes(request, spark):
    """Returns dataframes for use in test functions."""
    rows1 = spark.createDataFrame(
        pd.DataFrame([["0", 0, 0.1, DATE1, TIMESTAMP1]]),
        schema=StructType(
            [
                StructField("S", StringType(), True),
                StructField("I", LongType(), True),
                StructField("F", DoubleType(), True),
                StructField("D", DateType(), True),
                StructField("T", TimestampType(), True),
            ]
        ),
    )
    rows2 = spark.createDataFrame(
        pd.DataFrame([[0, "a"]]),
        schema=StructType(
            [
                StructField("I", LongType(), True),
                StructField("field", StringType(), True),
            ]
        ),
    )
    rows_infs_nans = spark.createDataFrame(
        pd.DataFrame(
            [[float("inf"), "string", float("nan")], [float("-inf"), None, 1.5]]
        ),
        schema=StructType(
            [
                StructField("inf", FloatType(), True),
                StructField("null", StringType(), True),
                StructField("nan", DoubleType(), True),
            ]
        ),
    )
    ids1 = spark.createDataFrame(
        pd.DataFrame([[1, "0", 0, 0.1, DATE1, TIMESTAMP1]]),
        schema=StructType(
            [
                StructField("id", LongType(), True),
                StructField("S", StringType(), True),
                StructField("I", LongType(), True),
                StructField("F", DoubleType(), True),
                StructField("D", DateType(), True),
                StructField("T", TimestampType(), True),
            ]
        ),
    )
    ids2 = spark.createDataFrame(
        pd.DataFrame([[1, 0, "a"]]),
        schema=StructType(
            [
                StructField("id", LongType(), True),
                StructField("I", LongType(), True),
                StructField("field", StringType(), True),
            ]
        ),
    )
    ids_infs_nans = spark.createDataFrame(
        pd.DataFrame(
            [[1, float("inf"), "string", float("nan")], [1, float("-inf"), None, 1.5]]
        ),
        schema=StructType(
            [
                StructField("id", LongType(), True),
                StructField("inf", FloatType(), True),
                StructField("null", StringType(), True),
                StructField("nan", DoubleType(), True),
            ]
        ),
    )

    ids_duplicates = spark.createDataFrame(
        pd.DataFrame(
            [
                [123, 123, 123, 456, 456, 456, 789, 789, 789, 789],
                ["A", "A", "A", "A", "B", "C", "A", "A", "B", "B"],
            ]
        ).transpose(),
        schema=StructType(
            [StructField("id", LongType(), True), StructField("St", StringType(), True)]
        ),
    )

    request.cls.input_data = {
        NamedTable("rows1"): rows1,
        NamedTable("rows2"): rows2,
        NamedTable("rows_infs_nans"): rows_infs_nans,
        TableCollection("ids"): {
            NamedTable("ids1"): ids1,
            NamedTable("ids2"): ids2,
            NamedTable("ids_infs_nans"): ids_infs_nans,
            NamedTable("ids_duplicates"): ids_duplicates,
        },
    }
    request.cls.dataframes = {
        "rows1": rows1,
        "rows2": rows2,
        "rows_infs_nans": rows_infs_nans,
        "ids1": ids1,
        "ids2": ids2,
        "ids_infs_nans": ids_infs_nans,
        "ids_duplicates": ids_duplicates,
    }


@pytest.fixture(scope="class")
def _catalog(request):
    """Returns a catalog for use in test functions."""
    catalog = Catalog()
    catalog.add_private_table(
        "rows1",
        {
            "S": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "I": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "F": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
            "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
            "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
        },
    )
    catalog.add_private_table(
        "rows2",
        {
            "I": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "field": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
    )
    catalog.add_private_table(
        "rows_infs_nans",
        {
            "inf": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_inf=True
            ),
            "null": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "nan": ColumnDescriptor(
                ColumnType.DECIMAL, allow_nan=True, allow_null=True
            ),
        },
    )
    catalog.add_private_table(
        "ids1",
        {
            "id": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "S": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "I": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "F": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
            "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
            "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
        },
        grouping_column="id",
    )
    catalog.add_private_table(
        "ids2",
        {
            "id": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "I": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "field": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
        grouping_column="id",
    )
    catalog.add_private_table(
        "ids_infs_nans",
        {
            "id": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "inf": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_inf=True
            ),
            "null": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "nan": ColumnDescriptor(
                ColumnType.DECIMAL, allow_nan=True, allow_null=True
            ),
        },
        grouping_column="id",
    )
    catalog.add_private_table(
        "ids_duplicates",
        {
            "id": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "St": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
        grouping_column="id",
    )
    catalog.add_public_table(
        "public",
        {
            "S": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "I": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "public": ColumnDescriptor(ColumnType.VARCHAR),
        },
    )
    request.cls.catalog = catalog


@pytest.fixture(scope="class")
def _visitor(spark, request):
    """Returns a TransformationVisitor for use in test functions."""
    input_domain = DictDomain(
        {
            NamedTable("rows1"): SparkDataFrameDomain(
                {
                    "S": SparkStringColumnDescriptor(allow_null=True),
                    "I": SparkIntegerColumnDescriptor(allow_null=True),
                    "F": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                    "D": SparkDateColumnDescriptor(allow_null=True),
                    "T": SparkTimestampColumnDescriptor(allow_null=True),
                }
            ),
            NamedTable("rows2"): SparkDataFrameDomain(
                {
                    "I": SparkIntegerColumnDescriptor(allow_null=True),
                    "field": SparkStringColumnDescriptor(allow_null=True),
                }
            ),
            NamedTable("rows_infs_nans"): SparkDataFrameDomain(
                {
                    "inf": SparkFloatColumnDescriptor(allow_null=True, allow_inf=True),
                    "null": SparkStringColumnDescriptor(allow_null=True),
                    "nan": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
                }
            ),
            TableCollection("ids"): DictDomain(
                {
                    NamedTable("ids1"): SparkDataFrameDomain(
                        {
                            "id": SparkIntegerColumnDescriptor(allow_null=True),
                            "S": SparkStringColumnDescriptor(allow_null=True),
                            "I": SparkIntegerColumnDescriptor(allow_null=True),
                            "F": SparkFloatColumnDescriptor(
                                allow_null=True, allow_nan=True, allow_inf=True
                            ),
                            "D": SparkDateColumnDescriptor(allow_null=True),
                            "T": SparkTimestampColumnDescriptor(allow_null=True),
                        }
                    ),
                    NamedTable("ids2"): SparkDataFrameDomain(
                        {
                            "id": SparkIntegerColumnDescriptor(allow_null=True),
                            "I": SparkIntegerColumnDescriptor(allow_null=True),
                            "field": SparkStringColumnDescriptor(allow_null=True),
                        }
                    ),
                    NamedTable("ids_infs_nans"): SparkDataFrameDomain(
                        {
                            "id": SparkIntegerColumnDescriptor(allow_null=True),
                            "inf": SparkFloatColumnDescriptor(
                                allow_null=True, allow_inf=True
                            ),
                            "null": SparkStringColumnDescriptor(allow_null=True),
                            "nan": SparkFloatColumnDescriptor(
                                allow_nan=True, allow_null=True
                            ),
                        }
                    ),
                    NamedTable("ids_duplicates"): SparkDataFrameDomain(
                        {
                            "id": SparkIntegerColumnDescriptor(allow_null=True),
                            "St": SparkStringColumnDescriptor(allow_null=True),
                        }
                    ),
                }
            ),
        }
    )
    input_metric = DictMetric(
        {
            NamedTable("rows1"): SymmetricDifference(),
            NamedTable("rows2"): SymmetricDifference(),
            NamedTable("rows_infs_nans"): SymmetricDifference(),
            TableCollection("ids"): AddRemoveKeys(
                {
                    NamedTable("ids1"): "id",
                    NamedTable("ids2"): "id",
                    NamedTable("ids_infs_nans"): "id",
                    NamedTable("ids_duplicates"): "id",
                }
            ),
        }
    )
    public_sources = {
        "public": spark.createDataFrame(
            pd.DataFrame({"S": ["0", "1"], "I": [0, 1], "public": ["x", "y"]}),
            schema=StructType(
                [
                    StructField("S", StringType(), True),
                    StructField("I", LongType(), True),
                    StructField("public", StringType(), False),
                ]
            ),
        )
    }
    visitor = TransformationVisitor(
        input_domain=input_domain,
        input_metric=input_metric,
        mechanism=NoiseMechanism.LAPLACE,
        public_sources=public_sources,
        table_constraints={
            NamedTable(t): []
            for t in (
                "rows1",
                "rows2",
                "rows_infs_nans",
                "ids1",
                "ids2",
                "ids_infs_nans",
                "ids_duplicates",
            )
        },
    )
    request.cls.visitor = visitor


@pytest.mark.usefixtures("_visitor", "_catalog", "_dataframes")
class TestTransformationVisitor:
    """Base class for transformation visitor tests."""

    visitor: TransformationVisitor
    catalog: Catalog
    input_data: Dict[Identifier, Union[DataFrame, Dict[Identifier, DataFrame]]]
    dataframes: Dict[str, DataFrame]

    def _get_result(self, t: Transformation, ref: TableReference) -> pd.DataFrame:
        return get_table_from_ref(t, ref)(self.input_data).toPandas()

    def _validate_result(
        self, t: Transformation, ref: TableReference, transformed_df: DataFrame
    ) -> None:
        assert isinstance(t.output_domain, DictDomain)
        assert isinstance(t.output_metric, (DictMetric, AddRemoveKeys))
        result_df = self._get_result(t, ref)
        assert_frame_equal_with_sort(result_df, transformed_df)


@pytest.fixture(scope="class")
def _nulls_catalog(request):
    """Returns a catalog for use in test functions for null/NaN/inf behavior."""
    df_columns = {
        "not_null": ColumnDescriptor(ColumnType.DECIMAL, allow_null=False),
        "null": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
        "nan": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True),
        "inf": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
        "null_nan": ColumnDescriptor(
            ColumnType.DECIMAL, allow_null=True, allow_nan=True
        ),
        "null_inf": ColumnDescriptor(
            ColumnType.DECIMAL, allow_null=True, allow_inf=True
        ),
        "nan_inf": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True, allow_inf=True),
        "null_nan_inf": ColumnDescriptor(
            ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
        ),
    }
    catalog = Catalog()
    catalog.add_private_table("rows", df_columns)
    catalog.add_private_table(
        "ids",
        {"id": ColumnDescriptor(ColumnType.INTEGER, allow_null=True), **df_columns},
        grouping_column="id",
    )
    request.cls.catalog = catalog


@pytest.fixture(scope="class")
def _nulls_visitor(request):
    """Returns a TransformationVisitor for tests of null/NaN/inf behavior."""
    df_columns: Dict[str, SparkColumnDescriptor] = {
        "not_null": SparkFloatColumnDescriptor(allow_null=False),
        "null": SparkFloatColumnDescriptor(allow_null=True),
        "nan": SparkFloatColumnDescriptor(allow_nan=True),
        "inf": SparkFloatColumnDescriptor(allow_inf=True),
        "null_nan": SparkFloatColumnDescriptor(allow_null=True, allow_nan=True),
        "null_inf": SparkFloatColumnDescriptor(allow_null=True, allow_inf=True),
        "nan_inf": SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True),
        "null_nan_inf": SparkFloatColumnDescriptor(
            allow_null=True, allow_nan=True, allow_inf=True
        ),
    }
    input_domain = DictDomain(
        {
            NamedTable("rows"): SparkDataFrameDomain(df_columns),
            TableCollection("ids"): DictDomain(
                {
                    NamedTable("ids"): SparkDataFrameDomain(
                        {
                            "id": SparkIntegerColumnDescriptor(allow_null=True),
                            **df_columns,
                        }
                    )
                }
            ),
        }
    )
    input_metric = DictMetric(
        {
            NamedTable("rows"): SymmetricDifference(),
            TableCollection("ids"): AddRemoveKeys({NamedTable("ids"): "id"}),
        }
    )
    visitor = TransformationVisitor(
        input_domain=input_domain,
        input_metric=input_metric,
        mechanism=NoiseMechanism.LAPLACE,
        public_sources={},
        table_constraints={NamedTable(t): [] for t in ("rows", "ids")},
    )
    request.cls.visitor = visitor


@pytest.mark.usefixtures("_nulls_visitor", "_nulls_catalog")
class TestTransformationVisitorNulls:
    """Base class for transformation visitor tests of null/NaN/inf behavior."""

    visitor: TransformationVisitor
    catalog: Catalog
