"""Unit tests for (v2) KeySet.from_dataframe."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from collections.abc import Sequence
from typing import ContextManager

import pandas as pd
import pytest
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet
from tmlt.analytics._schema import ColumnDescriptor, ColumnType


@parametrize(
    Case("one_column")(
        data=[["a1"], ["a2"]],
        schema=StructType([StructField("A", StringType(), False)]),
        expected_df=pd.DataFrame({"A": ["a1", "a2"]}),
        expected_schema={"A": ColumnDescriptor(ColumnType.VARCHAR)},
    ),
    Case("two_columns")(
        data=[["a1", "b1"], ["a2", "b2"], ["a3", "b3"]],
        schema=StructType(
            [
                StructField("A", StringType(), False),
                StructField("B", StringType(), False),
            ]
        ),
        expected_df=pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("mixed_types")(
        data=[
            [42, "foo", datetime.date.fromordinal(1)],
            [17, "bar", datetime.date.fromordinal(2)],
            [99, "baz", datetime.date.fromordinal(3)],
        ],
        schema=StructType(
            [
                StructField("int", LongType(), False),
                StructField("str", StringType(), False),
                StructField("date", DateType(), False),
            ]
        ),
        expected_df=pd.DataFrame(
            [
                [42, "foo", datetime.date.fromordinal(1)],
                [17, "bar", datetime.date.fromordinal(2)],
                [99, "baz", datetime.date.fromordinal(3)],
            ],
            columns=["int", "str", "date"],
        ),
        expected_schema={
            "int": ColumnDescriptor(ColumnType.INTEGER),
            "str": ColumnDescriptor(ColumnType.VARCHAR),
            "date": ColumnDescriptor(ColumnType.DATE),
        },
    ),
    Case("nulls")(
        data=[
            [None, "foo", datetime.date.fromordinal(1)],
            [17, None, datetime.date.fromordinal(2)],
            [99, "baz", None],
        ],
        schema=StructType(
            [
                StructField("int", LongType(), True),
                StructField("str", StringType(), True),
                StructField("date", DateType(), True),
            ]
        ),
        expected_df=pd.DataFrame(
            [
                [None, "foo", datetime.date.fromordinal(1)],
                [17, None, datetime.date.fromordinal(2)],
                [99, "baz", None],
            ],
            columns=["int", "str", "date"],
        ),
        expected_schema={
            "int": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "str": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "date": ColumnDescriptor(ColumnType.DATE, allow_null=True),
        },
    ),
    Case("duplicate_values")(
        data=[
            [None, None],
            [None, "foo"],
            [42, "bar"],
            [None, "foo"],
            [42, "bar"],
            [None, None],
        ],
        schema=StructType(
            [StructField("A", LongType(), True), StructField("B", StringType(), True)]
        ),
        expected_df=pd.DataFrame(
            [
                (None, None),
                (None, "foo"),
                (42, "bar"),
            ],
            columns=["A", "B"],
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
    ),
    Case("total")(
        data=[], schema=StructType([]), expected_df=pd.DataFrame(), expected_schema={}
    ),
    Case("empty")(
        data=[],
        schema=StructType(
            [StructField("A", LongType(), True), StructField("B", StringType(), True)]
        ),
        expected_df=pd.DataFrame([], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
    ),
)
def test_valid(
    data: Sequence[tuple],
    schema: StructType,
    expected_df: pd.DataFrame,
    expected_schema: dict[str, ColumnDescriptor],
    spark,
):
    """Valid parameters work as expected."""
    ks = KeySet.from_dataframe(spark.createDataFrame(data, schema=schema))
    assert ks.columns() == list(expected_schema.keys())
    assert ks.schema() == expected_schema
    if ks.columns():
        assert ks.size() == len(expected_df)
    else:
        assert ks.size() == 1
    assert_dataframe_equal(ks.dataframe(), expected_df)


@parametrize(Case("nullable")(nullable=True), Case("nonnullable")(nullable=False))
def test_dataframe_schema(nullable: bool, spark):
    """KeySet dataframes have the expected schema.

    Checks that nullability is preserved and also that IntegerType columns are
    converted to LongType.
    """
    ks = KeySet.from_dataframe(
        spark.createDataFrame(
            [(0, "a", datetime.date.today())],
            schema=StructType(
                [
                    StructField("int", IntegerType(), nullable=nullable),
                    StructField("str", StringType(), nullable=nullable),
                    StructField("date", DateType(), nullable=nullable),
                ]
            ),
        )
    )
    expected_schema = StructType(
        [
            StructField("int", LongType(), nullable=nullable),
            StructField("str", StringType(), nullable=nullable),
            StructField("date", DateType(), nullable=nullable),
        ]
    )
    assert ks.dataframe().schema == expected_schema


@parametrize(
    Case("empty_columns_with_rows")(
        data=[(), ()],
        schema=StructType([]),
        expectation=pytest.raises(
            ValueError, match="A KeySet with no columns must not have any rows"
        ),
    ),
    Case("float_column")(
        data=[],
        schema=StructType([StructField("A", DoubleType(), False)]),
        expectation=pytest.raises(
            ValueError, match="Column 'A' has type DECIMAL, but only allowed types"
        ),
    ),
    Case("datetime_column")(
        data=[],
        schema=StructType([StructField("A", TimestampType(), False)]),
        expectation=pytest.raises(
            ValueError, match="Column 'A' has type TIMESTAMP, but only allowed types"
        ),
    ),
    Case("empty_column_name")(
        data=[],
        schema=StructType([StructField("", LongType(), False)]),
        expectation=pytest.raises(
            ValueError, match="Empty column names are not allowed"
        ),
    ),
)
def test_invalid(
    data: Sequence[tuple], schema: StructType, expectation: ContextManager[None], spark
):
    """Invalid dataframes are rejected."""
    with expectation:
        KeySet.from_dataframe(spark.createDataFrame(data, schema=schema))
