"""Unit tests for (v2) KeySet.from_tuples."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import datetime
from collections.abc import Sequence
from typing import Any, ContextManager, Union

import pandas as pd
import pytest
from pyspark.sql.types import DateType, LongType, StringType, StructField, StructType
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet
from tmlt.analytics._schema import ColumnDescriptor, ColumnType


@parametrize(
    Case("one_column")(
        tuples=[("a1",), ("a2",)],
        columns=("A",),
        expected_df=pd.DataFrame({"A": ["a1", "a2"]}),
        expected_schema={"A": ColumnDescriptor(ColumnType.VARCHAR)},
    ),
    Case("two_columns")(
        tuples=[("a1", "b1"), ("a2", "b1"), ("a3", "b2")],
        columns=("A", "B"),
        expected_df=pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b1", "b2"]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("mixed_types")(
        tuples=[
            (42, "foo", datetime.date.fromordinal(1)),
            (17, "bar", datetime.date.fromordinal(2)),
            (99, "baz", datetime.date.fromordinal(3)),
        ],
        columns=("int_col", "string_col", "date_col"),
        expected_df=pd.DataFrame(
            {
                "int_col": [42, 17, 99],
                "string_col": ["foo", "bar", "baz"],
                "date_col": [
                    datetime.date.fromordinal(1),
                    datetime.date.fromordinal(2),
                    datetime.date.fromordinal(3),
                ],
            }
        ),
        expected_schema={
            "int_col": ColumnDescriptor(ColumnType.INTEGER),
            "string_col": ColumnDescriptor(ColumnType.VARCHAR),
            "date_col": ColumnDescriptor(ColumnType.DATE),
        },
    ),
    Case("nulls")(
        tuples=[
            (None, "foo", datetime.date.fromordinal(1)),
            (17, None, datetime.date.fromordinal(2)),
            (99, "baz", None),
        ],
        columns=("int_col", "string_col", "date_col"),
        expected_df=pd.DataFrame(
            {
                "int_col": [None, 17, 99],
                "string_col": ["foo", None, "baz"],
                "date_col": [
                    datetime.date.fromordinal(1),
                    datetime.date.fromordinal(2),
                    None,
                ],
            }
        ),
        expected_schema={
            "int_col": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "string_col": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "date_col": ColumnDescriptor(ColumnType.DATE, allow_null=True),
        },
    ),
    Case("duplicate_values")(
        tuples=[
            (None, None),
            (None, "foo"),
            (42, "bar"),
            (42, "bar"),
            (None, "foo"),
            (None, None),
        ],
        columns=("A", "B"),
        expected_df=pd.DataFrame({"A": [None, None, 42], "B": [None, "foo", "bar"]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
    ),
    Case("total")(
        tuples=[], columns=(), expected_df=pd.DataFrame(), expected_schema={}
    ),
)
def test_valid(
    tuples: Sequence[tuple[Union[str, int, datetime.date, None], ...]],
    columns: Sequence[str],
    expected_df: pd.DataFrame,
    expected_schema: dict[str, ColumnDescriptor],
):
    """Valid parameters work as expected."""
    ks = KeySet.from_tuples(tuples, columns)
    assert ks.columns() == list(columns)
    assert ks.schema() == expected_schema
    if ks.columns():
        assert ks.size() == len(expected_df)
    else:
        assert ks.size() == 1
    assert_dataframe_equal(ks.dataframe(), expected_df)


@parametrize(Case("nullable")(nullable=True), Case("nonnullable")(nullable=False))
def test_dataframe_schema(nullable: bool):
    """KeySet dataframes have the expected schema."""
    nulls = [(None, None, None)] if nullable else []
    ks = KeySet.from_tuples(
        [(0, "a", datetime.date.today()), *nulls], columns=["int", "str", "date"]
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
    Case("list")(columns=["A", "B"]),
    Case("tuple")(columns=("A", "B")),
)
def test_columns_type(columns: Sequence[str]):
    """Different collection types for columns parameter work."""
    ks = KeySet.from_tuples([("a1", "b1"), ("a2", "b2")], columns)
    assert ks.columns() == list(columns)
    assert_dataframe_equal(
        ks.dataframe(), pd.DataFrame([["a1", "b1"], ["a2", "b2"]], columns=["A", "B"])
    )


@parametrize(
    Case("non_tuple_element")(
        tuples=[("a1", "b1"), "a1"],
        columns=["A", "B"],
        expectation=pytest.raises(
            ValueError, match="Each element of tuples must be a tuple"
        ),
    ),
    Case("mismatched_tuple_length")(
        tuples=[("a1", "b1"), ("a2", "b2", "c2")],
        columns=["A", "B"],
        expectation=pytest.raises(
            ValueError, match="same number of values as there are columns"
        ),
    ),
    Case("float_column")(
        tuples=[(3.5,)],
        columns=["A"],
        expectation=pytest.raises(
            ValueError, match="Column 'A' has type DECIMAL, but only allowed types"
        ),
    ),
    Case("datetime_column")(
        tuples=[(datetime.datetime.now(),)],
        columns=["A"],
        expectation=pytest.raises(
            ValueError, match="Column 'A' has type TIMESTAMP, but only allowed types"
        ),
    ),
    Case("empty_columns_with_tuples")(
        tuples=[(), ()],
        columns=[],
        expectation=pytest.raises(
            ValueError, match="A KeySet with no columns must not have any rows"
        ),
    ),
    Case("empty_tuples_with_columns")(
        tuples=[],
        columns=["A"],
        expectation=pytest.raises(ValueError, match="Unable to infer column types"),
    ),
    Case("all_null_column")(
        tuples=[("a1", None), ("a2", None)],
        columns=["A", "B"],
        expectation=pytest.raises(
            ValueError, match="Column 'B' contains only null values"
        ),
    ),
    Case("empty_column_name")(
        tuples=[(1,)],
        columns=[""],
        expectation=pytest.raises(
            ValueError, match="Empty column names are not allowed"
        ),
    ),
    Case("non_string_column_name")(
        tuples=[(1,)],
        columns=[1],
        expectation=pytest.raises(ValueError, match="Column names must be strings"),
    ),
)
def test_invalid(tuples: Any, columns: Any, expectation: ContextManager[None]):
    """Invalid tuples/columns values are rejected."""
    with expectation:
        KeySet.from_tuples(tuples, columns)
