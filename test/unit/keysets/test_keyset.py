"""Unit tests for KeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
import os
import re
import tempfile
from typing import Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
import pyspark.sql.functions as sf
import pytest
from pyspark.sql import Column
from pyspark.sql.types import (
    DataType,
    DateType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics import ColumnDescriptor, ColumnType, KeySet

from ...conftest import assert_frame_equal_with_sort


@pytest.mark.parametrize(
    "d,expected_df",
    [
        ({"A": ["a1", "a2"]}, pd.DataFrame({"A": ["a1", "a2"]})),
        (
            {
                "A": ["a1", "a2"],
                "B": [0, 1, 2, 3],
                "C": ["c0"],
                "D": [datetime.date.fromordinal(1)],
            },
            pd.DataFrame(
                [
                    ["a1", 0, "c0", datetime.date.fromordinal(1)],
                    ["a1", 1, "c0", datetime.date.fromordinal(1)],
                    ["a1", 2, "c0", datetime.date.fromordinal(1)],
                    ["a1", 3, "c0", datetime.date.fromordinal(1)],
                    ["a2", 0, "c0", datetime.date.fromordinal(1)],
                    ["a2", 1, "c0", datetime.date.fromordinal(1)],
                    ["a2", 2, "c0", datetime.date.fromordinal(1)],
                    ["a2", 3, "c0", datetime.date.fromordinal(1)],
                ],
                columns=["A", "B", "C", "D"],
            ),
        ),
        ({"A": [0, 1, 2, 0]}, pd.DataFrame({"A": [0, 1, 2]})),
        (
            {"A": [0, 1], "B": [7, 8, 9, 7]},
            pd.DataFrame({"A": [0, 0, 0, 1, 1, 1], "B": [7, 8, 9, 7, 8, 9]}),
        ),
        (
            {"A": [None, 1, 2, 3], "B": [None, "b1"]},
            pd.DataFrame(
                [
                    [None, None],
                    [None, "b1"],
                    [1, None],
                    [1, "b1"],
                    [2, None],
                    [2, "b1"],
                    [3, None],
                    [3, "b1"],
                ],
                columns=["A", "B"],
            ),
        ),
        ({}, pd.DataFrame()),  # Empty KeySet with no columns is allowed
    ],
)
def test_from_dict(
    d: Mapping[
        str,
        Union[
            List[Optional[str]],
            List[Optional[int]],
            List[Optional[datetime.date]],
        ],
    ],
    expected_df: pd.DataFrame,
) -> None:
    """Test KeySet.from_dict works"""
    keyset = KeySet.from_dict(d)
    assert_frame_equal_with_sort(keyset.dataframe().toPandas(), expected_df)


@pytest.mark.parametrize(
    "d",
    [
        ({"A": []}),
        ({"A": [], "B": ["b1"]}),
        ({"A": [], "B": [0]}),
        ({"A": ["a1", "a2"], "B": []}),
        ({"A": [0, 1, 2, 3], "B": []}),
    ],
)
def test_from_dict_empty_list(
    d: Mapping[
        str,
        Union[
            List[Optional[str]],
            List[Optional[int]],
            List[Optional[datetime.date]],
        ],
    ]
) -> None:
    """Test that calls like ``KeySet.from_dict({'A': []})`` raise a friendly error."""
    with pytest.raises(
        ValueError,
        match="Unable to infer column types for an empty collection of values",
    ):
        KeySet.from_dict(d)


@pytest.mark.parametrize(
    "d,expected_err_msg",
    [
        ({"A": [3.1]}, "Column 'A' has type DECIMAL"),
        (
            {"A": [3.1], "B": [datetime.datetime.now()]},
            "Column 'A' has type DECIMAL",
        ),
        (
            {"A": [3], "B": [datetime.datetime.now()]},
            "Column 'B' has type TIMESTAMP",
        ),
    ],
)
def test_from_dict_invalid_types(d: Dict[str, List], expected_err_msg: str):
    """KeySet.from_dict raises an appropriate exception on invalid inputs."""
    with pytest.raises(ValueError, match=expected_err_msg):
        KeySet.from_dict(d)


@parametrize(
    Case("one column")(
        tuples=[("a1",), ("a2",)],
        columns=("A",),
        expected_df=pd.DataFrame({"A": ["a1", "a2"]}),
    ),
    Case("two columns")(
        tuples=[("a1", "b1"), ("a2", "b1"), ("a3", "b2")],
        columns=("A", "B"),
        expected_df=pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b1", "b2"]}),
    ),
    Case("columns is a list")(
        tuples=[("a1", "b1"), ("a2", "b1"), ("a3", "b2")],
        columns=["A", "B"],
        expected_df=pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b1", "b2"]}),
    ),
    Case("different types")(
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
    ),
    Case("None values")(
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
    ),
    Case("None & duplicated values")(
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
    ),
    Case("Empty KeySet")(tuples=[], columns=(), expected_df=pd.DataFrame()),
)
def test_from_tuples(
    tuples: List[Tuple[Optional[Union[str, int, datetime.date]], ...]],
    columns: Tuple[str, ...],
    expected_df: pd.DataFrame,
):
    """KeySet.from_tuples works as expected"""
    keyset = KeySet.from_tuples(tuples, columns)
    assert_frame_equal_with_sort(keyset.dataframe().toPandas(), expected_df)


@parametrize(
    Case("Tuple too big")(
        tuples=[("a1", "b2"), ("a2",)],
        columns=("A",),
        expected_err_msg=(
            "Tuples must contain the same number of values as there are columns"
        ),
    ),
    Case("Tuple too small")(
        tuples=[("a1", "b2"), ("a2",)],
        columns=("A", "B"),
        expected_err_msg=(
            "Tuples must contain the same number of values as there are columns"
        ),
    ),
    Case("Empty tuple")(
        tuples=[(), ("a1", "b2")],
        columns=("A", "B"),
        expected_err_msg=(
            "Tuples must contain the same number of values as there are columns"
        ),
    ),
    Case("Floats are forbidden")(
        tuples=[(42.17, "b2")],
        columns=("A", "B"),
        expected_err_msg="Column 'A' has type DECIMAL",
    ),
    Case("Timestamps are forbidden")(
        tuples=[(datetime.datetime.now(), "b2")],
        columns=("A", "B"),
        expected_err_msg="Column 'A' has type TIMESTAMP",
    ),
    Case("Mismatched types: str to int")(
        tuples=[("a1", "b2"), ("a2", 42)],
        columns=("A", "B"),
        expected_err_msg="Column 'B' contains values of multiple types",
    ),
    Case("Mismatched types: int to str")(
        tuples=[(42, "b1"), ("a2", "b2")],
        columns=("A", "B"),
        expected_err_msg="Column 'A' contains values of multiple types",
    ),
    Case("Mismatched types with None")(
        tuples=[(None, None), (None, "b2"), ("a1", None), ("a2", 42)],
        columns=("A", "B"),
        expected_err_msg="Column 'B' contains values of multiple types",
    ),
    Case("Columns full of None")(
        tuples=[(None, None), (None, "b2"), (None, None), (None, "b1")],
        columns=("A", "B"),
        expected_err_msg=(
            "Column 'A' contains only null values, unable to infer its type"
        ),
    ),
    Case("Empty tuples, no-empty columns")(
        tuples=[],
        columns=("A", "B"),
        expected_err_msg=(
            "Unable to infer column types for an empty collection of values"
        ),
    ),
)
def test_from_tuples_invalid_schema(
    tuples: List[Tuple[Optional[Union[str, int, datetime.date]], ...]],
    columns: Tuple[str, ...],
    expected_err_msg: str,
):
    """KeySet.from_tuples raises an appropriate exception on invalid inputs."""
    with pytest.raises(ValueError, match=re.escape(expected_err_msg)):
        KeySet.from_tuples(tuples, columns)


@pytest.mark.parametrize(
    "df_in",
    [(pd.DataFrame({"A": ["a1"]})), (pd.DataFrame({"A": ["a1", "a2"], "B": [0, 1]}))],
)
def test_from_dataframe(spark, df_in: pd.DataFrame) -> None:
    """Test KeySet.from_dataframe works."""
    keyset = KeySet.from_dataframe(spark.createDataFrame(df_in))
    assert_frame_equal_with_sort(keyset.dataframe().toPandas(), df_in)


@pytest.mark.parametrize(
    "df,expected_df",
    [
        (pd.DataFrame({"A": [0, 1, 2, 3, 0]}), pd.DataFrame({"A": [0, 1, 2, 3]})),
        (
            pd.DataFrame({"A": [0, 1, 0, 1, 0], "B": [0, 0, 1, 1, 1]}),
            pd.DataFrame({"A": [0, 1, 0, 1], "B": [0, 0, 1, 1]}),
        ),
    ],
)
def test_from_dataframe_nonunique(spark, df: pd.DataFrame, expected_df: pd.DataFrame):
    """Test KeySet.from_dataframe works on a dataframe with duplicate rows."""
    keyset = KeySet.from_dataframe(spark.createDataFrame(df))
    assert_frame_equal_with_sort(keyset.dataframe().toPandas(), expected_df)


@pytest.mark.parametrize(
    "df_in,schema",
    [
        (
            pd.DataFrame({"A": [1, 2, None]}),
            StructType([StructField("A", LongType(), nullable=True)]),
        ),
        (
            pd.DataFrame({"B": [None, "b2", "b3"]}),
            StructType([StructField("B", StringType(), nullable=True)]),
        ),
        (
            pd.DataFrame({"A": [1, 2, None], "B": [None, "b2", "b3"]}),
            StructType(
                [
                    StructField("A", LongType(), nullable=True),
                    StructField("B", StringType(), nullable=True),
                ]
            ),
        ),
    ],
)
def test_from_dataframe_with_null(
    spark, df_in: pd.DataFrame, schema: StructType
) -> None:
    """Test KeySet.from_dataframe allows nulls."""
    keyset = KeySet.from_dataframe(spark.createDataFrame(df_in, schema=schema))
    assert_frame_equal_with_sort(keyset.dataframe().toPandas(), df_in)


@pytest.mark.parametrize(
    "df,expected_err_msg",
    [
        (pd.DataFrame({"A": [3.1]}), "Column 'A' has type DECIMAL"),
        (
            pd.DataFrame({"A": [3.1], "B": [datetime.datetime.now()]}),
            "Column 'A' has type DECIMAL",
        ),
        (
            pd.DataFrame({"A": [3], "B": [datetime.datetime.now()]}),
            "Column 'B' has type TIMESTAMP",
        ),
    ],
)
def test_from_dataframe_invalid_types(spark, df: pd.DataFrame, expected_err_msg: str):
    """KeySet.from_dataframe raises an appropriate exception on invalid inputs."""
    sdf = spark.createDataFrame(df)
    with pytest.raises(ValueError, match=expected_err_msg):
        KeySet.from_dataframe(sdf)


@pytest.mark.parametrize(
    "keyset_df,condition,expected_df",
    [
        (
            pd.DataFrame([[0, "b0"], [1, "b0"], [2, "b0"]], columns=["A", "B"]),
            "A > 0",
            pd.DataFrame([[1, "b0"], [2, "b0"]], columns=["A", "B"]),
        ),
        (
            pd.DataFrame({"A": [10, 9, 8], "B": [-1, -2, -3]}),
            "B < 0",
            pd.DataFrame({"A": [10, 9, 8], "B": [-1, -2, -3]}),
        ),
        (
            pd.DataFrame({"A": ["a0", "a1", "a123456"]}),
            "length(A) > 3",
            pd.DataFrame({"A": ["a123456"]}),
        ),
    ],
)
def test_filter_str(
    spark,
    keyset_df: pd.DataFrame,
    condition: Union[Column, str],
    expected_df: pd.DataFrame,
) -> None:
    """Test KeySet.filter works"""
    keyset = KeySet.from_dataframe(spark.createDataFrame(keyset_df))
    filtered_keyset = keyset.filter(condition)
    assert_frame_equal_with_sort(filtered_keyset.dataframe().toPandas(), expected_df)


@pytest.mark.parametrize(
    "df_in,input_schema,expected_schema",
    [
        (
            pd.DataFrame({"A": [0, 1]}),
            StructType([StructField("A", IntegerType(), True)]),
            {"A": LongType()},
        ),
        (
            pd.DataFrame({"A": ["abc", "def"], "B": [2147483649, -42]}),
            StructType(
                [
                    StructField("A", StringType(), True),
                    StructField("B", LongType(), True),
                ]
            ),
            {"A": StringType(), "B": LongType()},
        ),
    ],
)
def test_type_coercion_from_dataframe(
    spark,
    df_in: pd.DataFrame,
    input_schema: StructType,
    expected_schema: Dict[str, DataType],
) -> None:
    """Test KeySet correctly coerces types in input DataFrames."""
    keyset = KeySet.from_dataframe(spark.createDataFrame(df_in, schema=input_schema))
    df_out = keyset.dataframe()
    for col in df_out.schema:
        assert col.dataType == expected_schema[col.name]


@pytest.mark.parametrize(
    "d_in,expected_schema",
    [
        ({"A": [0, 1, 2], "B": ["abc", "def"]}, {"A": LongType(), "B": StringType()}),
        (
            {
                "A": [123, 456, 789],
                "B": [2147483649, -1000000],
                "X": ["abc", "def"],
                "Y": [datetime.date.fromordinal(1)],
            },
            {"A": LongType(), "B": LongType(), "X": StringType(), "Y": DateType()},
        ),
        # Tests an empty dict, which is used for queries without a KeySet.
        ({}, {}),
    ],
)
def test_type_coercion_from_dict(
    d_in: Mapping[
        str,
        Union[
            List[Optional[str]],
            List[Optional[int]],
            List[Optional[datetime.date]],
        ],
    ],
    expected_schema: Dict[str, DataType],
) -> None:
    """Test KeySet correctly coerces types when created with ``from_dict``."""
    keyset = KeySet.from_dict(d_in)
    df_out = keyset.dataframe()
    for col in df_out.schema:
        assert col.dataType == expected_schema[col.name]


@pytest.mark.parametrize(
    "tuples,columns,expected_schema",
    [
        ([(None, None), (42, "foo")], ("A", "B"), {"A": LongType(), "B": StringType()}),
        (
            [
                (None, 2147483649, "foo", datetime.date.fromordinal(1)),
                (42, None, "bar", datetime.date.fromordinal(2)),
                (17, -1000000, None, datetime.date.fromordinal(3)),
                (None, None, None, None),
            ],
            ("A", "B", "X", "Y"),
            {"A": LongType(), "B": LongType(), "X": StringType(), "Y": DateType()},
        ),
        # Tests an empty tuple list, used for queries without a KeySet.
        ([], (), {}),
    ],
)
def test_type_coercion_from_tuples(
    tuples: List[Tuple[Optional[Union[str, int, datetime.date]], ...]],
    columns: Tuple[str, ...],
    expected_schema: Dict[str, DataType],
) -> None:
    """Test KeySet correctly coerces types when created with ``from_tuples``."""
    keyset = KeySet.from_tuples(tuples, columns)
    df_out = keyset.dataframe()
    for col in df_out.schema:
        assert col.dataType == expected_schema[col.name]


# This test is not parameterized because Column parameters are
# Python expressions containing the KeySet's DataFrame.
def test_filter_condition() -> None:
    """Test KeySet.filter with Columns conditions."""
    keyset = KeySet.from_dict({"A": ["abc", "def", "ghi"], "B": [0, 100]})
    filtered = keyset.filter(sf.col("B") > 0)
    expected = pd.DataFrame(
        [["abc", 100], ["def", 100], ["ghi", 100]], columns=["A", "B"]
    )
    assert_frame_equal_with_sort(filtered.dataframe().toPandas(), expected)

    filtered2 = keyset.filter(sf.col("A") != "string that is not there")
    assert_frame_equal_with_sort(
        filtered2.dataframe().toPandas(), keyset.dataframe().toPandas()
    )


# This test also uses a Column as a filter condition, and is not
# parameterized for the same reason as test_filter_condition.
def test_filter_to_empty() -> None:
    """Test when KeySet.filter should return an empty dataframe, it does"""
    keyset = KeySet.from_dict({"A": [-1, -2, -3]})
    filtered = keyset.filter("A > 0")
    pd_df = filtered.dataframe().toPandas()
    assert isinstance(pd_df, pd.DataFrame)
    assert pd_df.empty

    keyset2 = KeySet.from_dict({"A": ["a1", "a2", "a3"], "B": ["irrelevant"]})
    filtered2 = keyset2.filter(sf.col("A") == "string that is not there")
    pd_df2 = filtered2.dataframe().toPandas()
    assert isinstance(pd_df2, pd.DataFrame)
    assert pd_df2.empty


@pytest.mark.parametrize(
    "col,expected_df",
    [
        ("A", pd.DataFrame({"A": ["a1", "a2"]})),
        ("B", pd.DataFrame({"B": [0, 1, 2, 3]})),
    ],
)
def test_getitem_single(col: str, expected_df: pd.DataFrame) -> None:
    """Test KeySet[col] returns a keyset for only the requested column."""
    keyset = KeySet.from_dict({"A": ["a1", "a2"], "B": [0, 1, 2, 3]})
    got = keyset[col]
    assert_frame_equal_with_sort(got.dataframe().toPandas(), expected_df)


# This test is not parameterized because Python does not accept
# `obj[*tuple]` as valid syntax.
def test_getitem_multiple() -> None:
    """Test KeySet[col1, col2, ...] returns a keyset for requested columns."""
    keyset = KeySet.from_dict({"A": ["a1", "a2"], "B": ["b1"], "C": [0, 1]})
    got_ab = keyset["A", "B"]
    expected_ab = pd.DataFrame([["a1", "b1"], ["a2", "b1"]], columns=["A", "B"])
    assert_frame_equal_with_sort(got_ab.dataframe().toPandas(), expected_ab)

    got_bc = keyset["B", "C"]
    expected_bc = pd.DataFrame([["b1", 0], ["b1", 1]], columns=["B", "C"])
    assert_frame_equal_with_sort(got_bc.dataframe().toPandas(), expected_bc)

    got_abc = keyset["A", "B", "C"]
    assert_frame_equal_with_sort(
        got_abc.dataframe().toPandas(), keyset.dataframe().toPandas()
    )


@pytest.mark.parametrize(
    "l,expected_df",
    [
        (["A", "B"], pd.DataFrame([["a1", "b1"], ["a2", "b1"]], columns=["A", "B"])),
        (["B", "C"], pd.DataFrame([["b1", 0], ["b1", 1]], columns=["B", "C"])),
    ],
)
def test_getitem_list(l: List[str], expected_df: pd.DataFrame) -> None:
    """Test KeySet[[col1, col2, ...]] returns a keyset for requested columns."""
    keyset = KeySet.from_dict({"A": ["a1", "a2"], "B": ["b1"], "C": [0, 1]})
    got = keyset[l]
    assert_frame_equal_with_sort(got.dataframe().toPandas(), expected_df)


@pytest.mark.parametrize(
    "keys_df,columns,expected_df",
    [
        (
            pd.DataFrame([[0, 0, 0], [0, 1, 0], [0, 1, 1]], columns=["A", "B", "C"]),
            ["A", "B"],
            pd.DataFrame([[0, 0], [0, 1]], columns=["A", "B"]),
        ),
        (
            pd.DataFrame([[0, 0, 0], [0, 1, 0], [0, 1, 1]], columns=["A", "B", "C"]),
            ["B", "C"],
            pd.DataFrame([[0, 0], [1, 0], [1, 1]], columns=["B", "C"]),
        ),
    ],
)
def test_getitem_list_noncartesian(
    spark, keys_df: pd.DataFrame, columns: List[str], expected_df: pd.DataFrame
) -> None:
    """Test that indexing multiple columns works on non-Cartesian KeySets."""
    keyset = KeySet.from_dataframe(spark.createDataFrame(keys_df))
    actual_df = keyset[columns].dataframe().toPandas()
    assert_frame_equal_with_sort(actual_df, expected_df)


@pytest.mark.parametrize(
    "other,expected_df",
    [
        (
            KeySet.from_dict({"C": ["c1", "c2"]}),
            pd.DataFrame(
                [
                    ["a1", 0, "c1"],
                    ["a1", 0, "c2"],
                    ["a1", 1, "c1"],
                    ["a1", 1, "c2"],
                    ["a2", 0, "c1"],
                    ["a2", 0, "c2"],
                    ["a2", 1, "c1"],
                    ["a2", 1, "c2"],
                ],
                columns=["A", "B", "C"],
            ),
        ),
        (
            KeySet.from_dict({"C": [-1, -2], "D": ["d0"]}),
            pd.DataFrame(
                [
                    ["a1", 0, -1, "d0"],
                    ["a1", 0, -2, "d0"],
                    ["a1", 1, -1, "d0"],
                    ["a1", 1, -2, "d0"],
                    ["a2", 0, -1, "d0"],
                    ["a2", 0, -2, "d0"],
                    ["a2", 1, -1, "d0"],
                    ["a2", 1, -2, "d0"],
                ],
                columns=["A", "B", "C", "D"],
            ),
        ),
        (
            KeySet.from_dict({"Z": ["zzzzz"]}),
            pd.DataFrame(
                [
                    ["a1", 0, "zzzzz"],
                    ["a1", 1, "zzzzz"],
                    ["a2", 0, "zzzzz"],
                    ["a2", 1, "zzzzz"],
                ],
                columns=["A", "B", "Z"],
            ),
        ),
        (
            KeySet.from_dict({"Z": [None, "z1", "z2"]}),
            pd.DataFrame(
                [
                    ["a1", 0, None],
                    ["a1", 0, "z1"],
                    ["a1", 0, "z2"],
                    ["a1", 1, None],
                    ["a1", 1, "z1"],
                    ["a1", 1, "z2"],
                    ["a2", 0, None],
                    ["a2", 0, "z1"],
                    ["a2", 0, "z2"],
                    ["a2", 1, None],
                    ["a2", 1, "z1"],
                    ["a2", 1, "z2"],
                ],
                columns=["A", "B", "Z"],
            ),
        ),
    ],
)
def test_crossproduct(other: KeySet, expected_df: pd.DataFrame) -> None:
    """Test factored_df * factored_df returns the expected cross-product."""
    keyset = KeySet.from_dict({"A": ["a1", "a2"], "B": [0, 1]})
    product_left = keyset * other
    product_right = other * keyset
    assert_frame_equal_with_sort(product_left.dataframe().toPandas(), expected_df)
    assert_frame_equal_with_sort(product_right.dataframe().toPandas(), expected_df)


@pytest.mark.parametrize(
    "keyset,expected",
    [
        (
            KeySet.from_dict({"A": ["a1", "a2"]}),
            {"A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False)},
        ),
        (
            KeySet.from_dict({"A": [0, 1, 2], "B": ["abc"]}),
            {
                "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
            },
        ),
        (
            KeySet.from_dict(
                {"A": ["abc"], "B": [0], "C": ["def"], "D": [-1000000000, None]}
            ),
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
                "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                "C": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
                "D": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            },
        ),
    ],
)
def test_schema(keyset: KeySet, expected: Dict[str, ColumnDescriptor]) -> None:
    """Test KeySet.schema returns the expected schema."""
    assert keyset.schema() == expected


@pytest.mark.parametrize(
    "other_df,equal",
    [
        (
            pd.DataFrame(
                [["a1", 0], ["a1", 1], ["a2", 0], ["a2", 1]], columns=["A", "B"]
            ),
            True,
        ),
        (
            pd.DataFrame(
                [[1, "a2"], [1, "a1"], [0, "a2"], [0, "a1"]], columns=["B", "A"]
            ),
            True,
        ),
        (
            pd.DataFrame(
                [[1, "a2"], [1, "a1"], [0, "a2"], [0, "a1"]], columns=["Z", "A"]
            ),
            False,
        ),
    ],
)
def test_eq(spark, other_df: pd.DataFrame, equal: bool) -> None:
    """Test the equality operator."""
    keyset = KeySet.from_dict({"A": ["a1", "a2"], "B": [0, 1]})
    other = KeySet.from_dataframe(spark.createDataFrame(other_df))
    if equal:
        assert keyset == other
    else:
        assert keyset != other


def test_caching():
    """Tests that cache and unpersist methods work."""
    ks = KeySet.from_dict({"A": ["a1", "a2"]})

    # Assert that the KeySet is lazily evaluated.
    assert (
        repr(ks.dataframe().storageLevel)
        == "StorageLevel(False, False, False, False, 1)"
    )

    # Assert that caching adds the KeySet DataFrame to memory.
    ks.cache()
    assert (
        repr(ks.dataframe().storageLevel) == "StorageLevel(True, True, False, True, 1)"
    )

    # Assert that unpersisting removes the KeySet DataFrame from memory.
    ks.uncache()
    assert (
        repr(ks.dataframe().storageLevel)
        == "StorageLevel(False, False, False, False, 1)"
    )


@pytest.mark.parametrize(
    "_,keyset,expected",
    [
        ("Empty Keyset (Non-Groupby Queries use these)", KeySet.from_dict({}), 1),
        ("Single Item, Single Column", KeySet.from_dict({"A": [0]}), 1),
        ("Single Item, Two Columns", KeySet.from_dict({"A": [0], "B": [1]}), 1),
        ("Two Items, Two Columns", KeySet.from_dict({"A": [0], "B": [1, 2]}), 2),
        ("Two Items, One Columns", KeySet.from_dict({"A": [0, 1]}), 2),
    ],
)
def test_size_from_dict(_, keyset, expected):
    """Tests that the expected KeySet size is returned."""
    assert keyset.size() == expected


@pytest.mark.parametrize(
    "_,pd_df,expected_size,schema",
    [
        (
            "Empty Keyset (Non-Groupby Queries use these)",
            pd.DataFrame([], columns=["A"]),
            0,
            StructType([StructField("age", IntegerType(), True)]),
        ),
        ("Single Item, Single Column", pd.DataFrame({"A": [0]}), 1, None),
        ("Single Item, Two Columns", pd.DataFrame({"A": [0], "B": [1]}), 1, None),
        ("Two Items, Two Columns", pd.DataFrame({"A": [0, 1], "B": [1, 2]}), 2, None),
        ("Two Items, One Columns", pd.DataFrame({"A": [0, 1]}), 2, None),
    ],
)
def test_size_from_df(_, spark, pd_df, expected_size, schema):
    """Tests that the expected KeySet size is returned."""
    sdf = (
        spark.createDataFrame(pd_df)
        if not schema
        else spark.createDataFrame(pd_df, schema=schema)
    )
    keyset = KeySet.from_dataframe(sdf)
    assert keyset.size() == expected_size


@pytest.fixture(scope="module")
def _eq_hashing_test_data(spark):
    "Set up test data."
    pdf_ab = pd.DataFrame({"A": ["a1", "a2"], "B": [0, 1]})
    df_ab = spark.createDataFrame(pdf_ab)
    pdf_ac = pd.DataFrame({"A": ["a1", "a2"], "C": [0, 1]})
    df_ac = spark.createDataFrame(pdf_ac)
    temp_dir = tempfile.gettempdir()
    filepath1 = os.path.join(temp_dir, "keyset1.csv")
    pdf_ab.to_csv(filepath1, index=False, sep="|")
    filepath2 = os.path.join(temp_dir, "keyset2.csv")
    pdf_ac.to_csv(filepath2, index=False, sep="|")
    return {
        "df_ab": KeySet.from_dataframe(df_ab),
        "df_ab_duplicate": KeySet.from_dataframe(
            spark.createDataFrame(pd.DataFrame({"A": ["a1", "a2"], "B": [0, 1]}))
        ),
        "df_ac": KeySet.from_dataframe(df_ac),
        "df_ab_with_c": KeySet.from_dataframe(df_ab.withColumn("C", sf.lit(1))),
        "df_ab_with_d_int": KeySet.from_dataframe(df_ab.withColumn("D", sf.lit(2))),
        "df_ab_with_d_string": KeySet.from_dataframe(
            df_ab.withColumn("D", sf.lit("2"))
        ),
        "df_ab_from_file1": KeySet.from_dataframe(
            spark.read.csv(filepath1, header=True)
        ),
        "df_ac_from_file2": KeySet.from_dataframe(
            spark.read.csv(filepath2, header=True)
        ),
    }


@pytest.mark.parametrize(
    "key1, key2, equal",
    [
        ("df_ab", "df_ab", True),
        ("df_ab", "df_ac", False),
        ("df_ab", "df_ab_with_c", False),
        ("df_ab_with_d_string", "df_ab_with_d_int", False),
        ("df_ab_with_d_string", "df_ab_with_d_string", True),
        ("df_ab_from_file1", "df_ab_from_file1", True),
        ("df_ab_from_file1", "df_ac_from_file2", False),
    ],
)
def test_is_equivalent(_eq_hashing_test_data, key1, key2, equal):
    """Test the is_equivalent function for accuracy."""
    ks1 = _eq_hashing_test_data[key1]
    ks2 = _eq_hashing_test_data[key2]

    assert ks1.is_equivalent(ks2) == equal


def test_is_equivalent_same_file_read(spark):
    """Test the is_equivalent when same file read twice."""
    pdf_ab = pd.DataFrame({"A": ["a1", "a2"], "B": [0, 1]})
    temp_dir = tempfile.gettempdir()
    filepath1 = os.path.join(temp_dir, "keyset1.csv")
    pdf_ab.to_csv(filepath1, index=False, sep="|")

    ks1 = KeySet.from_dataframe(spark.read.csv(filepath1, header=True))
    ks2 = KeySet.from_dataframe(spark.read.csv(filepath1, header=True))

    assert ks1.is_equivalent(ks2)


@pytest.mark.parametrize(
    "key1, key2",
    [
        ("df_ab", "df_ab"),
        ("df_ab", "df_ab_duplicate"),
        ("df_ab_with_d_string", "df_ab_with_d_int"),
        ("df_ab_from_file1", "df_ac_from_file2"),
    ],
)
def test_hashing(_eq_hashing_test_data, key1, key2):
    """Tests that the hash function is hashing on DF schema."""
    ks_1 = _eq_hashing_test_data[key1]
    ks_2 = _eq_hashing_test_data[key2]

    if ks_1.dataframe().schema == ks_2.dataframe().schema:
        assert hash(ks_1) == hash(ks_2)
    else:
        assert hash(ks_1) != hash(ks_2)
