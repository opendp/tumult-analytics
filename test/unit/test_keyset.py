"""Unit tests for KeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Dict, List, Mapping, Optional, Union

import pandas as pd
import pytest
from pyspark.sql import Column
from pyspark.sql.types import (
    DataType,
    DateType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.keyset import KeySet, _check_df_schema, _check_dict_schema

from ..conftest import assert_frame_equal_with_sort


@pytest.mark.parametrize(
    "types",
    [
        (StructType([StructField("A", LongType(), False)])),
        (
            StructType(
                [
                    StructField("A", LongType(), False),
                    StructField("B", DateType(), False),
                    StructField("C", StringType(), False),
                ]
            )
        ),
    ],
)
def test_check_df_schema_valid(types: StructType):
    """_check_df_schema does not raise an exception on valid inputs."""
    _check_df_schema(types)


@pytest.mark.parametrize(
    "types,expected_err_msg",
    [
        (
            StructType([StructField("A", FloatType(), False)]),
            r"Column A has type FloatType\(?\)?, which is not allowed in KeySets",
        ),
        (
            StructType(
                [
                    StructField("A", LongType(), False),
                    StructField("B", TimestampType(), False),
                ]
            ),
            r"Column B has type TimestampType\(?\)?, which is not allowed in KeySets",
        ),
    ],
)
def test_check_df_schema_invalid(types: StructType, expected_err_msg: str):
    """_check_df_schema raises an appropriate exception on invalid inputs."""
    with pytest.raises(ValueError, match=expected_err_msg):
        _check_df_schema(types)


@pytest.mark.parametrize(
    "types", [({"A": int}), ({"A": int, "B": str, "C": datetime.date})]
)
def test_check_dict_schema_valid(types: Dict[str, type]):
    """_check_dict_schema does not raise an exception on valid inputs."""
    _check_dict_schema(types)


@pytest.mark.parametrize(
    "types,expected_err_msg",
    [
        ({"A": float}, "Column A has type float, which is not allowed in KeySets"),
        (
            {"A": int, "B": datetime.datetime},
            "Column B has type datetime, which is not allowed in KeySets",
        ),
    ],
)
def test_check_dict_schema_invalid(types: Dict[str, type], expected_err_msg: str):
    """_check_dict_schema raises an appropriate exception on invalid inputs."""
    with pytest.raises(ValueError, match=expected_err_msg):
        _check_dict_schema(types)


###TESTS FOR THE KEYSET CLASS###


@pytest.mark.parametrize(
    "df_in",
    [(pd.DataFrame({"A": ["a1"]})), (pd.DataFrame({"A": ["a1", "a2"], "B": [0, 1]}))],
)
def test_init(spark, df_in: pd.DataFrame) -> None:
    """Test that initialization works."""
    keyset = KeySet(spark.createDataFrame(df_in))
    assert_frame_equal_with_sort(keyset.dataframe().toPandas(), df_in)


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
    ],
)
def test_from_dict(
    d: Mapping[
        str,
        Union[
            List[str],
            List[Optional[str]],
            List[int],
            List[Optional[int]],
            List[datetime.date],
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
            List[str],
            List[Optional[str]],
            List[int],
            List[Optional[int]],
            List[datetime.date],
            List[Optional[datetime.date]],
        ],
    ]
) -> None:
    """Test that calls like ``KeySet.from_dict({'A': []})`` raise a friendly error."""
    with pytest.raises(ValueError):
        KeySet.from_dict(d)


@pytest.mark.parametrize(
    "d,expected_err_msg",
    [
        ({"A": [3.1]}, "Column A has type float, which is not allowed in KeySets"),
        (
            {"A": [3.1], "B": [datetime.datetime.now()]},
            "Column A has type float, which is not allowed in KeySets",
        ),
        (
            {"A": [3], "B": [datetime.datetime.now()]},
            "Column B has type datetime, which is not allowed in KeySets",
        ),
    ],
)
def test_from_dict_invalid_types(d: Dict[str, List], expected_err_msg: str):
    """KeySet.from_dict raises an appropriate exception on invalid inputs."""
    with pytest.raises(ValueError, match=expected_err_msg):
        KeySet.from_dict(d)


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
    keyset = KeySet(spark.createDataFrame(df_in, schema=schema))
    assert_frame_equal_with_sort(keyset.dataframe().toPandas(), df_in)


@pytest.mark.parametrize(
    "df,expected_err_msg",
    [
        (
            pd.DataFrame({"A": [3.1]}),
            r"Column A has type DoubleType\(?\)?, which is not allowed in KeySets",
        ),
        (
            pd.DataFrame({"A": [3.1], "B": [datetime.datetime.now()]}),
            r"Column A has type DoubleType\(?\)?, which is not allowed in KeySets",
        ),
        (
            pd.DataFrame({"A": [3], "B": [datetime.datetime.now()]}),
            r"Column B has type TimestampType\(?\)?, which is not allowed in KeySets",
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
    keyset = KeySet(spark.createDataFrame(keyset_df))
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
    keyset = KeySet(spark.createDataFrame(df_in, schema=input_schema))
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
    ],
)
def test_type_coercion_from_dict(
    d_in: Mapping[
        str,
        Union[
            List[str],
            List[Optional[str]],
            List[int],
            List[Optional[int]],
            List[datetime.date],
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


# This test is not parameterized because Column parameters are
# Python expressions containing the KeySet's DataFrame.
def test_filter_condition() -> None:
    """Test KeySet.filter with Columns conditions."""
    keyset = KeySet.from_dict({"A": ["abc", "def", "ghi"], "B": [0, 100]})
    filtered = keyset.filter(keyset.dataframe().B > 0)
    expected = pd.DataFrame(
        [["abc", 100], ["def", 100], ["ghi", 100]], columns=["A", "B"]
    )
    assert_frame_equal_with_sort(filtered.dataframe().toPandas(), expected)

    filtered2 = keyset.filter(keyset.dataframe().A != "string that is not there")
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
    filtered2 = keyset2.filter(keyset2.dataframe().A == "string that is not there")
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
            Schema({"A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)}),
        ),
        (
            KeySet.from_dict({"A": [0, 1, 2], "B": ["abc"]}),
            Schema(
                {
                    "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                    "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                }
            ),
        ),
        (
            KeySet.from_dict(
                {"A": ["abc"], "B": [0], "C": ["def"], "D": [-1000000000]}
            ),
            Schema(
                {
                    "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                    "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                    "C": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                    "D": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                }
            ),
        ),
    ],
)
def test_schema(keyset: KeySet, expected: Schema) -> None:
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
    other = KeySet(spark.createDataFrame(other_df))
    if equal:
        assert keyset == other
    else:
        assert keyset != other
