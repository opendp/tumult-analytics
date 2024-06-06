"""Tests for _ProductKeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024


import itertools
import re
from functools import reduce
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tmlt.analytics._schema import ColumnType, Schema
from tmlt.analytics.keyset import KeySet, _ProductKeySet

from ...conftest import assert_frame_equal_with_sort

# pylint: disable=unused-argument


def test_init_with_product_keyset(
    spark: SparkSession,
) -> None:
    """Test _ProductKeySet.__init__ with a _ProductKeySet as a factor."""
    ks1 = KeySet.from_dict({"A": [1, 2, 3]})
    ks2 = KeySet.from_dict({"B": ["a", "b", "c"]})
    ks3 = KeySet.from_dict({"C": [9, 8, 7]})
    product_2_and_3 = ks2 * ks3

    got = ks1 * product_2_and_3
    expected_df = ks1.dataframe().crossJoin(ks2.dataframe()).crossJoin(ks3.dataframe())
    assert_frame_equal_with_sort(got.dataframe().toPandas(), expected_df.toPandas())


def test_init_fails_with_duplicate_columns(spark: SparkSession) -> None:
    """Test _ProductKeySet.__init__ fails with duplicate columns in factors."""
    ks1 = KeySet.from_dict({"A": [1, 2, 3]})
    ks2 = KeySet.from_dict({"B": ["a"]})
    ks3 = KeySet.from_dict({"A": [1, 2, 3], "C": ["c"]})

    with pytest.raises(ValueError, match=re.escape("they share a column: A")):
        _ = ks1 * ks2 * ks3

    with pytest.raises(ValueError, match=re.escape("they share a column: A")):
        _ = ks1 * ks3


@pytest.mark.parametrize(
    "factor_dfs,expected_col_types",
    [
        (
            [pd.DataFrame({"A": ["a"]})],
            {"A": ColumnType.VARCHAR},
        ),
        (
            [pd.DataFrame({"B": [1]})],
            {"B": ColumnType.INTEGER},
        ),
        (
            [pd.DataFrame({"A": ["a"]}), pd.DataFrame({"B": [1]})],
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
        ),
        (
            [pd.DataFrame({"A": ["a"], "B": [1]})],
            {"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
        ),
    ],
)
def test_schema(
    spark: SparkSession,
    factor_dfs: List[pd.DataFrame],
    expected_col_types: Dict[str, ColumnType],
) -> None:
    """Test that the schema of the _ProductKeySet is as expected."""
    factors = [KeySet.from_dataframe(spark.createDataFrame(df)) for df in factor_dfs]
    product_keyset = reduce(lambda a, b: a * b, factors)
    expected_schema = Schema(expected_col_types, default_allow_null=True)
    got_schema = product_keyset.schema()
    assert got_schema == expected_schema


@pytest.mark.parametrize(
    "factors,select_col,expect_df",
    [
        (
            [
                KeySet.from_dict({"A": [1, 2, 3]}),
                KeySet.from_dict({"B": ["b"]}),
            ],
            "A",
            pd.DataFrame({"A": [1, 2, 3]}),
        ),
        (
            [
                KeySet.from_dict(
                    {
                        "A": [1, 2, 3],
                        "B": ["b"],
                    }
                ),
                KeySet.from_dict({"C": [9, 8, 7], "D": ["d"]}),
            ],
            "A",
            pd.DataFrame({"A": [1, 2, 3]}),
        ),
    ],
)
def test_getitem_single_column(
    spark: SparkSession, factors: List[KeySet], select_col: str, expect_df: pd.DataFrame
) -> None:
    """Test filtering with __getitem__."""
    product = reduce(lambda x, y: x * y, factors)
    filtered_product = product[select_col]
    assert_frame_equal_with_sort(filtered_product.dataframe().toPandas(), expect_df)


@pytest.mark.parametrize(
    "select_cols,expect_df",
    [
        (
            ("A", "B"),
            pd.DataFrame({"A": [1, 2, 3], "B": ["b", "b", "b"]}),
        ),
        (
            ("A", "C"),
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "C": ["c", "c", "c"],
                }
            ),
        ),
        (
            ("B", "C"),
            pd.DataFrame(
                {
                    "B": ["b"],
                    "C": ["c"],
                }
            ),
        ),
        (
            ("A", "B", "C"),
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": ["b", "b", "b"],
                    "C": ["c", "c", "c"],
                },
            ),
        ),
    ],
)
def test_getitem_multiple_columns_as_tuple(
    spark: SparkSession, select_cols: Tuple[str, ...], expect_df: pd.DataFrame
) -> None:
    """Test filtering multiple columns with __getitem__."""
    keyset_a = KeySet.from_dict({"A": [1, 2, 3]})
    keyset_b = KeySet.from_dict({"B": ["b"]})
    keyset_c = KeySet.from_dict({"C": ["c"]})

    product = keyset_a * keyset_b * keyset_c
    filtered = product[select_cols]
    assert_frame_equal_with_sort(filtered.dataframe().toPandas(), expect_df)


def test_getitem_from_subset_of_columns(spark: SparkSession) -> None:
    """Test selecting a subset of one factor's columns."""
    keyset1 = KeySet.from_dataframe(
        spark.createDataFrame(
            pd.DataFrame(
                [
                    [1, "a", "x"],
                    [2, "b", "y"],
                    [3, "c", "z"],
                ],
                columns=["A", "B", "C"],
            ),
        ),
    )
    keyset2 = KeySet.from_dict({"Z": [9, 8, 7]})
    product = keyset1 * keyset2
    filtered = product["A"]
    expect_df = pd.DataFrame({"A": [1, 2, 3]})
    assert_frame_equal_with_sort(filtered.dataframe().toPandas(), expect_df)

    filtered_ab = product["A", "B"]
    expect_df_ab = pd.DataFrame(
        [
            [1, "a"],
            [2, "b"],
            [3, "c"],
        ],
        columns=["A", "B"],
    )
    assert_frame_equal_with_sort(filtered_ab.dataframe().toPandas(), expect_df_ab)


@pytest.mark.parametrize(
    "columns",
    [
        "nonexistent_column",
        ("A", "B", "nonexistent_column"),
        ["A", "B", "nonexistent_column"],
        ["not a column", "neither is this one"],
    ],
)
def test_getitem_errors_with_missing_columns(
    spark: SparkSession, columns: Union[str, Tuple[str, ...], List[str]]
) -> None:
    """Test __getitem__ fails if the specified column doesn't exist."""
    keyset_a = KeySet.from_dict({"A": [1]})
    keyset_b = KeySet.from_dict({"B": ["b"]})
    product = keyset_a * keyset_b

    with pytest.raises(
        ValueError,
        match="Cannot select columns .* because those columns are not in this KeySet",
    ):
        _ = product[columns]


def test_getitem_errors_with_duplicate_columns(
    spark: SparkSession,
) -> None:
    """Test __getitem__ fails if you pass it the same column multiple times."""
    product = KeySet.from_dict({"A": [1]}) * KeySet.from_dict({"B": ["b"]})
    with pytest.raises(ValueError, match="duplicate columns were present"):
        _ = product["A", "A"]


@pytest.mark.parametrize(
    "factor_dfs,expected_df",
    [
        ([pd.DataFrame({"A": [1]})], pd.DataFrame({"A": [1]})),
        (
            [pd.DataFrame({"A": [1]}), pd.DataFrame({"B": ["a", "b"]})],
            pd.DataFrame({"A": [1, 1], "B": ["a", "b"]}),
        ),
        (
            [
                pd.DataFrame({"A": [1, 2], "B": ["a", "b"]}),
                pd.DataFrame({"C": [9, 8]}),
            ],
            pd.DataFrame(
                [
                    [1, "a", 9],
                    [1, "a", 8],
                    [2, "b", 9],
                    [2, "b", 8],
                ],
                columns=["A", "B", "C"],
            ),
        ),
    ],
)
def test_dataframe(
    spark: SparkSession, factor_dfs: List[pd.DataFrame], expected_df: pd.DataFrame
) -> None:
    """Test _ProductKeySet.dataframe()."""
    factors = [KeySet.from_dataframe(spark.createDataFrame(df)) for df in factor_dfs]
    product = reduce(lambda x, y: x * y, factors)
    got = product.dataframe()
    assert_frame_equal_with_sort(got.toPandas(), expected_df)


def test_complex_filter(spark: SparkSession) -> None:
    """Test a filter that applies *partially* to a factor."""
    keyset_a = KeySet.from_dict(
        {
            "A": [0, 1, 2],
        }
    )
    keyset_b = KeySet.from_dict({"B": [10, 9, 8]})

    product = keyset_a * keyset_b
    filtered = product.filter("A >= 1 AND B < 10")
    assert not isinstance(filtered, _ProductKeySet)

    expect_df = pd.DataFrame(
        [
            [1, 9],
            [1, 8],
            [2, 9],
            [2, 8],
        ],
        columns=["A", "B"],
    )
    assert_frame_equal_with_sort(filtered.dataframe().toPandas(), expect_df)


@pytest.mark.parametrize(
    "column_ordering",
    list(itertools.permutations(["A", "B", "C", "D"])),
)
def test_getitem_ordering(spark: SparkSession, column_ordering: Sequence[str]) -> None:
    """Test columns returned with __getitem__ are in the correct order."""
    keyset_a = KeySet.from_dict(
        {
            "A": [0, 1, 2],
        }
    )
    keyset_b = KeySet.from_dict(
        {
            "B": ["b1", "b2"],
        }
    )
    keyset_c = KeySet.from_dataframe(
        spark.createDataFrame(
            pd.DataFrame(
                [
                    [10, 0],
                    [9, 0],
                ],
                columns=["C", "D"],
            ),
        ),
    )

    product = keyset_a * keyset_b * keyset_c
    expect_df = pd.DataFrame(
        [
            [0, "b1", 10, 0],
            [0, "b1", 9, 0],
            [0, "b2", 10, 0],
            [0, "b2", 9, 0],
            [1, "b1", 10, 0],
            [1, "b1", 9, 0],
            [1, "b2", 10, 0],
            [1, "b2", 9, 0],
            [2, "b1", 10, 0],
            [2, "b1", 9, 0],
            [2, "b2", 10, 0],
            [2, "b2", 9, 0],
        ],
        columns=["A", "B", "C", "D"],
    )

    selected_keyset = product[column_ordering]
    assert list(column_ordering) == selected_keyset.columns()
    got_df = selected_keyset.dataframe()
    assert_frame_equal_with_sort(got_df.toPandas(), expect_df)
    assert list(column_ordering) == got_df.columns


# Defining KeySets for _ProductKeySet Equality Testing.
DF_A = KeySet.from_dict({"A": [1, 2, 3]})
DF_B = KeySet.from_dict({"B": ["a", "b", "c"]})
DF_C = KeySet.from_dict({"C": [9, 8, 7], "D": ["D", "E", "F"]})
DF_D = KeySet.from_dict({"E": [100, 200, 300], "F": ["G", "H", "I"]})


@pytest.mark.parametrize(
    "ks_a,ks_b,equal",
    [
        (DF_A * DF_B, DF_A * DF_B, True),
        (DF_A * DF_B, DF_B * DF_A, True),
        (DF_A * DF_B, DF_A * DF_C, False),
        (DF_A * DF_B, DF_A * DF_D, False),
        (DF_A * DF_C, DF_C * DF_A, True),
        (DF_A * DF_C, DF_D * DF_A, False),
        (DF_D * DF_C, DF_D * DF_C, True),
    ],
)
def test_equality(ks_a: _ProductKeySet, ks_b: _ProductKeySet, equal: bool):
    """Test custom equality function of two ProductKeySets."""
    assert (ks_a == ks_b) == equal


@pytest.mark.parametrize(
    "_,keyset,expected",
    [
        ("Empty Keyset", KeySet.from_dict({}) * KeySet.from_dict({}), 1),
        (
            "Single Item, Two Columns",
            KeySet.from_dict({"A": [0]}) * KeySet.from_dict({"B": [1]}),
            1,
        ),
        (
            "Two Items, Two Columns",
            KeySet.from_dict({"A": [0, 1]}) * KeySet.from_dict({"B": [1]}),
            2,
        ),
    ],
)
def test_size(_, keyset: KeySet, expected: int):
    """Tests that the expected KeySet size is returned."""
    assert keyset.size() == expected
