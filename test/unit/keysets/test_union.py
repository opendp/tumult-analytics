"""Unit tests for KeySet.union."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Callable, ContextManager, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StructField, StructType
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics import KeySet
from tmlt.analytics._schema import ColumnDescriptor, ColumnType
from tmlt.analytics.keyset._keyset import KeySetPlan


@parametrize(
    Case("single_column")(
        left=KeySet.from_tuples([(1,), (2,)], columns=["A"]),
        right=KeySet.from_tuples([(2,), (3,)], columns=["A"]),
        expected_df=pd.DataFrame({"A": [1, 2, 3]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("two_columns")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_tuples([(2, "b"), (3, "c")], columns=["A", "B"]),
        expected_df=pd.DataFrame(
            [(1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "c")],
            columns=["A", "B"],
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("complete_overlap")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        expected_df=pd.DataFrame(
            [(1, "a"), (2, "a"), (1, "b"), (2, "b")],
            columns=["A", "B"],
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("no_overlap")(
        left=KeySet.from_dict({"A": [1, 2]}),
        right=KeySet.from_dict({"A": [3, 4]}),
        expected_df=pd.DataFrame({"A": [1, 2, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("with_nulls")(
        left=KeySet.from_dict({"A": [1, None], "B": ["a", "b"]}),
        right=KeySet.from_tuples([(None, "b"), (3, None)], columns=["A", "B"]),
        expected_df=pd.DataFrame(
            [(1, "a"), (None, "a"), (1, "b"), (None, "b"), (3, None)],
            columns=["A", "B"],
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
    ),
    Case("empty_left")(
        left=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=KeySet.from_dict({"A": [1, 2]}),
        expected_df=pd.DataFrame({"A": [1, 2]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("empty_right")(
        left=KeySet.from_dict({"A": [1, 2]}),
        right=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": [1, 2]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("both_empty")(
        left=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("different_column_order")(
        left=KeySet.from_tuples([(1, "a"), (2, "b")], columns=["A", "B"]),
        right=KeySet.from_tuples([("c", 3), ("b", 2)], columns=["B", "A"]),
        expected_df=pd.DataFrame(
            [(1, "a"), (2, "b"), (3, "c")],
            columns=["A", "B"],
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
)
def test_valid(
    left: Union[KeySet, Callable[[SparkSession], DataFrame]],
    right: Union[KeySet, Callable[[SparkSession], DataFrame]],
    expected_df: pd.DataFrame,
    expected_schema: dict[str, ColumnDescriptor],
    spark,
):
    """Valid parameters work as expected."""
    if callable(left):
        left = KeySet.from_dataframe(left(spark))
    if callable(right):
        right = KeySet.from_dataframe(right(spark))
    ks = left.union(right)
    assert ks.columns() == list(expected_schema.keys())
    assert ks.schema() == expected_schema
    assert ks.size() == len(expected_df)
    assert_dataframe_equal(ks.dataframe(), expected_df)


@parametrize(
    Case("left_plan")(
        left=KeySet._detect(["A"]),
        right=KeySet.from_dict({"A": [1, 2]}),
        expected_columns=["A"],
    ),
    Case("right_plan")(
        left=KeySet.from_dict({"A": [1, 2]}),
        right=KeySet._detect(["A"]),
        expected_columns=["A"],
    ),
    Case("both_plan")(
        left=KeySet._detect(["A", "B"]),
        right=KeySet._detect(["A", "B"]),
        expected_columns=["A", "B"],
    ),
    Case("plan_with_two_columns")(
        left=KeySet._detect(["A", "B"]),
        right=KeySet.from_dict({"A": [1], "B": ["x"]}),
        expected_columns=["A", "B"],
    ),
)
def test_valid_plan(
    left: Union[KeySet, KeySetPlan],
    right: Union[KeySet, KeySetPlan],
    expected_columns: list[str],
):
    """Valid parameters including a KeySetPlan work as expected."""
    ks = left.union(right)
    assert isinstance(ks, KeySetPlan)
    assert ks.columns() == expected_columns


@parametrize(
    Case("different_columns")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=KeySet.from_tuples([(1,)], columns=["B"]),
        expectation=pytest.raises(
            ValueError,
            match="KeySet union operands must have the same columns",
        ),
    ),
    Case("different_column_count")(
        left=KeySet.from_tuples([(1, 2)], columns=["A", "B"]),
        right=KeySet.from_tuples([(1,)], columns=["A"]),
        expectation=pytest.raises(
            ValueError,
            match="KeySet union operands must have the same columns",
        ),
    ),
    Case("partially_overlapping_columns")(
        left=KeySet.from_tuples([(1, 2)], columns=["A", "B"]),
        right=KeySet.from_tuples([(2, 3)], columns=["B", "C"]),
        expectation=pytest.raises(
            ValueError,
            match="KeySet union operands must have the same columns",
        ),
    ),
    Case("mismatched_column_types")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_dict({"A": [1, 2], "B": [3, 4]}),
        expectation=pytest.raises(
            ValueError,
            match="KeySet union operands have mismatched column types",
        ),
    ),
    Case("invalid_right_operand")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=5,
        expectation=pytest.raises(
            ValueError,
            match="KeySet union expected another KeySet",
        ),
    ),
    Case("invalid_right_operand_string")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right="invalid",
        expectation=pytest.raises(
            ValueError,
            match="KeySet union expected another KeySet",
        ),
    ),
)
def test_invalid(left: KeySet, right: Any, expectation: ContextManager[None]):
    """Invalid parameters are rejected."""
    with expectation:
        _ = left.union(right)


def test_union_is_commutative():
    """Test that union is commutative (left.union(right) == right.union(left))."""
    left = KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]})
    right = KeySet.from_tuples([(2, "b"), (3, "c")], columns=["A", "B"])

    union_left_right = left.union(right)
    union_right_left = right.union(left)

    assert union_left_right == union_right_left


def test_union_multiple():
    """Test chaining multiple union operations."""
    ks1 = KeySet.from_dict({"A": [1]})
    ks2 = KeySet.from_dict({"A": [2]})
    ks3 = KeySet.from_dict({"A": [3]})

    result = ks1.union(ks2).union(ks3)
    expected_df = pd.DataFrame({"A": [1, 2, 3]})

    assert_dataframe_equal(result.dataframe(), expected_df)


def test_union_with_self():
    """Test that union with self works and deduplicates correctly."""
    ks = KeySet.from_dict({"A": [1, 2, 3]})
    result = ks.union(ks)

    assert result == ks
    assert result.size() == 3


def test_union_preserves_column_order():
    """Test that union preserves the column order of the left operand."""
    left = KeySet.from_tuples([(1, "a"), (2, "b")], columns=["A", "B"])
    right = KeySet.from_tuples([("c", 3)], columns=["B", "A"])

    result = left.union(right)

    # Result should have columns in the order of the left operand
    assert result.columns() == ["A", "B"]


def test_union_with_filtered_keysets():
    """Test union with filtered KeySets."""
    ks1 = KeySet.from_dict({"A": [1, 2, 3, 4]}).filter("A <= 2")
    ks2 = KeySet.from_dict({"A": [3, 4, 5]}).filter("A >= 4")

    result = ks1.union(ks2)
    expected_df = pd.DataFrame({"A": [1, 2, 4, 5]})

    assert_dataframe_equal(result.dataframe(), expected_df)
