"""Unit tests for KeySet.join."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
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
    Case("one_column")(
        left=KeySet.from_tuples([(1,), (2,)], columns=["A"]),
        right=KeySet.from_tuples([(2,), (3,)], columns=["A"]),
        expected_df=pd.DataFrame({"A": [2]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("two_column")(
        left=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        right=KeySet.from_tuples([(2, 5), (4, 6)], columns=["B", "C"]),
        expected_df=pd.DataFrame(
            [(1, 2, 5), (3, 4, 6)],
            columns=["A", "B", "C"],
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "C": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("mixed_types")(
        left=KeySet.from_tuples(
            [(5, None), (None, "str"), (1, "str")], columns=["int", "string"]
        ),
        right=KeySet.from_tuples(
            [
                ("str", datetime.date.fromordinal(1)),
                (None, datetime.date.fromordinal(2)),
            ],
            columns=["string", "date"],
        ),
        expected_df=pd.DataFrame(
            [
                (None, "str", datetime.date.fromordinal(1)),
                (1, "str", datetime.date.fromordinal(1)),
                (5, None, datetime.date.fromordinal(2)),
            ],
            columns=["int", "string", "date"],
        ),
        expected_schema={
            "int": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "string": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "date": ColumnDescriptor(ColumnType.DATE),
        },
    ),
    Case("nullable")(
        left=KeySet.from_dict({"A": [1, 2, None]}),
        right=KeySet.from_dict({"A": [1, None, 3]}),
        expected_df=pd.DataFrame({"A": [1, None]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        },
    ),
    Case("mixed_nullability")(
        left=KeySet.from_dict({"A": [1, 2, 3]}),
        right=KeySet.from_dict({"A": [1, None]}),
        expected_df=pd.DataFrame({"A": [1]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("dataframe_left")(
        left=lambda spark: spark.createDataFrame(
            [[1], [2]], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        expected_df=pd.DataFrame({"A": [1], "B": [2]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("dataframe_right")(
        left=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        right=lambda spark: spark.createDataFrame(
            [[3], [4]], schema=StructType([StructField("B", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": [3], "B": [4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("disjoint")(
        left=KeySet.from_dict({"A": [1, 2, 3]}),
        right=KeySet.from_dict({"A": [4, 5, 6]}),
        expected_df=pd.DataFrame({"A": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("empty_left")(
        left=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        expected_df=pd.DataFrame({"A": [], "B": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("empty_right")(
        left=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        right=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("B", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": [], "B": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("empty_both")(
        left=lambda spark: spark.createDataFrame(
            [],
            schema=StructType(
                [
                    StructField("A", LongType(), False),
                    StructField("B", LongType(), False),
                ]
            ),
        ),
        right=lambda spark: spark.createDataFrame(
            [],
            schema=StructType(
                [
                    StructField("B", LongType(), False),
                    StructField("C", LongType(), False),
                ]
            ),
        ),
        expected_df=pd.DataFrame({"A": [], "B": [], "C": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "C": ColumnDescriptor(ColumnType.INTEGER),
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
    ks = left.join(right)
    assert ks.columns() == list(expected_schema.keys())
    assert ks.schema() == expected_schema
    assert ks.size() == len(expected_df)
    assert_dataframe_equal(ks.dataframe(), expected_df)


@parametrize(
    Case("left_plan")(
        left=KeySet._detect(["A"]),
        right=KeySet.from_dict({"A": [0], "B": [1, 2]}),
        expected_columns=["A", "B"],
    ),
    Case("right_plan")(
        left=KeySet.from_dict({"A": [1, 2]}),
        right=KeySet._detect(["A", "B"]),
        expected_columns=["A", "B"],
    ),
    Case("both_plan")(
        left=KeySet._detect(["A", "B"]),
        right=KeySet._detect(["B", "C"]),
        expected_columns=["A", "B", "C"],
    ),
    Case("left_plan_swapped")(
        left=KeySet._detect(["B"]),
        right=KeySet.from_dict({"A": [1, 2], "B": [0]}),
        expected_columns=["B", "A"],
    ),
    Case("right_plan_swapped")(
        left=KeySet.from_dict({"B": [1, 2]}),
        right=KeySet._detect(["A", "B"]),
        expected_columns=["B", "A"],
    ),
    Case("both_plan_swapped")(
        left=KeySet._detect(["B", "C"]),
        right=KeySet._detect(["A", "B"]),
        expected_columns=["B", "C", "A"],
    ),
)
def test_valid_plan(
    left: Union[KeySet, KeySetPlan],
    right: Union[KeySet, KeySetPlan],
    expected_columns: list[str],
):
    """Valid parameters including a KeySetPlan work as expected."""
    ks = left.join(right)
    assert isinstance(ks, KeySetPlan)
    assert ks.columns() == expected_columns


@parametrize(
    Case("no_overlapping_columns")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=KeySet.from_tuples([(1,)], columns=["B"]),
        expectation=pytest.raises(
            ValueError,
            match="Unable to join KeySets, they have no overlapping columns",
        ),
    ),
    Case("column_type_mismatch")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=KeySet.from_tuples([("1",)], columns=["A"]),
        expectation=pytest.raises(
            ValueError,
            match="Unable to join KeySets, join column A does not have the same type",
        ),
    ),
    Case("invalid_right_operand")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=5,
        expectation=pytest.raises(
            ValueError,
            match="KeySet join expected another KeySet",
        ),
    ),
)
def test_invalid(left: KeySet, right: Any, expectation: ContextManager[None]):
    """Invalid tuples/columns values are rejected."""
    with expectation:
        _ = left.join(right)
