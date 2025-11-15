"""Unit tests for (v2) KeySet.__mul__."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from functools import reduce
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
        right=KeySet.from_tuples([(3,), (4,)], columns=["B"]),
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("one_column_swapped")(
        left=KeySet.from_tuples([(1,), (2,)], columns=["B"]),
        right=KeySet.from_tuples([(3,), (4,)], columns=["A"]),
        expected_df=pd.DataFrame({"B": [1, 1, 2, 2], "A": [3, 4, 3, 4]}),
        expected_schema={
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("two_column")(
        left=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        right=KeySet.from_tuples([(5, 6), (7, 8)], columns=["C", "D"]),
        expected_df=pd.DataFrame(
            [(1, 2, 5, 6), (1, 2, 7, 8), (3, 4, 5, 6), (3, 4, 7, 8)],
            columns=["A", "B", "C", "D"],
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "C": ColumnDescriptor(ColumnType.INTEGER),
            "D": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("mixed_types")(
        left=KeySet.from_tuples([(5, None), (None, "str")], columns=["int", "string"]),
        right=KeySet.from_tuples([(datetime.date.fromordinal(1),)], columns=["date"]),
        expected_df=pd.DataFrame(
            [
                (5, None, datetime.date.fromordinal(1)),
                (None, "str", datetime.date.fromordinal(1)),
            ],
            columns=["int", "string", "date"],
        ),
        expected_schema={
            "int": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "string": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "date": ColumnDescriptor(ColumnType.DATE),
        },
    ),
    Case("dataframe_left")(
        left=lambda spark: spark.createDataFrame(
            [[1], [2]], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=KeySet.from_tuples([(3,), (4,)], columns=["B"]),
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("dataframe_right")(
        left=KeySet.from_tuples([(1,), (2,)], columns=["A"]),
        right=lambda spark: spark.createDataFrame(
            [[3], [4]], schema=StructType([StructField("B", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("dataframe_both")(
        left=lambda spark: spark.createDataFrame(
            [[1], [2]], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=lambda spark: spark.createDataFrame(
            [[3], [4]], schema=StructType([StructField("B", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("total_left")(
        left=KeySet.from_tuples([], columns=[]),
        right=KeySet.from_tuples([(1,), (2,)], columns=["A"]),
        expected_df=pd.DataFrame({"A": [1, 2]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("total_right")(
        left=KeySet.from_tuples([(1,), (2,)], columns=["A"]),
        right=KeySet.from_tuples([], columns=[]),
        expected_df=pd.DataFrame({"A": [1, 2]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("total_both")(
        left=KeySet.from_tuples([], columns=[]),
        right=KeySet.from_tuples([], columns=[]),
        expected_df=pd.DataFrame({}),
        expected_schema={},
    ),
    Case("empty_left")(
        left=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=KeySet.from_tuples([(1,), (2,)], columns=["B"]),
        expected_df=pd.DataFrame({"A": [], "B": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("empty_right")(
        left=KeySet.from_tuples([(1,), (2,)], columns=["A"]),
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
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("B", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": [], "B": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("total_empty")(
        left=KeySet.from_tuples([], columns=[]),
        right=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        expected_df=pd.DataFrame({"A": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("empty_total")(
        left=lambda spark: spark.createDataFrame(
            [], schema=StructType([StructField("A", LongType(), False)])
        ),
        right=KeySet.from_tuples([], columns=[]),
        expected_df=pd.DataFrame({"A": []}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
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
    ks = left * right
    assert ks.columns() == left.columns() + right.columns()
    assert ks.schema() == expected_schema
    if ks.columns():
        assert ks.size() == len(expected_df)
    else:
        assert ks.size() == 1
    assert_dataframe_equal(ks.dataframe(), expected_df)


@parametrize(
    Case("left_plan")(
        left=KeySet._detect(["A"]),
        right=KeySet.from_dict({"B": [1, 2]}),
        expected_columns=["A", "B"],
    ),
    Case("right_plan")(
        left=KeySet.from_dict({"A": [1, 2]}),
        right=KeySet._detect(["B"]),
        expected_columns=["A", "B"],
    ),
    Case("both_plan")(
        left=KeySet._detect(["A"]),
        right=KeySet._detect(["B"]),
        expected_columns=["A", "B"],
    ),
    Case("left_plan_swapped")(
        left=KeySet._detect(["B"]),
        right=KeySet.from_dict({"A": [1, 2]}),
        expected_columns=["B", "A"],
    ),
    Case("right_plan_swapped")(
        left=KeySet.from_dict({"B": [1, 2]}),
        right=KeySet._detect(["A"]),
        expected_columns=["B", "A"],
    ),
    Case("both_plan_swapped")(
        left=KeySet._detect(["B"]),
        right=KeySet._detect(["A"]),
        expected_columns=["B", "A"],
    ),
)
def test_valid_plan(
    left: Union[KeySet, KeySetPlan],
    right: Union[KeySet, KeySetPlan],
    expected_columns: list[str],
):
    """Valid parameters including a KeySetPlan work as expected."""
    ks = left * right
    assert isinstance(ks, KeySetPlan)
    assert ks.columns() == expected_columns


@parametrize(
    Case("2^4")(factors=4, factor_size=2),
    Case("64^3")(factors=3, factor_size=64),
    Case("3^20", marks=pytest.mark.slow)(factors=20, factor_size=3),
    Case("64^5", marks=pytest.mark.slow)(factors=5, factor_size=64),
    Case("65536^2", marks=pytest.mark.slow)(factors=2, factor_size=65536),
)
def test_chained(factors: int, factor_size: int):
    """Chaining cross-joins works as expected."""
    keysets = [
        KeySet.from_tuples([(i,) for i in range(factor_size)], columns=[str(f)])
        for f in range(factors)
    ]
    ks = reduce(lambda l, r: l * r, keysets)
    assert ks.size() == factor_size**factors
    assert ks.dataframe().count() == factor_size**factors
    assert ks.columns() == [str(f) for f in range(factors)]
    assert ks.dataframe().columns == ks.columns()


def test_coalesce(spark):
    """Factors with many partitions are coalesced correctly."""
    size = 128
    partition_target = 2 * spark.sparkContext.defaultParallelism
    ks1 = KeySet.from_dataframe(
        spark.range(size, numPartitions=4 * partition_target).withColumnRenamed(
            "id", "A"
        )
    )
    ks2 = KeySet.from_dataframe(
        spark.range(size, numPartitions=4 * partition_target).withColumnRenamed(
            "id", "B"
        )
    )
    ks = ks1 * ks2

    assert ks.size() == size**2
    assert ks.dataframe().count() == size**2
    assert ks.columns() == ["A", "B"]
    assert ks.dataframe().columns == ["A", "B"]
    assert ks.dataframe().rdd.getNumPartitions() <= 2 * partition_target


@parametrize(
    Case("overlapping_columns")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=KeySet.from_tuples([(1,)], columns=["A"]),
        expectation=pytest.raises(
            ValueError,
            match="Unable to cross-join KeySets, they have overlapping columns",
        ),
    ),
    Case("partial_overlapping_columns")(
        left=KeySet.from_tuples([(1, 2)], columns=["A", "B"]),
        right=KeySet.from_tuples([(1, 2)], columns=["B", "C"]),
        expectation=pytest.raises(
            ValueError,
            match="Unable to cross-join KeySets, they have overlapping columns",
        ),
    ),
    Case("invalid_right_operand")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=5,
        expectation=pytest.raises(
            ValueError,
            match="KeySet multiplication expected another KeySet",
        ),
    ),
)
def test_invalid(left: KeySet, right: Any, expectation: ContextManager[None]):
    """Invalid cross-joins are rejected."""
    with expectation:
        _ = left * right
