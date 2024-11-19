"""Unit tests for (v2) KeySet.__mul__."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import datetime
from functools import reduce
from typing import Any, ContextManager

import pandas as pd
import pytest
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet
from tmlt.analytics._schema import ColumnDescriptor, ColumnType


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
    # TODO(tumult#3381, tumult#3382, tumult#3383): There's not currently a way
    #     to create a KeySet which contains columns but no rows. Once there is
    #     (any of the mentioned issues will add one), a test should be added
    #     here to ensure that crossing an empty KeySet with any other KeySet
    #     produces another empty KeySet.
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
)
def test_valid(
    left: KeySet,
    right: KeySet,
    expected_df: pd.DataFrame,
    expected_schema: dict[str, ColumnDescriptor],
):
    """Valid parameters work as expected."""
    ks = left * right
    assert ks.columns() == left.columns() + right.columns()
    assert_dataframe_equal(ks.dataframe(), expected_df)
    assert ks.schema() == expected_schema


@parametrize(
    Case("2^4")(factors=4, factor_size=2),
    Case("64^3")(factors=3, factor_size=64),
    Case("3^20", marks=pytest.mark.slow)(factors=20, factor_size=3),
    Case("63^5", marks=pytest.mark.slow)(factors=5, factor_size=63),
    Case("128^4", marks=pytest.mark.slow)(factors=4, factor_size=128),
)
def test_chained(factors: int, factor_size: int):
    """Chaining cross-joins works as expected."""
    keysets = [
        KeySet.from_tuples([(i,) for i in range(factor_size)], columns=[str(f)])
        for f in range(factors)
    ]
    ks = reduce(lambda l, r: l * r, keysets)
    assert ks.dataframe().count() == factor_size**factors


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
    """Invalid tuples/columns values are rejected."""
    with expectation:
        _ = left * right
