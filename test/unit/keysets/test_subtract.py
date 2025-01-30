"""Unit tests for (v2) KeySet.__sub__."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, ContextManager

import pandas as pd
import pytest
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics import KeySet
from tmlt.analytics._schema import ColumnDescriptor, ColumnType
from tmlt.analytics.keyset._keyset import KeySetPlan


@parametrize(
    Case("single_column")(
        left=KeySet.from_tuples([(1,), (2,)], columns=["A"]),
        right=KeySet.from_tuples([(2,)], columns=["A"]),
        expected_df=pd.DataFrame({"A": [1]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("one_of_two_column")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_tuples([("b",)], columns=["B"]),
        expected_df=pd.DataFrame([(1, "a"), (2, "a")], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("two_columns")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_tuples([(2, "b")], columns=["A", "B"]),
        expected_df=pd.DataFrame([(1, "a"), (2, "a"), (1, "b")], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("subtract_everything_subset_columns")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_tuples([("a",), ("b",)], columns=["B"]),
        expected_df=pd.DataFrame([], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("subtract_from_self")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        expected_df=pd.DataFrame([], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("non-overlapping_values")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_tuples(
            [
                ("b",),
                ("c",),
            ],
            columns=["B"],
        ),
        expected_df=pd.DataFrame([(1, "a"), (2, "a")], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("no_matching_values")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_tuples([("c",)], columns=["B"]),
        expected_df=pd.DataFrame(
            [(1, "a"), (2, "a"), (1, "b"), (2, "b")], columns=["A", "B"]
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("empty_rhs")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=(
            KeySet.from_tuples([("b",)], columns=["B"])
            - KeySet.from_tuples([("b",)], columns=["B"])
        ),
        expected_df=pd.DataFrame(
            [(1, "a"), (2, "a"), (1, "b"), (2, "b")], columns=["A", "B"]
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("empty_lhs")(
        left=(
            KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]})
            - KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]})
        ),
        right=KeySet.from_tuples([("b",)], columns=["B"]),
        expected_df=pd.DataFrame([], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("unsubtracted_null")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", None]}),
        right=KeySet.from_tuples([("a",)], columns=["B"]),
        expected_df=pd.DataFrame([(1, None), (2, None)], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
    ),
    Case("subtract_null")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", None]}),
        right=KeySet.from_tuples([(None,), ("c",)], columns=["B"]),
        expected_df=pd.DataFrame([(1, "a"), (2, "a")], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        },
    ),
    Case("subtract_null_from_nonull")(
        left=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        right=KeySet.from_tuples([(None,), ("b",)], columns=["B"]),
        expected_df=pd.DataFrame([(1, "a"), (2, "a")], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
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
    ks = left - right
    assert_dataframe_equal(ks.dataframe(), expected_df)
    assert ks.schema() == expected_schema


# pylint: disable=protected-access
@parametrize(
    Case("single_column")(
        left=KeySet._detect(["A"]),
        right=KeySet.from_tuples([(2,)], columns=["A"]),
        expected_columns=["A"],
    ),
    Case("one_of_two_column")(
        left=KeySet._detect(["A", "B"]),
        right=KeySet.from_tuples([("b",)], columns=["B"]),
        expected_columns=["A", "B"],
    ),
    Case("two_columns")(
        left=KeySet._detect(["A", "B"]),
        right=KeySet.from_tuples([(2, "b")], columns=["A", "B"]),
        expected_columns=["A", "B"],
    ),
    Case("empty_rhs")(
        left=KeySet._detect(["A", "B"]),
        right=(
            KeySet.from_tuples([("b",)], columns=["B"])
            - KeySet.from_tuples([("b",)], columns=["B"])
        ),
        expected_columns=["A", "B"],
    ),
)
# pylint: enable=protected-access
def test_valid_plan(
    left: KeySetPlan,
    right: KeySet,
    expected_columns: list[str],
):
    """Valid parameters including a KeySetPlan work as expected."""
    ks = left - right
    assert isinstance(ks, KeySetPlan)
    assert ks.columns() == expected_columns


@parametrize(
    Case("non_overlapping_columns")(
        left=KeySet.from_tuples([(1,)], columns=["A"]),
        right=KeySet.from_tuples([(1,)], columns=["B"]),
        expectation=pytest.raises(
            ValueError,
            match="right hand side has columns that do not exist in the left hand "
            "side",
        ),
    ),
    Case("non_overlapping_columns")(
        left=KeySet.from_tuples([(1, 2)], columns=["A", "B"]),
        right=KeySet.from_tuples([(2, 3)], columns=["B", "C"]),
        expectation=pytest.raises(
            ValueError,
            match="right hand side has columns that do not exist in the left hand "
            "side",
        ),
    ),
    Case("subtract_plan")(
        left=KeySet.from_tuples([(1, 2)], columns=["A", "B"]),
        right=KeySet._detect(columns=["A"]),
        expectation=pytest.raises(
            ValueError,
            match="Cannot subtract a KeySetPlan",
        ),
    ),
)
def test_invalid(left: KeySet, right: Any, expectation: ContextManager[None]):
    """Invalid tuples/columns values are rejected."""
    with expectation:
        _ = left - right
