"""Unit tests for (v2) KeySet projection."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, ContextManager, Sequence, Union

import pandas as pd
import pytest
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet, KeySetPlan
from tmlt.analytics._schema import ColumnDescriptor, ColumnType


@parametrize(
    Case("one_column_str")(
        base=KeySet.from_dict({"A": [1, 2], "B": [3, 4]}),
        columns="A",
        expected_df=pd.DataFrame({"A": [1, 2]}),
        expected_schema={"A": ColumnDescriptor(ColumnType.INTEGER)},
    ),
    Case("one_column")(
        base=KeySet.from_dict({"A": [1, 2], "B": [3, 4]}),
        columns=["A"],
        expected_df=pd.DataFrame({"A": [1, 2]}),
        expected_schema={"A": ColumnDescriptor(ColumnType.INTEGER)},
    ),
    Case("two_column")(
        base=KeySet.from_dict({"A": [1, 2], "B": [3, 4], "C": [5, 6]}),
        columns=["A", "B"],
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("all_columns")(
        base=KeySet.from_dict({"A": [1, 2], "B": [3, 4]}),
        columns=["A", "B"],
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("crossjoin_child")(
        base=KeySet.from_tuples([(1, 2, 3)], ["A", "B", "C"])
        * KeySet.from_tuples([(4, 5, 6)], ["D", "E", "F"]),
        columns=["A", "C", "E"],
        expected_df=pd.DataFrame({"A": [1], "C": [3], "E": [5]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "C": ColumnDescriptor(ColumnType.INTEGER),
            "E": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("crossjoin_child_reordered")(
        base=KeySet.from_tuples([(1, 2, 3)], ["A", "B", "C"])
        * KeySet.from_tuples([(4, 5, 6)], ["D", "E", "F"]),
        columns=["C", "A", "E"],
        expected_df=pd.DataFrame({"C": [3], "A": [1], "E": [5]}),
        expected_schema={
            "C": ColumnDescriptor(ColumnType.INTEGER),
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "E": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("remove_detect_columns")(
        base=KeySet.from_tuples([(1, 2, 3)], ["A", "B", "C"])
        * KeySet._detect(["D", "E", "F"]),  # pylint: disable=protected-access
        columns=["A", "B"],
        expected_df=pd.DataFrame({"A": [1], "B": [2]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
)
def test_valid(
    base: Union[KeySet, KeySetPlan],
    columns: Union[str, Sequence[str]],
    expected_df: pd.DataFrame,
    expected_schema: dict[str, ColumnDescriptor],
):
    """Valid parameters work as expected."""
    ks = base[columns]
    assert isinstance(ks, KeySet)
    assert ks.columns() == list(expected_schema.keys())
    assert ks.schema() == expected_schema
    if ks.columns():
        assert ks.size() == len(expected_df)
    else:
        assert ks.size() == 1
    assert_dataframe_equal(ks.dataframe(), expected_df)


# pylint: disable=protected-access
@parametrize(
    Case("one_column_str")(
        base=KeySet._detect(["A", "B"]),
        columns="A",
    ),
    Case("one_column")(
        base=KeySet._detect(["A", "B"]),
        columns=["A"],
    ),
    Case("two_column")(
        base=KeySet._detect(["A", "B", "C"]),
        columns=["A", "B"],
    ),
    Case("all_columns")(
        base=KeySet._detect(["A", "B"]),
        columns=["A", "B"],
    ),
    Case("crossjoin_child")(
        base=KeySet.from_tuples([(1, 2, 3)], ["A", "B", "C"])
        * KeySet._detect(["D", "E", "F"]),
        columns=["A", "C", "E"],
    ),
    Case("crossjoin_child_reordered")(
        base=KeySet.from_tuples([(1, 2, 3)], ["A", "B", "C"])
        * KeySet._detect(["D", "E", "F"]),
        columns=["C", "A", "E"],
    ),
)
# pylint: enable=protected-access
def test_valid_plan(
    base: KeySetPlan,
    columns: Union[str, Sequence[str]],
):
    """Valid parameters including a KeySetPlan work as expected."""
    ks = base[columns]
    if isinstance(columns, str):
        assert ks.columns() == [columns]
    else:
        assert ks.columns() == list(columns)


@parametrize(
    Case("missing_column")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        columns=["D"],
        expectation=pytest.raises(
            ValueError,
            match="Column D is not present in KeySet",
        ),
    ),
    Case("missing_columns")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        columns=["D", "E"],
        expectation=pytest.raises(
            ValueError,
            match="Columns D, E are not present in KeySet",
        ),
    ),
    Case("partial_missing_columns")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        columns=["C", "D", "E"],
        expectation=pytest.raises(
            ValueError,
            match="Columns D, E are not present in KeySet",
        ),
    ),
    Case("empty_project_columns")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        columns=[],
        expectation=pytest.raises(
            ValueError,
            match="At least one column must be kept when subscripting a KeySet.",
        ),
    ),
    Case("invalid_column_name")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        columns=["A", 1],
        expectation=pytest.raises(ValueError, match="Column names must be strings"),
    ),
    Case("empty_column_name")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        columns=["A", ""],
        expectation=pytest.raises(
            ValueError, match="Empty column names are not allowed"
        ),
    ),
)
def test_invalid(
    base: Union[KeySet, KeySetPlan], columns: Any, expectation: ContextManager[None]
):
    """Invalid tuples/columns values are rejected."""
    with expectation:
        _ = base[columns]
