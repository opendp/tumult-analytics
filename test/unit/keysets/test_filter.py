"""Unit tests for (v2) KeySet filtering."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024
from typing import Any, ContextManager, Sequence, Union

import pandas as pd
import pyspark.sql.functions as sf
import pytest
from pyspark.sql import Column
from pyspark.sql.utils import AnalysisException
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet, KeySetPlan
from tmlt.analytics._schema import ColumnDescriptor, ColumnType


@parametrize(
    Case("one_col_str")(
        base=KeySet.from_dict({"A": [1, 2, 3, 4]}),
        condition="A > 2",
        expected_df=pd.DataFrame({"A": [3, 4]}),
        expected_schema={"A": ColumnDescriptor(ColumnType.INTEGER)},
    ),
    Case("one_col_column")(
        base=KeySet.from_dict({"A": [1, 2, 3, 4]}),
        condition=sf.col("A") > 2,
        expected_df=pd.DataFrame({"A": [3, 4]}),
        expected_schema={"A": ColumnDescriptor(ColumnType.INTEGER)},
    ),
    Case("one_of_two_columns_str")(
        base=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        condition="B == 'b'",
        expected_df=pd.DataFrame([[1, "b"], [2, "b"]], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("one_of_two_columns_column")(
        base=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        condition=sf.col("B") == "b",
        expected_df=pd.DataFrame([[1, "b"], [2, "b"]], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("two_of_two_columns_str")(
        base=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        condition="B == 'b' and A < 2",
        expected_df=pd.DataFrame([[1, "b"]], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("two_of_two_columns_column")(
        base=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        condition=(sf.col("B") == "b") & (sf.col("A") < 2),
        expected_df=pd.DataFrame([[1, "b"]], columns=["A", "B"]),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("no_filtering_str")(
        base=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        condition="B != 'string_that_does_not_exist' and A > 0",
        expected_df=pd.DataFrame(
            [[1, "a"], [2, "a"], [1, "b"], [2, "b"]], columns=["A", "B"]
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
    Case("no_filtering_column")(
        base=KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]}),
        condition=(sf.col("B") != "string_that_does_not_exist") & (sf.col("A") > 0),
        expected_df=pd.DataFrame(
            [[1, "a"], [2, "a"], [1, "b"], [2, "b"]], columns=["A", "B"]
        ),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.VARCHAR),
        },
    ),
)
def test_valid(
    base: KeySet,
    condition: Union[str, Column],
    expected_df: pd.DataFrame,
    expected_schema: dict[str, ColumnDescriptor],
):
    """Valid parameters work as expected."""
    ks = base.filter(condition)
    assert ks.columns() == list(expected_schema.keys())
    assert ks.schema() == expected_schema
    if ks.columns():
        assert ks.size() == len(expected_df)
    else:
        assert ks.size() == 1
    assert_dataframe_equal(ks.dataframe(), expected_df)


# pylint: disable=protected-access
@parametrize(
    Case("one_col_str")(
        base=KeySet._detect(["A"]),
        condition="A > 2",
        expected_columns=["A"],
    ),
    Case("one_col_column")(
        base=KeySet._detect(["A"]),
        condition=sf.col("A") > 2,
        expected_columns=["A"],
    ),
    Case("one_of_two_columns_str")(
        base=KeySet._detect(["A", "B"]),
        condition="B == 'b'",
        expected_columns=["A", "B"],
    ),
    Case("one_of_two_columns_column")(
        base=KeySet._detect(["A", "B"]),
        condition=sf.col("B") == "b",
        expected_columns=["A", "B"],
    ),
    Case("two_of_two_columns_str")(
        base=KeySet._detect(["A", "B"]),
        condition="B == 'b' and A < 2",
        expected_columns=["A", "B"],
    ),
    Case("two_of_two_columns_column")(
        base=KeySet._detect(["A", "B"]),
        condition=(sf.col("B") == "b") & (sf.col("A") < 2),
        expected_columns=["A", "B"],
    ),
)
# pylint: enable=protected-access
def test_valid_plan(
    base: KeySetPlan,
    condition: Union[str, Column],
    expected_columns: Sequence[str],
):
    """Valid parameters including a KeySetPlan work as expected."""
    ks = base.filter(condition)
    assert ks.columns() == expected_columns


@parametrize(
    Case("empty_filter")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        condition="",
        expectation=pytest.raises(
            ValueError,
            match="empty condition",
        ),
    ),
    Case("bad_condition_str")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        condition="D == 1",
        expectation=pytest.raises(AnalysisException),
    ),
    Case("bad_condition_column")(
        base=KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"]),
        condition=sf.col("D") == 1,
        expectation=pytest.raises(AnalysisException),
    ),
)
def test_invalid(base: KeySet, condition: Any, expectation: ContextManager[None]):
    """Invalid tuples/columns values are rejected."""
    with expectation:
        result = base.filter(condition)
        _ = result.dataframe()


# TODO(3384): Add tests for invalid plans once get_keys() is implemented.
