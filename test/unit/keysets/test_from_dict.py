"""Unit tests for (v2) KeySet.from_dict."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024


import datetime
from typing import Any, ContextManager, Iterable, Mapping, Optional, Union

import pandas as pd
import pytest
from pyspark.sql.types import DateType, LongType, StringType, StructField, StructType
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet
from tmlt.analytics._schema import ColumnDescriptor, ColumnType


@parametrize(
    Case("one_column")(
        domains={"A": [1, 2]},
        expected_df=pd.DataFrame({"A": [1, 2]}),
        expected_schema={"A": ColumnDescriptor(ColumnType.INTEGER)},
    ),
    Case("two_columns")(
        domains={"A": [1, 2], "B": [3, 4]},
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("mixed_types")(
        domains={"int": [1], "str": ["a"], "date": [datetime.date.fromordinal(1)]},
        expected_df=pd.DataFrame(
            {"int": [1], "str": ["a"], "date": [datetime.date.fromordinal(1)]}
        ),
        expected_schema={
            "int": ColumnDescriptor(ColumnType.INTEGER),
            "str": ColumnDescriptor(ColumnType.VARCHAR),
            "date": ColumnDescriptor(ColumnType.DATE),
        },
    ),
    Case("nulls")(
        domains={
            "int": [1, None],
            "str": [None, "a"],
            "date": [datetime.date.fromordinal(1), None],
        },
        expected_df=pd.DataFrame({"int": [1, None]})
        .merge(pd.DataFrame({"str": [None, "a"]}), how="cross")
        .merge(
            pd.DataFrame({"date": [datetime.date.fromordinal(1), None]}), how="cross"
        ),
        expected_schema={
            "int": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "str": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "date": ColumnDescriptor(ColumnType.DATE, allow_null=True),
        },
    ),
    Case("duplicate_values")(
        domains={"A": [1, 2, 2, 1], "B": [3, 4]},
        expected_df=pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]}),
        expected_schema={
            "A": ColumnDescriptor(ColumnType.INTEGER),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    ),
    Case("total")(
        domains={},
        expected_df=pd.DataFrame(),
        expected_schema={},
    ),
    [
        Case(f"{type(v).__name__}_domain")(
            domains={"A": v},
            expected_df=pd.DataFrame({"A": [1, 2]}),
            expected_schema={"A": ColumnDescriptor(ColumnType.INTEGER)},
        )
        for v in [
            range(1, 3),
            (1, 2),
            {1, 2},
            (i for i in [1, 2]),
            map(int, ["1", "2"]),
        ]
    ],
)
def test_valid(
    domains: Mapping[
        str,
        Union[
            Iterable[Optional[str]],
            Iterable[Optional[int]],
            Iterable[Optional[datetime.date]],
        ],
    ],
    expected_df: pd.DataFrame,
    expected_schema: dict[str, ColumnDescriptor],
):
    """Valid parameters work as expected."""
    ks = KeySet.from_dict(domains)
    assert ks.columns() == list(domains.keys())
    assert_dataframe_equal(ks.dataframe(), expected_df)
    assert ks.schema() == expected_schema


@parametrize(Case("nullable")(nullable=True), Case("nonnullable")(nullable=False))
def test_dataframe_schema(nullable: bool):
    """KeySet dataframes have the expected schema."""
    null = [None] if nullable else []
    ks = KeySet.from_dict(
        {
            "int": [1, 2, *null],
            "str": ["a", "b", *null],
            "date": [datetime.date.today(), *null],
        }
    )
    expected_schema = StructType(
        [
            StructField("int", LongType(), nullable=nullable),
            StructField("str", StringType(), nullable=nullable),
            StructField("date", DateType(), nullable=nullable),
        ]
    )
    assert ks.dataframe().schema == expected_schema


@parametrize(
    Case("2^4")(factors=4, factor_size=2),
    Case("64^3")(factors=3, factor_size=64),
    Case("3^20", marks=pytest.mark.slow)(factors=20, factor_size=3),
    Case("63^5", marks=pytest.mark.slow)(factors=5, factor_size=63),
    Case("128^4", marks=pytest.mark.slow)(factors=4, factor_size=128),
)
def test_large(factors: int, factor_size: int):
    """Operations with large output KeySets work as expected."""
    ks = KeySet.from_dict({str(f): range(factor_size) for f in range(factors)})
    assert ks.dataframe().count() == factor_size**factors


@parametrize(
    Case("float_column")(
        domains={"A": [3.5]},
        expectation=pytest.raises(
            ValueError, match="Column 'A' has type DECIMAL, but only allowed types"
        ),
    ),
    Case("datetime_column")(
        domains={"A": [datetime.datetime.now()]},
        expectation=pytest.raises(
            ValueError, match="Column 'A' has type TIMESTAMP, but only allowed types"
        ),
    ),
    Case("empty_column")(
        domains={"A": []},
        expectation=pytest.raises(
            ValueError, match="Unable to infer column types for an empty collection"
        ),
    ),
    Case("all_null_column")(
        domains={"A": [None, None]},
        expectation=pytest.raises(
            ValueError, match="Column 'A' contains only null values"
        ),
    ),
    Case("empty_column_name")(
        domains={"": [1]},
        expectation=pytest.raises(
            ValueError, match="Empty column names are not allowed"
        ),
    ),
    Case("non_string_column_name")(
        domains={1: [1]},
        expectation=pytest.raises(ValueError, match="Column names must be strings"),
    ),
)
def test_invalid(domains: Any, expectation: ContextManager[None]):
    """Invalid domains are rejected."""
    with expectation:
        KeySet.from_dict(domains)
