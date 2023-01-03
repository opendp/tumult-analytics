"""Unit tests for schema."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
# pylint: disable=pointless-string-statement
import re

import pytest

from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema

"""Unit tests for Schema."""


def test_invalid_column_type() -> None:
    """Schema raises an exception when an invalid column type is used."""
    with pytest.raises(
        ValueError,
        match=r"Column types \{'BADTYPE'\} not supported; "
        r"use supported types \['[A-Z', ]+'\].",
    ):
        columns = {"Col1": "VARCHAR", "Col2": "BADTYPE", "Col3": "INTEGER"}
        Schema(columns)


def test_invalid_column_name() -> None:
    """Schema raises an exception if a column is named "" (empty string)."""
    with pytest.raises(
        ValueError,
        match=re.escape('"" (the empty string) is not a supported column name'),
    ):
        Schema({"col1": "VARCHAR", "": "VARCHAR"})


def test_valid_column_types() -> None:
    """Schema construction and py type translation succeeds with valid columns."""
    columns = {
        "1": "INTEGER",
        "2": "DECIMAL",
        "3": "VARCHAR",
        "4": "DATE",
        "5": "TIMESTAMP",
    }
    schema = Schema(columns)
    expected = {
        "1": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
        "2": ColumnDescriptor(ColumnType.DECIMAL, allow_null=False),
        "3": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
        "4": ColumnDescriptor(ColumnType.DATE, allow_null=False),
        "5": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=False),
    }
    assert expected == schema.column_descs


def test_schema_equality() -> None:
    """Make sure schema equality check works properly."""
    columns_1 = {"a": "VARCHAR", "b": "INTEGER"}
    columns_2 = {"a": "VARCHAR", "b": "INTEGER"}
    columns_3 = {"y": "VARCHAR", "z": "INTEGER"}
    columns_4 = {"a": "INTEGER", "b": "VARCHAR"}
    schema_1 = Schema(columns_1)
    schema_2 = Schema(columns_2)
    schema_3 = Schema(columns_3)
    schema_4 = Schema(columns_4)
    assert schema_1 == schema_2
    assert schema_1 != schema_3
    assert schema_1 != schema_4
