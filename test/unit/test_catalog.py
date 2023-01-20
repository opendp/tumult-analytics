"""Unit tests for catalog."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023


from typing import Optional

import pytest

from tmlt.analytics._catalog import Catalog, PrivateTable
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema


@pytest.mark.parametrize("grouping_column", ["A", (None)])
def test_add_private_table(grouping_column: Optional["str"]):
    """Adding a private table works as expected."""
    catalog = Catalog()
    catalog.add_private_table(
        source_id="private",
        col_types={"A": ColumnDescriptor(ColumnType.VARCHAR)},
        grouping_column=grouping_column,
    )
    assert len(catalog.tables) == 1
    private_table = catalog.tables["private"]
    assert isinstance(private_table, PrivateTable)
    assert private_table.source_id == "private"
    actual_schema = private_table.schema
    expected_schema = Schema({"A": ColumnType.VARCHAR}, grouping_column=grouping_column)
    assert actual_schema == expected_schema


def test_add_public_table():
    """Adding a public table works as expected."""
    catalog = Catalog()
    catalog.add_private_table(source_id="public", col_types={"A": ColumnType.VARCHAR})
    assert len(catalog.tables) == 1
    assert list(catalog.tables)[0] == "public"
    assert catalog.tables["public"].source_id == "public"
    actual_schema = catalog.tables["public"].schema
    expected_schema = Schema({"A": ColumnType.VARCHAR})
    assert actual_schema == expected_schema


def test_invalid_addition_private_table():
    """Adding a private table that already exists fails."""
    catalog = Catalog()
    source_id = "private"
    catalog.add_private_table(source_id=source_id, col_types={"A": ColumnType.VARCHAR})
    with pytest.raises(ValueError, match=f"{source_id} already exists in catalog."):
        catalog.add_private_table(
            source_id=source_id, col_types={"B": ColumnType.VARCHAR}
        )


def test_invalid_addition_public_table():
    """Adding a public table that already exists fails."""
    catalog = Catalog()
    source_id = "public"
    catalog.add_public_table(source_id, {"A": ColumnType.VARCHAR})
    with pytest.raises(ValueError, match=f"{source_id} already exists in catalog."):
        catalog.add_public_table(source_id, {"C": ColumnType.VARCHAR})
