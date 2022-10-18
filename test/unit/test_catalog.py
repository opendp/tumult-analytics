"""Unit tests for catalog."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022


from typing import Optional

import pytest

from tmlt.analytics._catalog import Catalog, PrivateTable
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema


@pytest.mark.parametrize("grouping_column", ["A", (None)])
def test_add_private_source(grouping_column: Optional["str"]):
    """Add private source."""
    catalog = Catalog()
    catalog.add_private_source(
        source_id="private",
        col_types={"A": ColumnDescriptor(ColumnType.VARCHAR)},
        stability=3,
        grouping_column=grouping_column,
    )
    assert len(catalog.tables) == 1
    private_table = catalog.tables["private"]
    assert isinstance(private_table, PrivateTable)
    assert private_table is catalog.private_table
    assert private_table.source_id == "private"
    actual_schema = private_table.schema
    expected_schema = Schema({"A": "VARCHAR"}, grouping_column=grouping_column)
    assert actual_schema == expected_schema
    assert private_table.stability == 3


def test_add_public_source():
    """Add public source."""
    catalog = Catalog()
    catalog.add_private_source(
        source_id="public", col_types={"A": "VARCHAR"}, stability=1
    )
    assert len(catalog.tables) == 1
    assert list(catalog.tables)[0] == "public"
    assert catalog.tables["public"].source_id == "public"
    actual_schema = catalog.tables["public"].schema
    expected_schema = Schema({"A": "VARCHAR"})
    assert actual_schema == expected_schema


def test_add_private_view():
    """Add private view."""
    catalog = Catalog()
    catalog.add_private_view(
        source_id="private_view", col_types={"A": "VARCHAR"}, stability=1
    )
    assert len(catalog.tables) == 1
    assert list(catalog.tables)[0] == "private_view"
    assert catalog.tables["private_view"].source_id == "private_view"
    actual_schema = catalog.tables["private_view"].schema
    expected_schema = Schema({"A": "VARCHAR"})
    assert actual_schema == expected_schema
    assert catalog.tables["private_view"].stability == 1


def test_private_source_already_exists():
    """Add invalid private source"""
    catalog = Catalog()
    source_id = "private"
    catalog.add_private_source(
        source_id=source_id, col_types=({"A": "VARCHAR"}), stability=1
    )
    with pytest.raises(RuntimeError, match="Cannot have more than one private source"):
        catalog.add_private_source(
            source_id=source_id, col_types={"B": "VARCHAR"}, stability=1
        )


def test_invalid_addition_private_view():
    """Add invalid private view"""
    catalog = Catalog()
    source_id = "private"
    catalog.add_private_source(
        source_id=source_id, col_types={"A": "VARCHAR"}, stability=1
    )
    with pytest.raises(ValueError, match=f"{source_id} already exists in catalog."):
        catalog.add_private_view(
            source_id=source_id, col_types={"B": "VARCHAR"}, stability=1
        )


def test_invalid_addition_public_source():
    """Add invalid public source"""
    catalog = Catalog()
    source_id = "public"
    catalog.add_public_source(source_id, {"A": "VARCHAR"})
    with pytest.raises(ValueError, match=f"{source_id} already exists in catalog."):
        catalog.add_public_source(source_id, {"C": "VARCHAR"})
