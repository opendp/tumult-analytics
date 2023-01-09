"""Tests for table identifier types."""

import pytest

from tmlt.analytics._table_identifiers import (
    NamedTable,
    TableCollection,
    TemporaryTable,
)
from tmlt.analytics._table_reference import TableReference


def test_table_equality():
    """Test that equality for table IDs works as expected."""
    # different NamedTables with the same name field are considered equal
    named_table = NamedTable(name="private1")
    other_named_table = NamedTable(name="private1")

    assert named_table == other_named_table

    # different TableCollections with the same name field are also considered equal

    table_collection = TableCollection(name="private1")
    other_table_collection = TableCollection(name="private1")

    assert table_collection == other_table_collection

    # different instances of TemporaryTables are not considered equal.

    temp_table = TemporaryTable()
    other_temp_table = TemporaryTable()

    assert temp_table != other_temp_table


def test_table_reference():
    """Test that TableReference behaves as expected."""
    test_path = [TemporaryTable(), TemporaryTable(), NamedTable(name="private")]

    reference = TableReference(path=test_path)

    assert reference.parent == TableReference(test_path[:-1])
    assert reference.identifier == test_path[-1]

    new_table = NamedTable(name="private2")

    old_ref = reference

    reference = reference / new_table

    assert reference.identifier == new_table
    assert reference.parent == old_ref
    # pylint: disable=unused-variable
    with pytest.raises(IndexError):
        _ = TableReference(path=[]).parent

    with pytest.raises(IndexError):
        _ = TableReference(path=[]).identifier
    # pylint: enable=unused-variable
