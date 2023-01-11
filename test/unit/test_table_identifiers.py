"""Tests for table identifier types."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import pytest

from tmlt.analytics._table_identifier import NamedTable, TableCollection, TemporaryTable
from tmlt.analytics._table_reference import TableReference


def test_table_equality():
    """Equality for table IDs works as expected."""
    assert NamedTable(name="private1") == NamedTable(name="private1")
    assert NamedTable(name="private1") != NamedTable(name="private2")

    assert TableCollection(name="private1") == TableCollection(name="private1")
    assert TableCollection(name="private1") != TableCollection(name="private2")

    temp_table = TemporaryTable()
    assert temp_table == temp_table  # pylint: disable=comparison-with-itself
    assert temp_table != TemporaryTable()


def test_table_reference():
    """TableReference behaves as expected."""
    test_path = [TemporaryTable(), TemporaryTable(), NamedTable(name="private")]
    reference = TableReference(path=test_path)

    assert reference.parent == TableReference(test_path[:-1])
    assert reference.identifier == test_path[-1]

    new_table = NamedTable(name="private2")
    old_ref = reference
    reference = reference / new_table

    assert reference.identifier == new_table
    assert reference.parent == old_ref


def test_table_reference_empty_errors():
    """Appropriate exceptions are raised when deconstructing empty references."""
    with pytest.raises(IndexError):
        _ = TableReference(path=[]).parent
    with pytest.raises(IndexError):
        _ = TableReference(path=[]).identifier
