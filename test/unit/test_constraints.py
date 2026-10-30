"""Tests for Constraints."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023-2025, and the Tumult Analytics Contributors 2025-present

from typing import Iterable, List

import pytest
from typeguard import TypeCheckError

from tmlt.analytics import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.constraints import simplify_constraints


def test_max_rows_per_id():
    """Test initialization of MaxRowsPerID constraints."""
    assert MaxRowsPerID(1).max == 1
    assert MaxRowsPerID(5).max == 5
    with pytest.raises(ValueError):
        MaxRowsPerID(0)
    with pytest.raises(ValueError):
        MaxRowsPerID(-5)


def test_max_groups_per_id():
    """Test initialization of MaxGroupsPerID constraints."""
    assert MaxGroupsPerID("grouping_column", 1).max == 1
    assert MaxGroupsPerID("grouping_column", 5).max == 5
    assert MaxGroupsPerID("grouping_column", 1).grouping_columns == frozenset(
        {"grouping_column"}
    )
    with pytest.raises(ValueError):
        MaxGroupsPerID("", 1)
    with pytest.raises(ValueError):
        MaxGroupsPerID("grouping_column", 0)
    with pytest.raises(ValueError):
        MaxGroupsPerID("grouping_column", -5)


def test_max_rows_per_group_per_id():
    """Test initialization of MaxRowsPerGroupPerID constraints."""
    assert MaxRowsPerGroupPerID("group_col", 1).max == 1
    assert MaxRowsPerGroupPerID("group_col", 5).max == 5
    with pytest.raises(ValueError):
        MaxRowsPerGroupPerID("group_col", 0)
    with pytest.raises(ValueError):
        MaxRowsPerGroupPerID("group_col", -5)
    with pytest.raises(TypeCheckError):
        MaxRowsPerGroupPerID(5, 10)  # type: ignore


@pytest.mark.parametrize("constraint_type", [MaxGroupsPerID, MaxRowsPerGroupPerID])
def test_multiple_grouping_columns(constraint_type):
    """Test initialization of constraints with multiple grouping columns."""
    c = constraint_type(["a", "b"], 2)
    assert c.max == 2
    assert c.grouping_columns == frozenset({"a", "b"})
    assert c == constraint_type(("a", "b"), 2)
    assert c == constraint_type(frozenset({"a", "b"}), 2)
    # The order in which the columns are given does not matter.
    assert c == constraint_type(["b", "a"], 2)
    assert hash(c) == hash(constraint_type(["b", "a"], 2))
    assert repr(constraint_type(["b", "a"], 2)) == (
        f"{constraint_type.__name__}(grouping_columns={{'a', 'b'}}, max=2)"
    )


@pytest.mark.parametrize("constraint_type", [MaxGroupsPerID, MaxRowsPerGroupPerID])
def test_single_grouping_column(constraint_type):
    """A single grouping column is stored as a one-element frozenset."""
    c = constraint_type("a", 2)
    assert c.grouping_columns == frozenset({"a"})
    assert c == constraint_type(["a"], 2)
    assert c == constraint_type(grouping_columns="a", max=2)
    assert repr(c) == f"{constraint_type.__name__}(grouping_columns={{'a'}}, max=2)"


@pytest.mark.parametrize("constraint_type", [MaxGroupsPerID, MaxRowsPerGroupPerID])
def test_deprecated_grouping_column(constraint_type):
    """The deprecated grouping_column property returns a single column name."""
    with pytest.warns(DeprecationWarning, match="grouping_column is deprecated"):
        assert constraint_type("a", 2).grouping_column == "a"
    with pytest.warns(DeprecationWarning, match="grouping_column is deprecated"):
        assert constraint_type(["a"], 2).grouping_column == "a"
    with pytest.warns(DeprecationWarning, match="grouping_column is deprecated"):
        with pytest.raises(ValueError, match="cannot be used with multiple grouping"):
            _ = constraint_type(["a", "b"], 2).grouping_column


@pytest.mark.parametrize("constraint_type", [MaxGroupsPerID, MaxRowsPerGroupPerID])
def test_keyword_arguments(constraint_type):
    """Constraints can be constructed with keyword arguments."""
    expected = constraint_type(["a", "b"], 2)
    assert constraint_type(grouping_columns=["a", "b"], max=2) == expected
    assert constraint_type(max=2, grouping_columns=["a", "b"]) == expected
    assert constraint_type(["a", "b"], max=2) == expected


@pytest.mark.parametrize("constraint_type", [MaxGroupsPerID, MaxRowsPerGroupPerID])
def test_deprecated_grouping_column_argument(constraint_type):
    """The deprecated grouping_column argument is an alias of grouping_columns."""
    with pytest.warns(DeprecationWarning, match="grouping_column argument"):
        c = constraint_type(grouping_column="a", max=2)
    assert c == constraint_type("a", 2)
    # Like the grouping_column property, the argument only supports one column.
    for grouping_column in (["a", "b"], ["a"]):
        with pytest.warns(DeprecationWarning, match="grouping_column argument"):
            with pytest.raises(TypeError, match="must be a single column name"):
                constraint_type(grouping_column=grouping_column, max=2)


@pytest.mark.parametrize("constraint_type", [MaxGroupsPerID, MaxRowsPerGroupPerID])
@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"max": 2}, "missing required argument grouping_columns"),
        ({"grouping_columns": "a"}, "missing required argument max"),
        ({}, "missing required argument"),
        (
            {"grouping_columns": "a", "grouping_column": "a", "max": 2},
            "got both grouping_columns and grouping_column",
        ),
    ],
)
def test_invalid_arguments(constraint_type, kwargs, message):
    """Missing or conflicting constructor arguments raise an error."""
    with pytest.raises(TypeError, match=message):
        constraint_type(**kwargs)


@pytest.mark.parametrize("constraint_type", [MaxGroupsPerID, MaxRowsPerGroupPerID])
@pytest.mark.parametrize(
    "grouping_columns,error_type,message",
    [
        ([], ValueError, "grouping_columns cannot be empty"),
        (["a", ""], ValueError, "grouping column names cannot be empty strings"),
        (["a", "a"], ValueError, "grouping_columns cannot contain duplicate columns"),
        (["a", 5], TypeCheckError, None),
        ([["a"], ["b"]], TypeCheckError, None),
    ],
)
def test_invalid_grouping_columns(
    constraint_type, grouping_columns, error_type, message
):
    """Invalid lists of grouping columns raise an error."""
    with pytest.raises(error_type, match=message):
        constraint_type(grouping_columns, 1)


@pytest.mark.parametrize(
    "constraints,expected_constraints",
    [
        ([], []),
        ([MaxRowsPerID(1)], [MaxRowsPerID(1)]),
        ([MaxRowsPerID(1), MaxRowsPerID(1)], [MaxRowsPerID(1)]),
        ([MaxRowsPerID(1), MaxRowsPerID(5)], [MaxRowsPerID(1)]),
        ([MaxRowsPerID(3), MaxRowsPerID(2), MaxRowsPerID(6)], [MaxRowsPerID(2)]),
        (
            [
                MaxGroupsPerID("grouping_column", 1),
                MaxGroupsPerID("grouping_column", 5),
            ],
            [MaxGroupsPerID("grouping_column", 1)],
        ),
        (
            [
                MaxGroupsPerID("grouping_column", 1),
                MaxGroupsPerID("other_grouping_column", 5),
                MaxGroupsPerID("grouping_column", 3),
            ],
            [
                MaxGroupsPerID("grouping_column", 1),
                MaxGroupsPerID("other_grouping_column", 5),
            ],
        ),
        (
            [MaxRowsPerID(1), MaxGroupsPerID("grouping_column", 1)],
            [MaxRowsPerID(1), MaxGroupsPerID("grouping_column", 1)],
        ),
        (
            [MaxRowsPerID(1), MaxGroupsPerID("grouping_column", 1), MaxRowsPerID(5)],
            [MaxRowsPerID(1), MaxGroupsPerID("grouping_column", 1)],
        ),
        (
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("grouping_column", 1),
                MaxGroupsPerID("grouping_column", 5),
                MaxGroupsPerID("other_grouping_column", 1),
            ],
            [
                MaxRowsPerID(1),
                MaxGroupsPerID("grouping_column", 1),
                MaxGroupsPerID("other_grouping_column", 1),
            ],
        ),
        (
            [MaxRowsPerGroupPerID("group_col", 1)],
            [MaxRowsPerGroupPerID("group_col", 1)],
        ),
        (
            [
                MaxRowsPerGroupPerID("group_col", 1),
                MaxRowsPerGroupPerID("group_col", 1),
            ],
            [MaxRowsPerGroupPerID("group_col", 1)],
        ),
        (
            [
                MaxRowsPerGroupPerID("group_col", 3),
                MaxRowsPerGroupPerID("group_col", 6),
            ],
            [MaxRowsPerGroupPerID("group_col", 3)],
        ),
        (
            [
                MaxRowsPerGroupPerID("group_col1", 1),
                MaxRowsPerGroupPerID("group_col2", 1),
                MaxRowsPerGroupPerID("group_col2", 5),
            ],
            [
                MaxRowsPerGroupPerID("group_col1", 1),
                MaxRowsPerGroupPerID("group_col2", 1),
            ],
        ),
        (
            [
                MaxRowsPerGroupPerID("group_col1", 1),
                MaxGroupsPerID("group_col1", 1),
                MaxRowsPerID(1),
            ],
            [
                MaxRowsPerGroupPerID("group_col1", 1),
                MaxGroupsPerID("group_col1", 1),
                MaxRowsPerID(1),
            ],
        ),
        (
            [
                MaxGroupsPerID(["a", "b"], 3),
                MaxGroupsPerID(["b", "a"], 2),
                MaxGroupsPerID("a", 1),
                MaxGroupsPerID(["a", "b", "c"], 4),
            ],
            [
                MaxGroupsPerID(["b", "a"], 2),
                MaxGroupsPerID("a", 1),
                MaxGroupsPerID(["a", "b", "c"], 4),
            ],
        ),
        (
            [
                MaxRowsPerGroupPerID(["a", "b"], 1),
                MaxRowsPerGroupPerID(["b", "a"], 2),
                MaxRowsPerGroupPerID(["a"], 3),
                MaxRowsPerGroupPerID("a", 5),
            ],
            [
                MaxRowsPerGroupPerID(["a", "b"], 1),
                MaxRowsPerGroupPerID("a", 3),
            ],
        ),
        (
            [
                MaxRowsPerID(1),
                MaxRowsPerID(2),
                MaxRowsPerGroupPerID("group_col1", 1),
                MaxRowsPerGroupPerID("group_col2", 2),
                MaxGroupsPerID("group_col1", 1),
                MaxGroupsPerID("group_col1", 5),
            ],
            [
                MaxRowsPerID(1),
                MaxRowsPerGroupPerID("group_col1", 1),
                MaxRowsPerGroupPerID("group_col2", 2),
                MaxGroupsPerID("group_col1", 1),
            ],
        ),
    ],
)
def test_simplify_constraints(
    constraints: Iterable[Constraint], expected_constraints: List[Constraint]
):
    """Test simplification of constraints."""
    assert simplify_constraints(constraints) == frozenset(expected_constraints)
