"""Tests for Constraints."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import List

import pytest

from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    simplify_constraints,
)


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
    assert MaxGroupsPerID("grouping_column", 1).grouping_column == "grouping_column"
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
    with pytest.raises(TypeError):
        MaxRowsPerGroupPerID(5, 10)  # type: ignore


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
    constraints: List[Constraint], expected_constraints: List[Constraint]
):
    """Test simplification of constraints."""
    assert set(simplify_constraints(constraints)) == set(expected_constraints)
