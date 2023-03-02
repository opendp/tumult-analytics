"""Tests for Constraints."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import List

import pytest

from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
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
    ],
)
def test_simplify_constraints(
    constraints: List[Constraint], expected_constraints: List[Constraint]
):
    """Test simplification of constraints."""
    assert simplify_constraints(constraints) == expected_constraints
