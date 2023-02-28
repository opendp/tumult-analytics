"""Tests for Constraints."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import List

import pytest

from tmlt.analytics.constraints import Constraint, MaxRowsPerID, simplify_constraints


def test_max_rows_per_id():
    """Test initialization of MaxRowsPerID constraints."""
    assert MaxRowsPerID(1).max == 1
    assert MaxRowsPerID(5).max == 5
    with pytest.raises(ValueError):
        MaxRowsPerID(0)
    with pytest.raises(ValueError):
        MaxRowsPerID(-5)


@pytest.mark.parametrize(
    "constraints,expected_constraints",
    [
        ([], []),
        ([MaxRowsPerID(1)], [MaxRowsPerID(1)]),
        ([MaxRowsPerID(1), MaxRowsPerID(1)], [MaxRowsPerID(1)]),
        ([MaxRowsPerID(1), MaxRowsPerID(5)], [MaxRowsPerID(1)]),
        ([MaxRowsPerID(3), MaxRowsPerID(2), MaxRowsPerID(6)], [MaxRowsPerID(2)]),
    ],
)
def test_simplify_constraints(
    constraints: List[Constraint], expected_constraints: List[Constraint]
):
    """Test simplification of constraints."""
    assert simplify_constraints(constraints) == expected_constraints
