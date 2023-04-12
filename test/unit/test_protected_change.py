"""Tests for :mod:`tmlt.analytics.protected_change`."""

# Copyright Tumult Labs 2023
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, List

import pytest

from tmlt.analytics.protected_change import (
    AddMaxRows,
    AddMaxRowsInMaxGroups,
    AddOneRow,
    AddRowsWithID,
)


def test_add_one_row():
    """For AddOneRow, max_rows = 1."""
    assert isinstance(AddOneRow(), AddMaxRows)
    assert AddOneRow().max_rows == 1


@pytest.mark.parametrize(
    "args,expectation",
    [
        ([1], does_not_raise()),
        ([5], does_not_raise()),
        ([0], pytest.raises(ValueError, match="^max_rows must be positive$")),
        ([-1], pytest.raises(ValueError, match="^max_rows must be positive$")),
        (
            ["a"],
            pytest.raises(
                TypeError, match="^type of max_rows must be int; got str instead$"
            ),
        ),
    ],
)
def test_add_max_rows(args: List[Any], expectation: ContextManager[None]):
    """Constructing AddMaxRows works as anticipated."""
    with expectation:
        AddMaxRows(*args)


@pytest.mark.parametrize(
    "args,expectation",
    [
        (["x", 1, 1], does_not_raise()),
        (["y", 10, 2], does_not_raise()),
        (["x", 0, 1], pytest.raises(ValueError, match="^max_groups must be positive$")),
        (
            ["x", 1, 0],
            pytest.raises(ValueError, match="^max_rows_per_group must be positive$"),
        ),
        (
            [1, 1, 1],
            pytest.raises(
                TypeError, match="^type of column must be str; got int instead$"
            ),
        ),
        (
            ["x", "y", 1],
            pytest.raises(
                TypeError, match="^type of max_groups must be int; got str instead$"
            ),
        ),
        (
            ["x", 1, "y"],
            pytest.raises(
                TypeError,
                match=(
                    r"^type of max_rows_per_group must be one of \(int, float\); got"
                    r" str instead$"
                ),
            ),
        ),
    ],
)
def test_add_max_rows_per_group_invalid(
    args: List[Any], expectation: ContextManager[None]
):
    """Invalid inputs are rejected by AddMaxRowsInMaxGroups."""
    with expectation:
        AddMaxRowsInMaxGroups(*args)


@pytest.mark.parametrize(
    "args,expectation",
    [
        (["x"], does_not_raise()),
        (["y"], does_not_raise()),
        (
            [1],
            pytest.raises(
                TypeError, match="^type of id_column must be str; got int instead$"
            ),
        ),
        (["x", "x_space"], does_not_raise()),
        (["x", ""], pytest.raises(ValueError, match="identifier must be non-empty")),
    ],
)
def test_add_rows_with_id_invalid(
    args: List[Any], expectation: ContextManager[None]
) -> None:
    """Invalid arguments are rejected by AddRowsWithID."""
    with expectation:
        AddRowsWithID(*args)
