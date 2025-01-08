"""Tests for KeySetOp tree rewriting operations.

This logic is to some degree tested by the other KeySet tests, but these tests
explicitly cover that rewrite rules don't change the output dataframes, and aim
to hit known-tricky pieces of the rewriting logic.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Callable
from unittest.mock import patch

from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet


@parametrize(
    Case("crossjoin_reorder")(
        ks=lambda: KeySet.from_dict({"A": [1], "C": [2], "B": [3]})
    ),
    Case("crossjoin_linearize")(
        ks=lambda: (
            (KeySet.from_dict({"A": [1]}) * KeySet.from_dict({"C": [1]}))
            * (KeySet.from_dict({"D": [1]}) * KeySet.from_dict({"B": [1]}))
        )
    ),
    Case("nested_project")(
        ks=lambda: KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"])[
            "A", "B", "C"
        ]["A", "B"]["A"]
    ),
)
def test_rewrite_equality(ks: Callable[[], KeySet]):
    """Rewritten KeySets have the same semantics as the original ones."""
    ks_rewritten = ks()
    with patch("tmlt.analytics._keyset_v2._keyset.rewrite", lambda op: op):
        ks_original = ks()

    # Ensure that rewriting actually happened
    # pylint: disable-next=protected-access
    assert ks_rewritten._op_tree != ks_original._op_tree

    assert ks_rewritten.columns() == ks_original.columns()
    assert ks_rewritten.schema() == ks_original.schema()
    assert_dataframe_equal(ks_rewritten.dataframe(), ks_original.dataframe())
