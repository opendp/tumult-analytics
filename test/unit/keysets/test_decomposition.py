"""Tests for KeySet _decompose method."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from functools import reduce
from typing import Collection, Optional, Sequence

from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics._keyset_v2 import KeySet


def _order_keysets(keysets: Collection[KeySet]) -> list[KeySet]:
    return sorted(keysets, key=lambda ks: tuple(sorted(ks.columns())))


def _assert_keyset_sequence_equivalent(
    actual: Sequence[KeySet], expected: Sequence[KeySet]
):
    actual = _order_keysets(actual)
    expected = _order_keysets(expected)
    actual_columns = [sorted(ks.columns()) for ks in actual]
    expected_columns = [sorted(ks.columns()) for ks in expected]
    assert actual_columns == expected_columns
    for a_ks, e_ks in zip(actual, expected):
        # The error message for just asserting that the KeySets are equivalent
        # is totally unhelpful, so generate something more useful.
        if not a_ks.is_equivalent(e_ks):
            assert_dataframe_equal(a_ks.dataframe(), e_ks.dataframe())
            assert False, "KeySets are equal, but could not be shown to be equivalent"


@parametrize(
    Case("from_tuples")(
        ks=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        expected_factors=[KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"])],
        expected_subtracted_values=[],
    ),
    Case("from_tuples_split")(
        ks=KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"]),
        split_columns=["A"],
        expected_factors=[KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"])],
        expected_subtracted_values=[],
    ),
    Case("from_dict")(
        ks=KeySet.from_dict({"A": [1, 2], "B": [3, 4], "C": [5, 6]}),
        expected_factors=[
            KeySet.from_tuples([(1,), (2,)], columns="A"),
            KeySet.from_tuples([(3,), (4,)], columns="B"),
            KeySet.from_tuples([(5,), (6,)], columns="C"),
        ],
        expected_subtracted_values=[],
    ),
    Case("from_dict_split")(
        ks=KeySet.from_dict({"A": [1, 2], "B": [3, 4], "C": [5, 6]}),
        split_columns=["B"],
        expected_factors=[
            KeySet.from_tuples([(1,), (2,)], columns="A"),
            KeySet.from_tuples([(3,), (4,)], columns="B"),
            KeySet.from_tuples([(5,), (6,)], columns="C"),
        ],
        expected_subtracted_values=[],
    ),
    Case("filter")(
        ks=KeySet.from_dict({"A": [1, 2], "B": [3, 4]}).filter("A == 1"),
        expected_factors=[
            KeySet.from_dict({"A": [1, 2], "B": [3, 4]}).filter("A == 1")
        ],
        expected_subtracted_values=[],
    ),
    Case("filter_split")(
        ks=KeySet.from_dict({"A": [1, 2], "B": [3, 4]}).filter("A == 1"),
        split_columns="A",
        expected_factors=[
            KeySet.from_dict({"A": [1, 2], "B": [3, 4]}).filter("A == 1")
        ],
        expected_subtracted_values=[],
    ),
    Case("subtract")(
        ks=KeySet.from_dict({"A": [1, 2, 3], "B": [4, 5], "C": [6]})
        - KeySet.from_tuples([(3, 5)], columns=["B", "C"])
        - KeySet.from_tuples([(1, 4, 6), (2, 5, 6)], columns=["A", "B", "C"]),
        expected_factors=[
            KeySet.from_dict({"A": [1, 2, 3]}),
            KeySet.from_dict({"B": [4, 5]}),
            KeySet.from_dict({"C": [6]}),
        ],
        expected_subtracted_values=[
            KeySet.from_tuples([(3, 5)], columns=["B", "C"]),
            KeySet.from_tuples([(1, 4, 6), (2, 5, 6)], columns=["A", "B", "C"]),
        ],
    ),
    Case("subtract_nested")(
        ks=KeySet.from_dict({"A": [1, 2, 3], "B": [4, 5], "C": [6]})
        - (
            KeySet.from_dict({"A": [1, 2], "B": [4]})
            - KeySet.from_tuples([(1, 4)], columns=["A", "B"])
        ),
        expected_factors=[
            KeySet.from_dict({"A": [1, 2, 3]}),
            KeySet.from_dict({"B": [4, 5]}),
            KeySet.from_dict({"C": [6]}),
        ],
        expected_subtracted_values=[
            KeySet.from_dict({"A": [1, 2], "B": [4]})
            - KeySet.from_tuples([(1, 4)], columns=["A", "B"])
        ],
    ),
    Case("subtract_single_factor")(
        ks=KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"])
        * KeySet.from_tuples([(3, 4), (4, 3)], columns=["C", "D"])
        - KeySet.from_tuples([(1, 1)], columns=["A", "B"]),
        expected_factors=[
            KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"])
            - KeySet.from_tuples([(1, 1)], columns=["A", "B"]),
            KeySet.from_tuples([(3, 4), (4, 3)], columns=["C", "D"]),
        ],
        expected_subtracted_values=[],
    ),
    Case("subtract_in_join")(
        ks=KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]).join(
            KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"])
            - KeySet.from_tuples([(1, 3)], columns=["B", "C"])
        ),
        expected_factors=[
            KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]).join(
                KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"])
                - KeySet.from_tuples([(1, 3)], columns=["B", "C"])
            )
        ],
        expected_subtracted_values=[],
    ),
    Case("subtract_in_join_split")(
        ks=KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]).join(
            KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"])
            - KeySet.from_tuples([(1, 3)], columns=["B", "C"])
        ),
        split_columns=["B"],
        expected_factors=[
            KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]),
            (
                KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"])
                - KeySet.from_tuples([(1, 3)], columns=["B", "C"])
            ),
        ],
        expected_subtracted_values=[],
    ),
    Case("subtract_in_join_split_other")(
        ks=KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]).join(
            KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"])
            - KeySet.from_tuples([(1, 3)], columns=["B", "C"])
        ),
        split_columns=["A"],
        expected_factors=[
            KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]).join(
                KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"])
                - KeySet.from_tuples([(1, 3)], columns=["B", "C"])
            )
        ],
        expected_subtracted_values=[],
    ),
    Case("crossjoin_subtract_in_join_split")(
        ks=KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]).join(
            KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"])
            * KeySet.from_dict({"D": [5, 6, 7]})
            - KeySet.from_tuples([(1, 6)], columns=["B", "D"])
        ),
        split_columns=["B"],
        expected_factors=[
            KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]),
            KeySet.from_tuples([(1, 3), (2, 4)], columns=["B", "C"]),
            KeySet.from_dict({"D": [5, 6, 7]}),
        ],
        expected_subtracted_values=[KeySet.from_tuples([(1, 6)], columns=["B", "D"])],
    ),
    Case("multi_join")(
        ks=KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"])
        .join(KeySet.from_tuples([(1, 3), (2, 3), (3, 4)], columns=["A", "C"]))
        .join(KeySet.from_tuples([(1, 5), (2, 6), (1, 7)], columns=["A", "D"]))
        .join(KeySet.from_tuples([(1, 8), (2, 8), (1, 9)], columns=["A", "E"]))
        .join(KeySet.from_tuples([(8, 10), (9, 10)], columns=["E", "F"])),
        expected_factors=[
            KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"])
            .join(KeySet.from_tuples([(1, 3), (2, 3), (3, 4)], columns=["A", "C"]))
            .join(KeySet.from_tuples([(1, 5), (2, 6), (1, 7)], columns=["A", "D"]))
            .join(KeySet.from_tuples([(1, 8), (2, 8), (1, 9)], columns=["A", "E"]))
            .join(KeySet.from_tuples([(8, 10), (9, 10)], columns=["E", "F"]))
        ],
        expected_subtracted_values=[],
    ),
    Case("multi_join_split")(
        ks=KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"])
        .join(KeySet.from_tuples([(1, 3), (2, 3), (3, 4)], columns=["A", "C"]))
        .join(KeySet.from_tuples([(1, 5), (2, 6), (1, 7)], columns=["A", "D"]))
        .join(KeySet.from_tuples([(1, 8), (2, 8), (1, 9)], columns=["A", "E"]))
        .join(KeySet.from_tuples([(8, 10), (9, 10)], columns=["E", "F"])),
        split_columns=["A"],
        expected_factors=[
            KeySet.from_tuples([(1, 1), (1, 2), (2, 1)], columns=["A", "B"]),
            KeySet.from_tuples([(1, 3), (2, 3), (3, 4)], columns=["A", "C"]),
            KeySet.from_tuples([(1, 5), (2, 6), (1, 7)], columns=["A", "D"]),
            KeySet.from_tuples([(1, 8), (2, 8), (1, 9)], columns=["A", "E"]).join(
                KeySet.from_tuples([(8, 10), (9, 10)], columns=["E", "F"])
            ),
        ],
        expected_subtracted_values=[],
    ),
)
def test_valid(
    ks: KeySet,
    split_columns: Optional[list[str]],
    expected_factors: list[KeySet],
    expected_subtracted_values: list[KeySet],
):
    # pylint: disable-next=protected-access
    factors, subtracted_values = ks._decompose(split_columns)

    _assert_keyset_sequence_equivalent(
        _order_keysets(factors), _order_keysets(expected_factors)
    )
    _assert_keyset_sequence_equivalent(
        _order_keysets(subtracted_values), _order_keysets(expected_subtracted_values)
    )

    def join(l, r):
        if set(l.columns()) & set(r.columns()):
            return l.join(r)
        return l * r

    reconstructed_ks = reduce(
        lambda l, r: l - r,
        subtracted_values,
        reduce(join, factors),
    )
    assert set(reconstructed_ks.columns()) == set(ks.columns())
    if not ks.is_equivalent(reconstructed_ks):
        # We could just assert that ks == reconstructed_ks directly, but the
        # error message is more helpful if we compare the dataframes.
        assert_dataframe_equal(
            # The column order isn't guaranteed to be preserved through this, so
            # reorder the reconstructed columns so that this comparison works.
            reconstructed_ks.dataframe().select(*ks.columns()),
            ks.dataframe(),
        )
