"""Tests for KeySet __eq__ and is_equivalent methods."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

# pylint: disable=comparison-with-itself

from typing import Optional, Union

from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics._keyset_v2 import KeySet, KeySetPlan

_KS_A = KeySet.from_dict({"A": [1, 2, 3]})
_KS_B = KeySet.from_dict({"B": [4, 5]})
_KS_C = KeySet.from_dict({"C": [6, 7]})
_KS_DEF = KeySet.from_tuples([(8, 9, 10), (11, 12, 13)], columns=["D", "E", "F"])
_KS_ABCDEF = _KS_A * _KS_B * _KS_C * _KS_DEF


@parametrize(
    Case("tuples_eq")(
        ks1=KeySet.from_tuples([(1, 2, 3), (4, 5, 6)], columns=["A", "B", "C"]),
        ks2=KeySet.from_tuples([(4, 5, 6), (1, 2, 3)], columns=["A", "B", "C"]),
        equal=True,
        equivalence_known=True,
    ),
    Case("tuples_ne_different_columns")(
        ks1=KeySet.from_tuples([(1, 2, 3), (4, 5, 6)], columns=["A", "B", "C"]),
        ks2=KeySet.from_tuples([(1, 2, 3), (4, 5, 6)], columns=["A", "B", "D"]),
        equal=False,
        equivalence_known=True,
    ),
    Case("tuples_ne_different_values")(
        ks1=KeySet.from_tuples([(1, 2, 3), (4, 5, 6)], columns=["A", "B", "C"]),
        ks2=KeySet.from_tuples([(1, 2, 4), (4, 5, 6)], columns=["A", "B", "C"]),
        equal=False,
    ),
    Case("project_eq")(
        ks1=_KS_DEF["D"],
        ks2=_KS_DEF["D"],
        equal=True,
        equivalence_known=True,
    ),
    Case("project_ne_different_column")(
        ks1=_KS_DEF["D"],
        ks2=_KS_DEF["E"],
        equal=False,
        equivalence_known=True,
    ),
    Case("project_ne_different_child")(
        ks1=_KS_DEF["D"],
        ks2=_KS_DEF.filter("E = 9")["D"],
        equal=False,
    ),
    Case("project_nested")(
        ks1=_KS_DEF["E", "F"]["E"],
        ks2=_KS_DEF["E"],
        equal=True,
        equivalence_known=True,
    ),
    Case("project_across_crossjoin")(
        ks1=_KS_ABCDEF["A", "C", "E"],
        ks2=_KS_A * _KS_C * _KS_DEF["E"],
        equal=True,
        equivalence_known=True,
    ),
    Case("crossjoin_commutative")(
        ks1=_KS_A * _KS_B,
        ks2=_KS_B * _KS_A,
        equal=True,
        equivalence_known=True,
    ),
    Case("crossjoin_associative")(
        ks1=(_KS_A * _KS_B) * _KS_C,
        ks2=_KS_A * (_KS_B * _KS_C),
        equal=True,
        equivalence_known=True,
    ),
    Case("from_dict_commutative")(
        ks1=KeySet.from_dict({"A": [1, 2], "B": [3, 4], "C": [5], "D": [6, 7, 8]}),
        ks2=KeySet.from_dict({"D": [6, 7, 8], "B": [3, 4], "A": [1, 2], "C": [5]}),
        equal=True,
        equivalence_known=True,
    ),
    Case("from_dict_ne_different_columns")(
        ks1=KeySet.from_dict({"A": [1, 2], "B": [3, 4], "C": [5], "D": [6, 7, 8]}),
        ks2=KeySet.from_dict({"X": [1, 2], "B": [3, 4], "C": [5], "D": [6, 7, 8]}),
        equal=False,
        equivalence_known=True,
    ),
    Case("from_dict_ne_different_values")(
        ks1=KeySet.from_dict({"A": [1, 2], "B": [3, 4], "C": [5], "D": [6, 7, 8]}),
        ks2=KeySet.from_dict({"A": [9, 2], "B": [3, 4], "C": [5], "D": [6, 7, 8]}),
        equal=False,
    ),
    Case("filter_eq")(
        ks1=_KS_DEF.filter("E = -1"),
        ks2=_KS_DEF.filter("E = -1"),
        equal=True,
        equivalence_known=True,
    ),
    Case("filter_eq_not_equivalent")(
        ks1=_KS_DEF.filter("E = -1"),
        ks2=_KS_DEF.filter("E = -2"),
        equal=True,
    ),
    Case("filter_ne_different_condition")(
        ks1=_KS_DEF.filter("E = 0"),
        ks2=_KS_DEF.filter("E != 0"),
        equal=False,
    ),
    Case("filter_ne_different_child")(
        ks1=_KS_DEF.filter("E = 0"),
        ks2=_KS_ABCDEF.filter("E = 0"),
        equal=False,
        equivalence_known=True,
    ),
    Case("subtraction_order")(
        ks1=_KS_ABCDEF
        - KeySet.from_dict({"A": [1, 2]})
        - KeySet.from_dict({"E": [9]})
        - KeySet.from_dict({"C": [6]}),
        ks2=_KS_ABCDEF
        - KeySet.from_dict({"C": [6]})
        - KeySet.from_dict({"E": [9]})
        - KeySet.from_dict({"A": [1, 2]}),
        equal=True,
        equivalence_known=True,
    ),
    Case("extract_crossjoin_from_join")(
        ks1=KeySet.from_tuples([(1, 2), (3, 4), (5, 6)], columns=["A", "B"]).join(
            KeySet.from_tuples([(2, 7), (4, 8)], columns=["B", "C"])
            * KeySet.from_dict({"D": [1, 2], "E": [3]})
        )
        * KeySet.from_dict({"F": [11, 12]}),
        ks2=KeySet.from_tuples([(1, 2), (3, 4), (5, 6)], columns=["A", "B"]).join(
            KeySet.from_tuples([(2, 7), (4, 8)], columns=["B", "C"])
        )
        * KeySet.from_dict({"D": [1, 2], "E": [3], "F": [11, 12]}),
        equal=True,
        equivalence_known=True,
    ),
)
def test_equivalence(
    ks1: KeySet, ks2: KeySet, equal: bool, equivalence_known: Optional[bool]
):
    """Equality and equivalence of KeySets work as expected.

    When ``equivalence_known`` is ``True``, the equivalence check between the
    two KeySets must match the equality check; otherwise, it may be ``None``,
    though it still must not get the opposite answer from the equality check.
    """
    assert ks1 == ks1
    assert ks2 == ks2
    assert (ks1 == ks2) == equal
    assert (ks2 == ks1) == equal
    assert ks1.is_equivalent(ks1)
    assert ks2.is_equivalent(ks2)
    if equivalence_known:
        assert ks1.is_equivalent(ks2) == equal
        assert ks2.is_equivalent(ks1) == equal
    else:
        assert ks1.is_equivalent(ks2) in {equal, None}
        assert ks2.is_equivalent(ks1) in {equal, None}


def test_equivalence_dataframes(spark):
    """Equality and equivalence of KeySets created from dataframes work as expected."""
    df1 = spark.range(10)
    df2 = spark.range(10)
    df3 = spark.range(12)

    assert df1 is not df2
    assert df1.sameSemantics(df2)

    ks1 = KeySet.from_dataframe(df1)
    ks2 = KeySet.from_dataframe(df2)
    ks3 = KeySet.from_dataframe(df3)

    assert ks1 == ks1
    assert ks1 == ks2
    assert ks2 == ks1
    assert ks1 != ks3
    assert ks3 != ks1
    assert ks1.is_equivalent(ks1)
    assert ks1.is_equivalent(ks2)
    assert ks2.is_equivalent(ks1)
    assert ks1.is_equivalent(ks3) is not True
    assert ks3.is_equivalent(ks1) is not True


# pylint: disable=protected-access
@parametrize(
    Case("detect_eq")(
        ks1=KeySet._detect(["A", "B"]),
        ks2=KeySet._detect(["B", "A"]),
        equivalent=True,
    ),
    Case("detect_ne")(
        ks1=KeySet._detect(["A", "B"]),
        ks2=KeySet._detect(["A", "B", "C"]),
        equivalent=False,
    ),
    Case("project_eq")(
        ks1=KeySet._detect(["A", "B"])["B"],
        ks2=KeySet._detect(["B", "A"])["B"],
        equivalent=True,
    ),
    Case("project_ne_different_child")(
        ks1=KeySet._detect(["A", "B"])["A"],
        ks2=KeySet._detect(["A", "B", "C"])["A"],
        equivalent={None, False},
    ),
    Case("project_ne_different_column")(
        ks1=KeySet._detect(["A", "B"])["A"],
        ks2=KeySet._detect(["A", "B"])["B"],
        equivalent=False,
    ),
    Case("projected_detect_ne")(
        ks1=KeySet._detect(["A", "B"])["A"],
        ks2=KeySet._detect(["A"]),
        equivalent={None, False},
    ),
    Case("crossjoin_eq")(
        ks1=KeySet._detect(["A", "B"]) * _KS_DEF,
        ks2=_KS_DEF * KeySet._detect(["A", "B"]),
        equivalent=True,
    ),
    Case("crossjoin_ne")(
        ks1=KeySet._detect(["A"]) * _KS_DEF,
        ks2=KeySet._detect(["A", "B"]) * _KS_DEF,
        equivalent=False,
    ),
    Case("crossjoin_ne_same_columns")(
        ks1=KeySet._detect(["A"]) * _KS_B * _KS_DEF,
        ks2=KeySet._detect(["A", "B"]) * _KS_DEF,
        equivalent={None, False},
    ),
    Case("filter_eq")(
        ks1=KeySet._detect(["A", "B", "C"]).filter("B = -1"),
        ks2=KeySet._detect(["A", "B", "C"]).filter("B = -1"),
        equivalent=True,
    ),
    Case("filter_eq_different_condition")(
        ks1=KeySet._detect(["A", "B", "C"]).filter("B = -1"),
        ks2=KeySet._detect(["A", "B", "C"]).filter("B = -2"),
        equivalent=None,
    ),
    Case("filter_ne_different_child")(
        ks1=KeySet._detect(["A", "B"]).filter("B = -1"),
        ks2=KeySet._detect(["A", "B", "C"]).filter("B = -1"),
        equivalent=False,
    ),
    Case("filter_ne_different_child_same_schema")(
        ks1=(KeySet._detect(["A", "B"]) * KeySet.from_dict({"C": [1]})).filter(
            "B = -1"
        ),
        ks2=KeySet._detect(["A", "B", "C"]).filter("B = -1"),
        equivalent={False, None},
    ),
)
# pylint: enable=protected-access
def test_plan_equivalence(
    ks1: KeySetPlan, ks2: KeySetPlan, equivalent: Union[None, bool, set[Optional[bool]]]
):
    """Equality and equivalence of KeySetPlans work as expected."""
    assert ks1 == ks1
    assert ks2 == ks2
    assert ks1.is_equivalent(ks1)
    assert ks2.is_equivalent(ks2)
    if isinstance(equivalent, set):
        assert (ks1 == ks2) in equivalent  # pylint: disable=superfluous-parens
        assert isinstance(ks1 == ks2, bool)
        assert ks1.is_equivalent(ks2) in equivalent
        assert ks2.is_equivalent(ks1) in equivalent
    else:
        assert (ks1 == ks2) == bool(equivalent)
        assert ks1.is_equivalent(ks2) == equivalent
        assert ks2.is_equivalent(ks1) == equivalent
