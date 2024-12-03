"""User-facing KeySet classes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from __future__ import annotations

import datetime
from collections.abc import Sequence
from functools import reduce
from typing import Iterable, Mapping, Optional, Union, overload

from pyspark.sql import DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, FrozenDict

from ._ops import CrossJoin, Detect, FromTuples, KeySetOp

KEYSET_COLUMN_TYPES = [ColumnType.INTEGER, ColumnType.DATE, ColumnType.VARCHAR]
"""Column types that are allowed in KeySets."""


def _validate_column_names(columns: Iterable[str]):
    for col in columns:
        if not isinstance(col, str):
            raise ValueError(
                f"Column names must be strings, not {type(col).__qualname__}."
            )
        if len(col) == 0:
            raise ValueError("Empty column names are not allowed.")


class KeySet:
    """A class containing a set of values for specific columns.

       An introduction to KeySet initialization and manipulation can be found in
       the :ref:`Group-by queries` tutorial.

    .. warning::
        If a column has null values dropped or replaced, then Analytics
        will raise an error if you use a KeySet that contains a null value for
        that column.
    """

    def __init__(self, op_tree: KeySetOp):
        """Constructor. @nodoc."""
        if not isinstance(op_tree, KeySetOp):
            raise ValueError(
                "KeySets should not be initialized using their constructor, "
                "use one of the various static initializer methods instead."
            )
        if op_tree.is_plan():
            raise AnalyticsInternalError(
                "KeySet should not be generated with a plan "
                "including partition selection."
            )
        self._op_tree = op_tree
        self._dataframe: Optional[DataFrame] = None
        self._cached = False

    @staticmethod
    def from_tuples(
        tuples: Iterable[tuple[Union[str, int, datetime.date, None], ...]],
        columns: Sequence[str],
    ) -> KeySet:
        """Creates a KeySet from a list of tuples and column names.

        Example:
            >>> tuples = [
            ...   ("a1", "b1"),
            ...   ("a2", "b1"),
            ...   ("a3", "b3"),
            ... ]
            >>> keyset = KeySet.from_tuples(tuples, ["A", "B"])
            >>> keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a2  b1
            2  a3  b3
        """
        _validate_column_names(columns)

        # Deduplicate the tuples
        tuple_set = frozenset(tuples)
        for t in tuple_set:
            if not isinstance(t, tuple):
                raise ValueError(
                    "Each element of tuples must be a tuple, but got "
                    f"{type(t)} instead."
                )
            if len(t) != len(columns):
                raise ValueError(
                    "Tuples must contain the same number of values "
                    "as there are columns.\n"
                    f"Columns: {', '.join(columns)}\n"
                    f"Mismatched tuple: {', '.join(map(str, t))}"
                )

        column_types: dict[str, set[type]] = {col: set() for col in columns}
        for t in tuple_set:
            for i, col in enumerate(columns):
                column_types[col].add(type(t[i]))

        schema = {}
        for col in column_types:
            types = column_types[col]
            if types == set():
                raise ValueError(
                    "Unable to infer column types for an empty collection of values."
                )
            if types == {type(None)}:
                raise ValueError(
                    f"Column '{col}' contains only null values, unable to "
                    "infer its type."
                )
            if len(types - {type(None)}) != 1:
                raise ValueError(
                    f"Column '{col}' contains values of multiple types: "
                    f"{', '.join(t.__name__ for t in types)}"
                )

            (col_type,) = types - {type(None)}
            # KeySets can't include floats, so no need to worry about nan/inf values.
            schema[col] = ColumnDescriptor(
                ColumnType(col_type), allow_null=type(None) in types
            )

        for col, desc in schema.items():
            if desc.column_type not in KEYSET_COLUMN_TYPES:
                raise ValueError(
                    f"Column '{col}' has type {desc.column_type}, but only allowed "
                    "types in KeySets are: "
                    f"{', '.join(t.name for t in KEYSET_COLUMN_TYPES)}"
                )

        return KeySet(FromTuples(tuple_set, FrozenDict.from_dict(schema)))

    @staticmethod
    def from_dict(
        domains: Mapping[
            str,
            Union[
                Iterable[Optional[str]],
                Iterable[Optional[int]],
                Iterable[Optional[datetime.date]],
            ],
        ]
    ) -> KeySet:
        """Creates a KeySet from a dictionary.

        The ``domains`` dictionary should map column names to the desired values
        for those columns. The KeySet returned is the cross-product of those
        columns. Duplicate values in the column domains are allowed, but only
        one of the duplicates is kept.

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": ["b1", "b2"],
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
        """
        if len(domains) == 0:
            return KeySet.from_tuples([], columns=[])

        domain_keysets = (
            KeySet.from_tuples(((v,) for v in values), columns=[col])
            for col, values in domains.items()
        )
        return reduce(lambda l, r: l * r, domain_keysets)

    # TODO(tumult-labs/tumult#3384): Make this public and fill in its docstring
    #     with an example of usage.
    @staticmethod
    def _detect(columns: Sequence[str]) -> KeySetPlan:
        """Detect the keys for a collection of columns."""
        column_set = frozenset(columns)
        if len(column_set) == 0:
            raise ValueError(
                "Detect must be used on a non-empty collection of columns."
            )
        _validate_column_names(column_set)
        return KeySetPlan(Detect(column_set))

    # Pydocstyle doesn't seem to understand overloads, so we need to disable the
    # check that a docstring exists for them.
    @overload
    def __mul__(self, other: KeySet) -> KeySet:  # noqa: D105
        ...

    @overload
    def __mul__(self, other: KeySetPlan) -> KeySetPlan:  # noqa: D105
        ...

    def __mul__(self, other):
        r"""The Cartesian product of the two KeySet or KeySetPlan factors.

        Multiplying two :class:`KeySet`\ s together produces another
        :class:`KeySet`; if either factor is a :class:`KeySetPlan`, then the
        result is a :class:`KeySetPlan`.

        Example:
            >>> keyset1 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset2 = KeySet.from_tuples([("b1",), ("b2",)], columns=["B"])
            >>> product = keyset1 * keyset2
            >>> product.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
        """
        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet multiplication expected another KeySet or KeySetPlan, not "
                f"{type(other).__qualname__}, as right-hand value."
            )
        overlapping_columns = set(self.columns()) & set(other.columns())
        if overlapping_columns:
            raise ValueError(
                "Unable to cross-join KeySets, they have "
                f"overlapping columns: {' '.join(overlapping_columns)}"
            )

        if isinstance(other, KeySet):
            return KeySet(CrossJoin(self._op_tree, other._op_tree))
        else:
            return KeySetPlan(CrossJoin(self._op_tree, other._op_tree))

    def columns(self) -> list[str]:
        """Returns the list of columns used in this KeySet."""
        return self._op_tree.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        # pylint: disable=line-too-long
        """Returns the KeySet's schema.

        Example:
            >>> keys = [
            ...     ("a1", 0),
            ...     ("a2", None),
            ... ]
            >>> keyset = KeySet.from_tuples(keys, columns=["A", "B"])
            >>> schema = keyset.schema()
            >>> schema # doctest: +NORMALIZE_WHITESPACE
            {'A': ColumnDescriptor(column_type=ColumnType.VARCHAR, allow_null=False, allow_nan=False, allow_inf=False),
             'B': ColumnDescriptor(column_type=ColumnType.INTEGER, allow_null=True, allow_nan=False, allow_inf=False)}
        """
        # pylint: enable=line-too-long
        return self._op_tree.schema()

    def dataframe(self) -> DataFrame:
        """Returns the dataframe associated with this KeySet.

        This dataframe contains every combination of values being selected in
        the KeySet, and its rows are guaranteed to be unique.
        """
        if not self._dataframe:
            self._dataframe = self._op_tree.dataframe()
            if self._cached:
                self._dataframe.cache()

        return self._dataframe

    def cache(self) -> None:
        """Caches the KeySet's dataframe in memory."""
        # Caching an already-cached dataframe produces a warning, so avoid doing
        # it by only caching the dataframe when the KeySet isn't already cached.
        if not self._cached:
            self._cached = True
            if self._dataframe:
                self._dataframe.cache()

    def uncache(self) -> None:
        """Removes the KeySet's dataframe from memory and disk."""
        self._cached = False
        if self._dataframe:
            self._dataframe.unpersist()


class KeySetPlan:
    """A plan for computing a KeySet based on values in a table.

    A :class:`.KeySetPlan` describes a plan for computing a set of group keys
    that may be used when computing a group-by query. This is similar to what a
    :class:`.KeySet` represents, with one key difference: a :class:`.KeySetPlan`
    requires spending some privacy budget with a :class:`.Session` to get back a
    specific :class:`.KeySet` for a particular table. The :class:`.KeySetPlan`
    alone cannot produce an equivalent dataframe and doesn't have a fixed
    schema.
    """

    def __init__(self, op_tree: KeySetOp):
        """Constructor. @nodoc."""
        if not isinstance(op_tree, KeySetOp):
            raise ValueError(
                "KeySets should not be initialized using their constructor, "
                "use one of the various static initializer methods instead."
            )
        if not op_tree.is_plan():
            raise AnalyticsInternalError(
                "KeySetPlan must be generated with a plan "
                "including partition selection."
            )
        self._op_tree = op_tree

    def columns(self) -> list[str]:
        """Returns the list of columns used in this KeySetPlan."""
        return self._op_tree.columns()

    def __mul__(self, other: Union[KeySet, KeySetPlan]) -> KeySetPlan:
        """The Cartesian product of the two KeySet or KeySetPlan factors.

        Example:
            >>> keyset1 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset2 = KeySet._detect(["B"])
            >>> product = keyset1 * keyset2
            >>> product.columns()
            ['A', 'B']
        """
        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet multiplication expected another KeySet or KeySetPlan, not "
                f"{type(other).__qualname__}, as right-hand value."
            )
        overlapping_columns = set(self.columns()) & set(other.columns())
        if overlapping_columns:
            raise ValueError(
                "Unable to cross-join KeySets, they have "
                f"overlapping columns: {' '.join(overlapping_columns)}"
            )

        return KeySetPlan(CrossJoin(self._op_tree, other._op_tree))
