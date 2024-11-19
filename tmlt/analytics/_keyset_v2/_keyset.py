"""User-facing KeySet classes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Optional, Union

from pyspark.sql import DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, FrozenDict

from ._ops import CrossJoin, FromTuples, KeySetOp

KEYSET_COLUMN_TYPES = [ColumnType.INTEGER, ColumnType.DATE, ColumnType.VARCHAR]
"""Column types that are allowed in KeySets."""


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
        tuples: Sequence[tuple[Union[str, int, datetime.date, None], ...]],
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
        for col in columns:
            if not isinstance(col, str):
                raise ValueError(f"Column names must be strings, not {type(col)}.")
            if len(col) == 0:
                raise ValueError("Empty column names are not allowed.")

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
                    "Unable to infer column types for empty collection of tuples."
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

    def __mul__(self, other: KeySet) -> KeySet:
        """A product (``KeySet * KeySet``) returns the cross-product of both KeySets.

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
        if not isinstance(other, KeySet):
            raise ValueError(
                "KeySet multiplication expected another KeySet, not "
                f"{type(other)}, as right-hand value."
            )
        overlapping_columns = set(self.columns()) & set(other.columns())
        if overlapping_columns:
            raise ValueError(
                "Unable to cross-join KeySets, they have "
                f"overlapping columns: {' '.join(overlapping_columns)}"
            )
        return KeySet(CrossJoin(self._op_tree, other._op_tree))

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
