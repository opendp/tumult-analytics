"""User-facing KeySet classes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import datetime
from collections.abc import Sequence
from functools import reduce
from typing import Any, Iterable, Mapping, Optional, Union, overload

from pyspark.sql import Column, DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, FrozenDict

from ._ops import (
    CrossJoin,
    Detect,
    Filter,
    FromSparkDataFrame,
    FromTuples,
    Join,
    KeySetOp,
    Project,
    Subtract,
    rewrite,
)


class KeySet:
    """A class containing a set of values for specific columns.

       An introduction to KeySet initialization and manipulation can be found in
       the :ref:`Group-by queries` tutorial.

    .. warning::
        If a column has null values dropped or replaced, then Analytics
        will raise an error if you use a KeySet that contains a null value for
        that column.
    """

    def __init__(self, op_tree: KeySetOp, columns: Sequence[str]):
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
        if len(columns) != len(set(columns)):
            column_counts: dict[str, int] = {}
            for c in columns:
                column_counts[c] = column_counts.get(c, 0) + 1
            raise AnalyticsInternalError(
                "KeySet columns are not all distict, duplicates are: "
                + " ".join(c for c, count in column_counts.items() if count > 1)
            )
        if op_tree.columns() != set(columns):
            raise AnalyticsInternalError(
                f"KeySet columns {columns} do not match "
                f"the columns of its op-tree {op_tree.columns()}."
            )
        self._op_tree = rewrite(op_tree)
        self._columns = columns
        self._dataframe: Optional[DataFrame] = None
        self._size: Optional[int] = None
        self._cached = False

    @staticmethod
    def from_dataframe(df: DataFrame) -> KeySet:
        """Creates a KeySet from a dataframe.

        This DataFrame should contain every combination of values being selected
        in the KeySet. If there are duplicate rows in the dataframe, only one
        copy of each will be kept.

        When creating KeySets with this method, it is the responsibility of the
        caller to ensure that the given dataframe remains valid for the lifetime
        of the KeySet. If the dataframe becomes invalid, for example because its
        Spark session is closed, this method or any uses of the resulting
        dataframe may raise exceptions or have other unanticipated effects.
        """
        return KeySet(FromSparkDataFrame(df), columns=df.columns)

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

        return KeySet(
            FromTuples(tuple_set, FrozenDict.from_dict(schema)), columns=columns
        )

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
        return KeySetPlan(Detect(frozenset(columns)), columns=columns)

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
        if isinstance(other, KeySet):
            return KeySet(
                CrossJoin((self._op_tree, other._op_tree)),
                columns=self.columns() + other.columns(),
            )
        else:
            return KeySetPlan(
                CrossJoin((self._op_tree, other._op_tree)),
                columns=self.columns() + other.columns(),
            )

    def __sub__(self, other: KeySet) -> KeySet:
        """Remove rows in this that match rows in another KeySet.

        Equivalent to a left anti-join between this KeySet and ``other``.

        ``other`` must have a subset of the columns of this KeySet. Any rows in this
        KeySet where the values in those columns match values in ``other`` are removed.

        Example:
            >>> keyset1 = KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]})
            >>> result = keyset1 - KeySet.from_tuples([(1, "b")], columns=["A", "B"])
            >>> result.dataframe().sort("A", "B").toPandas()
               A  B
            0  1  a
            1  2  a
            2  2  b
        """
        return KeySet(Subtract(self._op_tree, other._op_tree), self.columns())

    def __getitem__(self, desired_columns: Union[str, Sequence[str]]) -> KeySet:
        """``KeySet[col, col, ...]`` returns a KeySet with those columns only.

        The returned KeySet contains all unique combinations of values in the
        given columns that were present in the original KeySet.

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": ["b1", "b2"],
            ...     "C": ["c1", "c2"],
            ...     "D": [0, 1, 2, 3]
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> a_b_keyset = keyset["A", "B"]
            >>> a_b_keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
            >>> a_b_keyset = keyset[["A", "B"]]
            >>> a_b_keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
            >>> a_keyset = keyset["A"]
            >>> a_keyset.dataframe().sort("A").toPandas()
                A
            0  a1
            1  a2
        """
        if isinstance(desired_columns, str):
            desired_columns = [desired_columns]

        return KeySet(
            Project(self._op_tree, frozenset(desired_columns)), columns=desired_columns
        )

    # Pydocstyle doesn't seem to understand overloads, so we need to disable the
    # check that a docstring exists for them.
    @overload
    def join(self, other: KeySet) -> KeySet:  # noqa: D105
        ...

    @overload
    def join(self, other: KeySetPlan) -> KeySetPlan:  # noqa: D105
        ...

    def join(self, other):
        r"""The inner natural join of two KeySet or KeySetPlan objects.

        The two KeySets are inner joined on columns with matching names,
        treating nulls as equal to one another. Joining two :class:`KeySet`\ s
        together produces another :class:`KeySet`; if either operand is a
        :class:`KeySetPlan`, then the result is also a :class:`KeySetPlan`.

        Example:
            >>> keyset1 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset2 = KeySet.from_tuples(
            ...     [("a2", "b1"), ("a3", "b2")], columns=["A", "B"]
            ... )
            >>> keyset1.join(keyset2).dataframe().sort("A", "B").toPandas()
                A   B
            0  a2  b1
        """
        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet join expected another KeySet or KeySetPlan, not "
                f"{type(other).__qualname__}."
            )
        if isinstance(other, KeySet):
            return KeySet(
                # pylint: disable-next=protected-access
                Join(self._op_tree, other._op_tree),
                columns=list(dict.fromkeys(self.columns() + other.columns())),
            )
        else:
            return KeySetPlan(
                # pylint: disable-next=protected-access
                Join(self._op_tree, other._op_tree),
                columns=list(dict.fromkeys(self.columns() + other.columns())),
            )

    def filter(self, condition: Union[Column, str]) -> KeySet:
        """Filters this KeySet using some condition.

        This method accepts the same syntax as
        :meth:`pyspark.sql.DataFrame.filter`: valid conditions are those that
        can be used in a `WHERE clause
        <https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-where.html>`__
        in Spark SQL. Examples of valid conditions include:

        * ``age < 42``
        * ``age BETWEEN 17 AND 42``
        * ``age < 42 OR (age < 60 AND gender IS NULL)``
        * ``LENGTH(name) > 17``
        * ``favorite_color IN ('blue', 'red')``

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": [0, 1, 2, 3],
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> filtered_keyset = keyset.filter("B < 2")
            >>> filtered_keyset.dataframe().sort("A", "B").toPandas()
                A  B
            0  a1  0
            1  a1  1
            2  a2  0
            3  a2  1
            >>> import pyspark.sql.functions as sf
            >>> filtered_keyset = keyset.filter(sf.col("A") != "a1")
            >>> filtered_keyset.dataframe().sort("A", "B").toPandas()
                A  B
            0  a2  0
            1  a2  1
            2  a2  2
            3  a2  3
        """
        return KeySet(Filter(self._op_tree, condition), columns=self.columns())

    def columns(self) -> list[str]:
        """Returns the list of columns used in this KeySet."""
        return list(self._columns)

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
        schema = self._op_tree.schema()
        return {c: schema[c] for c in self.columns()}  # Reorder to match self.columns()

    def dataframe(self) -> DataFrame:
        """Returns the dataframe associated with this KeySet.

        This dataframe contains every combination of values being selected in
        the KeySet, and its rows are guaranteed to be unique.
        """
        if not self._dataframe:
            df = self._op_tree.dataframe()
            if set(df.columns) != set(self.columns()):
                raise AnalyticsInternalError(
                    f"KeySet op-tree dataframe produced columns {df.columns} that "
                    f"do not match its expected columns {self.columns()}."
                )
            # Reorder to match self.columns()
            self._dataframe = df.select(*self.columns())
            if self._cached:
                self._dataframe.cache()

        return self._dataframe

    def size(self) -> int:
        """Returns the number of groups included in this KeySet.

        Note that in some situations this method may need to count the elements
        in the KeySet's dataframe, which can be extremely slow.
        """
        if self._size is None:
            self._size = self._op_tree.size(fast=False)
        return self._size

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

    def is_equivalent(self, other: Union[KeySet, KeySetPlan]) -> Optional[bool]:
        """Determine if another KeySet is equivalent to this one, if possible.

        This method is an alternative to :meth:`KeySet.__eq__` which is
        guaranteed to never evaluate the full KeySet dataframe. This ensures
        that it is never time-consuming to call, but also means that it cannot
        always determine if two KeySets are equivalent. If the KeySets are
        neither definitely equivalent nor easily shown to not be equivalent,
        this method returns ``None``.
        """
        # A KeySet and a KeySetPlan can't be equivalent, but allowing either to
        # be passed could avoid some user confusion about what is/is not a plan.
        if not isinstance(other, KeySet):
            return False

        if self._op_tree == other._op_tree:  # pylint: disable=protected-access
            return True

        if self.schema() != other.schema():
            return False

        return None

    def __eq__(self, other: Any):
        """Determine if another KeySet is equal to this one.

        Two KeySets are equal if they contain the same values for the same
        columns; the rows and columns may appear in any order.

        Example:
            >>> ks1 = KeySet.from_dict({"A": [1, 2], "B": [3, 4]})
            >>> ks2 = KeySet.from_dict({"B": [3, 4], "A": [1, 2]})
            >>> ks3 = KeySet.from_dict({"B": [4, 3], "A": [2, 1]})
            >>> ks4 = KeySet.from_dict({"B": [4, 5], "A": [1, 2]})
            >>> ks1 == ks2
            True
            >>> ks1 == ks3
            True
            >>> ks1 == ks4
            False
        """
        if not isinstance(other, KeySet):
            return False

        equivalent = self.is_equivalent(other)
        if equivalent is not None:
            return equivalent

        # Reorder columns between the two dataframes to match
        columns = self.columns()
        self_df = self.dataframe().select(*columns)
        other_df = other.dataframe().select(*columns)
        if self_df.schema != other_df.schema:
            return False
        # other_df should contain all rows in self_df
        if self_df.exceptAll(other_df).count() != 0:
            return False
        # and vice versa
        if other_df.exceptAll(self_df).count() != 0:
            return False
        return True


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

    def __init__(self, op_tree: KeySetOp, columns: Sequence[str]):
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
        if op_tree.columns() != set(columns):
            raise AnalyticsInternalError(
                f"KeySet columns {columns} do not match "
                f"the columns of its op-tree {op_tree.columns()}."
            )
        self._op_tree = rewrite(op_tree)
        self._columns = columns

        if not self._op_tree.is_plan():
            raise AnalyticsInternalError(
                "KeySetPlan op-tree unexpectedly became not a plan."
            )

    def columns(self) -> list[str]:
        """Returns the list of columns used in this KeySetPlan."""
        return list(self._columns)

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
        return KeySetPlan(
            CrossJoin((self._op_tree, other._op_tree)),
            columns=self.columns() + other.columns(),
        )

    def __sub__(self, other: KeySet) -> KeySetPlan:
        """Remove rows in this that match rows in another KeySet.

        Equivalent to a left anti-join between this and other.

        other must have a subset of the columns of this KeySet. Any rows in this
        KeySet where the values in those columns match values in other are removed.

        Example:
            >>> keyset1 = KeySet._detect(["A", "B"])
            >>> result = keyset1 - KeySet.from_tuples([(1, "b")], columns=["A", "B"])
            >>> result.columns()
            ['A', 'B']
        """
        return KeySetPlan(
            Subtract(self._op_tree, other._op_tree), columns=self.columns()
        )

    def __getitem__(
        self, desired_columns: Union[str, Sequence[str]]
    ) -> Union[KeySet, KeySetPlan]:
        """``KeySetPlan[col, col, ...]`` returns a KeySetPlan with those columns only.

        The returned KeySetPlan contains all unique combinations of values in the
        given columns that were present in the original KeySetPlan.

        Example:
            >>> keyset = KeySet._detect(["A", "B", "C"])
            >>> keyset["A", "B"].columns()
            ['A', 'B']
            >>> keyset[["A", "B"]].columns()
            ['A', 'B']
            >>> keyset["A"].columns()
            ['A']
        """
        if isinstance(desired_columns, str):
            desired_columns = [desired_columns]

        # Projecting is the only operation that can turn a KeySetPlan into a
        # KeySet (by dropping the detected columns without a join or similar
        # operation that would force them to be computed anyway), so it works a
        # bit differently. Do the op-tree rewrite early and use its output to
        # figure out whether the result is a plan or not.
        op_tree = rewrite(Project(self._op_tree, frozenset(desired_columns)))
        if op_tree.is_plan():
            return KeySetPlan(op_tree, columns=desired_columns)
        return KeySet(op_tree, columns=desired_columns)

    def join(self, other: Union[KeySet, KeySetPlan]) -> KeySetPlan:
        r"""The inner natural join of two KeySet or KeySetPlan objects.

        The two KeySets are inner joined on columns with matching names,
        treating nulls as equal to one another. Joining a :class:`KeySetPlan` to
        any other value always produces a :class:`KeySetPlan`.

        Example:
            >>> keyset1 = KeySet._detect(["A", "B"])
            >>> keyset2 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset1.join(keyset2).columns()
            ['A', 'B']
        """
        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet join expected another KeySet or KeySetPlan, not "
                f"{type(other).__qualname__}."
            )

        return KeySetPlan(
            # pylint: disable-next=protected-access
            Join(self._op_tree, other._op_tree),
            columns=list(dict.fromkeys(self.columns() + other.columns())),
        )

    def filter(self, condition: Union[Column, str]) -> KeySetPlan:
        """Filters this KeySetPlan using some condition.

        This method accepts the same syntax as
        :meth:`pyspark.sql.DataFrame.filter`: valid conditions are those that
        can be used in a `WHERE clause
        <https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-where.html>`__
        in Spark SQL. Examples of valid conditions include:

        * ``age < 42``
        * ``age BETWEEN 17 AND 42``
        * ``age < 42 OR (age < 60 AND gender IS NULL)``
        * ``LENGTH(name) > 17``
        * ``favorite_color IN ('blue', 'red')``

        Example:
            >>> keyset = KeySet._detect(["A", "B"])
            >>> filtered_keyset = keyset.filter("B < 2")
            >>> filtered_keyset.columns()
            ['A', 'B']
        """
        return KeySetPlan(Filter(self._op_tree, condition), columns=self.columns())

    def is_equivalent(self, other: Union[KeySet, KeySetPlan]) -> Optional[bool]:
        """Determine if another KeySetPlan is equivalent to this one, if possible.

        This method is guaranteed to never evaluate any underlying dataframe,
        ensuring that it is never time-consuming to call. However, this means
        means that it cannot always determine if two KeySetPlans are
        equivalent. If the KeySetPlans are neither definitely equivalent nor
        easily shown to not be equivalent, this method returns ``None``.

        Example:
            >>> ks1 = KeySet._detect(["A", "B"]) * KeySet.from_dict({"C": [1,2]})
            >>> ks2 = KeySet.from_dict({"C": [1,2]}) * KeySet._detect(["A", "B"])
            >>> ks1.is_equivalent(ks2)
            True
        """
        # A KeySet and a KeySetPlan can't be equivalent, but allowing either to
        # be passed could avoid some user confusion about what is/is not a plan.
        if not isinstance(other, KeySetPlan):
            return False

        if self._op_tree == other._op_tree:  # pylint: disable=protected-access
            return True

        if self.columns() != other.columns():
            return False

        return None

    def __eq__(self, other: Any):
        r"""Determine if another :class:`KeySetPlan` is equal to this one.

        Unlike for :meth:`KeySet.__eq__`, there is no fallback full-dataframe
        comparison here -- this method only relies on :meth:`is_equivalent`, and
        returns ``False`` if it cannot determine whether the two
        :class:`KeySetPlan`\ s are equivalent.
        """
        if not isinstance(other, KeySetPlan):
            return False

        return bool(self.is_equivalent(other))
