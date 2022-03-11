"""A KeySet specifies a list of values for one or more columns.

For example, a KeySet could specify the values `["a1", "a2"]` for column A
and the values `[0, 1, 2, 3]` for column B.

Currently, KeySets are used as a simpler way to specify domains for groupby
transformations.
"""

# <placeholder: boilerplate>

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from pyspark.sql import Column, DataFrame

from tmlt.analytics._schema import Schema, spark_schema_to_analytics_columns
from tmlt.core.transformations.spark_transformations.groupby import (
    compute_full_domain_df,
)


class KeySet:
    """A class containing a set of values for specific columns."""

    def __init__(self, dataframe: Union[DataFrame, Callable[[], DataFrame]]) -> None:
        """Construct a new keyset.

        The :meth:`from_dict` and :meth:`from_dataframe` methods are preferred
        over directly using the constructor to create new KeySets.
        """
        self._dataframe = dataframe
        # TODO(#1707): Remove this
        self._public_id: Optional[str] = None

    def dataframe(self) -> DataFrame:
        """Return the dataframe associated with this KeySet.

        This dataframe contains every unique combination of values being
        selected in the KeySet.
        """
        if callable(self._dataframe):
            self._dataframe = self._dataframe()
        return self._dataframe

    @classmethod
    def from_dict(
        cls: Type[KeySet], domains: Dict[str, Union[List[str], List[int]]]
    ) -> KeySet:
        """Create a KeySet from a dictionary.

        The dictionary should map column names to the desired values for those columns.
        The KeySet returned is the cross-product of those columns.

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
        # compute_full_domain_df throws an IndexError if any list has length 0
        for v in domains.values():
            if len(v) == 0:
                raise ValueError("Every column should have a non-empty list of values.")
        return KeySet(lambda: compute_full_domain_df(domains))

    @classmethod
    def from_dataframe(cls: Type[KeySet], dataframe: DataFrame) -> KeySet:
        """Create a KeySet from a dataframe.

        This DataFrame should contain every unique combination of values being
        selected in the KeySet.

        When creating KeySets with this method, it is the responsibility of the
        caller to ensure that the given dataframe remains valid for the lifetime
        of the KeySet. If the dataframe becomes invalid, for example because its
        Spark session is closed, this method or any uses of the resulting
        dataframe may raise exceptions or have other unanticipated effects.
        """
        return KeySet(dataframe)

    # TODO(#1707): Remove this
    @classmethod
    def _from_public_source(cls, source_id: str) -> KeySet:
        """Create a KeySet based on a public source.

        Do not use this method in any new code. KeySets created with this method
        are not safe, and calling other methods on them will probably cause errors.
        """
        keyset = KeySet(None)
        keyset._public_id = source_id  # pylint: disable=protected-access
        return keyset

    def filter(self, expr: Union[Column, str]) -> KeySet:
        """Filter this domain using some expression.

        The expression should be one accepted by
        :meth:`~pyspark.sql.DataFrame.filter`.

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
            >>> filtered_keyset = keyset.filter(keyset.dataframe().A != "a1")
            >>> filtered_keyset.dataframe().sort("A", "B").toPandas()
                A  B
            0  a2  0
            1  a2  1
            2  a2  2
            3  a2  3
        """
        return KeySet(self.dataframe().filter(expr))

    def __getitem__(self, cols: Union[str, Tuple[str, ...], List[str]]) -> KeySet:
        """`KeySet[col, col, ...]` returns a KeySet with those columns only.

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
        if isinstance(cols, str):
            cols = (cols,)
        return KeySet(self.dataframe().select(*cols).dropDuplicates())

    def __mul__(self, other: KeySet) -> KeySet:
        """A product (`KeySet * KeySet`) returns the cross-product of both KeySets.

        Example:
            >>> keyset1 = KeySet.from_dict({"A": ["a1", "a2"]})
            >>> keyset2 = KeySet.from_dict({"B": ["b1", "b2"]})
            >>> product = keyset1 * keyset2
            >>> product.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
        """
        return KeySet(self.dataframe().crossJoin(other.dataframe()))

    def __eq__(self, other: object) -> bool:
        """Override equality.

        Two KeySets are equal if their dataframes contain the same values for
        the same columns (in any order).

        Example:
            >>> keyset1 = KeySet.from_dict({"A": ["a1", "a2"]})
            >>> keyset2 = KeySet.from_dict({"A": ["a1", "a2"]})
            >>> keyset3 = KeySet.from_dict({"A": ["a2", "a1"]})
            >>> keyset1 == keyset2
            True
            >>> keyset1 == keyset3
            True
            >>> different_keyset = KeySet.from_dict({"B": ["a1", "a2"]})
            >>> keyset1 == different_keyset
            False
        """
        if not isinstance(other, KeySet):
            return False
        # TODO(#1707): Remove this check
        if self._public_id is not None or other._public_id is not None:
            return self._public_id == other._public_id
        self_df = self.dataframe().toPandas()
        other_df = other.dataframe().toPandas()
        if sorted(self_df.columns) != sorted(other_df.columns):
            return False
        if self_df.empty and other_df.empty:
            return True
        sorted_columns = sorted(self_df.columns)
        self_df_sorted = self_df.set_index(sorted_columns).sort_index().reset_index()
        other_df_sorted = other_df.set_index(sorted_columns).sort_index().reset_index()
        return self_df_sorted.equals(other_df_sorted)

    def schema(self) -> Schema:
        """Returns a Schema based on the KeySet.

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": [0, 1, 2, 3],
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> schema = keyset.schema()
            >>> schema
            Schema({'A': 'VARCHAR', 'B': 'INTEGER'})
        """
        return Schema(spark_schema_to_analytics_columns(self.dataframe().schema))
