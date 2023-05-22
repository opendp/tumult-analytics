"""A KeySet specifies a list of values for one or more columns.

For example, a KeySet could specify the values ``["a1", "a2"]`` for column A
and the values ``[0, 1, 2, 3]`` for column B.

Currently, KeySets are used as a simpler way to specify domains for groupby
transformations.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from __future__ import annotations

import datetime
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import types as spark_types
from tmlt.core.transformations.spark_transformations.groupby import (
    compute_full_domain_df,
)
from tmlt.core.utils.type_utils import get_element_type

from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import Schema, spark_schema_to_analytics_columns


def _check_df_schema(types: spark_types.StructType):
    """Raise an exception if any of the given types are not allowed in a KeySet."""
    allowed_types = {
        spark_types.LongType(),
        spark_types.StringType(),
        spark_types.DateType(),
    }
    for field in types.fields:
        if field.dataType not in allowed_types:
            raise ValueError(
                f"Column {field.name} has type {field.dataType}, which is "
                "not allowed in KeySets. Allowed column types are: "
                f"{','.join(str(t) for t in allowed_types)}"
            )


def _check_dict_schema(types: Dict[str, type]) -> None:
    """Raise an exception if the dict contains a type not allowed in a KeySet."""
    allowed_types = {int, str, datetime.date}
    for col, dtype in types.items():
        if dtype not in allowed_types:
            raise ValueError(
                f"Column {col} has type {dtype.__qualname__}, which is "
                "not allowed in KeySets. Allowed column types are: "
                f"{','.join(t.__qualname__ for t in allowed_types)}"
            )


class KeySet:
    """A class containing a set of values for specific columns.

    .. warning::
        If a column has null values dropped or replaced, then Analytics
        will raise an error if you use a KeySet that contains a null value for
        that column.
    """

    def __init__(self, dataframe: Union[DataFrame, Callable[[], DataFrame]]) -> None:
        """Construct a new keyset.

        The :meth:`from_dict` and :meth:`from_dataframe` methods are preferred
        over directly using the constructor to create new KeySets. Directly
        constructing KeySets skips checks that guarantee the uniqueness of
        output rows.
        """
        self._dataframe: Union[DataFrame, Callable[[], DataFrame]]
        if isinstance(dataframe, DataFrame):
            self._dataframe = coerce_spark_schema_or_fail(dataframe)
            _check_df_schema(self._dataframe.schema)
        else:
            self._dataframe = dataframe

    def dataframe(self) -> DataFrame:
        """Return the dataframe associated with this KeySet.

        This dataframe contains every combination of values being selected in
        the KeySet, and its rows are guaranteed to be unique as long as the
        KeySet was constructed safely.
        """
        if callable(self._dataframe):
            self._dataframe = coerce_spark_schema_or_fail(self._dataframe())
            # Invalid column types should get caught before this, as it keeps
            # the exception closer to the user code that caused it, but in case
            # that is missed we check again here.
            _check_df_schema(self._dataframe.schema)
        return self._dataframe

    @classmethod
    def from_dict(
        cls: Type[KeySet],
        domains: Mapping[
            str,
            Union[
                Iterable[Optional[str]],
                Iterable[Optional[int]],
                Iterable[Optional[datetime.date]],
            ],
        ],
    ) -> KeySet:
        """Create a KeySet from a dictionary.

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
        # Mypy can't propagate the value type through this operation for some
        # reason -- it thinks the resulting type is Dict[str, List[object]].
        list_domains: Dict[
            str,
            Union[
                List[Optional[str]], List[Optional[int]], List[Optional[datetime.date]]
            ],
        ] = {
            c: list(set(d)) for c, d in domains.items()  # type: ignore
        }
        # compute_full_domain_df throws an IndexError if any list has length 0
        for v in list_domains.values():
            if len(v) == 0:
                raise ValueError("Every column should have a non-empty list of values.")
        _check_dict_schema({c: get_element_type(d) for c, d in list_domains.items()})
        return KeySet(lambda: compute_full_domain_df(list_domains))

    @classmethod
    def from_dataframe(cls: Type[KeySet], dataframe: DataFrame) -> KeySet:
        """Create a KeySet from a dataframe.

        This DataFrame should contain every combination of values being selected
        in the KeySet. If there are duplicate rows in the dataframe, only one
        copy of each will be kept.

        When creating KeySets with this method, it is the responsibility of the
        caller to ensure that the given dataframe remains valid for the lifetime
        of the KeySet. If the dataframe becomes invalid, for example because its
        Spark session is closed, this method or any uses of the resulting
        dataframe may raise exceptions or have other unanticipated effects.
        """
        return KeySet(coerce_spark_schema_or_fail(dataframe).dropDuplicates())

    def filter(self, condition: Union[Column, str]) -> KeySet:
        """Filter this KeySet using some condition.

        This method accepts the same syntax as
        :meth:`~pyspark.sql.DataFrame.filter`: valid conditions are those that
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
            >>> filtered_keyset = keyset.filter(keyset.dataframe().A != "a1")
            >>> filtered_keyset.dataframe().sort("A", "B").toPandas()
                A  B
            0  a2  0
            1  a2  1
            2  a2  2
            3  a2  3
        """
        return KeySet(self.dataframe().filter(condition))

    def __getitem__(self, columns: Union[str, Tuple[str, ...], List[str]]) -> KeySet:
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
        if isinstance(columns, str):
            columns = (columns,)
        return KeySet(self.dataframe().select(*columns).dropDuplicates())

    def __mul__(self, other: KeySet) -> KeySet:
        """A product (``KeySet * KeySet``) returns the cross-product of both KeySets.

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
        # TODO(#2107): Fix typing once Pandas has working type stubs
        self_df = self.dataframe().toPandas()
        other_df = other.dataframe().toPandas()
        if sorted(self_df.columns) != sorted(other_df.columns):  # type: ignore
            return False
        if self_df.empty and other_df.empty:  # type: ignore
            return True
        sorted_columns = sorted(self_df.columns)  # type: ignore
        self_df_sorted = (
            self_df.set_index(sorted_columns).sort_index().reset_index()  # type: ignore
        )
        other_df_sorted = (
            other_df.set_index(sorted_columns)  # type: ignore
            .sort_index()
            .reset_index()
        )
        return self_df_sorted.equals(other_df_sorted)

    def schema(self) -> Schema:
        # pylint: disable=line-too-long
        """Returns a Schema based on the KeySet.

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": [0, 1, 2, 3],
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> schema = keyset.schema()
            >>> schema # doctest: +NORMALIZE_WHITESPACE
            Schema({'A': ColumnDescriptor(column_type=ColumnType.VARCHAR, allow_null=True, allow_nan=False, allow_inf=False),
                    'B': ColumnDescriptor(column_type=ColumnType.INTEGER, allow_null=True, allow_nan=False, allow_inf=False)})
        """
        # pylint: enable=line-too-long
        return Schema(spark_schema_to_analytics_columns(self.dataframe().schema))
