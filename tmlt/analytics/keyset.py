"""A KeySet specifies a list of values for one or more columns.

For example, a KeySet could specify the values ``["a1", "a2"]`` for column A
and the values ``[0, 1, 2, 3]`` for column B.

Currently, KeySets are used as a simpler way to specify domains for groupby
transformations.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from copy import copy
from functools import partial, reduce
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as spark_types
from tmlt.core.transformations.spark_transformations.groupby import (
    compute_full_domain_df,
)
from tmlt.core.utils.type_utils import get_element_type

from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import (
    ColumnDescriptor,
    Schema,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics._utils import dataframe_is_empty


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


class KeySet(ABC):
    """A class containing a set of values for specific columns.

    .. warning::
        If a column has null values dropped or replaced, then Analytics
        will raise an error if you use a KeySet that contains a null value for
        that column.

    .. note::
        The :meth:`~.KeySet.from_dict` and :meth:`~.KeySet.from_dataframe` methods
        are the preferred way to construct KeySets. Directly constructing KeySets
        skips checks that guarantee the uniqueness of output rows, and ``__init__``
        methods are not guaranteed to work the same way between releases.
    """

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
        discrete_keysets: List[_MaterializedKeySet] = []
        for col_name, col_values in list_domains.items():
            # functools.partial will "freeze" the arguments in their current state
            # if you don't use functools.partial,
            # every keyset will use the same dictionary
            # corresponding to the last column iterated over
            func = partial(
                compute_full_domain_df, column_domains={col_name: col_values}
            )
            keyset = _MaterializedKeySet(func)
            discrete_keysets.append(keyset)
        return _ProductKeySet(discrete_keysets, list(domains.keys()))

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
        return _MaterializedKeySet(
            coerce_spark_schema_or_fail(dataframe).dropDuplicates()
        )

    @abstractmethod
    def dataframe(self) -> DataFrame:
        """Return the dataframe associated with this KeySet.

        This dataframe contains every combination of values being selected in
        the KeySet, and its rows are guaranteed to be unique as long as the
        KeySet was constructed safely.
        """

    @abstractmethod
    def __getitem__(self, columns: Union[str, Tuple[str], Sequence[str]]) -> KeySet:
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
        self_df = self.dataframe()
        other_df = other.dataframe()
        if sorted(self_df.columns) != sorted(other_df.columns):
            return False
        # Re-select the columns so that both dataframes have columns
        # in the same order
        self_df = self_df.select(sorted(self_df.columns))
        other_df = other_df.select(sorted(other_df.columns))
        if self_df.schema != other_df.schema:
            return False
        # other_df should contain all rows in self_df
        if self_df.exceptAll(other_df).count() != 0:
            return False
        # and vice versa
        if other_df.exceptAll(self_df).count() != 0:
            return False
        return True

    @abstractmethod
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
        return _ProductKeySet([self, other], self.columns() + other.columns())

    @abstractmethod
    def columns(self) -> List[str]:
        """Return the list of columns used by this KeySet."""

    @abstractmethod
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

    @abstractmethod
    def size(self) -> int:
        """Get the size of this KeySet."""

    def cache(self) -> None:
        """Caches the KeySet's dataframe in memory."""
        self.dataframe().cache()

    def uncache(self) -> None:
        """Removes the KeySet's dataframe from memory and disk."""
        self.dataframe().unpersist()


class _MaterializedKeySet(KeySet):
    """A class containing a set of values for specific columns.

    .. warning::
        If a column has null values dropped or replaced, then Analytics
        will raise an error if you use a KeySet that contains a null value for
        that column.
    """

    # Passing a function to __init__ allows you to construct a keyset
    # without creating the relevant Spark DataFrame right away.
    # This is useful for tests - as you can construct a list of test parameters
    # without having a Spark context yet - but should be avoided when possible.
    def __init__(
        self,
        dataframe: Union[DataFrame, Callable[[], DataFrame]],
    ) -> None:
        """Construct a new keyset.

        .. warning::
            The :meth:`from_dict` and :meth:`from_dataframe` methods are preferred
            over directly using the constructor to create new KeySets. Directly
            constructing KeySets skips checks that guarantee the uniqueness of
            output rows.
        """
        self._dataframe: Union[DataFrame, Callable[[], DataFrame]]
        if isinstance(dataframe, DataFrame):
            self._dataframe = coerce_spark_schema_or_fail(dataframe)
            self._columns: Optional[List[str]] = self._dataframe.columns
            _check_df_schema(self._dataframe.schema)
        else:
            self._dataframe = dataframe
            self._columns = None
        self._schema: Optional[Schema] = None
        self._size: Optional[int] = None

    def dataframe(self) -> DataFrame:
        """Return the dataframe associated with this KeySet."""
        if callable(self._dataframe):
            self._dataframe = coerce_spark_schema_or_fail(self._dataframe())
            # Invalid column types should get caught before this, as it keeps
            # the exception closer to the user code that caused it, but in case
            # that is missed we check again here.
            _check_df_schema(self._dataframe.schema)
        return self._dataframe

    def columns(self) -> List[str]:
        """What columns are used by this KeySet."""
        if self._columns is not None:
            return copy(self._columns)
        else:
            self._columns = self.dataframe().columns
            return self._columns

    def filter(self, condition: Union[Column, str]) -> KeySet:
        """Filter this KeySet using some condition."""
        return _MaterializedKeySet(self.dataframe().filter(condition))

    def __getitem__(
        self, columns: Union[str, Tuple[str, ...], Sequence[str]]
    ) -> KeySet:
        """``KeySet[col, col, ...]`` returns a KeySet with those columns only."""
        if isinstance(columns, str):
            columns = (columns,)
        if len(set(columns)) != len(columns):
            raise ValueError(
                f"Cannot select columns {columns} "
                "because duplicate columns were present"
            )
        return _MaterializedKeySet(self.dataframe().select(*columns).dropDuplicates())

    def schema(self) -> Schema:
        """Returns a Schema based on the KeySet."""
        if self._schema is not None:
            return self._schema
        self._schema = Schema(
            spark_schema_to_analytics_columns(self.dataframe().schema)
        )
        return self._schema

    def size(self) -> int:
        """Get the size of this KeySet."""
        if self._size is not None:
            return self._size
        self._size = self.dataframe().count()
        return self._size


class _ProductKeySet(KeySet):
    """A KeySet that is the product of a list of other KeySets."""

    def __init__(self, factors: Sequence[KeySet], column_order: List[str]):
        """Create a Product KeySet from a list of other KeySets.

        .. warning::
            The :meth:`from_dict` and :meth:`from_dataframe` methods are preferred
            over directly using the constructor to create new KeySets. Directly
            constructing KeySets skips checks that guarantee the uniqueness of
            output rows.
        """
        discrete_factors: List[_MaterializedKeySet] = []
        for factor in factors:
            if isinstance(factor, _ProductKeySet):
                for sub_factor in factor._factors:
                    discrete_factors.append(sub_factor)
            elif isinstance(factor, _MaterializedKeySet):
                discrete_factors.append(factor)
            else:
                df = factor.dataframe()
                discrete_factors.append(_MaterializedKeySet(df))
        self._factors: List[_MaterializedKeySet] = []
        columns: set[str] = set()
        for factor in discrete_factors:
            for col in factor.columns():
                if col in columns:
                    raise ValueError(
                        "Cannot multiply keysets together because "
                        f"they share a column: {col}"
                    )
                if col not in column_order:
                    raise ValueError(
                        f"Specified column ordering {column_order} "
                        f"does not contain column {col}"
                    )
                columns.add(col)
            self._factors.append(factor)
        self._columns: List[str] = column_order
        self._dataframe: Optional[DataFrame] = None
        self._schema: Optional[Schema] = None
        self._size: Optional[int] = None

    def schema(self) -> Schema:
        """Return a Schema for this KeySet."""
        if self._schema is not None:
            return self._schema
        analytics_columns: Dict[str, ColumnDescriptor] = {}
        for factor in self._factors:
            factor_schema = factor.schema()
            for col_name in factor_schema.columns:
                analytics_columns[col_name] = factor_schema[col_name]
        self._schema = Schema(analytics_columns)
        return self._schema

    def columns(self) -> List[str]:
        """Return the list of columns used by this KeySet."""
        return copy(self._columns)

    # pylint: disable=line-too-long
    def __getitem__(
        self, desired_columns: Union[str, Tuple[str, ...], Sequence[str]]
    ) -> _ProductKeySet:
        """``_ProductKeySet[col, col, ...]`` returns a KeySet with those columns only."""
        # pylint: enable=line-too-long
        if isinstance(desired_columns, str):
            desired_columns = [desired_columns]
        desired_column_set = set(desired_columns)
        if len(desired_column_set) != len(desired_columns):
            raise ValueError(
                f"Cannot select columns {desired_columns} "
                "because duplicate columns were present"
            )
        if any((col not in self.columns() for col in desired_column_set)):
            missing_cols = desired_column_set - set(self.columns())
            raise ValueError(
                f"Cannot select columns {missing_cols} "
                "because those columns are not in this KeySet"
            )

        new_factors: List[KeySet] = []
        for keyset in self._factors:
            if set(keyset.columns()).isdisjoint(desired_column_set):
                continue
            if set(keyset.columns()) < desired_column_set:
                new_factors.append(keyset)
            else:
                applicable_columns = tuple(set(keyset.columns()) & desired_column_set)
                new_factors.append(keyset[applicable_columns])
        return _ProductKeySet(new_factors, list(desired_columns))

    def filter(self, condition: Union[Column, str]) -> KeySet:
        """Filter this KeySet using some condition."""
        df = self.dataframe()
        return _MaterializedKeySet(df).filter(condition)

    def dataframe(self) -> DataFrame:
        """Get the dataframe corresponding to this KeySet."""
        if self._dataframe is not None:
            return self._dataframe
        # Use Spark to join together all results if the final dataframe is very large
        if self.size() > 10**6:
            dataframe = reduce(
                lambda acc, df: acc.crossJoin(df),
                [factor.dataframe() for factor in self._factors],
            )
            self._dataframe = dataframe.select(self._columns)
            return self._dataframe

        # Get combined domains of all *single-column* keysets
        column_domains: Dict[
            str,
            Union[
                List[str],
                List[Optional[str]],
                List[int],
                List[Optional[int]],
                List[datetime.date],
                List[Optional[datetime.date]],
            ],
        ] = {}
        # If a factor has *multiple columns*, you can't just add it to column_domains
        # For example:
        # pd.DataFrame({"A": [1,2], "B": ["a", "b"]}) x pd.DataFrame({"C": [9]})
        # should produce
        # pd.DataFrame({"A": [1,2], "B": ["a", "b"], "C": [9, 9]})
        # So instead, we keep track of factors that contain multiple columns
        # and do the cross-product later
        multi_column_factors: List[_MaterializedKeySet] = []
        for keyset in self._factors:
            if len(keyset.columns()) > 1:
                multi_column_factors.append(keyset)
                continue
            df = keyset.dataframe()
            for col in df.columns:
                domain_values = df.agg(sf.collect_list(col)).collect()[0][0]
                # Workaround because collect_list doesn't put nulls in the output list
                if not dataframe_is_empty(df.where(sf.col(col).isNull())):
                    domain_values.append(None)
                column_domains[col] = domain_values
        dataframe = compute_full_domain_df(column_domains)
        for keyset in multi_column_factors:
            if dataframe_is_empty(dataframe):
                dataframe = keyset.dataframe()
            else:
                dataframe = dataframe.crossJoin(keyset.dataframe())
        self._dataframe = dataframe.select(self._columns)
        return self._dataframe

    def size(self) -> int:
        """Get the size of this KeySet.

        Note: A KeySet with an empty DataFrame will have a size of 1 because queries
        with an empty KeySet will return one row with the query applied to the total
        dataset.
        """
        if self._size is not None:
            return self._size
        self._size = reduce(lambda acc, keyset: acc * keyset.size(), self._factors, 1)
        return self._size
