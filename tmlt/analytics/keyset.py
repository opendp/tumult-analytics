"""A KeySet specifies a list of values for one or more columns.

They are used as input to the
:meth:`~tmlt.analytics.query_builder.QueryBuilder.groupby` method to build
group-by queries. An introduction to KeySets can be found in the
:ref:`Group-by queries` tutorial.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from copy import copy
from functools import partial, reduce
from typing import (
    Any,
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

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import types as spark_types
from tmlt.core.transformations.spark_transformations.groupby import (
    compute_full_domain_df,
)
from tmlt.core.utils.type_utils import get_element_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import ColumnDescriptor, spark_schema_to_analytics_columns
from tmlt.analytics._utils import dataframe_is_empty


def _check_df_schema(types: spark_types.StructType):
    """Raises an exception if any of the given types are not allowed in a KeySet."""
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


def _check_dict_schema(types: Dict[str, type]):
    """Raises an exception if the dict contains a type not allowed in a KeySet."""
    allowed_types = {int, str, datetime.date}
    for col, dtype in types.items():
        if dtype not in allowed_types:
            raise ValueError(
                f"Column {col} has type {dtype.__qualname__}, which is "
                "not allowed in KeySets. Allowed column types are: "
                f"{','.join(t.__qualname__ for t in allowed_types)}"
            )


def _check_tuples_schema(
    tuples: List[Tuple[Optional[Union[str, int, datetime.date]], ...]],
    columns: Tuple[str, ...],
):
    """Raises an exception if the tuples schema is faulty.

    More specifically, raises an exception of a tuple contains a a type not
    allowed in a KeySet, if the size or types of the tuples are inconsistent, or
    if a column only has None values (which would prevent Spark from inferring
    its type).
    """
    allowed_types = {int, str, datetime.date}
    num_columns = len(columns)
    expected_types: List[Optional[type]] = [None for _ in columns]
    for t in tuples:
        if len(t) != num_columns:
            raise ValueError(
                f"Mismatch between tuple {t}, which has {len(t)} elements, and "
                f"columns argument {columns}, which has {num_columns}."
            )
        for i, elt in enumerate(t):
            if elt is None:
                continue
            dtype = type(elt)
            if dtype not in allowed_types:
                raise ValueError(
                    f"Element {elt} of tuple {t} has type {dtype.__qualname__}, "
                    "which is not allowed in KeySets. Allowed column types are: "
                    f"{','.join(t.__qualname__ for t in allowed_types)}."
                )
            expected = expected_types[i]
            if expected is None:
                expected_types[i] = dtype
            elif expected != dtype:
                raise ValueError(
                    f"Element {elt} of tuple {t} (for column '{columns[i]}') "
                    f"has type {dtype.__qualname__}, expected type "
                    f"{expected.__qualname__} from a previous tuple."
                )
    for i, expected in enumerate(expected_types):
        if expected is None:
            raise ValueError(
                f"Could not infer type for column '{columns[i]}': all its "
                "values are None."
            )


LOW_SIZE = 10**6
"""An arbitrary threshold below which a KeySet is considered small."""


class KeySet(ABC):
    """A class containing a set of values for specific columns.

       An introduction to KeySet initialization and manipulation can be found in
       the :ref:`Group-by queries` tutorial.

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
        # Manage case of len(0) KeySet by creating a KeySet with an empty DataFrame.
        if len(domains) == 0:
            return _MaterializedKeySet(
                SparkSession.builder.getOrCreate().createDataFrame(
                    [], spark_types.StructType([])
                ),
                size=1,
            )

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
        for k, v in list_domains.items():
            if not v:
                raise ValueError(f"Column {k} has an empty list of values.")
        _check_dict_schema({c: get_element_type(d) for c, d in list_domains.items()})
        discrete_keysets: List[_MaterializedKeySet] = []
        for col_name, col_values in list_domains.items():
            # functools.partial will "freeze" the arguments in their current state
            # if we don't use functools.partial, every keyset will use the same
            # dictionary corresponding to the last column iterated over.
            func = partial(
                compute_full_domain_df,
                column_domains={col_name: col_values},
            )
            keyset = _MaterializedKeySet(func, size=len(col_values))
            discrete_keysets.append(keyset)

        # Avoids making a _ProductKeySet if the list of factors only contains one
        # element. Context:
        # https://gitlab.com/tumult-labs/analytics-upstream/-/merge_requests/342#note_2134174669
        if len(discrete_keysets) == 1:
            return discrete_keysets[0]
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
        if not isinstance(dataframe, DataFrame):
            raise ValueError(
                "Expected a Spark DataFrame, but got "
                f"{type(dataframe).__module__}.{type(dataframe).__name__}"
            )
        return _MaterializedKeySet(
            coerce_spark_schema_or_fail(dataframe).dropDuplicates()
        )

    @classmethod
    def from_tuples(
        cls: Type[KeySet],
        tuples: List[Tuple[Optional[Union[str, int, datetime.date]], ...]],
        columns: Tuple[str, ...],
    ) -> KeySet:
        """Create a KeySet from a list of tuples and column names.

        Example:
            >>> tuples = [
            ...   ("a1", "b1"),
            ...   ("a2", "b1"),
            ...   ("a3", "b3"),
            ... ]
            >>> keyset = KeySet.from_tuples(tuples, ("A", "B"))
            >>> keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a2  b1
            2  a3  b3
        """
        if not tuples:
            # Initializing an empty KeySet with no columns is allowed.
            if not columns:
                return KeySet.from_dict({})
            # Initializing an empty KeySet with columns is forbidden, as we're
            # missing type information for columns.
            raise ValueError(
                "Cannot initialize a KeySet using from_tuples with no tuples "
                "and non-zero columns. If you want to create an empty KeySet, "
                "use from_dataframe with an empty DataFrame instead."
            )
        _check_tuples_schema(tuples, columns)
        spark = SparkSession.builder.getOrCreate()
        df = spark.createDataFrame(tuples, schema=columns)
        return KeySet.from_dataframe(df)

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
    def _fast_equality_check(self, other: Any) -> bool:
        """Checks KeySets for equality without performing expensive DataFrame checks.

        .. note::
            This method ensures a quick evaluation of KeySet equality. It is guaranteed
            to return False if the KeySets are not equal, but there may be cases where
            it returns False for equal KeySets.
        """

    @abstractmethod
    def schema(self) -> Dict[str, ColumnDescriptor]:
        # pylint: disable=line-too-long
        """Returns the KeySet's schema.

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": [0, 1, 2, 3],
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> schema = keyset.schema()
            >>> schema # doctest: +NORMALIZE_WHITESPACE
            {'A': ColumnDescriptor(column_type=ColumnType.VARCHAR, allow_null=True, allow_nan=False, allow_inf=False),
             'B': ColumnDescriptor(column_type=ColumnType.INTEGER, allow_null=True, allow_nan=False, allow_inf=False)}
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

    def __hash__(self) -> int:
        """Hashes a KeySet on the underlying DataFrame's schema."""
        return hash(self.dataframe().schema)


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
        size: Optional[int] = None,
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
        self._schema: Optional[Dict[str, ColumnDescriptor]] = None
        self._size = size

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

    def _fast_equality_check(self, other: Any):
        """Checks KeySets for equality without performing expensive DataFrame checks.

        .. note::
            This method ensures a quick evaluation of KeySet equality. It is guaranteed
            to return False if the KeySets are not equal, but there may be cases where
            it returns False for equal KeySets.
        """
        if other is self:
            return True

        # Any KeySet with no columns equals any other keyset with no columns.
        if len(self.columns()) == 0 and len(other.columns()) == 0:
            return True

        if self.columns() != other.columns():
            return False

        if self.dataframe().schema != other.dataframe().schema:
            return False

        if self.dataframe().sameSemantics(other.dataframe()):
            return True

        # pylint: disable=protected-access
        if self._size is not None and other._size is not None:
            self_size = self._size
            other_size = other._size
            # pylint: enable=protected-access

            if self_size != other_size:
                return False
            elif self_size < LOW_SIZE:
                # This comparison is required because KeySets produced by from_dict({})
                # aren't equal according to sameSemantics. This ensures we can still
                # successfully compare small KeySets defined inline. The size threshold
                # ensures the check remains fast.
                return (
                    self.dataframe()  # type: ignore
                    .toPandas()
                    .equals(other.dataframe().toPandas())
                )
        return False

    def schema(self) -> Dict[str, ColumnDescriptor]:
        """Returns a Schema based on the KeySet."""
        if self._schema is not None:
            return self._schema
        self._schema = spark_schema_to_analytics_columns(self.dataframe().schema)
        return self._schema

    def size(self) -> int:
        """Get the size of this KeySet."""
        if self._size is not None:
            return self._size
        self._size = self.dataframe().count()
        if self._size == 0 and len(self.columns()) == 0:
            # A query with a KeySet of 0 cols and 0 rows returns a total for the query.
            # This total query has on output, so the KeySet size is set to 1.
            self._size = 1
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
        self._schema: Optional[Dict[str, ColumnDescriptor]] = None
        factors_size: Optional[int] = None
        if all(factor._size is not None for factor in self._factors):
            factors_size = 1
            for factor in self._factors:
                if factor._size is None:  # appease mypy
                    raise AnalyticsInternalError("KeySet size is None")
                factors_size *= factor._size
        self._size = factors_size

    def schema(self) -> Dict[str, ColumnDescriptor]:
        """Return a Schema for this KeySet."""
        if self._schema is not None:
            return self._schema
        analytics_columns: Dict[str, ColumnDescriptor] = {}
        for factor in self._factors:
            factor_schema = factor.schema()
            for col_name in factor_schema:
                analytics_columns[col_name] = factor_schema[col_name]
        self._schema = analytics_columns
        return self._schema

    def columns(self) -> List[str]:
        """Return the list of columns used by this KeySet."""
        return copy(self._columns)

    # pylint: disable=line-too-long
    def __getitem__(
        self, desired_columns: Union[str, Tuple[str, ...], Sequence[str]]
    ) -> KeySet:
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
            if set(keyset.columns()) <= desired_column_set:
                new_factors.append(keyset)
            else:
                applicable_columns = tuple(set(keyset.columns()) & desired_column_set)
                new_factors.append(keyset[applicable_columns])
        # Avoids making a _ProductKeySet if the list of factors only contains
        # one element. Context:
        # https://gitlab.com/tumult-labs/analytics-upstream/-/merge_requests/342#note_2134174669
        if len(new_factors) == 1:
            return new_factors[0]
        return _ProductKeySet(new_factors, list(desired_columns))

    def _fast_equality_check(self, other: Any):
        """Checks KeySets for equality without performing expensive DataFrame checks.

        .. note::
            This method ensures a quick evaluation of KeySet equality. It is guaranteed
            to return False if the KeySets are not equal, but there may be cases where
            it returns False for equal KeySets.
        """
        if other is self:
            return True

        # pylint: disable=protected-access
        # For product keysets if someone passes a materialized keyset,
        # call materialized keyset's fast equality check.
        # Context: https://gitlab.com/tumult-labs/analytics-upstream/-/merge_requests/342#note_2140238892
        if isinstance(other, _MaterializedKeySet):
            return other._fast_equality_check(self)

        # Any KeySet with no columns equals any other keyset with no columns.
        if len(self.columns()) == 0 and len(other.columns()) == 0:
            return True

        if self.columns() != other.columns():
            return False

        if len(self._factors) != len(other._factors):
            return False

        # Check if all factors are equal. This returns false if column order is different
        return all(
            self_factor._fast_equality_check(other_factor)
            for self_factor, other_factor in zip(self._factors, other._factors)
        )
        # pylint: enable=protected-access

    def filter(self, condition: Union[Column, str]) -> KeySet:
        """Filter this KeySet using some condition."""
        df = self.dataframe()
        return _MaterializedKeySet(df).filter(condition)

    def dataframe(self) -> DataFrame:
        """Get the dataframe corresponding to this KeySet."""
        if self._dataframe is not None:
            return self._dataframe
        # Use Spark to join together all results if the final dataframe is very large or we don't know the size
        if self._size is None or self._size > LOW_SIZE:
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
