"""Tests for QueryExpr."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

# pylint: disable=too-many-arguments, pointless-string-statement

import datetime
import re
from typing import Any, Callable, Dict, List, Mapping, Tuple, Type, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType
from typeguard import TypeCheckError

from tmlt.analytics import (
    AnalyticsInternalError,
    KeySet,
    QueryBuilder,
    TruncationStrategy,
)
from tmlt.analytics._query_expr import (
    DropInfinity,
    DropNullAndNan,
    Filter,
    FlatMap,
    FlatMapByID,
    GroupByBoundedAverage,
    GroupByBoundedStdev,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByQuantile,
    JoinPrivate,
    JoinPublic,
    Map,
    PrivateSource,
    QueryExpr,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
    SuppressAggregates,
)
from tmlt.analytics._schema import FrozenDict, Schema

from ..conftest import (
    GROUPBY_AGGREGATION_QUERIES,
    NON_GROUPBY_AGGREGATION_QUERIES,
    SIMPLE_TRANSFORMATION_QUERIES,
    assert_frame_equal_with_sort,
)

"""Tests for invalid attributes on dataclasses."""


@pytest.mark.parametrize(
    "invalid_source_id,exception_type,expected_error_msg",
    [
        (1001, TypeCheckError, None),
        (" ", ValueError, "source_id must be a valid Python identifier."),
        ("space present", ValueError, "source_id must be a valid Python identifier."),
        (
            "2startsWithNumber",
            ValueError,
            "source_id must be a valid Python identifier.",
        ),
    ],
)
def test_invalid_private_source(
    invalid_source_id: str, exception_type: Type[Exception], expected_error_msg: str
):
    """Tests that invalid private source errors on post-init."""
    with pytest.raises(exception_type, match=expected_error_msg):
        PrivateSource(invalid_source_id)


@pytest.mark.parametrize(
    "column_mapper",
    [
        (True,),
        ({"A": 123},),
    ],
)
def test_invalid_rename(column_mapper: Dict[str, str]):
    """Tests that invalid Rename errors on post-init."""
    with pytest.raises(TypeCheckError):
        Rename(PrivateSource("private"), FrozenDict.from_dict(column_mapper))


def test_invalid_rename_empty_string():
    """Test that rename doesn't allow you to rename columns to "" (empty string)."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Cannot rename column A to "" (the empty string): columns named ""'
            " are not allowed"
        ),
    ):
        Rename(PrivateSource("private"), FrozenDict.from_dict({"A": ""}))


def test_invalid_filter():
    """Tests that invalid Filter errors on post-init."""
    with pytest.raises(TypeCheckError):
        Filter(PrivateSource("private"), 0)  # type: ignore


@pytest.mark.parametrize(
    "columns",
    [
        (True,),
        (tuple([1]),),
        (("A", "B", "B"),),
    ],
)
def test_invalid_select(
    columns: Tuple[str, ...],
):
    """Tests that invalid Rename errors on post-init."""
    with pytest.raises((ValueError, TypeCheckError)):
        Select(PrivateSource("private"), columns)


@pytest.mark.parametrize(
    "func,schema_new_columns,augment,expected_error_msg",
    [
        (  # Invalid augument
            lambda row: {"C": 2 * str(row["B"])},
            Schema({"C": "VARCHAR"}),
            1.0,
            None,
        ),
        (  # Invalid Schema
            lambda row: {"C": 2 * str(row["B"])},
            {"C": "VARCHAR"},
            True,
            None,
        ),
        (  # Grouping column in schema
            lambda row: {"C": 2 * str(row["B"])},
            Schema({"C": "VARCHAR"}, grouping_column="C"),
            True,
            "Map cannot be be used to create grouping columns",
        ),
    ],
)
def test_invalid_map(
    func: Callable, schema_new_columns: Schema, augment: bool, expected_error_msg: str
):
    """Tests that invalid Map errors on post-init."""
    with pytest.raises((TypeCheckError, ValueError), match=expected_error_msg):
        Map(PrivateSource("private"), func, schema_new_columns, augment)


@pytest.mark.parametrize(
    "child,func,max_rows,schema_new_columns,augment,expected_error_msg",
    [
        (  # Invalid max_rows
            PrivateSource("private"),
            lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            -1,
            Schema({"i": "INTEGER"}),
            False,
            "Limit on number of rows '-1' must be non-negative.",
        ),
        (  # Invalid augment
            FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
                schema_new_columns=Schema({"Repeat": "INTEGER"}),
                augment=True,
                max_rows=1,
            ),
            lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            2,
            Schema({"i": "INTEGER"}),
            1.0,
            None,
        ),
        (  # Invalid grouping result
            PrivateSource("private"),
            lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            2,
            Schema({"i": "INTEGER", "j": "INTEGER"}, grouping_column="i"),
            False,
            (
                "schema_new_columns contains 2 columns, "
                "grouping flat map can only result in 1 new column"
            ),
        ),
    ],
)
def test_invalid_flatmap(
    child: QueryExpr,
    func: Callable,
    max_rows: int,
    schema_new_columns: Schema,
    augment: bool,
    expected_error_msg: str,
):
    """Tests that invalid FlatMap errors on post-init."""
    with pytest.raises((TypeCheckError, ValueError), match=expected_error_msg):
        FlatMap(child, func, schema_new_columns, augment, max_rows)


@pytest.mark.parametrize(
    "schema_new_columns,expected_exc",
    [
        (Schema({"i": "INTEGER"}, grouping_column="i"), AnalyticsInternalError),
        (Schema({"i": "INTEGER"}, id_column="i"), AnalyticsInternalError),
    ],
)
def test_invalid_flat_map_by_id(
    schema_new_columns: Schema, expected_exc: Type[Exception]
):
    """FlatMapByID raises an exception when given invalid parameters."""
    with pytest.raises(expected_exc):
        FlatMapByID(
            PrivateSource("private"),
            lambda rows: rows,
            schema_new_columns=schema_new_columns,
        )


@pytest.mark.parametrize(
    "join_columns,expected_error_msg",
    [
        ([], "Provided join columns must not be empty"),
        (["A", "A"], "Join columns must be distinct"),
    ],
)
def test_invalid_join_columns(join_columns: List[str], expected_error_msg: str):
    """Tests that JoinPrivate, JoinPublic error with invalid join columns."""
    with pytest.raises(ValueError, match=expected_error_msg):
        JoinPrivate(
            PrivateSource("private"),
            PrivateSource("private2"),
            TruncationStrategy.DropExcess(1),
            TruncationStrategy.DropExcess(1),
            tuple(join_columns),
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            JoinPublic(PrivateSource("private"), "public", tuple(join_columns))


def test_invalid_how():
    """Tests that JoinPublic, JoinPublic error with invalid how."""
    with pytest.raises(
        ValueError, match="Invalid join type 'invalid': must be 'inner' or 'left'"
    ):
        JoinPublic(PrivateSource("private"), "public", tuple("A"), how="invalid")


@pytest.mark.parametrize(
    "replace_with,expected_error_msg",
    [
        ({"str": 100.0}, None),
        ({"str": [100.0, 100.0]}, None),
        ([], None),
        (
            {"A": (-100.0,)},
            re.escape("'A'"),
        ),
    ],
)
def test_invalid_replace_infinity(replace_with: Any, expected_error_msg: str) -> None:
    """Test ReplaceInfinity with invalid arguments."""
    with pytest.raises((TypeCheckError), match=expected_error_msg):
        QueryBuilder("private").replace_infinity(replace_with)


@pytest.mark.parametrize(
    "columns",
    [
        "A",
        ["A", "B"],
        tuple([1]),
    ],
)
def test_invalid_drop_null_and_nan(columns: Any) -> None:
    """Test DropNullAndNan with invalid arguments."""
    with pytest.raises(TypeCheckError):
        DropNullAndNan(PrivateSource("private"), columns)


@pytest.mark.parametrize(
    "columns",
    [
        "A",
        ["A", "B"],
        tuple([1]),
    ],
)
def test_invalid_drop_infinity(columns: Any) -> None:
    """Test DropInfinity with invalid arguments."""
    with pytest.raises(TypeCheckError):
        DropInfinity(PrivateSource("private"), columns)


@pytest.mark.parametrize(
    "child,keys,output_column",
    [
        (
            PrivateSource("private"),
            KeySet.from_dict({}),
            123,
        )
    ],
)
def test_invalid_groupbycount(child: QueryExpr, keys: KeySet, output_column: str):
    """Tests that invalid GroupByCount errors on post-init."""
    with pytest.raises(TypeCheckError):
        GroupByCount(child, keys, output_column)


@pytest.mark.parametrize(
    "keys,measure_column,low,high,expected_error_msg",
    [
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            "1.0",
            10.0,
            None,
        ),
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            10.0,
            1,
            "Lower bound '10.0' must be less than the upper bound '1.0'.",
        ),
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            1.0,
            1.0,
            "Lower bound '1.0' must be less than the upper bound '1.0'.",
        ),
    ],
)
def test_invalid_groupbyagg(
    keys: KeySet, measure_column: str, low: float, high: float, expected_error_msg: str
):
    """Test invalid GroupBy aggregates errors on post-init."""
    for DataClass in [
        GroupByBoundedSum,
        GroupByBoundedAverage,
        GroupByBoundedVariance,
        GroupByBoundedStdev,
    ]:
        with pytest.raises((TypeCheckError, ValueError), match=expected_error_msg):
            DataClass(PrivateSource("private"), keys, measure_column, low, high)


@pytest.mark.parametrize(
    "keys,measure_column,quantile,low,high,expected_error_msg",
    [
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            "0",
            8.0,
            10.0,
            None,
        ),
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            -1,
            8.0,
            10.0,
            "Quantile must be between 0 and 1, and not ",
        ),
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            1.1,
            8.0,
            10.0,
            "Quantile must be between 0 and 1, and not ",
        ),
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            0.5,
            "1.0",
            10.0,
            None,
        ),
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            0.5,
            10.0,
            1.0,
            "Lower bound '10.0' must be less than the upper bound '1.0'.",
        ),
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            0.5,
            1.0,
            1.0,
            "Lower bound '1.0' must be less than the upper bound '1.0'.",
        ),
    ],
)
def test_invalid_groupbyquantile(
    keys: KeySet,
    measure_column: str,
    quantile: float,
    low: float,
    high: float,
    expected_error_msg: str,
):
    """Test invalid GroupByQuantile."""
    with pytest.raises((TypeCheckError, ValueError), match=expected_error_msg):
        GroupByQuantile(
            PrivateSource("private"), keys, measure_column, quantile, low, high
        )


"""Tests for valid attributes on dataclasses."""


@pytest.mark.parametrize("source_id", ["private_source", "_Private", "no_space2"])
def test_valid_private_source(source_id: str):
    """Tests valid private source does not error."""
    PrivateSource(source_id)


@pytest.mark.parametrize("low,high", [(8.0, 10.0), (1, 10), (1.0, 10)])
def test_clamping_bounds_casting(low: float, high: float):
    """Test type of clamping bounds match on post-init."""
    for DataClass in [
        GroupByBoundedSum,
        GroupByBoundedAverage,
        GroupByBoundedVariance,
        GroupByBoundedStdev,
    ]:
        query = DataClass(
            PrivateSource("private"),
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            low,
            high,
        )
        assert isinstance(
            query,
            (
                GroupByBoundedSum,
                GroupByBoundedAverage,
                GroupByBoundedVariance,
                GroupByBoundedStdev,
            ),
        )
        assert type(query.low) == type(query.high)


@pytest.mark.parametrize(
    "child,replace_with",
    [
        (PrivateSource("private"), {"col": "value", "col2": "value2"}),
        (PrivateSource("private"), {}),
        (
            PrivateSource("private"),
            {
                "A": 1,
                "B": 2.0,
                "C": "c1",
                "D": datetime.date(2020, 1, 1),
                "E": datetime.datetime(2020, 1, 1),
            },
        ),
    ],
)
def test_valid_replace_null_and_nan(
    child: QueryExpr,
    replace_with: Mapping[
        str, Union[int, float, str, datetime.date, datetime.datetime]
    ],
):
    """Test ReplaceNullAndNan creation with valid values."""
    ReplaceNullAndNan(child, FrozenDict.from_dict(replace_with))


@pytest.mark.parametrize(
    "child,replace_with",
    [
        (PrivateSource("private"), {}),
        (PrivateSource("private"), {"A": (-100.0, 100.0)}),
        (PrivateSource("private"), {"A": (-999.9, 999.9), "B": (123.45, 678.90)}),
    ],
)
def test_valid_replace_infinity(
    child: QueryExpr, replace_with: Dict[str, Tuple[float, float]]
) -> None:
    """Test ReplaceInfinity with valid values."""
    query = ReplaceInfinity(child, FrozenDict.from_dict(replace_with))
    for v in query.replace_with.values():
        # Check that values got converted to floats
        assert len(v) == 2
        assert isinstance(v[0], float)
        assert isinstance(v[1], float)


@pytest.mark.parametrize(
    "child,columns",
    [
        (PrivateSource("private"), []),
        (PrivateSource("private"), ["A"]),
        (PrivateSource("different_private_source"), ["A", "B"]),
    ],
)
def test_valid_drop_null_and_nan(child: QueryExpr, columns: List[str]) -> None:
    """Test DropNullAndNan with valid values."""
    DropInfinity(child, tuple(columns))


@pytest.mark.parametrize(
    "child,columns",
    [
        (PrivateSource("private"), []),
        (PrivateSource("private"), ["A"]),
        (PrivateSource("different_private_source"), ["A", "B"]),
    ],
)
def test_valid_drop_infinity(child: QueryExpr, columns: List[str]) -> None:
    """Test DropInfinity with valid values."""
    DropInfinity(child, tuple(columns))


"""Tests for JoinPublic with a Spark DataFrame as the public table."""


def test_join_public_string_nan(spark):
    """Test that the string "NaN" is allowed in string-valued columns."""
    df = spark.createDataFrame(pd.DataFrame({"col": ["nan", "NaN", "NAN", "Nan"]}))
    query_expr = JoinPublic(PrivateSource("a"), df)
    assert isinstance(query_expr.public_table, DataFrame)
    assert_frame_equal_with_sort(query_expr.public_table.toPandas(), df.toPandas())


def test_join_public_dataframe_validation_column_type(spark):
    """Unsupported column types are rejected in JoinPublic."""
    data = [{"bytes": b"some bytes"}]
    schema = StructType([StructField("bytes", BinaryType(), False)])
    df = spark.createDataFrame(data, schema)

    with pytest.raises(ValueError, match="^Unsupported Spark data type.*"):
        JoinPublic(PrivateSource("a"), df)


@pytest.mark.parametrize(
    "child,column,threshold,expected_error_msg",
    [
        (
            PrivateSource("P"),
            "count",
            0,
            "SuppressAggregates is only supported on aggregates that are GroupByCounts",
        ),
        (
            GroupByCount(PrivateSource("P"), KeySet.from_dict({})),
            -17,
            0,
            None,
        ),
        (
            GroupByCount(PrivateSource("P"), KeySet.from_dict({})),
            "count",
            "not an int",
            None,
        ),
    ],
)
def test_invalid_suppress_aggregates(
    spark: SparkSession,  # pylint: disable=unused-argument
    child: GroupByCount,
    column: str,
    threshold: int,
    expected_error_msg: str,
) -> None:
    """Test that SuppressAggregates rejects invalid arguments."""
    with pytest.raises((TypeError, TypeCheckError), match=expected_error_msg):
        SuppressAggregates(child, column, threshold)


@pytest.mark.parametrize(
    "queryexpr",
    SIMPLE_TRANSFORMATION_QUERIES
    + NON_GROUPBY_AGGREGATION_QUERIES
    + [
        func2(func1)
        for func1 in SIMPLE_TRANSFORMATION_QUERIES
        for func2 in GROUPBY_AGGREGATION_QUERIES
    ],
)
def test_queryexpr_hashing(queryexpr):
    """Tests that each query expression has enabled hashing and eq."""
    test_dict = {queryexpr: 1}
    assert test_dict[queryexpr] == 1
    assert queryexpr == queryexpr  # pylint: disable=comparison-with-itself
