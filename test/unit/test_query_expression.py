"""Tests for QueryExpr."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=too-many-arguments, unidiomatic-typecheck, pointless-string-statement

import datetime
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, StructField, StructType

from tmlt.analytics._schema import Schema
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_expr import (
    DropInfinity,
    DropNullAndNan,
    Filter,
    FlatMap,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
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
)
from tmlt.analytics.truncation_strategy import TruncationStrategy

from ..conftest import assert_frame_equal_with_sort

"""Tests for invalid attributes on dataclasses."""


@pytest.mark.parametrize(
    "invalid_source_id,exception_type,expected_error_msg",
    [
        (1001, TypeError, "type of source_id must be str; got int instead"),
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
    "column_mapper,expected_error_msg",
    [
        (True, "type of column_mapper must be a dict; got bool instead"),
        ({"A": 123}, r"type of column_mapper\['A'] must be str; got int instead"),
    ],
)
def test_invalid_rename(column_mapper: Dict[str, str], expected_error_msg: str):
    """Tests that invalid Rename errors on post-init."""
    with pytest.raises(TypeError, match=expected_error_msg):
        Rename(PrivateSource("private"), column_mapper)


def test_invalid_rename_empty_string():
    """Test that rename doesn't allow you to rename columns to "" (empty string)."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Cannot rename column A to "" (the empty string): columns named ""'
            " are not allowed"
        ),
    ):
        Rename(PrivateSource("private"), {"A": ""})


def test_invalid_filter():
    """Tests that invalid Filter errors on post-init."""
    with pytest.raises(TypeError):
        Filter(PrivateSource("private"), 0)  # type: ignore


@pytest.mark.parametrize(
    "columns, expected_error_msg",
    [
        (True, "type of columns must be a list; got bool instead"),
        ([1], "type of columns[0] must be str; got int instead"),
        (["A", "B", "B"], "Column name appears more than once in ['A', 'B', 'B']"),
    ],
)
def test_invalid_select(columns: List[str], expected_error_msg: str):
    """Tests that invalid Rename errors on post-init."""
    with pytest.raises((ValueError, TypeError), match=re.escape(expected_error_msg)):
        Select(PrivateSource("private"), columns)


@pytest.mark.parametrize(
    "func,schema_new_columns,augment,expected_error_msg",
    [
        (  # Invalid augument
            lambda row: {"C": 2 * str(row["B"])},
            Schema({"C": "VARCHAR"}),
            1.0,
            "type of augment must be bool; got float instead",
        ),
        (  # Invalid Schema
            lambda row: {"C": 2 * str(row["B"])},
            {"C": "VARCHAR"},
            True,
            (
                "type of schema_new_columns must be "
                "tmlt.analytics._schema.Schema; got dict instead"
            ),
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
    with pytest.raises((TypeError, ValueError), match=expected_error_msg):
        Map(PrivateSource("private"), func, schema_new_columns, augment)


@pytest.mark.parametrize(
    "child,func,max_num_rows,schema_new_columns,augment,expected_error_msg",
    [
        (  # Invalid max_num_rows
            PrivateSource("private"),
            lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            -1,
            Schema({"i": "INTEGER"}),
            False,
            "Limit on number of rows '-1' must be nonnegative.",
        ),
        (  # Invalid augment
            FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
                schema_new_columns=Schema({"Repeat": "INTEGER"}),
                augment=True,
                max_num_rows=1,
            ),
            lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
            2,
            Schema({"i": "INTEGER"}),
            1.0,
            "type of augment must be bool; got float instead",
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
    max_num_rows: int,
    schema_new_columns: Schema,
    augment: bool,
    expected_error_msg: str,
):
    """Tests that invalid FlatMap errors on post-init."""
    with pytest.raises((TypeError, ValueError), match=expected_error_msg):
        FlatMap(child, func, schema_new_columns, augment, max_num_rows)


@pytest.mark.parametrize(
    "join_columns,expected_error_msg",
    [
        ([], "Provided join columns must not be empty"),
        (["A", "A"], "Join columns must be distinct"),
    ],
)
def test_invalid_join_columns(
    join_columns: Optional[List[str]], expected_error_msg: str
):
    """Tests that JoinPrivate, JoinPublic error with invalid join columns."""
    with pytest.raises(ValueError, match=expected_error_msg):
        JoinPrivate(
            PrivateSource("private"),
            PrivateSource("private2"),
            TruncationStrategy.DropExcess(1),
            TruncationStrategy.DropExcess(1),
            join_columns,
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            JoinPublic(PrivateSource("private"), "public", join_columns)


@pytest.mark.parametrize(
    "replace_with,expected_error_msg",
    [
        (
            {"str": 100.0},
            re.escape(
                r"type of replace_with['str'] must be a tuple; got float instead"
            ),
        ),
        (
            {"str": [100.0, 100.0]},
            re.escape(r"type of replace_with['str'] must be a tuple; got list instead"),
        ),
        ([], re.escape(r"type of replace_with must be a dict; got list instead")),
        (
            {"A": (-100.0,)},
            re.escape(
                r"replace_with['A'] has wrong number of elements (expected 2, got 1"
                r" instead)"
            ),
        ),
    ],
)
def test_invalid_replace_infinity(replace_with: Any, expected_error_msg: str) -> None:
    """Test ReplaceInfinity with invalid arguments."""
    with pytest.raises(TypeError, match=expected_error_msg):
        ReplaceInfinity(PrivateSource("private"), replace_with)


@pytest.mark.parametrize(
    "columns,expected_error_msg",
    [
        ("A", "type of columns must be a list; got str instead"),
        (("A", "B"), "type of columns must be a list; got tuple instead"),
        ([1], re.escape(r"type of columns[0] must be str; got int instead")),
    ],
)
def test_invalid_drop_null_and_nan(columns: Any, expected_error_msg: str) -> None:
    """Test DropNullAndNan with invalid arguments."""
    with pytest.raises(TypeError, match=expected_error_msg):
        DropNullAndNan(PrivateSource("private"), columns)


@pytest.mark.parametrize(
    "columns,expected_error_msg",
    [
        ("A", "type of columns must be a list; got str instead"),
        (("A", "B"), "type of columns must be a list; got tuple instead"),
        ([1], re.escape(r"type of columns[0] must be str; got int instead")),
    ],
)
def test_invalid_drop_infinity(columns: Any, expected_error_msg: str) -> None:
    """Test DropInfinity with invalid arguments."""
    with pytest.raises(TypeError, match=expected_error_msg):
        DropInfinity(PrivateSource("private"), columns)


@pytest.mark.parametrize(
    "child,keys,expected_error_msg,output_column",
    [
        (
            PrivateSource("private"),
            KeySet.from_dict({}),
            "type of output_column must be str; got int instead",
            123,
        )
    ],
)
def test_invalid_groupbycount(
    child: QueryExpr, keys: KeySet, expected_error_msg: str, output_column: str
):
    """Tests that invalid GroupByCount errors on post-init."""
    with pytest.raises(TypeError, match=expected_error_msg):
        GroupByCount(child, keys, output_column)


@pytest.mark.parametrize(
    "keys,measure_column,low,high,expected_error_msg",
    [
        (
            KeySet.from_dict({"A": ["0", "1"]}),
            "B",
            "1.0",
            10.0,
            "type of low must be either float or int; got str instead",
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
        GroupByBoundedSTDEV,
    ]:
        with pytest.raises((TypeError, ValueError), match=expected_error_msg):
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
            "type of quantile must be either float or int; got str instead",
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
            "type of low must be either float or int; got str instead",
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
    with pytest.raises((TypeError, ValueError), match=expected_error_msg):
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
        GroupByBoundedSTDEV,
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
                GroupByBoundedSTDEV,
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
    ReplaceNullAndNan(child, replace_with)


@pytest.mark.parametrize(
    "child,replace_with",
    [
        (PrivateSource("private"), {}),
        (PrivateSource("private"), {"A": (-100.0, 100.0)}),
        (PrivateSource("private"), {"A": (-1, 1)}),
        (PrivateSource("private"), {"A": (-999.9, 999.9), "B": (123.45, 678.90)}),
    ],
)
def test_valid_replace_infinity(
    child: QueryExpr, replace_with: Dict[str, Tuple[float, float]]
) -> None:
    """Test ReplaceInfinity with valid values."""
    query = ReplaceInfinity(child, replace_with)
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
    DropInfinity(child, columns)


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
    DropInfinity(child, columns)


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
