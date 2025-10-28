"""System tests for tables with special values (nulls, nans, infinities)."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pytest
from numpy import sqrt
from pyspark.sql import DataFrame
from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics import (
    AddOneRow,
    AddRowsWithID,
    ColumnDescriptor,
    ColumnType,
    MaxRowsPerID,
    ProtectedChange,
    PureDPBudget,
    Query,
    QueryBuilder,
    Session,
)
from tmlt.analytics._schema import Schema, analytics_to_spark_schema

from ...conftest import assert_frame_equal_with_sort


@pytest.fixture(name="sdf_special_values", scope="module")
def null_setup(spark):
    """Set up test data for sessions with special values."""
    sdf_col_types = {
        "string_nulls": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        "int_no_null": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
        "int_nulls": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        "float_no_special": ColumnDescriptor(
            ColumnType.DECIMAL,
            allow_null=False,
            allow_nan=False,
            allow_inf=False,
        ),
        "float_nulls": ColumnDescriptor(
            ColumnType.DECIMAL,
            allow_null=True,
            allow_nan=False,
            allow_inf=False,
        ),
        "float_nans": ColumnDescriptor(
            ColumnType.DECIMAL,
            allow_null=False,
            allow_nan=True,
            allow_inf=False,
        ),
        "float_infs": ColumnDescriptor(
            ColumnType.DECIMAL,
            allow_null=False,
            allow_nan=False,
            allow_inf=True,
        ),
        "float_all_special": ColumnDescriptor(
            ColumnType.DECIMAL,
            allow_null=True,
            allow_nan=True,
            allow_inf=True,
        ),
        "date_nulls": ColumnDescriptor(ColumnType.DATE, allow_null=True),
        "time_nulls": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
    }
    date = datetime.date(2000, 1, 1)
    time = datetime.datetime(2020, 1, 1)
    sdf = spark.createDataFrame(
        [(f"normal_{i}", 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, date, time) for i in range(20)]
        + [
            # Rows with nulls
            (None, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, date, time),
            ("u2", 1, None, 1.0, 1.0, 1.0, 1.0, 1.0, date, time),
            ("u3", 1, 1, 1.0, None, 1.0, 1.0, None, date, time),
            ("u4", 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, None, time),
            ("u5", 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, date, None),
            # Rows with nans
            ("a6", 1, 1, 1.0, 1.0, float("nan"), 1.0, float("nan"), date, time),
            # Rows with infinities
            ("i7", 1, 1, 1.0, 1.0, 1.0, float("inf"), float("inf"), date, time),
            ("i8", 1, 1, 1.0, 1.0, 1.0, -float("inf"), -float("inf"), date, time),
            ("i9", 1, 1, 1.0, 1.0, 1.0, float("inf"), 1.0, date, time),
            ("i10", 1, 1, 1.0, 1.0, 1.0, -float("inf"), 1.0, date, time),
        ],
        schema=analytics_to_spark_schema(Schema(sdf_col_types)),
    )
    return sdf


@parametrize(
    [
        Case("int_noop")(
            # Column "int_no_null" has only non-null values, all equal to 1
            replace_with={"int_no_null": 42},
            column="int_no_null",
            low=0,
            high=1,
            expected_average=1,
        ),
        Case("int_replace_null")(
            # Column "int_nulls" has one null value and 29 1s.
            replace_with={"int_nulls": 31},
            column="int_nulls",
            low=0,
            high=100,
            expected_average=2.0,  # (29+31)/30
        ),
        Case("float_replace_null")(
            # Column "float_nulls" has one null value and 29 1s.
            replace_with={"float_nulls": 61},
            column="float_nulls",
            low=0,
            high=100,
            expected_average=3.0,  # (29+61)/30
        ),
        Case("float_replace_nan")(
            # Column "float_nulls" has one null value and 29 1s.
            replace_with={"float_nans": 91},
            column="float_nans",
            low=0,
            high=100,
            expected_average=4.0,  # (29+91)/30
        ),
        Case("float_replace_both")(
            # Column "float_all_special" has 26 1s, one null value, one nan-value, one
            # negative infinity (clamped to 0), one positive infinity (clamped to 34).
            replace_with={"float_all_special": 15},
            column="float_all_special",
            low=0,
            high=34,
            expected_average=3.0,  # (26+15+15+34)/30
        ),
        Case("replace_all_with_none")(
            # When called with no argument, replace_null_and_nan should replace all null
            # values by analytics defaults, e.g. 0.
            replace_with=None,
            column="float_nulls",
            low=0,
            high=1,
            expected_average=29.0 / 30,
        ),
        Case("replace_all_with_empty_dict")(
            # Same thing with an empty dict and with nan values.
            replace_with={},
            column="float_nans",
            low=0,
            high=1,
            expected_average=29.0 / 30,
        ),
    ]
)
def test_replace_null_and_nan(
    sdf_special_values: DataFrame,
    replace_with: Optional[Dict[str, Union[int, float]]],
    column: str,
    low: Union[int, float],
    high: Union[int, float],
    expected_average: Union[int, float],
):
    inf_budget = PureDPBudget(float("inf"))
    sess = Session.from_dataframe(
        inf_budget,
        "private",
        sdf_special_values,
        AddOneRow(),
    )
    base_query = QueryBuilder("private")
    query = base_query.replace_null_and_nan(replace_with).average(column, low, high)
    result = sess.evaluate(query, inf_budget)
    expected_df = pd.DataFrame([[expected_average]], columns=[column + "_average"])
    assert_frame_equal_with_sort(result.toPandas(), expected_df)


@parametrize(
    [
        # All columns have 30 rows, all non-special values are equal to 1.
        Case("int_noop")(
            # Column "int_no_null" has only regular values.
            affected_columns=["int_no_null"],
            measure_column="int_no_null",
            low=0,
            high=1,
            expected_sum=30,
        ),
        Case("int_drop_nulls")(
            # Column "int_nulls" has one null value and 29 1s.
            affected_columns=["int_nulls"],
            measure_column="int_nulls",
            low=0,
            high=100,
            expected_sum=29,
        ),
        Case("float_drop_nulls")(
            # Column "float_nulls" has one null value and 29 1s.
            affected_columns=["float_nulls"],
            measure_column="float_nulls",
            low=0,
            high=100,
            expected_sum=29,
        ),
        Case("float_drop_nan")(
            # Column "float_nans" has one nan value and 29 1s.
            affected_columns=["float_nans"],
            measure_column="float_nans",
            low=0,
            high=100,
            expected_sum=29,
        ),
        Case("float_drop_both")(
            # Column "float_all_special" has 26 1s, one null, one nan, one negative
            # infinity (clamped to 0), one positive infinity (clamped to 100).
            affected_columns=["float_all_special"],
            measure_column="float_all_special",
            low=0,
            high=100,
            expected_sum=126,
        ),
        Case("drop_other_columns")(
            # Column "float_infs" has 26 1s, two negative infinities (clamped to 0) and
            # two positive infinities (clamped to 100). But dropping rows from columns
            # "string_nulls", "float_nulls", "float_nans", "date_nulls" and "time_nulls"
            # should remove five rows, leaving just 21 normal values.
            affected_columns=[
                "string_nulls",
                "float_nulls",
                "float_nans",
                "date_nulls",
                "time_nulls",
            ],
            measure_column="float_infs",
            low=0,
            high=100,
            expected_sum=221,
        ),
        Case("drop_all_with_none")(
            # When called with no argument, replace_null_and_nan should drop all rows
            # that have null/nan values anywhere, which leaves 24 1s even if we're
            # summing a column without nulls.
            affected_columns=None,
            measure_column="int_no_null",
            low=0,
            high=1,
            expected_sum=24,
        ),
        Case("drop_all_with_empty_list")(
            # Same thing with an empty list.
            affected_columns=[],
            measure_column="float_nulls",
            low=0,
            high=1,
            expected_sum=24.0,
        ),
    ]
)
def test_drop_null_and_nan(
    sdf_special_values: DataFrame,
    affected_columns: Optional[List[str]],
    measure_column: str,
    low: Union[int, float],
    high: Union[int, float],
    expected_sum: Union[int, float],
):
    inf_budget = PureDPBudget(float("inf"))
    sess = Session.from_dataframe(
        inf_budget,
        "private",
        sdf_special_values,
        AddOneRow(),
    )
    base_query = QueryBuilder("private")
    query = base_query.drop_null_and_nan(affected_columns).sum(
        measure_column, low, high
    )
    result = sess.evaluate(query, inf_budget)
    expected_df = pd.DataFrame([[expected_sum]], columns=[measure_column + "_sum"])
    assert_frame_equal_with_sort(result.toPandas(), expected_df)


@parametrize(
    # All these tests compute the average of the "float_infs" column of the input table,
    # in which there are:
    # - 26 non-infinity values, all equal to 1
    # - two negative infinity
    # - two positive infinity
    # We test this using average and not sum to distinguish between infinities being
    # removed from infinities being changed to 0.
    [
        Case("replace_no_clamp")(
            replace_with={"float_infs": (0, 17)},
            low=-100,
            high=100,
            # 26+0+0+17+17 = 60, divided by 30 is 2
            expected_average=2.0,
        ),
        Case("replace_clamp")(
            replace_with={"float_infs": (-4217, 300)},
            low=-5,
            high=22,
            # 26-5+5+22+22 = 60, divided by 30 is 2
            expected_average=2.0,
        ),
        Case("replace_unrelated_column")(
            # If we don't explicitly replace infinity in the measure column, then
            # infinities should be clamped to the bounds.
            replace_with={"float_all_special": (-4217, 300)},
            low=-10,
            high=27,
            # 26-10+10+27+27 = 60, divided by 30 is 2
            expected_average=2.0,
        ),
        Case("replace_with_none")(
            # If used without any argument, replace_infinity transforms all infinity
            # values in all columns of the table to 0.
            replace_with=None,
            low=-10,
            high=10,
            expected_average=26.0 / 30.0,
        ),
        Case("replace_with_empty_dict")(
            # Same with an empty dict.
            replace_with={},
            low=-10,
            high=10,
            expected_average=26.0 / 30.0,
        ),
    ]
)
def test_replace_infinity_average(
    sdf_special_values: DataFrame,
    replace_with: Optional[Dict[str, Tuple[float, float]]],
    low: float,
    high: float,
    expected_average: float,
):
    inf_budget = PureDPBudget(float("inf"))
    sess = Session.from_dataframe(
        inf_budget,
        "private",
        sdf_special_values,
        AddOneRow(),
    )
    base_query = QueryBuilder("private")
    query = base_query.replace_infinity(replace_with).average("float_infs", low, high)
    result = sess.evaluate(query, inf_budget)
    expected_df = pd.DataFrame([[expected_average]], columns=["float_infs_average"])
    assert_frame_equal_with_sort(result.toPandas(), expected_df)


@parametrize(
    [
        Case("all_ones")(
            replace_with={"float_infs": (1, 1)},
            expected_sum=30.0,
            expected_stdev=0,
            expected_variance=0,
        ),
        Case("one_zero_one_one")(
            # If we don't replace infinities in the measure column, then infinity values
            # should be clamped to the bounds, namely 0 and 1.
            replace_with={"float_all_special": (1, 1)},
            expected_sum=28.0,
            expected_stdev=sqrt((2 * (28.0 / 30) ** 2 + 28 * (2.0 / 30) ** 2) / 29),
            expected_variance=(2 * (28.0 / 30) ** 2 + 28 * (2.0 / 30) ** 2) / 29,
        ),
        Case("all_zeroes")(
            # Without argument, all infinities are replaced by 0
            replace_with=None,
            expected_sum=26.0,
            expected_stdev=sqrt((4 * (26.0 / 30) ** 2 + 26 * (4.0 / 30) ** 2) / 29),
            expected_variance=(4 * (26.0 / 30) ** 2 + 26 * (4.0 / 30) ** 2) / 29,
        ),
    ]
)
def test_replace_infinity_other_aggregations(
    sdf_special_values: DataFrame,
    replace_with: Optional[Dict[str, Tuple[float, float]]],
    expected_sum: float,
    expected_stdev: float,
    expected_variance: float,
):
    inf_budget = PureDPBudget(float("inf"))
    sess = Session.from_dataframe(
        inf_budget,
        "private",
        sdf_special_values,
        protected_change=AddOneRow(),
    )

    query_sum = (
        QueryBuilder("private").replace_infinity(replace_with).sum("float_infs", 0, 1)
    )
    result_sum = sess.evaluate(query_sum, inf_budget)
    expected_df = pd.DataFrame([[expected_sum]], columns=["float_infs_sum"])
    assert_frame_equal_with_sort(result_sum.toPandas(), expected_df)

    query_stdev = (
        QueryBuilder("private").replace_infinity(replace_with).stdev("float_infs", 0, 1)
    )
    result_stdev = sess.evaluate(query_stdev, inf_budget)
    expected_df = pd.DataFrame([[expected_stdev]], columns=["float_infs_stdev"])
    assert_frame_equal_with_sort(result_stdev.toPandas(), expected_df)

    query_variance = (
        QueryBuilder("private")
        .replace_infinity(replace_with)
        .variance("float_infs", 0, 1)
    )
    result_variance = sess.evaluate(query_variance, inf_budget)
    expected_df = pd.DataFrame([[expected_variance]], columns=["float_infs_variance"])
    assert_frame_equal_with_sort(result_variance.toPandas(), expected_df)


@parametrize(
    # All these tests compute the sum of the "float_infs" column of the input table.
    [
        Case("drop_rows_in_column")(
            # There are 26 non-infinity values in the "float_infs" column.
            columns=["float_infs"],
            expected_sum=26.0,
        ),
        Case("drop_no_rows")(
            # The call to drop_infinity is a no-op. In the "float_infs" column, there
            # are two rows with positive infinities (clamped to 1), and two with
            # negative infinities (clamped to 0).
            columns=["float_no_special"],
            expected_sum=28.0,
        ),
        Case("drop_some_rows_due_to_other_columns")(
            # Two rows with infinite values in the "float_infs" column also have
            # infinite values in the "float_all_special" column. We end up with one
            # positive infinity value, clamped to 1, and one negative, clamped to 0.
            columns=["float_all_special"],
            expected_sum=27.0,
        ),
        Case("drop_rows_in_all_columns")(
            # If used without any argument, drop_infinity removes all infinity values in
            # all columns of the table.
            columns=None,
            expected_sum=26.0,
        ),
    ]
)
def test_drop_infinity(
    sdf_special_values: DataFrame,
    columns: Optional[List[str]],
    expected_sum: float,
):
    inf_budget = PureDPBudget(float("inf"))
    sess = Session.from_dataframe(
        inf_budget,
        "private",
        sdf_special_values,
        AddOneRow(),
    )
    base_query = QueryBuilder("private")
    query = base_query.drop_infinity(columns).sum("float_infs", 0, 1)
    result = sess.evaluate(query, inf_budget)
    expected_df = pd.DataFrame([[expected_sum]], columns=["float_infs_sum"])
    assert_frame_equal_with_sort(result.toPandas(), expected_df)


@parametrize(
    [
        Case("works_with_nulls")(
            # Check that
            query=QueryBuilder("private").get_bounds("int_nulls"),
            expected_df=pd.DataFrame(
                [[-1, 1]],
                columns=["int_nulls_lower_bound", "int_nulls_upper_bound"],
            ),
        ),
        Case("works_with_nan")(
            # Check that
            query=QueryBuilder("private").get_bounds("float_nans"),
            expected_df=pd.DataFrame(
                [[-1, 1]],
                columns=["float_nans_lower_bound", "float_nans_upper_bound"],
            ),
        ),
        Case("works_with_infinity")(
            # Check that
            query=QueryBuilder("private").get_bounds("float_infs"),
            expected_df=pd.DataFrame(
                [[-1, 1]],
                columns=["float_infs_lower_bound", "float_infs_upper_bound"],
            ),
        ),
        Case("drop_and_replace")(
            # Dropping nulls & nans removes 6/30 values, replacing 4 infinity values by
            # (-3,3) guarantees ensures get_bounds should get the interval corresponding
            # to next power of 2, namely 4
            query=(
                QueryBuilder("private")
                .drop_null_and_nan()
                .replace_infinity({"float_infs": (-3, 3)})
                .get_bounds("float_infs")
            ),
            expected_df=pd.DataFrame(
                [[-4, 4]],
                columns=["float_infs_lower_bound", "float_infs_upper_bound"],
            ),
        ),
    ]
)
def test_get_bounds(
    sdf_special_values: DataFrame, query: Query, expected_df: pd.DataFrame
):
    inf_budget = PureDPBudget(float("inf"))
    sess = Session.from_dataframe(
        inf_budget,
        "private",
        sdf_special_values,
        AddOneRow(),
    )
    result = sess.evaluate(query, inf_budget)
    assert_frame_equal_with_sort(result.toPandas(), expected_df)

# TODO: add tests with AddRowsWithID, checking behavior when removing/replacing nulls in
# the ID column

# TODO: add tests for public and private joins

# TODO: add changelog
