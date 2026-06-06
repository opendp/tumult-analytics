"""Query tests flat-map-by-id on tables using ID-based protected changes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Callable, Dict, List, Union

import pandas as pd
import pytest
from py4j.protocol import Py4JJavaError
from pyspark.sql import Row
from tmlt.core.utils.testing import assert_dataframe_equal

from tmlt.analytics import (
    AddRowsWithID,
    ColumnDescriptor,
    ColumnType,
    MaxRowsPerID,
    PureDPBudget,
    QueryBuilder,
)

from .conftest import make_session


def test_simple(spark):
    """Flat-map-by-ID works as expected for simple cases."""
    budget = PureDPBudget(float("inf"))
    input_df = spark.createDataFrame(
        pd.DataFrame({"id": [1, 1, 2, 3], "x": [2, 6, 3, 4]})
    )
    sess = make_session(budget, {"t": (input_df, AddRowsWithID("id"))})

    q = (
        QueryBuilder("t")
        .flat_map_by_id(
            lambda rs: [{"sum": sum(r["x"] for r in rs)}],
            {"sum": ColumnType.INTEGER},
        )
        .enforce(MaxRowsPerID(1))
        .sum("sum", low=0, high=10, name="sum")
    )
    expected = pd.DataFrame({"sum": [15]})
    result = sess.evaluate(q, budget)
    assert_dataframe_equal(result, expected)


def test_map_inputs(spark):
    """The groups of rows passed to the mapping function are as expected."""
    budget = PureDPBudget(float("inf"))
    input_df = spark.createDataFrame(
        pd.DataFrame({"id": [1, 1, 2, 3], "x": [2, 6, 3, 4]})
    )
    sess = make_session(budget, {"t": (input_df, AddRowsWithID("id"))})

    def f(rows):
        if rows[0]["id"] == 1:
            assert rows == [Row(id=1, x=2), Row(id=1, x=6)]
        elif rows[0]["id"] == 2:
            assert rows == [Row(id=2, x=3)]
        elif rows[0]["id"] == 3:
            assert rows == [Row(id=3, x=4)]
        else:
            assert False, "Unexpected ID value passed to mapping function"
        return []

    q = QueryBuilder("t").flat_map_by_id(f, {}).enforce(MaxRowsPerID(1)).count()
    expected = pd.DataFrame({"count": [0]})
    result = sess.evaluate(q, budget)
    assert_dataframe_equal(result, expected)


def test_id_conflict(spark):
    """Map operations that would overwrite the ID column are not allowed."""
    budget = PureDPBudget(float("inf"))
    input_df = spark.createDataFrame(
        pd.DataFrame({"id": [1, 1, 2, 3], "x": [2, 6, 3, 4]})
    )
    sess = make_session(budget, {"t": (input_df, AddRowsWithID("id"))})

    q = (
        QueryBuilder("t")
        .flat_map_by_id(lambda rs: [{"id": 1}], {"id": ColumnType.INTEGER})
        .enforce(MaxRowsPerID(1))
        .count()
    )
    with pytest.raises(ValueError):
        sess.evaluate(q, budget)


@pytest.mark.parametrize(
    "schema,expected_schema",
    [
        (
            {"v": ColumnType.INTEGER},
            {
                "id": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                "v": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            },
        ),
        (
            {"v": ColumnDescriptor(ColumnType.INTEGER, allow_null=False)},
            {
                "id": ColumnDescriptor(ColumnType.INTEGER),
                "v": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            },
        ),
        (
            {"v": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
            {
                "id": ColumnDescriptor(ColumnType.INTEGER),
                "v": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            },
        ),
        (
            {"v": ColumnType.DECIMAL},
            {
                "id": ColumnDescriptor(ColumnType.INTEGER),
                "v": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_inf=True, allow_nan=True
                ),
            },
        ),
        (
            {
                "v": ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=False,
                    allow_nan=False,
                    allow_inf=False,
                )
            },
            {
                "id": ColumnDescriptor(ColumnType.INTEGER),
                "v": ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=True,
                    allow_nan=False,
                    allow_inf=False,
                ),
            },
        ),
        (
            {
                "v": ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=True,
                    allow_nan=True,
                    allow_inf=False,
                )
            },
            {
                "id": ColumnDescriptor(ColumnType.INTEGER),
                "v": ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=True,
                    allow_nan=True,
                    allow_inf=False,
                ),
            },
        ),
        (
            {
                "v": ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=False,
                    allow_nan=True,
                    allow_inf=False,
                )
            },
            {
                "id": ColumnDescriptor(ColumnType.INTEGER),
                "v": ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=True,
                    allow_nan=True,
                    allow_inf=False,
                ),
            },
        ),
    ],
)
def test_schema(
    spark,
    schema: Dict[str, Union[ColumnType, ColumnDescriptor]],
    expected_schema: Dict[str, Union[ColumnType, ColumnDescriptor]],
):
    """Flat-map-by-ID works with ColumnDescriptor and has expected output schema."""
    budget = PureDPBudget(float("inf"))
    input_df = spark.createDataFrame(
        pd.DataFrame({"id": [1, 1, 2, 3], "x": [2, 6, 3, 4]}),
    )
    input_df.schema["id"].nullable = False
    sess = make_session(budget, {"t": (input_df, AddRowsWithID("id"))})

    view_q = QueryBuilder("t").flat_map_by_id(lambda rs: [{"v": 0}], schema)
    sess.create_view(view_q, "view", cache=False)
    assert sess.get_schema("view") == expected_schema


@pytest.mark.parametrize(
    "base_query,input_df,expected_df",
    [
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1 if rs[0]["id"] != 1 else None}],
                {"v": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
            )
            .filter("isnull(v)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [1]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1 if rs[0]["id"] != 1 else None}],
                {"v": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
            )
            .filter("isnan(v)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [0]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else None}],
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True)},
            )
            .filter("isnull(v)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [1]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else None}],
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True)},
            )
            .filter("isnan(v)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [0]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else None}],
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True)},
            )
            .filter("v == CAST('inf' AS DOUBLE) OR v == CAST('-inf' AS DOUBLE)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [0]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else float("nan")}],
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True)},
            )
            .filter("isnull(v)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [0]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else float("nan")}],
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True)},
            )
            .filter("isnan(v)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [1]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else float("nan")}],
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True)},
            )
            .filter("v == CAST('inf' AS DOUBLE) OR v == CAST('-inf' AS DOUBLE)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [0]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else float("inf")}],
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True)},
            )
            .filter("v == CAST('inf' AS DOUBLE)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [1]}),
        ),
        (
            QueryBuilder("t")
            .flat_map_by_id(
                lambda rs: (
                    len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else float("-inf")}]
                ),
                {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True)},
            )
            .filter("v == CAST('-inf' AS DOUBLE)"),
            pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
            pd.DataFrame({"count": [1]}),
        ),
    ],
)
def test_nulls_nans_infs_allowed(
    spark,
    base_query: QueryBuilder,
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Flat-map-by-ID behaves as expected when nulls/NaNs/infs are produced."""
    budget = PureDPBudget(float("inf"))
    sess = make_session(
        budget, {"t": (spark.createDataFrame(input_df), AddRowsWithID("id"))}
    )

    result_df = sess.evaluate(base_query.enforce(MaxRowsPerID(10)).count(), budget)
    assert_dataframe_equal(result_df, expected_df)


@pytest.mark.xfail(reason="tumult-labs/tumult#3298")
@pytest.mark.parametrize(
    "f,schema",
    [
        pytest.param(
            lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else None}],
            {"v": ColumnDescriptor(ColumnType.INTEGER, allow_null=False)},
            marks=pytest.mark.xfail(reason="tumult-labs/tumult#3297"),
        ),
        (
            lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else None}],
            {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_null=False)},
        ),
        (
            lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else float("nan")}],
            {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=False)},
        ),
        (
            lambda rs: len(rs) * [{"v": 1.0 if rs[0]["id"] != 1 else float("inf")}],
            {"v": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=False)},
        ),
    ],
)
def test_nulls_nans_infs_disallowed(
    spark,
    f: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    schema: Dict[str, Union[ColumnType, ColumnDescriptor]],
):
    """Flat-map-by-ID raises an error when unexpected nulls/NaNs/infs are produced."""
    budget = PureDPBudget(float("inf"))
    input_df = spark.createDataFrame(
        pd.DataFrame({"id": [1, 2, 2, 3, 3, 3]}),
    )
    sess = make_session(budget, {"t": (input_df, AddRowsWithID("id"))})

    q = QueryBuilder("t").flat_map_by_id(f, schema).enforce(MaxRowsPerID(10)).count()
    with pytest.raises(Py4JJavaError):
        sess.evaluate(q, budget)


def test_no_output_columns(spark):
    """Flat-map-by-ID works when mapping function produces no columns."""
    budget = PureDPBudget(float("inf"))
    input_df = spark.createDataFrame(
        pd.DataFrame({"id": [1, 1, 2, 3], "x": [2, 6, 3, 4]})
    )
    sess = make_session(budget, {"t": (input_df, AddRowsWithID("id"))})

    q = (
        QueryBuilder("t")
        .flat_map_by_id(
            lambda rs: [{} for r in rs for _ in range(r["x"])],
            {},
        )
        .enforce(MaxRowsPerID(10))
        .count()
    )
    expected = pd.DataFrame({"count": [15]})
    result = sess.evaluate(q, budget)
    assert_dataframe_equal(result, expected)
