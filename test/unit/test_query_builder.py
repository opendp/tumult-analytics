"""Unit tests for :mod:`~tmlt.analytics.query_builder`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
# pylint: disable=no-self-use
import datetime
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame

from tmlt.analytics._schema import Schema
from tmlt.analytics.binning_spec import BinningSpec
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_builder import ColumnDescriptor, ColumnType, QueryBuilder
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
    GroupByCountDistinct,
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

PRIVATE_ID = "private"

Row = Dict[str, Any]


###DEFINE ROOT BUILDER###
def root_builder():
    """Set up QueryBuilder."""
    root_built = QueryBuilder(PRIVATE_ID)
    return root_built


# pylint throws a lot of spurious no-member errors in this file,
# because QueryBuilders return a QueryExpr, which doesn't have a .child and so on
# pylint: disable=no-member


@pytest.mark.parametrize("join_columns", [(None), (["B"])])
def test_join_public(join_columns: Optional[List[str]]):
    """QueryBuilder.join_public works as expected with a public source ID."""

    join_table = "public"
    query = (
        root_builder()
        .join_public(join_table, join_columns)
        .groupby(KeySet.from_dict({"A + B": ["0", "1", "2"]}))
        .count()
    )

    assert query.child.join_columns == join_columns

    # Check query expression
    assert isinstance(query, GroupByCount)

    join_expr = query.child
    assert isinstance(join_expr, JoinPublic)
    assert join_expr.public_table == join_table

    root_expr = join_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


@pytest.mark.parametrize("join_columns", [(None), (["B"])])
def test_join_public_dataframe(spark, join_columns: Optional[List[str]]):
    """QueryBuilder.join_public works as expected when used with a dataframe."""

    join_table = spark.createDataFrame(pd.DataFrame({"A": [1, 2]}))
    query = (
        root_builder()
        .join_public(join_table, join_columns)
        .groupby(KeySet.from_dict({"A + B": ["0", "1", "2"]}))
        .count()
    )

    assert query.child.join_columns == join_columns

    # Check query expression
    assert isinstance(query, GroupByCount)

    join_expr = query.child
    assert isinstance(join_expr, JoinPublic)
    assert isinstance(join_expr.public_table, DataFrame)
    assert_frame_equal_with_sort(
        join_expr.public_table.toPandas(), join_table.toPandas()
    )

    root_expr = join_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


@pytest.mark.parametrize("join_columns", [(None), (["B"])])
def test_join_private(join_columns: Optional[Sequence[str]]):
    """Tests that join_private works as expected with/without named join columns."""
    query = (
        root_builder()
        .join_private(
            right_operand=QueryBuilder("private_2"),
            truncation_strategy_left=TruncationStrategy.DropExcess(1),
            truncation_strategy_right=TruncationStrategy.DropExcess(2),
            join_columns=join_columns,
        )
        .groupby(KeySet.from_dict({"A": ["1", "2"]}))
        .count()
    )
    assert isinstance(query, GroupByCount)
    private_join_expr = query.child
    assert isinstance(private_join_expr, JoinPrivate)
    assert private_join_expr.join_columns == join_columns
    assert private_join_expr.truncation_strategy_left == TruncationStrategy.DropExcess(
        1
    )
    assert private_join_expr.truncation_strategy_right == TruncationStrategy.DropExcess(
        2
    )
    right_operand_expr = private_join_expr.right_operand_expr
    assert isinstance(right_operand_expr, PrivateSource)
    assert right_operand_expr.source_id == "private_2"

    assert isinstance(query, GroupByCount)


def test_join_private_str() -> None:
    """Test join_private("table_name") works as expected."""
    query = (
        root_builder()
        .join_private(
            right_operand="private_2",
            truncation_strategy_left=TruncationStrategy.DropExcess(1),
            truncation_strategy_right=TruncationStrategy.DropExcess(2),
            join_columns=None,
        )
        .groupby(KeySet.from_dict({"A": ["1", "2"]}))
        .count()
    )

    assert isinstance(query, GroupByCount)
    private_join_expr = query.child
    assert isinstance(private_join_expr, JoinPrivate)
    assert private_join_expr.join_columns is None
    assert private_join_expr.truncation_strategy_left == TruncationStrategy.DropExcess(
        1
    )
    assert private_join_expr.truncation_strategy_right == TruncationStrategy.DropExcess(
        2
    )
    right_operand_expr = private_join_expr.right_operand_expr
    assert isinstance(right_operand_expr, PrivateSource)
    assert right_operand_expr.source_id == "private_2"

    assert isinstance(query, GroupByCount)


def test_rename():
    """QueryBuilder rename works as expected."""
    column_mapper = {"A": "Z"}
    query = (
        root_builder()
        .rename(column_mapper)
        .groupby(KeySet.from_dict({"Z": ["1", "2"]}))
        .count()
    )

    # Check query expression
    assert isinstance(query, GroupByCount)

    rename_expr = query.child
    assert isinstance(rename_expr, Rename)
    assert rename_expr.column_mapper == column_mapper

    root_expr = rename_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


def test_filter():
    """QueryBuilder filter works as expected."""
    condition = "A == '0'"
    query = root_builder().filter(condition).count()

    # Check query expression
    assert isinstance(query, GroupByCount)

    filter_expr = query.child
    assert isinstance(filter_expr, Filter)
    assert filter_expr.condition == condition

    root_expr = filter_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


def test_select():
    """QueryBuilder select works as expected."""
    columns = ["A"]
    query = (
        root_builder()
        .select(columns)
        .groupby(KeySet.from_dict({"Z": ["1", "2"]}))
        .count()
    )

    # Check query expression
    assert isinstance(query, GroupByCount)

    select_expr = query.child
    assert isinstance(select_expr, Select)
    assert select_expr.columns == columns

    root_expr = select_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


def test_invalid_map():
    """QueryBuilder.map doesn't allow columns named ""."""

    # this is a map function that returns a column named ""
    def new_empty_column(_: Row) -> Row:
        return {"": 2 * "B"}

    # which should raise an error
    with pytest.raises(
        ValueError,
        match=re.escape('"" (the empty string) is not a supported column name'),
    ):
        root_builder().map(
            f=new_empty_column, new_column_types={"": "VARCHAR"}, augment=False
        )

    # this should also raise an error if augment is true
    with pytest.raises(
        ValueError,
        match=re.escape('"" (the empty string) is not a supported column name'),
    ):
        root_builder().map(
            f=new_empty_column, new_column_types={"": "VARCHAR"}, augment=True
        )


def test_map_augment_is_false():
    """QueryBuilder map works as expected with augment=False."""

    def double_row(_: Row) -> Row:
        """Return row with doubled value."""
        return {"C": 2 * "B"}

    query = (
        root_builder()
        .map(double_row, new_column_types={"C": "VARCHAR"}, augment=False)
        .groupby(KeySet.from_dict({"C": ["0", "1"]}))
        .count()
    )

    # Check query expression
    assert isinstance(query, GroupByCount)

    map_expr = query.child
    assert isinstance(map_expr, Map)
    assert getattr(map_expr, "f") is double_row
    assert map_expr.schema_new_columns.column_types == {"C": "VARCHAR"}
    assert not map_expr.augment

    root_expr = map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


def test_map_augment_is_true():
    """QueryBuilder map works as expected with augment=True."""

    def double_row(_: Row) -> Row:
        """Return row with doubled value."""
        return {"C": 2 * "B"}

    query = (
        root_builder()
        .map(double_row, new_column_types={"C": "VARCHAR"}, augment=True)
        .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["0", "1"]}))
        .count()
    )

    # Check query expression
    assert isinstance(query, GroupByCount)

    map_expr = query.child
    assert isinstance(map_expr, Map)
    assert getattr(map_expr, "f") is double_row
    assert map_expr.schema_new_columns.column_types == {"C": "VARCHAR"}
    assert map_expr.augment

    root_expr = map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


def test_invalid_flat_map() -> None:
    """QueryBuilder flat_map does not allow columns named ""."""

    def duplicate_rows(_: Row) -> List[Row]:
        return [{"": "0"}, {"": "1"}]

    # This should fail whether augment and grouping are true or false
    with pytest.raises(
        ValueError,
        match=re.escape('"" (the empty string) is not a supported column name'),
    ):
        root_builder().flat_map(
            f=duplicate_rows,
            new_column_types={"": "VARCHAR"},
            augment=False,
            grouping=False,
            max_num_rows=2,
        )
    with pytest.raises(
        ValueError,
        match=re.escape('"" (the empty string) is not a supported column name'),
    ):
        root_builder().flat_map(
            f=duplicate_rows,
            new_column_types={"": "VARCHAR"},
            augment=False,
            grouping=True,
            max_num_rows=2,
        )
    with pytest.raises(
        ValueError,
        match=re.escape('"" (the empty string) is not a supported column name'),
    ):
        root_builder().flat_map(
            f=duplicate_rows,
            new_column_types={"": "VARCHAR"},
            augment=True,
            grouping=False,
            max_num_rows=2,
        )
    with pytest.raises(
        ValueError,
        match=re.escape('"" (the empty string) is not a supported column name'),
    ):
        root_builder().flat_map(
            f=duplicate_rows,
            new_column_types={"": "VARCHAR"},
            augment=True,
            grouping=True,
            max_num_rows=2,
        )


def test_flat_map_augment_is_false():
    """QueryBuilder flat_map works as expected with augment=False."""

    def duplicate_rows(_: Row) -> List[Row]:
        """Duplicate each row, with one copy having C=0, and the other C=1."""
        return [{"C": "0"}, {"C": "1"}]

    query = (
        root_builder()
        .flat_map(
            duplicate_rows,
            new_column_types={"C": "VARCHAR"},
            augment=False,
            max_num_rows=2,
        )
        .groupby(KeySet.from_dict({"C": ["0", "1"]}))
        .count()
    )

    # Check query expression
    assert isinstance(query, GroupByCount)

    flat_map_expr = query.child
    assert isinstance(flat_map_expr, FlatMap)
    assert getattr(flat_map_expr, "f") is duplicate_rows
    assert flat_map_expr.max_num_rows == 2
    assert flat_map_expr.schema_new_columns == Schema(
        {
            "C": ColumnDescriptor(
                ColumnType.VARCHAR, allow_null=True, allow_nan=True, allow_inf=True
            )
        }
    )
    assert not flat_map_expr.augment

    root_expr = flat_map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


def test_flat_map_augment_is_true():
    """QueryBuilder flat_map works as expected with augment=True."""

    def duplicate_rows(_: Row) -> List[Row]:
        """Duplicate each row, with one copy having C=0, and the other C=1."""
        return [{"C": "0"}, {"C": "1"}]

    query = (
        root_builder()
        .flat_map(
            duplicate_rows,
            new_column_types={"C": "VARCHAR"},
            augment=True,
            max_num_rows=2,
        )
        .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["0", "1"]}))
        .count()
    )

    # Check query expression
    assert isinstance(query, GroupByCount)

    flat_map_expr = query.child
    assert isinstance(flat_map_expr, FlatMap)
    assert getattr(flat_map_expr, "f") is duplicate_rows
    assert flat_map_expr.max_num_rows == 2
    assert flat_map_expr.schema_new_columns == Schema(
        {
            "C": ColumnDescriptor(
                ColumnType.VARCHAR, allow_null=True, allow_nan=True, allow_inf=True
            )
        }
    )
    assert flat_map_expr.augment

    root_expr = flat_map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID


def test_flat_map_grouping_is_true():
    """QueryBuilder flat_map works as expected with grouping=True."""

    def duplicate_rows(_: Row) -> List[Row]:
        return [{"C": "0"}, {"C": "1"}]

    query = (
        root_builder()
        .flat_map(
            duplicate_rows,
            new_column_types={
                "C": ColumnDescriptor(
                    ColumnType.VARCHAR, allow_null=True, allow_nan=True, allow_inf=True
                )
            },
            grouping=True,
            max_num_rows=2,
        )
        .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["0", "1"]}))
        .count()
    )

    # Check query expression
    assert isinstance(query, GroupByCount)

    flat_map_expr = query.child
    assert isinstance(flat_map_expr, FlatMap)
    assert flat_map_expr.schema_new_columns == Schema(
        {
            "C": ColumnDescriptor(
                ColumnType.VARCHAR, allow_null=True, allow_nan=True, allow_inf=True
            )
        },
        grouping_column="C",
    )


def test_bin_column():
    """QueryBuilder.bin_column works as expected."""
    spec = BinningSpec([0, 5, 10])
    query = root_builder().bin_column("A", spec).count()
    assert isinstance(query, GroupByCount)
    map_expr = query.child
    assert isinstance(map_expr, Map)
    assert map_expr.schema_new_columns == Schema(
        {
            "A_binned": ColumnDescriptor(
                ColumnType.VARCHAR, allow_null=True, allow_nan=False, allow_inf=False
            )
        }
    )
    assert map_expr.augment
    root_expr = map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    # Verify the behavior of the map function, since there's no direct way
    # to check if it's the right one.
    assert getattr(map_expr, "f")({"A": 3}) == {"A_binned": "[0, 5]"}
    assert getattr(map_expr, "f")({"A": 7}) == {"A_binned": "(5, 10]"}


def test_bin_column_options():
    """QueryBuilder.bin_column works as expected with options."""
    spec = BinningSpec([0.0, 1.0, 2.0], names=[0, 1])
    query = root_builder().bin_column("A", spec, name="rounded").count()
    assert isinstance(query, GroupByCount)
    map_expr = query.child
    assert isinstance(map_expr, Map)
    assert map_expr.schema_new_columns == Schema(
        {
            "rounded": ColumnDescriptor(
                ColumnType.INTEGER, allow_null=True, allow_nan=False, allow_inf=False
            )
        }
    )
    assert map_expr.augment
    root_expr = map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    assert getattr(map_expr, "f")({"A": 0.5}) == {"rounded": 0}
    assert getattr(map_expr, "f")({"A": 1.5}) == {"rounded": 1}


def test_histogram():
    """QueryBuilder.histogram works as expected."""
    spec = BinningSpec([0, 5, 10])

    query = root_builder().histogram("A", spec)

    assert isinstance(query, GroupByCount)
    map_expr = query.child
    assert isinstance(map_expr, Map)

    assert map_expr.schema_new_columns == Schema(
        {
            "A_binned": ColumnDescriptor(
                ColumnType.VARCHAR, allow_null=True, allow_nan=False, allow_inf=False
            )
        }
    )

    assert map_expr.augment
    root_expr = map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    assert getattr(map_expr, "f")({"A": 3}) == {"A_binned": "[0, 5]"}
    assert getattr(map_expr, "f")({"A": 7}) == {"A_binned": "(5, 10]"}


def test_histogram_options():
    """QueryBuilder.histogram works as expected, with options."""

    query = root_builder().histogram("A", [0, 5, 10], name="New")

    assert isinstance(query, GroupByCount)
    map_expr = query.child
    assert isinstance(map_expr, Map)

    assert map_expr.schema_new_columns == Schema(
        {
            "New": ColumnDescriptor(
                ColumnType.VARCHAR, allow_null=True, allow_nan=False, allow_inf=False
            )
        }
    )

    assert map_expr.augment
    root_expr = map_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    assert getattr(map_expr, "f")({"A": 3}) == {"New": "[0, 5]"}
    assert getattr(map_expr, "f")({"A": 7}) == {"New": "(5, 10]"}


@pytest.mark.parametrize(
    "replace_with",
    [
        ({}),
        (None),
        ({"A": datetime.date.today()}),
        ({"A": "new_string"}),
        (
            {
                "A": "new_string",
                "B": 999,
                "C": -123.45,
                "D": datetime.date(1999, 1, 1),
                "E": datetime.datetime(2020, 1, 1),
            }
        ),
    ],
)
def test_replace_null_and_nan(
    replace_with: Optional[
        Mapping[str, Union[int, float, str, datetime.date, datetime.datetime]]
    ]
) -> None:
    """QueryBuilder.replace_null_and_nan works as expected."""
    query = root_builder().replace_null_and_nan(replace_with).count()
    assert isinstance(query, GroupByCount)
    replace_expr = query.child
    assert isinstance(replace_expr, ReplaceNullAndNan)

    root_expr = replace_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    expected_replace_with: Mapping[
        str, Union[int, float, str, datetime.date, datetime.datetime]
    ] = {}
    if replace_with is not None:
        expected_replace_with = replace_with

    assert replace_expr.replace_with == expected_replace_with


@pytest.mark.parametrize(
    "replace_with",
    [
        ({}),
        (None),
        ({"A": (-100.0, 100.0)}),
        ({"A": (-999.9, 999.9), "B": (123.45, 678.90)}),
    ],
)
def test_replace_infinity(
    replace_with: Optional[Dict[str, Tuple[float, float]]]
) -> None:
    """QueryBuilder.replace_infinity works as expected."""
    query = root_builder().replace_infinity(replace_with).count()
    # You want to use both of these assert statements:
    # - `self.assertIsInstance` will print a helpful error message if it isn't true
    # - `assert isinstance` tells mypy what type this has
    assert isinstance(query, GroupByCount)
    replace_expr = query.child
    assert isinstance(replace_expr, ReplaceInfinity)

    root_expr = replace_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    expected_replace_with: Dict[str, Tuple[float, float]] = {}
    if replace_with is not None:
        expected_replace_with = replace_with
    assert replace_expr.replace_with == expected_replace_with


@pytest.mark.parametrize("columns", [([]), (None), (["A"]), (["A", "B"])])
def test_drop_null_and_nan(columns: Optional[List[str]]) -> None:
    """QueryBuilder.drop_null_and_nan works as expected."""
    query = root_builder().drop_null_and_nan(columns).count()
    assert isinstance(query, GroupByCount)
    drop_expr = query.child
    assert isinstance(drop_expr, DropNullAndNan)

    root_expr = drop_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    expected_columns: List[str] = []
    if columns is not None:
        expected_columns = columns
    assert drop_expr.columns == expected_columns


@pytest.mark.parametrize("columns", [([]), (None), (["A"]), (["A", "B"])])
def test_drop_infinity(columns: Optional[List[str]]) -> None:
    """QueryBuilder.drop_infinity works as expected."""
    query = root_builder().drop_infinity(columns).count()
    assert isinstance(query, GroupByCount)
    drop_expr = query.child
    assert isinstance(drop_expr, DropInfinity)

    root_expr = drop_expr.child
    assert isinstance(root_expr, PrivateSource)
    assert root_expr.source_id == PRIVATE_ID

    expected_columns: List[str] = []
    if columns is not None:
        expected_columns = columns
    assert drop_expr.columns == expected_columns


class _TestAggregationsData:
    """Some extra data used in parameterizing tests in TestAggregations."""

    # This lives in a separate class because the pytest.mark.parametrize() call
    # can't use attributes of TestAggregations in its parameters, but they're
    # all logically related.

    keyset_test_cases: Tuple[pd.DataFrame, ...] = (
        pd.DataFrame(),
        pd.DataFrame({"A": ["0", "1"]}),
        pd.DataFrame({"A": ["0", "1", "0", "1"], "B": ["0", "0", "1", "1"]}),
        pd.DataFrame({"A": ["0", "1", "0"], "B": ["0", "0", "1"]}),
        pd.DataFrame({"A": [0, 1]}),
    )

    domains_test_cases: Tuple[Tuple[Dict, pd.DataFrame], ...] = (
        ({}, pd.DataFrame()),
        ({"A": ["0", "1"]}, pd.DataFrame({"A": ["0", "1"]})),
        (
            {"A": ["0", "1"], "B": ["2", "3"]},
            pd.DataFrame({"A": ["0", "1", "0", "1"], "B": ["2", "2", "3", "3"]}),
        ),
        (
            {"A": ["0", "1"], "B": [2, 3]},
            pd.DataFrame({"A": ["0", "1", "0", "1"], "B": [2, 2, 3, 3]}),
        ),
    )


class TestAggregations:
    """Tests for QueryBuilder, GroupedQueryBuilder Aggregations."""

    def _keys_from_pandas(self, spark, df: pd.DataFrame):
        """Convert Pandas df to KeySet."""
        return KeySet.from_dataframe(
            # This conditional works around an odd behavior in Spark where
            # converting an empty pandas dataframe with no columns will fail.
            spark.createDataFrame(df)
            if not df.empty
            else spark.createDataFrame([], "")
        )

    def assert_root_expr(self, root_expr: QueryExpr):
        """Confirm the root expr is correct."""
        assert isinstance(root_expr, PrivateSource)
        assert root_expr.source_id == PRIVATE_ID

    def assert_count_query_correct(
        self,
        query: QueryExpr,
        expected_groupby_keys: KeySet,
        expected_output_column: str,
    ):
        """Confirm that a count query is constructed correctly."""
        assert isinstance(query, GroupByCount)
        assert query.groupby_keys == expected_groupby_keys
        assert query.output_column == expected_output_column

        self.assert_root_expr(query.child)

    @pytest.mark.parametrize(
        "name,expected_name", [(None, "count"), ("total", "total")]
    )
    def test_count_ungrouped(self, spark, name: Optional[str], expected_name: str):
        """Query returned by ungrouped count is correct."""
        query = root_builder().count(name)
        self.assert_count_query_correct(
            query, self._keys_from_pandas(spark, pd.DataFrame()), expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "count"), ("total", "total"))
        ),
    )
    def test_count_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and count is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = root_builder().groupby(keys).count(name)
        self.assert_count_query_correct(query, keys, expected_name)

    def assert_count_distinct_query_correct(
        self,
        query: QueryExpr,
        expected_groupby_keys: KeySet,
        expected_columns: Optional[List[str]],
        expected_output_column: str,
    ):
        """Confirm that a count_distinct query is constructed correctly."""
        assert isinstance(query, GroupByCountDistinct)
        assert query.columns_to_count == expected_columns
        assert query.groupby_keys == expected_groupby_keys
        assert query.output_column == expected_output_column

        self.assert_root_expr(query.child)

    @pytest.mark.parametrize(
        "name,expected_name,columns",
        [
            (None, "count_distinct", None),
            ("total", "total", ["Col1", "Col2"]),
            (None, "count_distinct(A, B)", ["A", "B"]),
        ],
    )
    def test_count_distinct_ungrouped(
        self, spark, name: Optional[str], expected_name: str, columns: List[str]
    ):
        """Query returned by ungrouped count_distinct is correct."""
        query = root_builder().count_distinct(columns=columns, name=name)
        self.assert_count_distinct_query_correct(
            query, self._keys_from_pandas(spark, pd.DataFrame()), columns, expected_name
        )

    @pytest.mark.parametrize("columns", [(["A"]), (["col1", "col2"])])
    def test_count_distinct_raises_warnings(self, columns: List[str]):
        """Test that count_distinct raises warning when ``cols`` is provided."""
        with pytest.warns(
            DeprecationWarning, match=re.escape("`cols` argument is deprecated")
        ):
            root_builder().count_distinct(cols=columns)

        keys = KeySet.from_dict({e: ["a"] for e in columns})
        with pytest.warns(
            DeprecationWarning, match=re.escape("`cols` argument is deprecated")
        ):
            root_builder().groupby(keys).count_distinct(cols=columns)

    @pytest.mark.parametrize("columns", [(["A"]), (["col1", "col2"])])
    def test_count_distinct_raises_error(self, columns: List[str]):
        """Test that count_distinct raises error with both ``cols`` and ``columns``."""
        with pytest.raises(
            ValueError,
            match=re.escape("cannot provide both `cols` and `columns` arguments"),
        ):
            root_builder().count_distinct(columns=columns, cols=columns)

        keys = KeySet.from_dict({e: ["a"] for e in columns})
        with pytest.raises(
            ValueError,
            match=re.escape("cannot provide both `cols` and `columns` arguments"),
        ):
            root_builder().groupby(keys).count_distinct(columns=columns, cols=columns)

    @pytest.mark.parametrize(
        "keys_df,name,expected_name,columns",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in (
                (None, "count_distinct", None),
                ("total", "total", ["Col1", "Col2"]),
                (None, "count_distinct(X, Y)", ["X", "Y"]),
            )
        ),
    )
    def test_count_distinct_keyset(
        self,
        spark,
        keys_df: pd.DataFrame,
        name: Optional[str],
        expected_name: str,
        columns: List[str],
    ):
        """Query returned by groupby with KeySet and count_distinct is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = root_builder().groupby(keys).count_distinct(columns=columns, name=name)
        self.assert_count_distinct_query_correct(query, keys, columns, expected_name)

    def assert_common_query_fields_correct(
        self,
        query: Union[
            GroupByBoundedSum,
            GroupByQuantile,
            GroupByBoundedAverage,
            GroupByBoundedVariance,
            GroupByBoundedSTDEV,
        ],
        expected_groupby_keys: KeySet,
        expected_measure_column: str,
        expected_low: float,
        expected_high: float,
        expected_output_column: str,
    ):
        """Confirm that common fields in different types of queries are correct."""
        assert query.groupby_keys == expected_groupby_keys
        assert query.measure_column == expected_measure_column
        assert query.low == expected_low
        assert query.high == expected_high
        assert query.output_column == expected_output_column

        self.assert_root_expr(query.child)

    @pytest.mark.parametrize(
        "name,expected_name,quantile",
        [(None, "B_quantile(0.5)", 0.5), ("custom_name", "custom_name", 0.25)],
    )
    def test_quantile_ungrouped(
        self, spark, name: Optional[str], expected_name: str, quantile: float
    ):
        """Query returned by ungrouped quantile is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().quantile(
            column="B", quantile=quantile, low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == quantile
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "name,expected_name", [(None, "B_min"), ("custom_name", "custom_name")]
    )
    def test_quantile_min_ungrouped(
        self, spark, name: Optional[str], expected_name: str
    ):
        """Query returned by an ungrouped min is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().min(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == 0.0
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "name,expected_name", [(None, "B_max"), ("custom_name", "custom_name")]
    )
    def test_quantile_max_ungrouped(
        self, spark, name: Optional[str], expected_name: str
    ):
        """Query returned by an ungrouped max is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().max(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == 1.0
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "name,expected_name", [(None, "B_median"), ("custom_name", "custom_name")]
    )
    def test_quantile_median_ungrouped(
        self, spark, name: Optional[str], expected_name: str
    ):
        """Query returned by an ungrouped median is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().median(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == 0.5
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name,quantile",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in (
                (None, "B_quantile(0.5)", 0.5),
                ("custom_name", "custom_name", 0.25),
            )
        ),
    )
    def test_quantile_keyset(
        self,
        spark,
        keys_df: pd.DataFrame,
        name: Optional[str],
        expected_name: str,
        quantile: float,
    ):
        """Query returned by groupby with KeySet and quantile is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder()
            .groupby(keys)
            .quantile(column="B", quantile=quantile, low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == quantile
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "B_min"), ("custom_name", "custom_name"))
        ),
    )
    def test_quantile_min_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and min is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder().groupby(keys).min(column="B", low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == 0.0
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "B_max"), ("custom_name", "custom_name"))
        ),
    )
    def test_quantile_max_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and max is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder().groupby(keys).max(column="B", low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == 1.0
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "B_median"), ("custom_name", "custom_name"))
        ),
    )
    def test_quantile_median_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and median is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder()
            .groupby(keys)
            .median(column="B", low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByQuantile)
        assert query.quantile == 0.5
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "name,expected_name", [(None, "B_sum"), ("total", "total")]
    )
    def test_sum_ungrouped(self, spark, name: Optional[str], expected_name: str):
        """Query returned by ungrouped sum is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().sum(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedSum)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "B_sum"), ("total", "total"))
        ),
    )
    def test_sum_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and sum is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder().groupby(keys).sum(column="B", low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByBoundedSum)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize("name,expected_name", [(None, "B_average"), ("M", "M")])
    def test_average_ungrouped(self, spark, name: Optional[str], expected_name: str):
        """Query returned by ungrouped average is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().average(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedAverage)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "B_average"), ("M", "M"))
        ),
    )
    def test_average_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and average is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder()
            .groupby(keys)
            .average(column="B", low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByBoundedAverage)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "name,expected_name", [(None, "B_variance"), ("var", "var")]
    )
    def test_variance_ungrouped(self, spark, name: Optional[str], expected_name: str):
        """Query returned by ungrouped variance is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().variance(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedVariance)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "B_variance"), ("var", "var"))
        ),
    )
    def test_variance_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and variance is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder()
            .groupby(keys)
            .variance(column="B", low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByBoundedVariance)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize("name,expected_name", [(None, "B_stdev"), ("std", "std")])
    def test_stdev_ungrouped(self, spark, name: Optional[str], expected_name: str):
        """Query returned by ungrouped stdev is correct."""
        keys = self._keys_from_pandas(spark, pd.DataFrame())
        query = root_builder().stdev(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedSTDEV)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @pytest.mark.parametrize(
        "keys_df,name,expected_name",
        (
            (keys_df, *options)
            for keys_df in _TestAggregationsData.keyset_test_cases
            for options in ((None, "B_stdev"), ("std", "std"))
        ),
    )
    def test_stdev_keyset(
        self, spark, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and stdev is correct."""
        keys = self._keys_from_pandas(spark, keys_df)
        query = (
            root_builder().groupby(keys).stdev(column="B", low=0.0, high=1.0, name=name)
        )
        assert isinstance(query, GroupByBoundedSTDEV)
        self.assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )
