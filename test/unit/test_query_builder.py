"""Unit tests for :mod:`~tmlt.analytics.query_builder`."""

# SPDX-License-Identifier: Apache-2.0

import datetime
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from parameterized import parameterized
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
from tmlt.core.utils.testing import PySparkTest

PRIVATE_ID = "private"

Row = Dict[str, Any]


class TestTransformations(PySparkTest):
    """Unit tests for QueryBuilder transformations."""

    def setUp(self) -> None:
        """Set up QueryBuilder."""
        self.root_builder = QueryBuilder(PRIVATE_ID)

    @parameterized.expand([(None,), (["B"],)])
    def test_join_public(self, join_columns: Optional[List[str]]):
        """QueryBuilder.join_public works as expected with a public source ID.

        Args:
            join_columns: the specified columns used for joining the two DataFrames.
        """
        join_table = "public"
        query = (
            self.root_builder.join_public(join_table, join_columns)
            .groupby(KeySet.from_dict({"A + B": ["0", "1", "2"]}))
            .count()
        )

        self.assertEqual(query.child.join_columns, join_columns)  # type: ignore

        # Check query expression
        assert isinstance(query, GroupByCount)

        join_expr = query.child
        assert isinstance(join_expr, JoinPublic)
        self.assertEqual(join_expr.public_table, join_table)

        root_expr = join_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    @parameterized.expand([(None,), (["B"],)])
    def test_join_public_dataframe(self, join_columns: Optional[List[str]]):
        """QueryBuilder.join_public works as expected when used with a dataframe.

        Args:
            join_columns: the specified columns used for joining the two DataFrames.
        """
        join_table = self.spark.createDataFrame(pd.DataFrame({"A": [1, 2]}))
        query = (
            self.root_builder.join_public(join_table, join_columns)
            .groupby(KeySet.from_dict({"A + B": ["0", "1", "2"]}))
            .count()
        )

        self.assertEqual(query.child.join_columns, join_columns)  # type: ignore

        # Check query expression
        assert isinstance(query, GroupByCount)

        join_expr = query.child
        assert isinstance(join_expr, JoinPublic)
        assert isinstance(join_expr.public_table, DataFrame)
        self.assert_frame_equal_with_sort(
            join_expr.public_table.toPandas(), join_table.toPandas()
        )

        root_expr = join_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    @parameterized.expand([(None,), (["B"],)])
    def test_join_private(self, join_columns: Optional[Sequence[str]] = None):
        """Tests that join_private works as expected with/without named join columns.

        Args:
            join_columns: the specified columns used for joining the two DataFrames.
        """
        query = (
            self.root_builder.join_private(
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
        self.assertEqual(private_join_expr.join_columns, join_columns)
        self.assertEqual(
            private_join_expr.truncation_strategy_left, TruncationStrategy.DropExcess(1)
        )
        self.assertEqual(
            private_join_expr.truncation_strategy_right,
            TruncationStrategy.DropExcess(2),
        )
        right_operand_expr = private_join_expr.right_operand_expr
        assert isinstance(right_operand_expr, PrivateSource)
        self.assertEqual(right_operand_expr.source_id, "private_2")

        assert isinstance(query, GroupByCount)

    def test_rename(self):
        """QueryBuilder rename works as expected."""
        column_mapper = {"A": "Z"}
        query = (
            self.root_builder.rename(column_mapper)
            .groupby(KeySet.from_dict({"Z": ["1", "2"]}))
            .count()
        )

        # Check query expression
        assert isinstance(query, GroupByCount)

        rename_expr = query.child
        assert isinstance(rename_expr, Rename)
        self.assertEqual(rename_expr.column_mapper, column_mapper)

        root_expr = rename_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def test_filter(self):
        """QueryBuilder filter works as expected."""
        predicate = "A == '0'"
        query = self.root_builder.filter(predicate).count()

        # Check query expression
        assert isinstance(query, GroupByCount)

        filter_expr = query.child
        assert isinstance(filter_expr, Filter)
        self.assertEqual(filter_expr.predicate, predicate)

        root_expr = filter_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def test_select(self):
        """QueryBuilder select works as expected."""
        columns = ["A"]
        query = (
            self.root_builder.select(columns)
            .groupby(KeySet.from_dict({"Z": ["1", "2"]}))
            .count()
        )

        # Check query expression
        assert isinstance(query, GroupByCount)

        select_expr = query.child
        assert isinstance(select_expr, Select)
        self.assertEqual(select_expr.columns, columns)

        root_expr = select_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def test_invalid_map(self):
        """QueryBuilder.map doesn't allow columns named ""."""

        # this is a map function that returns a column named ""
        def new_empty_column(_: Row) -> Row:
            return {"": 2 * "B"}

        # which should raise an error
        with self.assertRaisesRegex(
            ValueError,
            re.escape('"" (the empty string) is not a supported column name'),
        ):
            self.root_builder.map(
                f=new_empty_column, new_column_types={"": "VARCHAR"}, augment=False
            )

        # this should also raise an error if augment is true
        with self.assertRaisesRegex(
            ValueError,
            re.escape('"" (the empty string) is not a supported column name'),
        ):
            self.root_builder.map(
                f=new_empty_column, new_column_types={"": "VARCHAR"}, augment=True
            )

    def test_map_augment_is_false(self):
        """QueryBuilder map works as expected with augment=False."""

        def double_row(_: Row) -> Row:
            """Return row with doubled value

            Args:
                _: Row to apply function to.
            """
            return {"C": 2 * "B"}

        query = (
            self.root_builder.map(
                double_row, new_column_types={"C": "VARCHAR"}, augment=False
            )
            .groupby(KeySet.from_dict({"C": ["0", "1"]}))
            .count()
        )

        # Check query expression
        assert isinstance(query, GroupByCount)

        map_expr = query.child
        assert isinstance(map_expr, Map)
        self.assertIs(getattr(map_expr, "f"), double_row)
        self.assertEqual(map_expr.schema_new_columns.column_types, {"C": "VARCHAR"})
        self.assertFalse(map_expr.augment)

        root_expr = map_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def test_map_augment_is_true(self):
        """QueryBuilder map works as expected with augment=True."""

        def double_row(_: Row) -> Row:
            """Return row with doubled value

            Args:
                _: Row to apply function to.
            """
            return {"C": 2 * "B"}

        query = (
            self.root_builder.map(
                double_row, new_column_types={"C": "VARCHAR"}, augment=True
            )
            .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["0", "1"]}))
            .count()
        )

        # Check query expression
        assert isinstance(query, GroupByCount)

        map_expr = query.child
        assert isinstance(map_expr, Map)
        self.assertIs(getattr(map_expr, "f"), double_row)
        self.assertEqual(map_expr.schema_new_columns.column_types, {"C": "VARCHAR"})
        self.assertTrue(map_expr.augment)

        root_expr = map_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def test_invalid_flat_map(self) -> None:
        """QueryBuilder flat_map does not allow columns named ""."""

        def duplicate_rows(_: Row) -> List[Row]:
            return [{"": "0"}, {"": "1"}]

        # This should fail whether augment and grouping are true or false
        with self.assertRaisesRegex(
            ValueError,
            re.escape('"" (the empty string) is not a supported column name'),
        ):
            self.root_builder.flat_map(
                f=duplicate_rows,
                max_num_rows=2,
                new_column_types={"": "VARCHAR"},
                augment=False,
                grouping=False,
            )
        with self.assertRaisesRegex(
            ValueError,
            re.escape('"" (the empty string) is not a supported column name'),
        ):
            self.root_builder.flat_map(
                f=duplicate_rows,
                max_num_rows=2,
                new_column_types={"": "VARCHAR"},
                augment=False,
                grouping=True,
            )
        with self.assertRaisesRegex(
            ValueError,
            re.escape('"" (the empty string) is not a supported column name'),
        ):
            self.root_builder.flat_map(
                f=duplicate_rows,
                max_num_rows=2,
                new_column_types={"": "VARCHAR"},
                augment=True,
                grouping=False,
            )
        with self.assertRaisesRegex(
            ValueError,
            re.escape('"" (the empty string) is not a supported column name'),
        ):
            self.root_builder.flat_map(
                f=duplicate_rows,
                max_num_rows=2,
                new_column_types={"": "VARCHAR"},
                augment=True,
                grouping=True,
            )

    def test_flat_map_augment_is_false(self):
        """QueryBuilder flat_map works as expected with augment=False."""

        def duplicate_rows(_: Row) -> List[Row]:
            """Duplicate each row, with one copy having C=0, and the other C=1.

            Args:
                _: Row to apply function to.
            """
            return [{"C": "0"}, {"C": "1"}]

        query = (
            self.root_builder.flat_map(
                duplicate_rows, 2, new_column_types={"C": "VARCHAR"}, augment=False
            )
            .groupby(KeySet.from_dict({"C": ["0", "1"]}))
            .count()
        )

        # Check query expression
        assert isinstance(query, GroupByCount)

        flat_map_expr = query.child
        assert isinstance(flat_map_expr, FlatMap)
        self.assertIs(getattr(flat_map_expr, "f"), duplicate_rows)
        self.assertEqual(flat_map_expr.max_num_rows, 2)
        self.assertEqual(
            flat_map_expr.schema_new_columns,
            Schema(
                {
                    "C": ColumnDescriptor(
                        ColumnType.VARCHAR,
                        allow_null=True,
                        allow_nan=True,
                        allow_inf=True,
                    )
                }
            ),
        )
        self.assertFalse(flat_map_expr.augment)

        root_expr = flat_map_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def test_flat_map_augment_is_true(self):
        """QueryBuilder flat_map works as expected with augment=True."""

        def duplicate_rows(_: Row) -> List[Row]:
            """Duplicate each row, with one copy having C=0, and the other C=1.

            Args:
                _: Row to apply function to.
            """
            return [{"C": "0"}, {"C": "1"}]

        query = (
            self.root_builder.flat_map(
                duplicate_rows, 2, new_column_types={"C": "VARCHAR"}, augment=True
            )
            .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["0", "1"]}))
            .count()
        )

        # Check query expression
        assert isinstance(query, GroupByCount)

        flat_map_expr = query.child
        assert isinstance(flat_map_expr, FlatMap)
        self.assertIs(getattr(flat_map_expr, "f"), duplicate_rows)
        self.assertEqual(flat_map_expr.max_num_rows, 2)
        self.assertEqual(
            flat_map_expr.schema_new_columns,
            Schema(
                {
                    "C": ColumnDescriptor(
                        ColumnType.VARCHAR,
                        allow_null=True,
                        allow_nan=True,
                        allow_inf=True,
                    )
                }
            ),
        )
        self.assertTrue(flat_map_expr.augment)

        root_expr = flat_map_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def test_flat_map_grouping_is_true(self):
        """QueryBuilder flat_map works as expected with grouping=True."""

        def duplicate_rows(_: Row) -> List[Row]:
            """Duplicate each row, with one copy having C=0, and the other C=1.

            Args:
                _: Row to apply function to.
            """
            return [{"C": "0"}, {"C": "1"}]

        query = (
            self.root_builder.flat_map(
                duplicate_rows,
                2,
                new_column_types={
                    "C": ColumnDescriptor(
                        ColumnType.VARCHAR,
                        allow_null=True,
                        allow_nan=True,
                        allow_inf=True,
                    )
                },
                grouping=True,
            )
            .groupby(KeySet.from_dict({"A": ["0", "1"], "C": ["0", "1"]}))
            .count()
        )

        # Check query expression
        assert isinstance(query, GroupByCount)

        flat_map_expr = query.child
        assert isinstance(flat_map_expr, FlatMap)
        self.assertEqual(
            flat_map_expr.schema_new_columns,
            Schema(
                {
                    "C": ColumnDescriptor(
                        ColumnType.VARCHAR,
                        allow_null=True,
                        allow_nan=True,
                        allow_inf=True,
                    )
                },
                grouping_column="C",
            ),
        )

    def test_bin_column(self):
        """QueryBuilder.bin_column works as expected."""
        spec = BinningSpec([0, 5, 10])
        query = self.root_builder.bin_column("A", spec).count()
        assert isinstance(query, GroupByCount)
        map_expr = query.child
        assert isinstance(map_expr, Map)
        self.assertEqual(
            map_expr.schema_new_columns,
            Schema(
                {
                    "A_binned": ColumnDescriptor(
                        ColumnType.VARCHAR,
                        allow_null=True,
                        allow_nan=True,
                        allow_inf=True,
                    )
                }
            ),
        )
        self.assertTrue(map_expr.augment)
        root_expr = map_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

        # Verify the behavior of the map function, since there's no direct way
        # to check if it's the right one.
        self.assertEqual(getattr(map_expr, "f")({"A": 3}), {"A_binned": "[0, 5]"})
        self.assertEqual(getattr(map_expr, "f")({"A": 7}), {"A_binned": "(5, 10]"})

    def test_bin_column_options(self):
        """QueryBuilder.bin_column works as expected with options."""
        spec = BinningSpec([0.0, 1.0, 2.0], names=[0, 1])
        query = self.root_builder.bin_column("A", spec, name="rounded").count()
        assert isinstance(query, GroupByCount)
        map_expr = query.child
        assert isinstance(map_expr, Map)
        self.assertEqual(
            map_expr.schema_new_columns,
            Schema(
                {
                    "rounded": ColumnDescriptor(
                        ColumnType.INTEGER,
                        allow_null=True,
                        allow_nan=True,
                        allow_inf=True,
                    )
                }
            ),
        )
        self.assertTrue(map_expr.augment)
        root_expr = map_expr.child
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

        self.assertEqual(getattr(map_expr, "f")({"A": 0.5}), {"rounded": 0})
        self.assertEqual(getattr(map_expr, "f")({"A": 1.5}), {"rounded": 1})

    @parameterized.expand(
        [
            ({},),
            (None,),
            ({"A": datetime.date.today()},),
            ({"A": "new_string"},),
            (
                {
                    "A": "new_string",
                    "B": 999,
                    "C": -123.45,
                    "D": datetime.date(1999, 1, 1),
                    "E": datetime.datetime(2020, 1, 1),
                },
            ),
        ]
    )
    def test_replace_null_and_nan(
        self,
        replace_with: Optional[
            Mapping[str, Union[int, float, str, datetime.date, datetime.datetime]]
        ],
    ) -> None:
        """QueryBuilder.replace_null_and_nan works as expected."""
        query = self.root_builder.replace_null_and_nan(replace_with).count()
        # You want to use both of these assert statements:
        # - `self.assertIsInstance` will print a helpful error message if it isn't true
        # - `assert isinstance` helps mypy
        self.assertIsInstance(query, GroupByCount)
        assert isinstance(query, GroupByCount)
        replace_expr = query.child
        self.assertIsInstance(replace_expr, ReplaceNullAndNan)
        assert isinstance(replace_expr, ReplaceNullAndNan)

        root_expr = replace_expr.child
        self.assertIsInstance(root_expr, PrivateSource)
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

        expected_replace_with: Mapping[
            str, Union[int, float, str, datetime.date, datetime.datetime]
        ] = {}
        if replace_with is not None:
            expected_replace_with = replace_with

        self.assertEqual(replace_expr.replace_with, expected_replace_with)

    @parameterized.expand(
        [
            ({},),
            (None,),
            ({"A": (-100.0, 100.0)},),
            ({"A": (-999.9, 999.9), "B": (123.45, 678.90)},),
        ]
    )
    def test_replace_infinity(
        self, replace_with: Optional[Dict[str, Tuple[float, float]]]
    ) -> None:
        """QueryBuilder.replace_infinity works as expected."""
        query = self.root_builder.replace_infinity(replace_with).count()
        # You want to use both of these assert statements:
        # - `self.assertIsInstance` will print a helpful error message if it isn't true
        # - `assert isinstance` tells mypy what type this has
        self.assertIsInstance(query, GroupByCount)
        assert isinstance(query, GroupByCount)
        replace_expr = query.child
        self.assertIsInstance(replace_expr, ReplaceInfinity)
        assert isinstance(replace_expr, ReplaceInfinity)

        root_expr = replace_expr.child
        self.assertIsInstance(root_expr, PrivateSource)
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

        expected_replace_with: Dict[str, Tuple[float, float]] = {}
        if replace_with is not None:
            expected_replace_with = replace_with
        self.assertEqual(replace_expr.replace_with, expected_replace_with)

    @parameterized.expand([([],), (None,), (["A"],), (["A", "B"],)])
    def test_drop_null_and_nan(self, columns: Optional[List[str]]) -> None:
        """QueryBuilder.drop_null_and_nan works as expected."""
        query = self.root_builder.drop_null_and_nan(columns).count()
        self.assertIsInstance(query, GroupByCount)
        assert isinstance(query, GroupByCount)
        drop_expr = query.child
        self.assertIsInstance(drop_expr, DropNullAndNan)
        assert isinstance(drop_expr, DropNullAndNan)

        root_expr = drop_expr.child
        self.assertIsInstance(root_expr, PrivateSource)
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

        expected_columns: List[str] = []
        if columns is not None:
            expected_columns = columns
        self.assertEqual(drop_expr.columns, expected_columns)

    @parameterized.expand([([],), (None,), (["A"],), (["A", "B"],)])
    def test_drop_infinity(self, columns: Optional[List[str]]) -> None:
        """QueryBuilder.drop_infinity works as expected."""
        query = self.root_builder.drop_infinity(columns).count()
        self.assertIsInstance(query, GroupByCount)
        assert isinstance(query, GroupByCount)
        drop_expr = query.child
        self.assertIsInstance(drop_expr, DropInfinity)
        assert isinstance(drop_expr, DropInfinity)

        root_expr = drop_expr.child
        self.assertIsInstance(root_expr, PrivateSource)
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

        expected_columns: List[str] = []
        if columns is not None:
            expected_columns = columns
        self.assertEqual(drop_expr.columns, expected_columns)


class _TestAggregationsData:
    """Some extra data used in parameterizing tests in TestAggregations."""

    # This lives in a separate class because the parameterized.expand() call
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


class TestAggregations(PySparkTest):
    """Unit tests for QueryBuilder and GroupedQueryBuilder aggregations."""

    def setUp(self) -> None:
        """Set up QueryBuilder."""
        self.root_builder = QueryBuilder("private")

    def _keys_from_pandas(self, df: pd.DataFrame):
        """Convert a Pandas dataframe into a KeySet object."""
        return KeySet.from_dataframe(
            # This conditional works around an odd behavior in Spark where
            # converting an empty pandas dataframe with no columns will fail.
            self.spark.createDataFrame(df)
            if not df.empty
            else self.spark.createDataFrame([], "")
        )

    def _assert_root_expr(self, root_expr: QueryExpr):
        """Confirm the root expr is correct."""
        assert isinstance(root_expr, PrivateSource)
        self.assertEqual(root_expr.source_id, PRIVATE_ID)

    def _assert_count_query_correct(
        self,
        query: QueryExpr,
        expected_groupby_keys: KeySet,
        expected_output_column: str,
    ):
        """Confirm that a count query is constructed correctly."""
        assert isinstance(query, GroupByCount)
        self.assertEqual(query.groupby_keys, expected_groupby_keys)
        self.assertEqual(query.output_column, expected_output_column)

        self._assert_root_expr(query.child)

    @parameterized.expand([(None, "count"), ("total", "total")])
    def test_count_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by ungrouped count is correct."""
        query = self.root_builder.count(name)
        self._assert_count_query_correct(
            query, self._keys_from_pandas(pd.DataFrame()), expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "count"), ("total", "total"))
    )
    def test_count_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and count is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).count(name)
        self._assert_count_query_correct(query, keys, expected_name)

    def _assert_count_distinct_query_correct(
        self,
        query: QueryExpr,
        expected_groupby_keys: KeySet,
        expected_columns: Optional[List[str]],
        expected_output_column: str,
    ):
        """Confirm that a count_distinct query is constructed correctly."""
        assert isinstance(query, GroupByCountDistinct)
        self.assertEqual(query.columns_to_count, expected_columns)
        self.assertEqual(query.groupby_keys, expected_groupby_keys)
        self.assertEqual(query.output_column, expected_output_column)

        self._assert_root_expr(query.child)

    @parameterized.expand(
        [
            (None, "count_distinct", None),
            ("total", "total", ["Col1", "Col2"]),
            (None, "count_distinct(A, B)", ["A", "B"]),
        ]
    )
    def test_count_distinct_ungrouped(
        self, name: Optional[str], expected_name: str, columns: List[str]
    ):
        """Query returned by ungrouped count_distinct is correct."""
        query = self.root_builder.count_distinct(cols=columns, name=name)
        self._assert_count_distinct_query_correct(
            query, self._keys_from_pandas(pd.DataFrame()), columns, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in (
            (None, "count_distinct", None),
            ("total", "total", ["Col1", "Col2"]),
            (None, "count_distinct(X, Y)", ["X", "Y"]),
        )
    )
    def test_count_distinct_keyset(
        self,
        keys_df: pd.DataFrame,
        name: Optional[str],
        expected_name: str,
        columns: List[str],
    ):
        """Query returned by groupby with KeySet and count_distinct is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).count_distinct(cols=columns, name=name)
        self._assert_count_distinct_query_correct(query, keys, columns, expected_name)

    def _assert_common_query_fields_correct(
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
        self.assertEqual(query.groupby_keys, expected_groupby_keys)
        self.assertEqual(query.measure_column, expected_measure_column)
        self.assertEqual(query.low, expected_low)
        self.assertEqual(query.high, expected_high)
        self.assertEqual(query.output_column, expected_output_column)

        self._assert_root_expr(query.child)

    @parameterized.expand(
        [(None, "B_quantile(0.5)", 0.5), ("custom_name", "custom_name", 0.25)]
    )
    def test_quantile_ungrouped(
        self, name: Optional[str], expected_name: str, quantile: float
    ):
        """Query returned by ungrouped quantile is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.quantile(
            column="B", quantile=quantile, low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, quantile)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand([(None, "B_min"), ("custom_name", "custom_name")])
    def test_quantile_min_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by an ungrouped min is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.min(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, 0.0)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand([(None, "B_max"), ("custom_name", "custom_name")])
    def test_quantile_max_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by an ungrouped max is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.max(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, 1.0)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand([(None, "B_median"), ("custom_name", "custom_name")])
    def test_quantile_median_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by an ungrouped median is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.median(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, 0.5)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in (
            (None, "B_quantile(0.5)", 0.5),
            ("custom_name", "custom_name", 0.25),
        )
    )
    def test_quantile_keyset(
        self,
        keys_df: pd.DataFrame,
        name: Optional[str],
        expected_name: str,
        quantile: float,
    ):
        """Query returned by groupby with KeySet and quantile is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).quantile(
            column="B", quantile=quantile, low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, quantile)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "B_min"), ("custom_name", "custom_name"))
    )
    def test_quantile_min_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and min is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).min(
            column="B", low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, 0.0)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "B_max"), ("custom_name", "custom_name"))
    )
    def test_quantile_max_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and max is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).max(
            column="B", low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, 1.0)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "B_median"), ("custom_name", "custom_name"))
    )
    def test_quantile_median_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and median is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).median(
            column="B", low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByQuantile)
        self.assertEqual(query.quantile, 0.5)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand([(None, "B_sum"), ("total", "total")])
    def test_sum_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by ungrouped sum is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.sum(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedSum)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "B_sum"), ("total", "total"))
    )
    def test_sum_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and sum is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).sum(
            column="B", low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByBoundedSum)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand([(None, "B_average"), ("M", "M")])
    def test_average_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by ungrouped average is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.average(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedAverage)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "B_average"), ("M", "M"))
    )
    def test_average_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and average is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).average(
            column="B", low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByBoundedAverage)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand([(None, "B_variance"), ("var", "var")])
    def test_variance_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by ungrouped variance is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.variance(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedVariance)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "B_variance"), ("var", "var"))
    )
    def test_variance_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and variance is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).variance(
            column="B", low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByBoundedVariance)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand([(None, "B_stdev"), ("std", "std")])
    def test_stdev_ungrouped(self, name: Optional[str], expected_name: str):
        """Query returned by ungrouped stdev is correct."""
        keys = self._keys_from_pandas(pd.DataFrame())
        query = self.root_builder.stdev(column="B", low=0.0, high=1.0, name=name)
        assert isinstance(query, GroupByBoundedSTDEV)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )

    @parameterized.expand(
        (keys_df, *options)
        for keys_df in _TestAggregationsData.keyset_test_cases
        for options in ((None, "B_stdev"), ("std", "std"))
    )
    def test_stdev_keyset(
        self, keys_df: pd.DataFrame, name: Optional[str], expected_name: str
    ):
        """Query returned by groupby with KeySet and stdev is correct."""
        keys = self._keys_from_pandas(keys_df)
        query = self.root_builder.groupby(keys).stdev(
            column="B", low=0.0, high=1.0, name=name
        )
        assert isinstance(query, GroupByBoundedSTDEV)
        self._assert_common_query_fields_correct(
            query, keys, "B", 0.0, 1.0, expected_name
        )
