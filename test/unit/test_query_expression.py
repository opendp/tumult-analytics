"""Tests for QueryExpr."""

# <placeholder: boilerplate>

# pylint: disable=too-many-arguments

import datetime
import re
import unittest
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import pandas as pd
from parameterized import parameterized
from pyspark.sql.types import BinaryType, StructField, StructType

from tmlt.analytics._schema import Schema
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_expr import (
    DropInvalid,
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
from tmlt.core.utils.testing import PySparkTest


class TestInvalidAttributes(unittest.TestCase):
    """Tests for invalid attributes on dataclasses."""

    @parameterized.expand(
        [
            (1001, TypeError, "type of source_id must be str; got int instead"),
            (" ", ValueError, "source_id must be a valid Python identifier."),
            (
                "space present",
                ValueError,
                "source_id must be a valid Python identifier.",
            ),
            (
                "2startsWithNumber",
                ValueError,
                "source_id must be a valid Python identifier.",
            ),
        ]
    )
    def test_invalid_private_source(
        self,
        invalid_source_id: str,
        exception_type: Type[Exception],
        expected_error_msg: str,
    ):
        """Tests that invalid private source errors on post-init."""
        with self.assertRaisesRegex(exception_type, expected_error_msg):
            PrivateSource(invalid_source_id)

    @parameterized.expand(
        [
            (True, "type of column_mapper must be a dict; got bool instead"),
            ({"A": 123}, r"type of column_mapper\['A'] must be str; got int instead"),
        ]
    )
    def test_invalid_rename(
        self, column_mapper: Dict[str, str], expected_error_msg: str
    ):
        """Tests that invalid Rename errors on post-init."""
        with self.assertRaisesRegex(TypeError, expected_error_msg):
            Rename(PrivateSource("private"), column_mapper)

    def test_invalid_rename_empty_string(self):
        """Test that rename doesn't allow you to rename columns to "" (empty string)."""
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                'Cannot rename column A to "" (the empty string): columns named ""'
                " are not allowed"
            ),
        ):
            Rename(PrivateSource("private"), {"A": ""})

    def test_invalid_filter(self):
        """Tests that invalid Filter errors on post-init."""
        with self.assertRaises(TypeError):
            Filter(PrivateSource("private"), 0)

    @parameterized.expand(
        [
            (True, "type of columns must be a list; got bool instead"),
            ([1], "type of columns[0] must be str; got int instead"),
            (["A", "B", "B"], "Column name appears more than once in ['A', 'B', 'B']"),
        ]
    )
    def test_invalid_select(self, columns: List[str], expected_error_msg: str):
        """Tests that invalid Rename errors on post-init."""
        with self.assertRaisesRegex(
            (ValueError, TypeError), re.escape(expected_error_msg)
        ):
            Select(PrivateSource("private"), columns)

    @parameterized.expand(
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
                "type of schema_new_columns must be "
                "tmlt.analytics._schema.Schema; got dict instead",
            ),
            (  # Grouping column in schema
                lambda row: {"C": 2 * str(row["B"])},
                Schema({"C": "VARCHAR"}, grouping_column="C"),
                True,
                "Map cannot be be used to create grouping columns",
            ),
        ]
    )
    def test_invalid_map(
        self,
        func: Callable,
        schema_new_columns: Schema,
        augment: bool,
        expected_error_msg: str,
    ):
        """Tests that invalid Map errors on post-init."""
        with self.assertRaisesRegex((TypeError, ValueError), expected_error_msg):
            Map(PrivateSource("private"), func, schema_new_columns, augment)

    @parameterized.expand(
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
                    max_num_rows=1,
                    schema_new_columns=Schema({"Repeat": "INTEGER"}),
                    augment=True,
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
                "schema_new_columns contains 2 columns, "
                "grouping flat map can only result in 1 new column",
            ),
        ]
    )
    def test_invalid_flatmap(
        self,
        child: QueryExpr,
        func: Callable,
        max_num_rows: int,
        schema_new_columns: Schema,
        augment: bool,
        expected_error_msg: str,
    ):
        """Tests that invalid FlatMap errors on post-init."""
        with self.assertRaisesRegex((TypeError, ValueError), expected_error_msg):
            FlatMap(child, func, max_num_rows, schema_new_columns, augment)

    @parameterized.expand(
        [
            ([], "Provided join columns must not be empty"),
            (["A", "A"], "Join columns must be distinct"),
        ]
    )
    def test_invalid_join_columns(
        self, join_columns: Optional[List[str]], expected_error_msg: str
    ):
        """Tests that JoinPrivate, JoinPublic error with invalid join columns."""
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            JoinPrivate(
                PrivateSource("private"),
                PrivateSource("private2"),
                TruncationStrategy.DropExcess(1),
                TruncationStrategy.DropExcess(1),
                join_columns,
            )
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            JoinPublic(PrivateSource("private"), "public", join_columns)

    @parameterized.expand(
        [
            (
                {"str": 100.0},
                re.escape(
                    r"type of replace_with['str'] must be a tuple; got float instead"
                ),
            ),
            (
                {"str": [100.0, 100.0]},
                re.escape(
                    r"type of replace_with['str'] must be a tuple; got list instead"
                ),
            ),
            ([], re.escape(r"type of replace_with must be a dict; got list instead")),
            (
                {"A": (-100.0,)},
                re.escape(
                    r"replace_with['A'] has wrong number of elements (expected 2, got 1"
                    r" instead)"
                ),
            ),
        ]
    )
    def test_invalid_replace_infinity(
        self, replace_with: Any, expected_error_msg: str
    ) -> None:
        """Test ReplaceInfinity with invalid arguments."""
        with self.assertRaisesRegex(TypeError, expected_error_msg):
            ReplaceInfinity(PrivateSource("private"), replace_with)

    @parameterized.expand(
        [
            ("A", "type of columns must be a list; got str instead"),
            (("A", "B"), "type of columns must be a list; got tuple instead"),
            ([1], re.escape(r"type of columns[0] must be str; got int instead")),
        ]
    )
    def test_invalid_drop_invalid(self, columns: Any, expected_error_msg: str) -> None:
        """Test DropInvalid with invalid arguments."""
        with self.assertRaisesRegex(TypeError, expected_error_msg):
            DropInvalid(PrivateSource("private"), columns)

    @parameterized.expand(
        [
            (
                PrivateSource("private"),
                KeySet.from_dict({}),
                "type of output_column must be str; got int instead",
                123,
            )
        ]
    )
    def test_invalid_groupbycount(
        self,
        child: QueryExpr,
        keys: KeySet,
        expected_error_msg: str,
        output_column: str = "count",
    ):
        """Tests that invalid GroupByCount errors on post-init."""
        with self.assertRaisesRegex(TypeError, expected_error_msg):
            GroupByCount(child, keys, output_column)

    @parameterized.expand(
        [
            (
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                "1.0",
                10.0,
                "type of low must be either float or int; got str instead",
            ),  # invalid lower dtype
            (
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                10.0,
                1,
                "Lower bound '10.0' can not be greater than the upper bound '1.0'.",
            ),  # lower > upper
        ]
    )
    def test_invalid_groupbyagg(
        self,
        keys: KeySet,
        measure_column: str,
        low: float,
        high: float,
        expected_error_msg: str,
    ):
        """Test invalid GroupBy aggregates errors on post-init."""
        for DataClass in [
            GroupByBoundedSum,
            GroupByBoundedAverage,
            GroupByBoundedVariance,
            GroupByBoundedSTDEV,
        ]:
            with self.assertRaisesRegex((TypeError, ValueError), expected_error_msg):
                DataClass(  # type: ignore
                    PrivateSource("private"), keys, measure_column, low, high
                )

    @parameterized.expand(
        [
            (
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                "0",
                8.0,
                10.0,
                "type of quantile must be either float or int; got str instead",
            ),  # invalid quantile dtype
            (
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                -1,
                8.0,
                10.0,
                "Quantile must be between 0 and 1, and not ",
            ),  # invalid quantile q < 0
            (
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                1.1,
                8.0,
                10.0,
                "Quantile must be between 0 and 1, and not ",
            ),  # invalid quantile q = 1
            (
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                0.5,
                "1.0",
                10.0,
                "type of low must be either float or int; got str instead",
            ),  # invalid lower dtype
            (
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                0.5,
                10.0,
                1.0,
                "Lower bound '10.0' can not be greater than the upper bound '1.0'.",
            ),  # lower > upper
        ]
    )
    def test_invalid_groupbyquantile(
        self,
        keys: KeySet,
        measure_column: str,
        quantile: float,
        low: float,
        high: float,
        expected_error_msg: str,
    ):
        """Test invalid GroupByQuantile."""
        with self.assertRaisesRegex((TypeError, ValueError), expected_error_msg):
            GroupByQuantile(
                PrivateSource("private"), keys, measure_column, quantile, low, high
            )


class TestValidAttributes(unittest.TestCase):
    """Tests for valid attributes on dataclasses."""

    @parameterized.expand([("private_source",), ("_Private",), ("no_space2",)])
    def test_valid_private_source(self, source_id: str):  # pylint: disable=no-self-use
        """Tests valid private source does not error."""
        PrivateSource(source_id)

    @parameterized.expand(
        [
            (8.0, 10.0),  # Both float
            (1, 10),  # Both int
            (1.0, 10),  # Different types; cast to float
        ]
    )
    def test_clamping_bounds_casting(self, low: float, high: float):
        """Test type of clamping bounds match on post-init."""
        for DataClass in [
            GroupByBoundedSum,
            GroupByBoundedAverage,
            GroupByBoundedVariance,
            GroupByBoundedSTDEV,
        ]:
            query = DataClass(  # type: ignore
                PrivateSource("private"),
                KeySet.from_dict({"A": ["0", "1"]}),
                "B",
                low,
                high,
            )
            # Help out mypy.
            assert isinstance(
                query,
                (
                    GroupByBoundedSum,
                    GroupByBoundedAverage,
                    GroupByBoundedVariance,
                    GroupByBoundedSTDEV,
                ),
            )
            self.assertEqual(type(query.low), type(query.high))

    @parameterized.expand(
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
        ]
    )
    def test_valid_replace_null_and_nan(
        self,
        child: QueryExpr,
        replace_with: Mapping[
            str, Union[int, float, str, datetime.date, datetime.datetime]
        ],
    ):  # pylint: disable=no-self-use
        """Test ReplaceNullAndNan creation with valid values."""
        # this should not raise an error
        ReplaceNullAndNan(child, replace_with)

    @parameterized.expand(
        [
            (PrivateSource("private"), {}),
            (PrivateSource("private"), {"A": (-100.0, 100.0)}),
            (PrivateSource("private"), {"A": (-1, 1)}),
            (PrivateSource("private"), {"A": (-999.9, 999.9), "B": (123.45, 678.90)}),
        ]
    )
    def test_valid_replace_infinity(
        self, child: QueryExpr, replace_with: Dict[str, Tuple[float, float]]
    ) -> None:
        """Test ReplaceInfinity with valid values."""
        query = ReplaceInfinity(child, replace_with)
        for v in query.replace_with.values():
            # Check that values got converted to floats
            self.assertEqual(len(v), 2)
            self.assertIsInstance(v[0], float)
            self.assertIsInstance(v[1], float)

    # pylint: disable=no-self-use
    @parameterized.expand(
        [
            (PrivateSource("private"), []),
            (PrivateSource("private"), ["A"]),
            (PrivateSource("different_private_source"), ["A", "B"]),
        ]
    )
    def test_valid_drop_invalid(self, child: QueryExpr, columns: List[str]) -> None:
        """Test DropInvalid with valid values."""
        DropInvalid(child, columns)

    # pylint: enable=no-self-use


class TestJoinPublicDataframe(PySparkTest):
    """Tests for JoinPublic with a Spark DataFrame as the public table."""

    def test_join_public_string_nan(self):
        """Test that the string "NaN" is allowed in string-valued columns."""
        df = self.spark.createDataFrame(
            pd.DataFrame({"col": ["nan", "NaN", "NAN", "Nan"]})
        )
        query_expr = JoinPublic(PrivateSource("a"), df)
        self.assert_frame_equal_with_sort(
            query_expr.public_table.toPandas(), df.toPandas()
        )

    def test_join_public_dataframe_validation_column_type(self):
        """Unsupported column types are rejected in JoinPublic."""
        data = [{"bytes": b"some bytes"}]
        schema = StructType([StructField("bytes", BinaryType(), False)])
        df = self.spark.createDataFrame(data, schema)

        with self.assertRaisesRegex(ValueError, "^Unsupported Spark data type.*"):
            JoinPublic(PrivateSource("a"), df)
