"""Tests for TransformationVisitor."""

# <placeholder: boilerplate>

import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import pandas as pd
from parameterized import parameterized
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.query_expr import AnalyticsDefault
from tmlt.analytics.query_expr import DropInfinity as DropInfExpr
from tmlt.analytics.query_expr import (
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
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.metrics import DictMetric, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.dictionary import GetValue
from tmlt.core.transformations.spark_transformations.filter import (
    Filter as FilterTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoin as PrivateJoinTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    PublicJoin as PublicJoinTransformation,
)
from tmlt.core.transformations.spark_transformations.join import (
    TruncationStrategy as CoreTruncationStrategy,
)
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap as FlatMapTransformation,
)
from tmlt.core.transformations.spark_transformations.map import GroupingFlatMap
from tmlt.core.transformations.spark_transformations.map import Map as MapTransformation
from tmlt.core.transformations.spark_transformations.nan import (
    DropInfs as DropInfTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropNaNs,
    DropNulls,
    ReplaceInfs,
    ReplaceNaNs,
    ReplaceNulls,
)
from tmlt.core.transformations.spark_transformations.rename import (
    Rename as RenameTransformation,
)
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.testing import PySparkTest


def chain_to_list(t: ChainTT) -> List[Transformation]:
    """Turns a ChainTT's tree into a list, in order from left to right."""
    left: List[Transformation]
    if not isinstance(t.transformation1, ChainTT):
        left = [t.transformation1]
    else:
        left = chain_to_list(t.transformation1)
    right: List[Transformation]
    if not isinstance(t.transformation2, ChainTT):
        right = [t.transformation2]
    else:
        right = chain_to_list(t.transformation2)
    return left + right


class TestTransformationVisitor(PySparkTest):
    """Test the TransformationVisitor."""

    def setUp(self) -> None:
        input_domain = DictDomain(
            {
                "private": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(allow_null=True),
                        "B": SparkIntegerColumnDescriptor(allow_null=True),
                        "X": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True, allow_inf=True
                        ),
                        "D": SparkDateColumnDescriptor(allow_null=True),
                        "T": SparkTimestampColumnDescriptor(allow_null=True),
                    }
                ),
                "private_2": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkIntegerColumnDescriptor(allow_null=True),
                    }
                ),
            }
        )
        input_metric = DictMetric(
            {"private": SymmetricDifference(), "private_2": SymmetricDifference()}
        )
        public_sources = {
            "public": self.spark.createDataFrame(
                pd.DataFrame({"A": ["zero", "one"], "B": [0, 1]}),
                schema=StructType(
                    [
                        StructField("A", StringType(), False),
                        StructField("B", LongType(), True),
                    ]
                ),
            )
        }
        self.visitor = TransformationVisitor(
            input_domain=input_domain,
            input_metric=input_metric,
            mechanism=NoiseMechanism.LAPLACE,
            public_sources=public_sources,
        )
        self.base_query = PrivateSource(source_id="private")

        self.catalog = Catalog()
        self.catalog.add_private_source(
            "private",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                "X": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
                ),
                "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
            },
            stability=3,
        )
        self.catalog.add_private_view(
            "private_2",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                "C": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            },
            stability=3,
        )
        self.catalog.add_public_source(
            "public",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR),
                "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            },
        )

    def _validate_transform_basics(self, t: Transformation, query: QueryExpr) -> None:
        self.assertEqual(t.input_domain, self.visitor.input_domain)
        self.assertEqual(t.input_metric, self.visitor.input_metric)
        self.assertIsInstance(t, ChainTT)
        assert isinstance(t, ChainTT)
        first_transform = chain_to_list(t)[0]
        self.assertIsInstance(first_transform, GetValue)

        expected_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        expected_output_metric = (
            SymmetricDifference()
            if expected_schema.grouping_column is None
            else IfGroupedBy(
                expected_schema.grouping_column, self.visitor.inner_metric()
            )
        )
        self.assertEqual(t.output_domain, expected_output_domain)
        self.assertEqual(t.output_metric, expected_output_metric)

    def _assert_dict_equal_without_ordering(
        self, d1: Mapping[Any, Any], d2: Mapping[Any, Any]
    ) -> None:
        """Assert that two dictionaries map the same keys to the same values.

        In Python, two dictionaries are equal if they contain the same key-value
        pairs *and* those pairs were created in the same order:
        `{'A': 'a1', 'B': 'b1'} != {'B': 'b1', 'A': 'a1'}`.
        """
        self.assertEqual(sorted(list(d1.keys())), sorted(list(d2.keys())))
        for k in list(d1.keys()):
            self.assertEqual(d1[k], d2[k])

    @parameterized.expand([("private",), ("private_2",)])
    def test_visit_private_source(self, source_id: "str") -> None:
        """Test visit_private_source"""
        query = PrivateSource(source_id=source_id)
        transformation = self.visitor.visit_private_source(query)
        self.assertIsInstance(transformation, GetValue)
        assert isinstance(transformation, GetValue)
        self.assertEqual(transformation.key, source_id)
        self.assertEqual(transformation.input_domain, self.visitor.input_domain)
        self.assertEqual(transformation.input_metric, self.visitor.input_metric)
        self.assertEqual(
            transformation.output_domain, self.visitor.input_domain[source_id]
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())

    def test_invalid_private_source(self) -> None:
        """Test visiting an invalid private source."""
        query = PrivateSource(source_id="source_that_does_not_exist")
        with self.assertRaises(ValueError):
            self.visitor.visit_private_source(query)

    @parameterized.expand([({"A": "columnA"},), ({"A": "aaaaa"},)])
    def test_visit_rename(self, mapper: Dict[str, str]) -> None:
        """Test visit_rename."""
        query = Rename(column_mapper=mapper, child=self.base_query)
        transformation = self.visitor.visit_rename(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, RenameTransformation)
        assert isinstance(transformation.transformation2, RenameTransformation)
        self.assertEqual(transformation.transformation2.rename_mapping, mapper)

    def test_visit_invalid_rename(self) -> None:
        """Test visit_rename with an invalid query."""
        query = Rename(
            column_mapper={"column_that_does_not_exit": "asdf"}, child=self.base_query
        )
        with self.assertRaises(ValueError):
            self.visitor.visit_rename(query)

    @parameterized.expand([("B > X",), ("A = 'ABC'",)])
    def test_visit_filter(self, filter_expr: str) -> None:
        """Test visit_filter."""
        query = Filter(predicate=filter_expr, child=self.base_query)
        transformation = self.visitor.visit_filter(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, FilterTransformation)
        assert isinstance(transformation.transformation2, FilterTransformation)
        self.assertEqual(transformation.transformation2.filter_expr, filter_expr)

    def test_visit_invalid_filter(self) -> None:
        """Test visit_filter with an invalid query."""
        query = Filter(predicate="not a valid predicate", child=self.base_query)
        with self.assertRaises(ValueError):
            self.visitor.visit_filter(query)

    @parameterized.expand([(["A"],), (["A", "B", "X"],)])
    def test_visit_select(self, columns: List[str]) -> None:
        """Test visit_select."""
        query = Select(columns=columns, child=self.base_query)
        transformation = self.visitor.visit_select(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, SelectTransformation)
        assert isinstance(transformation.transformation2, SelectTransformation)
        self.assertEqual(transformation.transformation2.columns, columns)

    def test_visit_invalid_select(self) -> None:
        """Test visit_select with invalid query."""
        query = Select(columns=["column_that_does_not_exist"], child=self.base_query)
        with self.assertRaises(ValueError):
            self.visitor.visit_select(query)

    @parameterized.expand(
        [
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {"C": 2 * str(row["B"])},
                    schema_new_columns=Schema({"C": "VARCHAR"}),
                    augment=True,
                ),
            ),
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {"C": 2 * str(row["B"])},
                    schema_new_columns=Schema({"C": "VARCHAR"}),
                    augment=False,
                ),
            ),
        ]
    )
    def test_visit_map(self, query: Map) -> None:
        """Test visit_map."""
        transformation = self.visitor.visit_map(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, MapTransformation)
        assert isinstance(transformation.transformation2, MapTransformation)
        transformer = transformation.transformation2.row_transformer
        self.assertEqual(transformer.augment, query.augment)

    @parameterized.expand(
        [
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda row: [{"A_is_zero": 1 if row["A"] == "zero" else 2}],
                    max_num_rows=1,
                    schema_new_columns=Schema({"A_is_zero": "INTEGER"}),
                    augment=True,
                ),
            ),
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda row: [{"i": n for n in range(row["B"] + 1)}],
                    max_num_rows=10,
                    schema_new_columns=Schema({"i": "DECIMAL"}),
                    augment=False,
                ),
            ),
        ]
    )
    def test_visit_flat_map_without_grouping(self, query: FlatMap) -> None:
        """Test visit_flat_map when query has no grouping_column."""
        transformation = self.visitor.visit_flat_map(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, FlatMapTransformation)
        assert isinstance(transformation.transformation2, FlatMapTransformation)
        flat_map_transform = transformation.transformation2
        self.assertEqual(flat_map_transform.max_num_rows, query.max_num_rows)
        self.assertEqual(flat_map_transform.row_transformer.augment, query.augment)

    @parameterized.expand(
        [
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda row: [{"Group": 0 if row["X"] == 0 else 17}],
                    max_num_rows=2,
                    schema_new_columns=Schema(
                        {"Group": ColumnDescriptor(ColumnType.INTEGER)},
                        grouping_column="Group",
                    ),
                    augment=True,
                ),
            )
        ]
    )
    def test_visit_flat_map_with_grouping(self, query: FlatMap) -> None:
        """Test visit_flat_map when query has a grouping_column."""
        transformation = self.visitor.visit_flat_map(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, GroupingFlatMap)
        assert isinstance(transformation.transformation2, GroupingFlatMap)
        group_map_transform = transformation.transformation2
        self.assertEqual(group_map_transform.max_num_rows, query.max_num_rows)
        self.assertEqual(group_map_transform.row_transformer.augment, query.augment)

    @parameterized.expand(
        [
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropExcess(10),
                ),
                CoreTruncationStrategy.TRUNCATE,
                3,
                CoreTruncationStrategy.TRUNCATE,
                10,
                ["A"],
            ),
            (
                JoinPrivate(
                    child=PrivateSource("private_2"),
                    right_operand_expr=PrivateSource("private"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropNonUnique(),
                    join_columns=["A"],
                ),
                CoreTruncationStrategy.TRUNCATE,
                3,
                CoreTruncationStrategy.DROP,
                1,
                ["A"],
            ),
        ]
    )
    def test_visit_join_private(
        self,
        query: JoinPrivate,
        expected_left_truncation_strategy: CoreTruncationStrategy,
        expected_left_truncation_threshold: int,
        expected_right_truncation_strategy: CoreTruncationStrategy,
        expected_right_truncation_threshold: int,
        expected_join_cols: List[str],
    ) -> None:
        """Test visit_join_private."""
        transformation = self.visitor.visit_join_private(query)

        self.assertEqual(transformation.input_domain, self.visitor.input_domain)
        self.assertEqual(transformation.input_metric, self.visitor.input_metric)
        self.assertIsInstance(transformation, ChainTT)

        expected_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        expected_output_metric = (
            SymmetricDifference()
            if expected_schema.grouping_column is None
            else IfGroupedBy(
                expected_schema.grouping_column, self.visitor.inner_metric()
            )
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.output_metric, expected_output_metric)

        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, PrivateJoinTransformation)
        assert isinstance(transformation.transformation2, PrivateJoinTransformation)
        private_join_transform = cast(
            PrivateJoinTransformation, transformation.transformation2
        )
        self.assertEqual(
            private_join_transform.left_truncation_strategy,
            expected_left_truncation_strategy,
        )
        self.assertEqual(
            private_join_transform.right_truncation_strategy,
            expected_right_truncation_strategy,
        )
        self.assertEqual(
            private_join_transform.left_truncation_threshold,
            expected_left_truncation_threshold,
        )
        self.assertEqual(
            private_join_transform.right_truncation_threshold,
            expected_right_truncation_threshold,
        )

        self.assertEqual(private_join_transform.join_cols, expected_join_cols)

    def test_visit_join_private_with_invalid_truncation_strategy(self) -> None:
        """Test visit_join_private raises an error with an invalid strategy."""

        class InvalidStrategy(TruncationStrategy.Type):
            """An invalid truncation strategy."""

        query1 = JoinPrivate(
            child=self.base_query,
            right_operand_expr=PrivateSource("private_2"),
            truncation_strategy_left=InvalidStrategy(),
            truncation_strategy_right=TruncationStrategy.DropExcess(3),
        )
        expected_error_msg = (
            f"Truncation strategy type {InvalidStrategy.__qualname__} is not supported."
        )
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            self.visitor.visit_join_private(query1)

        query2 = JoinPrivate(
            child=self.base_query,
            right_operand_expr=PrivateSource("private_2"),
            truncation_strategy_left=TruncationStrategy.DropExcess(2),
            truncation_strategy_right=InvalidStrategy(),
        )
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            self.visitor.visit_join_private(query2)

    @parameterized.expand([("public", None), ("public", ["A", "B"])])
    def test_visit_join_public_str(
        self, source_id: str, join_columns: Optional[List[str]]
    ) -> None:
        """Test visit_join_public with a string identifying the public source."""
        query = JoinPublic(
            child=self.base_query, public_table=source_id, join_columns=join_columns
        )
        transformation = self.visitor.visit_join_public(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, PublicJoinTransformation)
        assert isinstance(transformation.transformation2, PublicJoinTransformation)
        public_join_transform = transformation.transformation2
        if join_columns is not None:
            self.assertEqual(public_join_transform.join_cols, join_columns)
        got_df = public_join_transform.public_df
        self.assert_frame_equal_with_sort(
            got_df.toPandas(), self.visitor.public_sources[source_id].toPandas()
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": ["asdf", "qwer"], "B": [0, 1]}),
                StructType(
                    [
                        StructField("A", StringType(), False),
                        StructField("B", LongType(), False),
                    ]
                ),
                ["A", "B"],
                False,
            ),
            (
                pd.DataFrame({"A": [None, "abc", "def"], "new_column": [0, 1, 2]}),
                StructType(
                    [
                        StructField("A", StringType(), True),
                        StructField("new_column", LongType(), False),
                    ]
                ),
                ["A"],
                True,
            ),
        ]
    )
    def test_visit_join_public_df(
        self,
        df: pd.DataFrame,
        df_schema: StructType,
        expected_join_cols: List[str],
        expected_join_on_null: bool,
    ) -> None:
        """Test visit_join_public with a dataframe."""
        public_df = self.spark.createDataFrame(df, schema=df_schema)
        query = JoinPublic(child=self.base_query, public_table=public_df)
        transformation = self.visitor.visit_join_public(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        self.assertIsInstance(transformation.transformation2, PublicJoinTransformation)
        assert isinstance(transformation.transformation2, PublicJoinTransformation)
        public_join_transform = transformation.transformation2
        self.assertEqual(public_join_transform.join_cols, expected_join_cols)
        got_df = public_join_transform.public_df
        self.assert_frame_equal_with_sort(got_df.toPandas(), public_df.toPandas())
        # pylint: disable=protected-access
        self.assertEqual(public_join_transform._join_on_nulls, expected_join_on_null)
        # pylint: enable=protected-access

    @parameterized.expand(
        [
            (
                {},
                {
                    "A": "",
                    "B": 0,
                    "X": 0.0,
                    "D": datetime.date.fromtimestamp(0),
                    "T": datetime.datetime.fromtimestamp(0),
                },
                True,
            ),
            ({"X": 0}, {"X": 0.0}, True),
            ({"A": "replacement_str"}, {"A": "replacement_str"}, False),
        ]
    )
    def test_visit_replace_null_and_nan(
        self,
        replace_with: Mapping[
            str, Union[int, float, str, datetime.date, datetime.datetime]
        ],
        expected_replace_with: Mapping[
            str, Union[int, float, str, datetime.date, datetime.datetime]
        ],
        expect_nan_replacement: bool,
    ):
        """Test visit_replace_null_and_nan."""
        query = ReplaceNullAndNan(child=self.base_query, replace_with=replace_with)
        transformation = self.visitor.visit_replace_null_and_nan(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        expected_output_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            schema=analytics_to_spark_columns_descriptor(expected_output_schema)
        )
        transformations = chain_to_list(transformation)

        replace_transform: ReplaceNulls
        if expect_nan_replacement:
            self.assertEqual(len(transformations), 3)
            self.assertIsInstance(transformations[1], ReplaceNulls)
            assert isinstance(transformations[1], ReplaceNulls)
            replace_transform = transformations[1]

            nan_transform = transformations[2]
            self.assertIsInstance(nan_transform, ReplaceNaNs)
            assert isinstance(nan_transform, ReplaceNaNs)
            expected_nan_replace = {
                k: v
                for k, v in expected_replace_with.items()
                if expected_output_schema[k].column_type == ColumnType.DECIMAL
            }
            self._assert_dict_equal_without_ordering(
                expected_nan_replace, nan_transform.replace_map
            )
        else:
            self.assertEqual(len(transformations), 2)
            self.assertIsInstance(transformations[1], ReplaceNulls)
            assert isinstance(transformations[1], ReplaceNulls)
            replace_transform = transformations[1]

        expected_output_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            schema=analytics_to_spark_columns_descriptor(expected_output_schema)
        )
        self.assertEqual(expected_output_domain, transformation.output_domain)
        self._assert_dict_equal_without_ordering(
            expected_replace_with, replace_transform.replace_map
        )

    def test_visit_replace_null_and_nan_with_grouping_column(self) -> None:
        """Test behavior of visit_replace_null_and_nan with IfGroupedBy metric."""
        flatmap_query = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Group": 0 if row["X"] == 0 else 17}],
            max_num_rows=2,
            schema_new_columns=Schema(
                {"Group": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
                grouping_column="Group",
            ),
            augment=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "Cannot replace null values in column Group, because it is being used as a"
            " grouping column",
        ):
            invalid_replace_query = ReplaceNullAndNan(
                child=flatmap_query, replace_with={"Group": -10}
            )
            self.visitor.visit_replace_null_and_nan(invalid_replace_query)
        valid_replace_query = ReplaceNullAndNan(child=flatmap_query, replace_with={})
        expected_replace_with = {
            "A": "",
            "B": 0,
            "X": 0.0,
            "D": datetime.date.fromtimestamp(0),
            "T": datetime.datetime.fromtimestamp(0),
        }
        transformation = self.visitor.visit_replace_null_and_nan(valid_replace_query)
        self._validate_transform_basics(transformation, valid_replace_query)
        self.assertIsInstance(transformation, ChainTT)
        assert isinstance(transformation, ChainTT)
        transformations = chain_to_list(transformation)
        self.assertIsInstance(transformations[0], GetValue)
        self.assertIsInstance(transformations[1], GroupingFlatMap)
        self.assertIsInstance(transformations[2], ReplaceNulls)
        assert isinstance(transformations[2], ReplaceNulls)
        replace_nulls = transformations[2]
        self._assert_dict_equal_without_ordering(
            replace_nulls.replace_map, expected_replace_with
        )
        expected_replace_nan = {"X": 0.0}
        self.assertEqual(len(transformations), 4)
        self.assertIsInstance(transformations[3], ReplaceNaNs)
        assert isinstance(transformations[3], ReplaceNaNs)
        replace_nans = transformations[3]
        self.assertEqual(replace_nans.replace_map, expected_replace_nan)

    @parameterized.expand(
        [
            ({}, {"X": (AnalyticsDefault.DECIMAL, AnalyticsDefault.DECIMAL)}),
            ({"X": (-100.0, 100.0)}, {"X": (-100.0, 100.0)}),
        ]
    )
    def test_visit_replace_infinity(
        self,
        replace_with: Dict[str, Tuple[float, float]],
        expected_replace_with: Dict[str, Tuple[float, float]],
    ):
        """Test visit_replace_infinity."""
        query = ReplaceInfinity(child=self.base_query, replace_with=replace_with)
        transformation = self.visitor.visit_replace_infinity(query)
        self._validate_transform_basics(transformation, query)
        assert isinstance(transformation, ChainTT)
        expected_output_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            schema=analytics_to_spark_columns_descriptor(expected_output_schema)
        )
        self.assertIsInstance(transformation.transformation2, ReplaceInfs)
        assert isinstance(transformation.transformation2, ReplaceInfs)
        replace_transform = transformation.transformation2

        self.assertEqual(expected_output_domain, replace_transform.output_domain)
        self._assert_dict_equal_without_ordering(
            expected_replace_with, replace_transform.replace_map
        )

    def test_visit_drop_null_and_nan_with_grouping_column(self) -> None:
        """Test behavior of visit_drop_null_and_nan with IfGroupedBy metric."""
        flatmap_query = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Group": 0 if row["X"] == 0 else 17}],
            max_num_rows=2,
            schema_new_columns=Schema(
                {"Group": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
                grouping_column="Group",
            ),
            augment=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "Cannot drop null values in column Group, because it is being used as a"
            " grouping column",
        ):
            invalid_drop_query = DropNullAndNan(child=flatmap_query, columns=["Group"])
            self.visitor.visit_drop_null_and_nan(invalid_drop_query)
        valid_drop_query = DropNullAndNan(child=flatmap_query, columns=[])
        expected_columns = ["A", "B", "X", "D", "T"]
        t = self.visitor.visit_drop_null_and_nan(valid_drop_query)
        self._validate_transform_basics(t, valid_drop_query)
        self.assertIsInstance(t, ChainTT)
        assert isinstance(t, ChainTT)
        transformations = chain_to_list(t)
        self.assertIsInstance(transformations[0], GetValue)
        self.assertIsInstance(transformations[1], GroupingFlatMap)
        self.assertIsInstance(transformations[2], DropNulls)
        assert isinstance(transformations[2], DropNulls)
        self.assertEqual(
            sorted(set(transformations[2].columns)), sorted(set(expected_columns))
        )

    def test_visit_drop_infinity_with_grouping_column(self) -> None:
        """Test behavior of visit_drop_infinity with IfGroupedBy metric."""
        flatmap_query = FlatMap(
            child=PrivateSource("private"),
            f=lambda row: [{"Group": 0 if row["X"] == 0 else 17}],
            max_num_rows=2,
            schema_new_columns=Schema(
                {"Group": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
                grouping_column="Group",
            ),
            augment=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "Cannot drop infinite values in column Group, because it is being used as a"
            " grouping column",
        ):
            invalid_drop_query = DropInfExpr(child=flatmap_query, columns=["Group"])
            self.visitor.visit_drop_infinity(invalid_drop_query)
        valid_drop_query = DropInfExpr(child=flatmap_query, columns=[])
        expected_columns = ["X"]
        t = self.visitor.visit_drop_infinity(valid_drop_query)
        self._validate_transform_basics(t, valid_drop_query)
        self.assertIsInstance(t, ChainTT)
        assert isinstance(t, ChainTT)
        transformations = chain_to_list(t)
        self.assertIsInstance(transformations[0], GetValue)
        self.assertIsInstance(transformations[1], GroupingFlatMap)
        self.assertIsInstance(transformations[2], DropInfTransformation)
        assert isinstance(transformations[2], DropInfTransformation)
        self.assertEqual(
            sorted(set(transformations[2].columns)), sorted(set(expected_columns))
        )

    def test_measurement_visits(self):
        """Test that visiting measurement queries raises an error."""
        with self.assertRaises(NotImplementedError):
            self.visitor.visit_groupby_count(
                GroupByCount(groupby_keys=KeySet.from_dict({}), child=self.base_query)
            )

        with self.assertRaises(NotImplementedError):
            self.visitor.visit_groupby_count_distinct(
                GroupByCountDistinct(
                    groupby_keys=KeySet.from_dict({}), child=self.base_query
                )
            )

        with self.assertRaises(NotImplementedError):
            self.visitor.visit_groupby_quantile(
                GroupByQuantile(
                    child=self.base_query,
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    quantile=0.1,
                    low=0,
                    high=1,
                )
            )

        with self.assertRaises(NotImplementedError):
            self.visitor.visit_groupby_bounded_sum(
                GroupByBoundedSum(
                    child=self.base_query,
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )

        with self.assertRaises(NotImplementedError):
            self.visitor.visit_groupby_bounded_average(
                GroupByBoundedAverage(
                    child=self.base_query,
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )

        with self.assertRaises(NotImplementedError):
            self.visitor.visit_groupby_bounded_variance(
                GroupByBoundedVariance(
                    child=self.base_query,
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )

        with self.assertRaises(NotImplementedError):
            self.visitor.visit_groupby_bounded_stdev(
                GroupByBoundedSTDEV(
                    child=self.base_query,
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )


class TestTransformationVisitorWithComplexSchema(PySparkTest):
    """Test the TransformationVisitor with a complicated schema."""

    def setUp(self) -> None:
        input_domain = DictDomain(
            {
                "private": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(allow_null=True),
                        "B": SparkIntegerColumnDescriptor(allow_null=True),
                        "NOTNULL": SparkFloatColumnDescriptor(allow_null=False),
                        "null": SparkFloatColumnDescriptor(allow_null=True),
                        "nan": SparkFloatColumnDescriptor(allow_nan=True),
                        "inf": SparkFloatColumnDescriptor(allow_inf=True),
                        "null_and_nan": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True
                        ),
                        "null_and_inf": SparkFloatColumnDescriptor(
                            allow_null=True, allow_inf=True
                        ),
                        "nan_and_inf": SparkFloatColumnDescriptor(
                            allow_nan=True, allow_inf=True
                        ),
                        "null_and_nan_and_inf": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True, allow_inf=True
                        ),
                        "D": SparkDateColumnDescriptor(allow_null=True),
                        "T": SparkTimestampColumnDescriptor(allow_null=True),
                    }
                )
            }
        )
        input_metric = DictMetric({"private": SymmetricDifference()})
        self.visitor = TransformationVisitor(
            input_domain=input_domain,
            input_metric=input_metric,
            mechanism=NoiseMechanism.LAPLACE,
            public_sources={},
        )
        self.base_query = PrivateSource(source_id="private")

        self.catalog = Catalog()
        self.catalog.add_private_source(
            "private",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                "NOTNULL": ColumnDescriptor(ColumnType.DECIMAL, allow_null=False),
                "null": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
                "nan": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True),
                "inf": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
                "null_and_nan": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True
                ),
                "null_and_inf": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_inf=True
                ),
                "nan_and_inf": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_nan=True, allow_inf=True
                ),
                "null_and_nan_and_inf": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
                ),
                "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
            },
            stability=3,
        )

    def _validate_transform_basics(self, t: Transformation, query: QueryExpr) -> None:
        """Check the basics of a transformation."""
        self.assertEqual(t.input_domain, self.visitor.input_domain)
        self.assertEqual(t.input_metric, self.visitor.input_metric)
        first_transform: Transformation
        if isinstance(t, ChainTT):
            first_transform = chain_to_list(t)[0]
        else:
            first_transform = t
        self.assertIsInstance(first_transform, GetValue)

        expected_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        expected_output_metric = (
            SymmetricDifference()
            if expected_schema.grouping_column is None
            else IfGroupedBy(
                expected_schema.grouping_column, self.visitor.inner_metric()
            )
        )
        self.assertEqual(t.output_domain, expected_output_domain)
        self.assertEqual(t.output_metric, expected_output_metric)

    @parameterized.expand(
        [
            (["A"], ["A"], []),
            (["A", "B", "D", "T"], ["A", "B", "D", "T"], []),
            (["NOTNULL"], [], []),
            (["null", "nan", "inf"], ["null"], ["nan"]),
            (
                ["null_and_nan", "null_and_inf", "nan_and_inf"],
                ["null_and_nan", "null_and_inf"],
                ["null_and_nan", "nan_and_inf"],
            ),
            (
                ["null", "nan", "inf", "null_and_nan_and_inf"],
                ["null", "null_and_nan_and_inf"],
                ["nan", "null_and_nan_and_inf"],
            ),
            (
                [],
                [
                    "A",
                    "B",
                    "null",
                    "null_and_nan",
                    "null_and_inf",
                    "null_and_nan_and_inf",
                    "D",
                    "T",
                ],
                ["nan", "null_and_nan", "nan_and_inf", "null_and_nan_and_inf"],
            ),
        ]
    )
    def test_visit_drop_null_and_nan(
        self,
        query_columns: List[str],
        expected_null_cols: List[str],
        expected_nan_cols: List[str],
    ) -> None:
        """Test visit_drop_invalid."""
        query = DropNullAndNan(child=PrivateSource("private"), columns=query_columns)
        transform = self.visitor.visit_drop_null_and_nan(query)
        self._validate_transform_basics(transform, query)
        if not expected_null_cols and not expected_nan_cols:
            # There should just be a GetValue transformation
            self.assertIsInstance(transform, GetValue)
            # nothing else to test!
            return
        self.assertIsInstance(transform, ChainTT)
        assert isinstance(transform, ChainTT)
        transformations = chain_to_list(transform)
        # Pop the get_value transformation off the front of the list
        # (_validate_transform_basics checks that the first transformation
        # is a GetValue transformation)
        transformations.pop(0)

        # We expect transformations to happen in this order:
        # DropNulls -> DropNaNs
        # but each one will only be present if it makes sense
        if expected_null_cols:
            null_transform = transformations.pop(0)
            self.assertIsInstance(null_transform, DropNulls)
            assert isinstance(null_transform, DropNulls)
            self.assertEqual(sorted(null_transform.columns), sorted(expected_null_cols))
        if expected_nan_cols:
            nan_transform = transformations.pop(0)
            self.assertIsInstance(nan_transform, DropNaNs)
            assert isinstance(nan_transform, DropNaNs)
            self.assertEqual(sorted(nan_transform.columns), sorted(expected_nan_cols))

    @parameterized.expand(
        [
            (["inf"], ["inf"]),
            (["null", "nan", "inf"], ["null", "nan", "inf"]),
            (
                ["null_and_nan", "null_and_inf", "nan_and_inf"],
                ["null_and_nan", "null_and_inf", "nan_and_inf"],
            ),
            (
                ["null", "nan", "inf", "null_and_nan_and_inf"],
                ["null", "nan", "inf", "null_and_nan_and_inf"],
            ),
            ([], ["inf", "null_and_inf", "nan_and_inf", "null_and_nan_and_inf"]),
        ]
    )
    def test_visit_drop_infinity(
        self, query_columns: List[str], expected_inf_cols: List[str]
    ) -> None:
        """Test visit_drop_infinity."""
        query = DropInfExpr(child=PrivateSource("private"), columns=query_columns)
        transform = self.visitor.visit_drop_infinity(query)
        self._validate_transform_basics(transform, query)
        self.assertIsInstance(transform, ChainTT)
        assert isinstance(transform, ChainTT)
        transformations = chain_to_list(transform)
        # Pop the get_value transformation off the front of the list
        # (_validate_transform_basics checks that the first transformation
        # is a GetValue transformation)
        transformations.pop(0)

        inf_transform = transformations.pop(0)
        self.assertIsInstance(inf_transform, DropInfTransformation)
        assert isinstance(inf_transform, DropInfTransformation)
        self.assertEqual(sorted(inf_transform.columns), sorted(expected_inf_cols))
