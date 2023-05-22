"""Tests for TransformationVisitor."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, StringType, StructField, StructType
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.dictionary import AugmentDictTransformation
from tmlt.core.transformations.identity import Identity as IdentityTransformation

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
from tmlt.analytics._table_identifier import Identifier
from tmlt.analytics._table_reference import TableReference, lookup_domain, lookup_metric
from tmlt.analytics.keyset import KeySet
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

from .conftest import (
    DATE1,
    TIMESTAMP1,
    TestTransformationVisitor,
    TestTransformationVisitorNulls,
    chain_to_list,
)


class TestAddRows(TestTransformationVisitor):
    """Tests for the transformation visitor using for AddMaxRows."""

    visitor: TransformationVisitor
    catalog: Catalog
    input_data: Dict[Identifier, Union[DataFrame, Dict[Identifier, DataFrame]]]
    dataframes: Dict[str, DataFrame]

    def _validate_transform_basics(
        self, t: Transformation, reference: TableReference, query: QueryExpr
    ) -> None:
        assert t.input_domain == self.visitor.input_domain
        assert t.input_metric == self.visitor.input_metric
        assert isinstance(t, ChainTT)
        first_transform = chain_to_list(t)[0]
        assert isinstance(first_transform, IdentityTransformation)

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
        assert lookup_domain(t.output_domain, reference) == expected_output_domain
        assert lookup_metric(t.output_metric, reference) == expected_output_metric

    @pytest.mark.parametrize("source_id", ["rows1", "rows2"])
    def test_visit_private_source(self, source_id: str) -> None:
        """Test visit_private_source"""
        query = PrivateSource(source_id=source_id)
        transformation, reference, constraints = query.accept(self.visitor)
        assert isinstance(transformation, IdentityTransformation)
        assert isinstance(reference, TableReference)
        assert isinstance(reference.identifier, Identifier)
        assert lookup_domain(transformation.output_domain, reference) == lookup_domain(
            self.visitor.input_domain, reference
        )
        assert (
            lookup_metric(transformation.output_metric, reference)
            == SymmetricDifference()
        )
        assert constraints == []

    def test_invalid_private_source(self) -> None:
        """Test visiting an invalid private source."""
        query = PrivateSource(source_id="source_that_does_not_exist")
        with pytest.raises((KeyError, ValueError)):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "mapper,expected_df",
        [
            (
                {"S": "columnS"},
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1]],
                    columns=["columnS", "I", "F", "D", "T"],
                ),
            ),
            (
                {"D": "date", "T": "time"},
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1]],
                    columns=["S", "I", "F", "date", "time"],
                ),
            ),
        ],
    )
    def test_visit_rename(self, mapper: Dict[str, str], expected_df: DataFrame) -> None:
        """Test visit_rename."""
        query = Rename(column_mapper=mapper, child=PrivateSource(source_id="rows1"))
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        # check dataframe renamed as expected
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_invalid_rename(self) -> None:
        """Test visit_rename with an invalid query."""
        query = Rename(
            column_mapper={"column_that_does_not_exit": "asdf"},
            child=PrivateSource(source_id="rows1"),
        )
        with pytest.raises(ValueError):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "filter_expr,expected_df",
        [
            (
                "F > I",
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1]],
                    columns=["S", "I", "F", "D", "T"],
                ),
            ),
            ("S = 'ABC'", pd.DataFrame(columns=["S", "I", "F", "D", "T"])),
        ],
    )
    def test_visit_filter(self, filter_expr: str, expected_df: DataFrame) -> None:
        """Test visit_filter."""
        query = Filter(condition=filter_expr, child=PrivateSource(source_id="rows1"))
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_invalid_filter(self) -> None:
        """Test visit_filter with an invalid query."""
        query = Filter(
            condition="not a valid condition", child=PrivateSource(source_id="rows1")
        )
        with pytest.raises(ValueError):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "columns,expected_df",
        [
            (["S"], pd.DataFrame([["0"]], columns=["S"])),
            (["S", "I", "F"], pd.DataFrame([["0", 0, 0.1]], columns=["S", "I", "F"])),
        ],
    )
    def test_visit_select(self, columns: List[str], expected_df: DataFrame) -> None:
        """Test visit_select."""
        query = Select(columns=columns, child=PrivateSource(source_id="rows1"))
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_invalid_select(self) -> None:
        """Test visit_select with invalid query."""
        query = Select(
            columns=["column_that_does_not_exist"],
            child=PrivateSource(source_id="rows1"),
        )
        with pytest.raises(ValueError):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                Map(
                    child=PrivateSource("rows1"),
                    f=lambda row: {"X": 2 * str(row["S"])},
                    schema_new_columns=Schema({"X": "VARCHAR"}),
                    augment=True,
                ),
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, "00"]],
                    columns=["S", "I", "F", "D", "T", "X"],
                ),
            ),
            (
                Map(
                    child=PrivateSource("rows1"),
                    f=lambda row: {"X": 2 * str(row["S"])},
                    schema_new_columns=Schema({"X": "VARCHAR"}),
                    augment=False,
                ),
                pd.DataFrame([["00"]], columns=["X"]),
            ),
        ],
    )
    def test_visit_map(self, query: Map, expected_df: DataFrame) -> None:
        """Test visit_map."""
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                FlatMap(
                    child=PrivateSource("rows1"),
                    f=lambda row: [{"S_is_zero": 1 if row["S"] == "0" else 2}],
                    schema_new_columns=Schema({"S_is_zero": "INTEGER"}),
                    augment=True,
                    max_num_rows=1,
                ),
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, 1]],
                    columns=["S", "I", "F", "D", "T", "S_is_zero"],
                ),
            ),
            (
                FlatMap(
                    child=PrivateSource("rows1"),
                    f=lambda row: [{"i": n for n in range(row["I"] + 1)}],
                    schema_new_columns=Schema({"i": "INTEGER"}),
                    augment=False,
                    max_num_rows=10,
                ),
                pd.DataFrame([[0]], columns=["i"]),
            ),
            (
                FlatMap(
                    child=PrivateSource("rows1"),
                    f=lambda row: [{"i": n} for n in range(row["I"] + 10)],
                    schema_new_columns=Schema({"i": "INTEGER"}),
                    augment=False,
                    max_num_rows=3,
                ),
                pd.DataFrame([[0], [1], [2]], columns=["i"]),
            ),
        ],
    )
    def test_visit_flat_map_without_grouping(
        self, query: FlatMap, expected_df: DataFrame
    ) -> None:
        """Test visit_flat_map when query has no grouping_column."""
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                FlatMap(
                    child=PrivateSource("rows1"),
                    f=lambda row: [{"group": 0 if row["F"] == 0 else 17}],
                    schema_new_columns=Schema(
                        {"group": ColumnDescriptor(ColumnType.INTEGER)},
                        grouping_column="group",
                    ),
                    augment=True,
                    max_num_rows=2,
                ),
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, 17]],
                    columns=["S", "I", "F", "D", "T", "group"],
                ),
            )
        ],
    )
    def test_visit_flat_map_with_grouping(
        self, query: FlatMap, expected_df: DataFrame
    ) -> None:
        """Test visit_flat_map when query has a grouping_column."""
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_flat_map_invalid(self) -> None:
        """Test visit_flat_map with invalid query."""
        query = FlatMap(
            child=PrivateSource("rows1"),
            f=lambda row: [{"group": 0 if row["F"] == 0 else 17}],
            schema_new_columns=Schema(
                {"group": ColumnDescriptor(ColumnType.INTEGER)}, grouping_column="group"
            ),
            augment=True,
            max_num_rows=None,
        )
        with pytest.raises(
            ValueError,
            match=(
                "Flat maps on tables without IDs must have"
                " a defined max_num_rows parameter."
            ),
        ):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                JoinPrivate(
                    child=PrivateSource("rows1"),
                    right_operand_expr=PrivateSource("rows2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropExcess(10),
                ),
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, "a"]],
                    columns=["S", "I", "F", "D", "T", "field"],
                ),
            ),
            (
                JoinPrivate(
                    child=PrivateSource("rows2"),
                    right_operand_expr=PrivateSource("rows1"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropNonUnique(),
                    join_columns=["I"],
                ),
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, "a"]],
                    columns=["S", "I", "F", "D", "T", "field"],
                ),
            ),
        ],
    )
    def test_visit_join_private(
        self, query: JoinPrivate, expected_df: DataFrame
    ) -> None:
        """Test visit_join_private."""
        transformation, reference, constraints = query.accept(self.visitor)
        assert transformation.input_domain == self.visitor.input_domain
        assert transformation.input_metric == self.visitor.input_metric

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
        assert (
            lookup_domain(transformation.output_domain, reference)
            == expected_output_domain
        )
        assert (
            lookup_metric(transformation.output_metric, reference)
            == expected_output_metric
        )

        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_join_private_with_invalid_truncation_strategy(self) -> None:
        """Test visit_join_private raises an error with an invalid strategy."""

        class InvalidStrategy(TruncationStrategy.Type):
            """An invalid truncation strategy."""

        query1 = JoinPrivate(
            child=PrivateSource("rows1"),
            right_operand_expr=PrivateSource("rows2"),
            truncation_strategy_left=InvalidStrategy(),
            truncation_strategy_right=TruncationStrategy.DropExcess(3),
        )
        expected_error_msg = (
            f"Truncation strategy type {InvalidStrategy.__qualname__} is not supported."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            query1.accept(self.visitor)

        query2 = JoinPrivate(
            child=PrivateSource("rows1"),
            right_operand_expr=PrivateSource("rows2"),
            truncation_strategy_left=TruncationStrategy.DropExcess(2),
            truncation_strategy_right=InvalidStrategy(),
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            query2.accept(self.visitor)

        query3 = JoinPrivate(
            child=PrivateSource("rows1"),
            right_operand_expr=PrivateSource("rows2"),
            truncation_strategy_left=None,
            truncation_strategy_right=None,
        )
        with pytest.raises(
            ValueError,
            match="When joining without IDs, truncation strategies are required.",
        ):
            self.visitor.visit_join_private(query3)

    @pytest.mark.parametrize(
        "source_id,join_columns,expected_df",
        [
            (
                "public",
                None,
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, "x"]],
                    columns=["S", "I", "F", "D", "T", "public"],
                ),
            ),
            (
                "public",
                ["S", "I"],
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, "x"]],
                    columns=["S", "I", "F", "D", "T", "public"],
                ),
            ),
        ],
    )
    def test_visit_join_public_str(
        self, source_id: str, join_columns: Optional[List[str]], expected_df: DataFrame
    ) -> None:
        """Test visit_join_public with a string identifying the public source."""
        query = JoinPublic(
            child=PrivateSource(source_id="rows1"),
            public_table=source_id,
            join_columns=join_columns,
        )
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "df,df_schema,expected_df",
        [
            (
                pd.DataFrame({"S": ["asdf", "qwer"], "I": [0, 1]}),
                StructType(
                    [
                        StructField("S", StringType(), True),
                        StructField("I", LongType(), True),
                    ]
                ),
                pd.DataFrame([], columns=["S", "I", "F", "D", "T"]),
            ),
            (
                pd.DataFrame({"S": [None, "0", "def"], "new_column": [0, 1, 2]}),
                StructType(
                    [
                        StructField("S", StringType(), True),
                        StructField("new_column", LongType(), False),
                    ]
                ),
                pd.DataFrame(
                    [["0", 0, 0.1, DATE1, TIMESTAMP1, 1]],
                    columns=["S", "I", "F", "D", "T", "new_column"],
                ),
            ),
        ],
    )
    def test_visit_join_public_df(
        self, spark, df: pd.DataFrame, df_schema: StructType, expected_df: DataFrame
    ) -> None:
        """Test visit_join_public with a dataframe."""
        public_df = spark.createDataFrame(df, schema=df_schema)
        query = JoinPublic(
            child=PrivateSource(source_id="rows1"), public_table=public_df
        )
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "replace_with,expected_df",
        [
            (
                {},
                pd.DataFrame(
                    [[float("inf"), "string", 0.0], [float("-inf"), "", 1.5]],
                    columns=["inf", "null", "nan"],
                ),
            ),
            (
                {"nan": 0.0},
                pd.DataFrame(
                    [[float("inf"), "string", 0.0], [float("-inf"), None, 1.5]],
                    columns=["inf", "null", "nan"],
                ),
            ),
            (
                {"null": "not null"},
                pd.DataFrame(
                    [
                        [float("inf"), "string", float("nan")],
                        [float("-inf"), "not null", 1.5],
                    ],
                    columns=["inf", "null", "nan"],
                ),
            ),
        ],
    )
    def test_visit_replace_null_and_nan(
        self,
        replace_with: Mapping[
            str, Union[int, float, str, datetime.date, datetime.datetime]
        ],
        expected_df: DataFrame,
    ):
        """Test visit_replace_null_and_nan."""
        query = ReplaceNullAndNan(
            child=PrivateSource(source_id="rows_infs_nans"), replace_with=replace_with
        )
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        expected_output_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            schema=analytics_to_spark_columns_descriptor(expected_output_schema)
        )
        expected_output_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            schema=analytics_to_spark_columns_descriptor(expected_output_schema)
        )
        assert expected_output_domain == lookup_domain(
            transformation.output_domain, reference
        )
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_replace_null_and_nan_with_grouping_column(self) -> None:
        """Test behavior of visit_replace_null_and_nan with IfGroupedBy metric."""
        flatmap_query = FlatMap(
            child=PrivateSource("rows_infs_nans"),
            f=lambda row: [{"group": 0 if row["inf"] < 0 else 17}],
            schema_new_columns=Schema(
                {"group": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
                grouping_column="group",
            ),
            augment=True,
            max_num_rows=2,
        )
        with pytest.raises(
            ValueError,
            match=(
                "Cannot replace null values in column group, because it is being used"
                " as a grouping column"
            ),
        ):
            invalid_replace_query = ReplaceNullAndNan(
                child=flatmap_query, replace_with={"group": -10}
            )
            invalid_replace_query.accept(self.visitor)

        valid_replace_query = ReplaceNullAndNan(child=flatmap_query, replace_with={})
        transformation, reference, constraints = valid_replace_query.accept(
            self.visitor
        )
        self._validate_transform_basics(transformation, reference, valid_replace_query)
        assert isinstance(transformation, ChainTT)
        transformations = chain_to_list(transformation)
        assert isinstance(transformations[0], IdentityTransformation)
        assert isinstance(transformations[1], AugmentDictTransformation)
        assert isinstance(transformations[2], AugmentDictTransformation)
        expected_df = pd.DataFrame(
            [[float("inf"), "string", 0, 17], [float("-inf"), "", 1.5, 0]],
            columns=["inf", "null", "nan", "group"],
        )
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "replace_with,expected_df",
        [
            (
                {},
                pd.DataFrame(
                    [[0.0, "string", float("nan")], [0.0, None, 1.5]],
                    columns=["inf", "null", "nan"],
                ),
            ),
            (
                {"inf": (-100.0, 100.0)},
                pd.DataFrame(
                    [[100.0, "string", float("nan")], [-100.0, None, 1.5]],
                    columns=["inf", "null", "nan"],
                ),
            ),
        ],
    )
    def test_visit_replace_infinity(
        self, replace_with: Dict[str, Tuple[float, float]], expected_df: DataFrame
    ):
        """Test visit_replace_infinity."""
        query = ReplaceInfinity(
            child=PrivateSource(source_id="rows_infs_nans"), replace_with=replace_with
        )
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)

        expected_output_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            schema=analytics_to_spark_columns_descriptor(expected_output_schema)
        )
        assert (
            lookup_domain(transformation.output_domain, reference)
            == expected_output_domain
        )

        assert isinstance(transformation, ChainTT)
        assert isinstance(transformation.transformation2, AugmentDictTransformation)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_drop_null_and_nan_with_grouping_column(self) -> None:
        """Test behavior of visit_drop_null_and_nan with IfGroupedBy metric."""
        flatmap_query = FlatMap(
            child=PrivateSource("rows_infs_nans"),
            f=lambda row: [{"group": 0 if row["inf"] < 0 else 17}],
            schema_new_columns=Schema(
                {"group": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
                grouping_column="group",
            ),
            augment=True,
            max_num_rows=2,
        )
        with pytest.raises(
            ValueError,
            match=(
                "Cannot drop null values in column group, because it is being used as a"
                " grouping column"
            ),
        ):
            invalid_drop_query = DropNullAndNan(child=flatmap_query, columns=["group"])
            invalid_drop_query.accept(self.visitor)
        valid_drop_query = DropNullAndNan(child=flatmap_query, columns=[])
        transformation, reference, constraints = valid_drop_query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, valid_drop_query)
        assert isinstance(transformation, ChainTT)
        transformations = chain_to_list(transformation)
        assert len(transformations) == 3
        assert isinstance(transformations[0], IdentityTransformation)
        assert isinstance(transformations[1], AugmentDictTransformation)
        assert isinstance(transformations[2], AugmentDictTransformation)
        ###expect group col added, row dropped
        expected_df = pd.DataFrame(columns=["inf", "null", "nan", "group"])
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_drop_infinity_with_grouping_column(self) -> None:
        """Test behavior of visit_drop_infinity with IfGroupedBy metric."""
        flatmap_query = FlatMap(
            child=PrivateSource("rows_infs_nans"),
            f=lambda row: [{"group": 0 if row["inf"] < 0 else 17}],
            schema_new_columns=Schema(
                {"group": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
                grouping_column="group",
            ),
            augment=True,
            max_num_rows=2,
        )
        with pytest.raises(
            ValueError,
            match=(
                "Cannot drop infinite values in column group, because it is being used"
                " as a grouping column"
            ),
        ):
            invalid_drop_query = DropInfExpr(child=flatmap_query, columns=["group"])
            invalid_drop_query.accept(self.visitor)
        valid_drop_query = DropInfExpr(child=flatmap_query, columns=[])
        transformation, reference, constraints = valid_drop_query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, valid_drop_query)
        assert isinstance(transformation, ChainTT)
        transformations = chain_to_list(transformation)
        assert isinstance(transformations[0], IdentityTransformation)
        assert isinstance(transformations[1], AugmentDictTransformation)
        assert isinstance(transformations[2], AugmentDictTransformation)
        expected_df = pd.DataFrame(columns=["inf", "null", "nan", "group"])

        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_measurement_visits(self):
        """Test that visiting measurement queries raises an error."""
        with pytest.raises(NotImplementedError):
            self.visitor.visit_groupby_count(
                GroupByCount(
                    groupby_keys=KeySet.from_dict({}),
                    child=PrivateSource(source_id="rows1"),
                )
            )

        with pytest.raises(NotImplementedError):
            self.visitor.visit_groupby_count_distinct(
                GroupByCountDistinct(
                    groupby_keys=KeySet.from_dict({}),
                    child=PrivateSource(source_id="rows1"),
                )
            )

        with pytest.raises(NotImplementedError):
            self.visitor.visit_groupby_quantile(
                GroupByQuantile(
                    child=PrivateSource(source_id="rows1"),
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    quantile=0.1,
                    low=0,
                    high=1,
                )
            )

        with pytest.raises(NotImplementedError):
            self.visitor.visit_groupby_bounded_sum(
                GroupByBoundedSum(
                    child=PrivateSource(source_id="rows1"),
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )

        with pytest.raises(NotImplementedError):
            self.visitor.visit_groupby_bounded_average(
                GroupByBoundedAverage(
                    child=PrivateSource(source_id="rows1"),
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )

        with pytest.raises(NotImplementedError):
            self.visitor.visit_groupby_bounded_variance(
                GroupByBoundedVariance(
                    child=PrivateSource(source_id="rows1"),
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )

        with pytest.raises(NotImplementedError):
            self.visitor.visit_groupby_bounded_stdev(
                GroupByBoundedSTDEV(
                    child=PrivateSource(source_id="rows1"),
                    groupby_keys=KeySet.from_dict({}),
                    measure_column="A",
                    low=0,
                    high=1,
                )
            )


class TestAddRowsNulls(TestTransformationVisitorNulls):
    """Test the TransformationVisitor with nulls, NaNs, and infs."""

    visitor: TransformationVisitor
    catalog: Catalog

    def _validate_transform_basics(
        self, t: Transformation, reference: TableReference, query: QueryExpr
    ) -> None:
        """Check the basics of a transformation."""
        assert t.input_domain == self.visitor.input_domain
        assert t.input_metric == self.visitor.input_metric

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
        assert lookup_domain(t.output_domain, reference) == expected_output_domain
        assert lookup_metric(t.output_metric, reference) == expected_output_metric

    @pytest.mark.parametrize(
        "query_columns,expected_null_cols,expected_nan_cols",
        [
            (
                ["not_null"],
                ["null", "null_nan", "null_inf", "null_nan_inf"],
                ["nan", "null_nan", "nan_inf", "null_nan_inf"],
            ),
            (
                ["null", "nan", "inf"],
                ["null_nan", "null_inf", "null_nan_inf"],
                ["null_nan", "nan_inf", "null_nan_inf"],
            ),
            (
                ["null_nan", "null_inf", "nan_inf"],
                ["null", "null_nan_inf"],
                ["nan", "null_nan_inf"],
            ),
            (
                ["null", "nan", "inf", "null_nan_inf"],
                ["null_nan", "null_inf"],
                ["null_nan", "nan_inf"],
            ),
            ([], [], []),
        ],
    )
    def test_visit_drop_null_and_nan(
        self,
        query_columns: List[str],
        expected_null_cols: List[str],
        expected_nan_cols: List[str],
    ) -> None:
        """Test generating transformations from a DropNullAndNan."""
        query = DropNullAndNan(PrivateSource("rows"), query_columns)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert constraints == []

        output_domain = lookup_domain(transformation.output_domain, reference)
        assert isinstance(output_domain, SparkDataFrameDomain)

        for col in output_domain.schema:
            col_descriptor = output_domain.schema[col]
            assert col_descriptor.allow_null == (col in expected_null_cols)
            if isinstance(col_descriptor, SparkFloatColumnDescriptor):
                assert col_descriptor.allow_nan == (col in expected_nan_cols)

    @pytest.mark.parametrize(
        "query_columns,expected_inf_cols",
        [
            (["not_null"], ["inf", "null_inf", "nan_inf", "null_nan_inf"]),
            (["null", "nan", "inf"], ["null_inf", "nan_inf", "null_nan_inf"]),
            (["null_nan", "null_inf", "nan_inf"], ["inf", "null_nan_inf"]),
            (["null", "nan", "inf", "null_nan_inf"], ["null_inf", "nan_inf"]),
        ],
    )
    def test_visit_drop_infinity(
        self, query_columns: List[str], expected_inf_cols: List[str]
    ) -> None:
        """Test generating transformations from a DropInfinity."""
        query = DropInfExpr(child=PrivateSource("rows"), columns=query_columns)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert constraints == []

        output_domain = lookup_domain(transformation.output_domain, reference)
        assert isinstance(output_domain, SparkDataFrameDomain)

        for col in output_domain.schema:
            col_descriptor = output_domain.schema[col]
            if isinstance(col_descriptor, SparkFloatColumnDescriptor):
                assert col_descriptor.allow_inf == (col in expected_inf_cols)
