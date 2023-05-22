"""Tests for TransformationVisitor on tables with AddRemoveKeys metrics."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Dict, List, Mapping, Tuple, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.identity import Identity as IdentityTransformation

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._schema import Schema, analytics_to_spark_columns_descriptor
from tmlt.analytics._table_identifier import Identifier, NamedTable, TableCollection
from tmlt.analytics._table_reference import TableReference, lookup_domain
from tmlt.analytics._transformation_utils import get_table_from_ref
from tmlt.analytics.query_expr import (
    DropInfinity,
    DropNullAndNan,
    Filter,
    FlatMap,
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


class TestAddKeys(TestTransformationVisitor):
    """Tests for the transformation visitor using for AddOneID."""

    visitor: TransformationVisitor
    catalog: Catalog
    input_data: Dict[Identifier, Union[DataFrame, Dict[Identifier, DataFrame]]]
    dataframes: Dict[str, DataFrame]

    def _validate_transform_basics(
        self,
        transformation: Transformation,
        reference: TableReference,
        query: QueryExpr,
        grouping_column: str = "id",
    ) -> None:
        assert transformation.input_domain == self.visitor.input_domain
        assert transformation.input_metric == self.visitor.input_metric
        assert isinstance(transformation, ChainTT)
        first_transform = chain_to_list(transformation)[0]
        assert isinstance(first_transform, IdentityTransformation)

        expected_schema = query.accept(OutputSchemaVisitor(self.catalog))
        assert expected_schema.grouping_column == grouping_column

        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        expected_output_metric = IfGroupedBy(grouping_column, SymmetricDifference())

        table_transform = get_table_from_ref(transformation, reference)
        assert table_transform.output_domain == expected_output_domain
        assert table_transform.output_metric == expected_output_metric

    @pytest.mark.parametrize("source_id", ["ids1", "ids2"])
    def test_visit_private_source(self, source_id: str) -> None:
        """Test generating transformations from a PrivateSource."""
        query = PrivateSource(source_id)
        transformation, reference, constraints = query.accept(self.visitor)
        assert reference.path == [TableCollection("ids"), NamedTable(source_id)]
        assert isinstance(transformation, IdentityTransformation)
        assert constraints == []

    def test_invalid_private_source(self) -> None:
        """Test that invalid PrivateSource expressions are handled."""
        query = PrivateSource("nonexistent")
        with pytest.raises(ValueError, match="Table 'nonexistent' does not exist"):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "mapper,expected_df,grouping_column",
        [
            (
                {"S": "columnS"},
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1]],
                    columns=["id", "columnS", "I", "F", "D", "T"],
                ),
                "id",
            ),
            (
                {"D": "date", "T": "time"},
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1]],
                    columns=["id", "S", "I", "F", "date", "time"],
                ),
                "id",
            ),
            (
                {"id": "id2"},
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1]],
                    columns=["id2", "S", "I", "F", "D", "T"],
                ),
                "id2",
            ),
        ],
    )
    def test_visit_rename(
        self, mapper: Dict[str, str], expected_df: DataFrame, grouping_column: str
    ) -> None:
        """Test generating transformations from a Rename."""
        query = Rename(PrivateSource("ids1"), mapper)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(
            transformation, reference, query, grouping_column
        )
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "filter_expr,expected_df",
        [
            (
                "F > I",
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1]],
                    columns=["id", "S", "I", "F", "D", "T"],
                ),
            ),
            ("S = 'ABC'", pd.DataFrame(columns=["id", "S", "I", "F", "D", "T"])),
        ],
    )
    def test_visit_filter(self, filter_expr: str, expected_df: DataFrame) -> None:
        """Test visit_filter."""
        query = Filter(PrivateSource(source_id="ids1"), filter_expr)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "columns,expected_df",
        [
            (["id"], pd.DataFrame([[1]], columns=["id"])),
            (
                ["id", "S", "I", "F"],
                pd.DataFrame([[1, "0", 0, 0.1]], columns=["id", "S", "I", "F"]),
            ),
        ],
    )
    def test_visit_select(self, columns: List[str], expected_df: DataFrame) -> None:
        """Test generating transformations from a Select."""
        query = Select(PrivateSource(source_id="ids1"), columns)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                Map(
                    PrivateSource("ids1"),
                    lambda row: {"X": 2 * str(row["S"])},
                    Schema({"X": "VARCHAR"}),
                    augment=True,
                ),
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1, "00"]],
                    columns=["id", "S", "I", "F", "D", "T", "X"],
                ),
            ),
            (
                Map(
                    PrivateSource("ids1"),
                    lambda row: {"X": 2 * str(row["S"]), "Y": row["I"] + 2 * row["F"]},
                    Schema({"X": "VARCHAR", "Y": "DECIMAL"}),
                    augment=True,
                ),
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1, "00", 0.2]],
                    columns=["id", "S", "I", "F", "D", "T", "X", "Y"],
                ),
            ),
        ],
    )
    def test_visit_map(self, query: Map, expected_df: DataFrame) -> None:
        """Test generating transformations from a Map."""
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_map_invalid(self) -> None:
        """Test that invalid Map expressions are handled."""
        query = Map(PrivateSource("ids1"), lambda row: {}, Schema({}), augment=False)
        with pytest.raises(ValueError, match="Maps on tables.*must be augmenting"):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                FlatMap(
                    child=PrivateSource("ids1"),
                    f=lambda row: [{"S_is_zero": 1 if row["S"] == "0" else 2}],
                    schema_new_columns=Schema({"S_is_zero": "INTEGER"}),
                    augment=True,
                    max_num_rows=1,
                ),
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1, 1]],
                    columns=["id", "S", "I", "F", "D", "T", "S_is_zero"],
                ),
            ),
            (
                FlatMap(
                    child=PrivateSource("ids1"),
                    f=lambda row: [{"X": n} for n in range(row["I"] + 10)],
                    schema_new_columns=Schema({"X": "INTEGER"}),
                    augment=True,
                    max_num_rows=3,
                ),
                pd.DataFrame(
                    [
                        [1, "0", 0, 0.1, DATE1, TIMESTAMP1, 0],
                        [1, "0", 0, 0.1, DATE1, TIMESTAMP1, 1],
                        [1, "0", 0, 0.1, DATE1, TIMESTAMP1, 2],
                    ],
                    columns=["id", "S", "I", "F", "D", "T", "X"],
                ),
            ),
        ],
    )
    def test_visit_flat_map(self, query: FlatMap, expected_df: DataFrame) -> None:
        """Test generating transformations from a non-grouping FlatMap."""
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_flatmap_invalid(self) -> None:
        """Test that invalid FlatMap expressions are handled."""
        query = FlatMap(
            PrivateSource("ids1"),
            lambda row: [{}],
            Schema({}),
            augment=False,
            max_num_rows=1,
        )
        with pytest.raises(ValueError, match="Flat maps on tables.*must be augmenting"):
            query.accept(self.visitor)

        query = FlatMap(
            PrivateSource("ids1"),
            lambda row: [{"X": row["I"]}],
            Schema({"X": "INTEGER"}, "X"),
            augment=True,
            max_num_rows=1,
        )
        with pytest.raises(ValueError, match="Flat maps on tables.*cannot be grouping"):
            query.accept(self.visitor)

        query = FlatMap(
            PrivateSource("ids1"),
            lambda row: [{"X": row["I"]}],
            Schema({}),
            augment=True,
            max_num_rows=1,
        )
        with pytest.warns(
            UserWarning, match="the max_num_rows parameter is not required."
        ):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                JoinPrivate(
                    PrivateSource("ids1"), PrivateSource("ids2"), None, None, None
                ),
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1, "a"]],
                    columns=["id", "S", "I", "F", "D", "T", "field"],
                ),
            ),
            (
                JoinPrivate(
                    PrivateSource("ids1"), PrivateSource("ids2"), None, None, ["id"]
                ),
                pd.DataFrame(
                    [[1, "0", 0, 0, 0.1, DATE1, TIMESTAMP1, "a"]],
                    columns=["id", "S", "I_left", "I_right", "F", "D", "T", "field"],
                ),
            ),
            (
                JoinPrivate(
                    PrivateSource("ids1"), PrivateSource("ids2"), join_columns=["id"]
                ),
                pd.DataFrame(
                    [[1, "0", 0, 0, 0.1, DATE1, TIMESTAMP1, "a"]],
                    columns=["id", "S", "I_left", "I_right", "F", "D", "T", "field"],
                ),
            ),
        ],
    )
    def test_visit_join_private(
        self, query: JoinPrivate, expected_df: DataFrame
    ) -> None:
        """Test generating transformations from a JoinPrivate."""
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "query",
        [
            JoinPrivate(
                PrivateSource("ids1"),
                PrivateSource("ids2"),
                TruncationStrategy.DropExcess(1),
                TruncationStrategy.DropExcess(1),
                ["id"],
            ),
            JoinPrivate(
                PrivateSource("ids1"),
                PrivateSource("ids2"),
                TruncationStrategy.DropExcess(1),
                None,
                ["id"],
            ),
            JoinPrivate(
                PrivateSource("ids1"),
                PrivateSource("ids2"),
                None,
                TruncationStrategy.DropExcess(1),
                ["id"],
            ),
        ],
    )
    def test_visit_join_private_raises_warning(self, query) -> None:
        """Test that warnings are raised appropriately for JoinPrivate expressions."""
        with pytest.warns(
            UserWarning,
            match=(
                "When joining with IDs, truncation strategies are not required."
                " Provided truncation parameters will be ignored."
            ),
        ):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "query,expected_df",
        [
            (
                JoinPublic(PrivateSource("ids1"), "public", None),
                pd.DataFrame(
                    [[1, "0", 0, 0.1, DATE1, TIMESTAMP1, "x"]],
                    columns=["id", "S", "I", "F", "D", "T", "public"],
                ),
            ),
            (
                JoinPublic(PrivateSource("ids1"), "public", ["S"]),
                pd.DataFrame(
                    [[1, "0", 0, 0, 0.1, DATE1, TIMESTAMP1, "x"]],
                    columns=["id", "S", "I_left", "I_right", "F", "D", "T", "public"],
                ),
            ),
        ],
    )
    def test_visit_join_public_str(
        self, query: JoinPublic, expected_df: DataFrame
    ) -> None:
        """Test generating transformations from a JoinPublic."""
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    def test_visit_join_public_df(self) -> None:
        """Test generating transformations from a JoinPublic using a dataframe."""
        query = JoinPublic(
            PrivateSource("ids1"), self.visitor.public_sources["public"], None
        )
        expected_df = pd.DataFrame(
            [[1, "0", 0, 0.1, DATE1, TIMESTAMP1, "x"]],
            columns=["id", "S", "I", "F", "D", "T", "public"],
        )
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

    @pytest.mark.parametrize(
        "replace_with,expected_df",
        [
            (
                {},
                pd.DataFrame(
                    [[1, float("inf"), "string", 0.0], [1, float("-inf"), "", 1.5]],
                    columns=["id", "inf", "null", "nan"],
                ),
            ),
            (
                {"nan": 0.0},
                pd.DataFrame(
                    [[1, float("inf"), "string", 0.0], [1, float("-inf"), None, 1.5]],
                    columns=["id", "inf", "null", "nan"],
                ),
            ),
            (
                {"null": "not null"},
                pd.DataFrame(
                    [
                        [1, float("inf"), "string", float("nan")],
                        [1, float("-inf"), "not null", 1.5],
                    ],
                    columns=["id", "inf", "null", "nan"],
                ),
            ),
            (
                {"inf": 10},
                pd.DataFrame(
                    [
                        [1, float("inf"), "string", float("nan")],
                        [1, float("-inf"), None, 1.5],
                    ],
                    columns=["id", "inf", "null", "nan"],
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
        """Test generating transformations from a ReplaceNullAndNan."""
        query = ReplaceNullAndNan(PrivateSource("ids_infs_nans"), replace_with)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

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

    @pytest.mark.parametrize(
        "replace_with,expected_df",
        [
            (
                {},
                pd.DataFrame(
                    [[1, 0.0, "string", float("nan")], [1, 0.0, None, 1.5]],
                    columns=["id", "inf", "null", "nan"],
                ),
            ),
            (
                {"inf": (-100.0, 100.0)},
                pd.DataFrame(
                    [[1, 100.0, "string", float("nan")], [1, -100.0, None, 1.5]],
                    columns=["id", "inf", "null", "nan"],
                ),
            ),
        ],
    )
    def test_visit_replace_infinity(
        self, replace_with: Dict[str, Tuple[float, float]], expected_df: DataFrame
    ):
        """Test generating transformations from a ReplaceInfinity."""
        query = ReplaceInfinity(PrivateSource("ids_infs_nans"), replace_with)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        self._validate_result(transformation, reference, expected_df)
        assert constraints == []

        expected_output_schema = query.accept(OutputSchemaVisitor(self.catalog))
        expected_output_domain = SparkDataFrameDomain(
            schema=analytics_to_spark_columns_descriptor(expected_output_schema)
        )
        assert (
            lookup_domain(transformation.output_domain, reference)
            == expected_output_domain
        )


class TestAddKeysNulls(TestTransformationVisitorNulls):
    """Test the TransformationVisitor with nulls, NaNs, and infs."""

    visitor: TransformationVisitor
    catalog: Catalog

    def _validate_transform_basics(
        self,
        transformation: Transformation,
        reference: TableReference,
        query: QueryExpr,
    ) -> None:
        assert transformation.input_domain == self.visitor.input_domain
        assert transformation.input_metric == self.visitor.input_metric
        assert isinstance(transformation, ChainTT)
        first_transform = chain_to_list(transformation)[0]
        assert isinstance(first_transform, IdentityTransformation)

        expected_schema = query.accept(OutputSchemaVisitor(self.catalog))
        assert expected_schema.grouping_column == "id"

        expected_output_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(expected_schema)
        )
        expected_output_metric = IfGroupedBy("id", SymmetricDifference())

        table_transform = get_table_from_ref(transformation, reference)
        assert table_transform.output_domain == expected_output_domain
        assert table_transform.output_metric == expected_output_metric

    @pytest.mark.parametrize(
        "query_columns,expected_null_cols,expected_nan_cols",
        [
            (
                ["not_null"],
                ["id", "null", "null_nan", "null_inf", "null_nan_inf"],
                ["nan", "null_nan", "nan_inf", "null_nan_inf"],
            ),
            (
                ["null", "nan", "inf"],
                ["id", "null_nan", "null_inf", "null_nan_inf"],
                ["null_nan", "nan_inf", "null_nan_inf"],
            ),
            (
                ["null_nan", "null_inf", "nan_inf"],
                ["id", "null", "null_nan_inf"],
                ["nan", "null_nan_inf"],
            ),
            (
                ["null", "nan", "inf", "null_nan_inf"],
                ["id", "null_nan", "null_inf"],
                ["null_nan", "nan_inf"],
            ),
            ([], ["id"], []),
        ],
    )
    def test_visit_drop_null_and_nan(
        self,
        query_columns: List[str],
        expected_null_cols: List[str],
        expected_nan_cols: List[str],
    ) -> None:
        """Test generating transformations from a DropNullAndNan."""
        query = DropNullAndNan(PrivateSource("ids"), query_columns)
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
            ([], []),
        ],
    )
    def test_visit_drop_infinity(
        self, query_columns: List[str], expected_inf_cols: List[str]
    ) -> None:
        """Test generating transformations from a DropInfinity."""
        query = DropInfinity(PrivateSource("ids"), query_columns)
        transformation, reference, constraints = query.accept(self.visitor)
        self._validate_transform_basics(transformation, reference, query)
        assert constraints == []

        output_domain = lookup_domain(transformation.output_domain, reference)
        assert isinstance(output_domain, SparkDataFrameDomain)

        for col in output_domain.schema:
            col_descriptor = output_domain.schema[col]
            if isinstance(col_descriptor, SparkFloatColumnDescriptor):
                assert col_descriptor.allow_inf == (col in expected_inf_cols)
