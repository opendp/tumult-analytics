"""Tests for OutputSchemaVisitor."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Dict, List, Type

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    spark_schema_to_analytics_columns,
)
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

# Convenience lambda functions to create dataframes for KeySets
GET_PUBLIC = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    [],
    schema=StructType(
        [StructField("A", LongType(), False), StructField("A+B", LongType(), False)]
    ),
)
GET_GROUPBY_COLUMN_A = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    [], schema=StructType([StructField("A", StringType(), False)])
)
GET_GROUPBY_COLUMN_B = lambda: SparkSession.builder.getOrCreate().createDataFrame(
    [], schema=StructType([StructField("B", LongType(), False)])
)
GET_GROUPBY_NON_EXISTING_COLUMN = (
    lambda: SparkSession.builder.getOrCreate().createDataFrame(
        [], schema=StructType([StructField("yay", LongType(), False)])
    )
)
GET_GROUPBY_COLUMN_WRONG_TYPE = (
    lambda: SparkSession.builder.getOrCreate().createDataFrame(
        [], schema=StructType([StructField("A", LongType(), False)])
    )
)

OUTPUT_SCHEMA_INVALID_QUERY_TESTS = [
    (  # Query references public source instead of private source
        PrivateSource("public"),
        "Attempted query on table 'public', which is not a private table",
    ),
    (  # JoinPublic has invalid public_id
        JoinPublic(child=PrivateSource("private"), public_table="private"),
        "Attempted public join on table 'private', which is not a public table",
    ),
    (  # JoinPublic references invalid private source
        JoinPublic(
            child=PrivateSource("private_source_not_in_catalog"), public_table="public"
        ),
        "Query references nonexistent table 'private_source_not_in_catalog'",
    ),
    (  # JoinPublic on columns not common to both tables
        JoinPublic(
            child=PrivateSource("private"), public_table="public", join_columns=["B"]
        ),
        "Join columns must be common to both tables",
    ),
    (  # JoinPrivate on columns not common to both tables
        JoinPrivate(
            PrivateSource("private"),
            Rename(PrivateSource("private"), {"B": "Q"}),
            TruncationStrategy.DropExcess(1),
            TruncationStrategy.DropExcess(1),
            join_columns=["B"],
        ),
        "Join columns must be common to both tables",
    ),
    (  # JoinPublic on tables with no common columns
        JoinPublic(
            child=Rename(PrivateSource("private"), {"A": "Q"}), public_table="public"
        ),
        "Tables have no common columns to join on",
    ),
    (  # JoinPrivate on tables with no common columns
        JoinPrivate(
            PrivateSource("private"),
            Rename(Select(PrivateSource("private"), ["A"]), {"A": "Z"}),
            TruncationStrategy.DropExcess(1),
            TruncationStrategy.DropExcess(1),
        ),
        "Tables have no common columns to join on",
    ),
    (  # JoinPublic on column with mismatched types
        JoinPublic(
            child=PrivateSource("private"), public_table="public", join_columns=["A"]
        ),
        (
            "Join columns must have identical types on both tables, "
            "but column 'A' does not"
        ),
    ),
    (  # JoinPrivate on column with mismatched types
        JoinPrivate(
            PrivateSource("private"),
            Rename(Rename(PrivateSource("private"), {"A": "Q"}), {"B": "A"}),
            TruncationStrategy.DropExcess(1),
            TruncationStrategy.DropExcess(1),
            join_columns=["A"],
        ),
        (
            "Join columns must have identical types on both tables, "
            "but column 'A' does not"
        ),
    ),
    (  # Filter on invalid column
        Filter(child=PrivateSource("private"), condition="NONEXISTENT>1"),
        "Invalid filter condition 'NONEXISTENT>1'.*",
    ),
    (  # Rename on non-existent column
        Rename(child=PrivateSource("private"), column_mapper={"NONEXISTENT": "Z"}),
        "Nonexistent columns {'NONEXISTENT'} in rename query",
    ),
    (  # Rename when column exists
        Rename(child=PrivateSource("private"), column_mapper={"A": "B"}),
        "Cannot rename 'A' to 'B': column 'B' already exists",
    ),
    (  # Select non-existent column
        Select(child=PrivateSource("private"), columns=["NONEXISTENT"]),
        "Nonexistent columns {'NONEXISTENT'} in select query",
    ),
    (  # Nested grouping FlatMap
        FlatMap(
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
                schema_new_columns=Schema({"i": "INTEGER"}, grouping_column="i"),
                augment=True,
                max_num_rows=2,
            ),
            f=lambda row: [{"j": row["X"]} for i in range(row["Repeat"])],
            schema_new_columns=Schema({"j": "INTEGER"}, grouping_column="j"),
            augment=True,
            max_num_rows=2,
        ),
        (
            "Multiple grouping transformations are used in this query. "
            "Only one grouping transformation is allowed."
        ),
    ),
    (  # FlatMap with inner grouping FlatMap but outer augment=False
        FlatMap(
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
                schema_new_columns=Schema({"i": "INTEGER"}, grouping_column="i"),
                augment=True,
                max_num_rows=2,
            ),
            f=lambda row: [{"j": row["X"]} for i in range(row["Repeat"])],
            schema_new_columns=Schema({"j": "INTEGER"}),
            augment=False,
            max_num_rows=2,
        ),
        "Flat map must set augment=True to ensure that grouping column 'i' is not lost",
    ),
    (  # Map with inner grouping FlatMap but outer augment=False
        Map(
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"i": row["X"]} for i in range(row["Repeat"])],
                schema_new_columns=Schema({"i": "INTEGER"}, grouping_column="i"),
                augment=True,
                max_num_rows=2,
            ),
            f=lambda row: {"C": 2 * str(row["B"])},
            schema_new_columns=Schema({"C": "VARCHAR"}),
            augment=False,
        ),
        "Map must set augment=True to ensure that grouping column 'i' is not lost",
    ),
    (  # ReplaceNullAndNan with a column that doesn't exist
        ReplaceNullAndNan(
            child=PrivateSource("private"), replace_with={"bad": "new_string"}
        ),
        r"Column 'bad' does not exist in this table, available columns are \[.*\]",
    ),
    (
        # ReplaceNullAndNan with bad replacement type
        ReplaceNullAndNan(
            child=PrivateSource("private"), replace_with={"B": "not_an_int"}
        ),
        "Column 'B' cannot have nulls replaced with 'not_an_int', as .* type INTEGER",
    ),
    (
        # ReplaceInfinity with nonexistent column
        ReplaceInfinity(
            child=PrivateSource("private"), replace_with={"wrong": (-1, 1)}
        ),
        r"Column 'wrong' does not exist in this table, available columns are \[.*\]",
    ),
    (
        #  ReplaceInfinity with non-decimal column
        ReplaceInfinity(child=PrivateSource("private"), replace_with={"A": (-1, 1)}),
        r"Column 'A' has a replacement value provided.*of type VARCHAR \(not DECIMAL\) "
        "and so cannot contain infinite values",
    ),
    (
        # DropNullAndNan with column that doesn't exist
        DropNullAndNan(child=PrivateSource("private"), columns=["bad"]),
        r"Column 'bad' does not exist in this table, available columns are \[.*\]",
    ),
    (
        # DropInfinity with column that doesn't exist
        DropInfinity(child=PrivateSource("private"), columns=["bad"]),
        r"Column 'bad' does not exist in this table, available columns are \[.*\]",
    ),
    (  # Type mismatch for the measure column of GroupByQuantile
        GroupByQuantile(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({"B": [0, 1, 2]}),
            measure_column="A",
            quantile=0.5,
            low=10,
            high=20,
            output_column="out",
        ),
        (
            "Quantile query's measure column 'A' has invalid type "
            "'VARCHAR'. Expected types: 'INTEGER' or 'DECIMAL'."
        ),
    ),
    (  # Type mismatch for the measure column of GroupByBoundedAverage
        GroupByBoundedAverage(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="A",
            low=0.0,
            high=1.0,
        ),
        (
            "measure column 'A' has invalid type 'VARCHAR'. "
            "Expected types: 'INTEGER' or 'DECIMAL'"
        ),
    ),
    (  # Grouping column is set in a FlatMap but not used in a later GroupBy
        GroupByCount(
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"i": row["B"]} for i in range(row["Repeat"])],
                schema_new_columns=Schema({"i": "INTEGER"}, grouping_column="i"),
                augment=True,
                max_num_rows=2,
            ),
            groupby_keys=KeySet.from_dict({"B": [0, 1, 2]}),
        ),
        "Column 'i' produced by grouping transformation is not in groupby columns",
    ),
    (  # Grouping column is set but not used in a later groupby_public_source
        GroupByCount(
            child=FlatMap(
                child=PrivateSource("private"),
                f=lambda row: [{"i": row["B"]} for i in range(row["Repeat"])],
                schema_new_columns=Schema({"i": "INTEGER"}, grouping_column="i"),
                augment=True,
                max_num_rows=2,
            ),
            groupby_keys=KeySet(dataframe=GET_GROUPBY_COLUMN_A),
        ),
        "Column 'i' produced by grouping transformation is not in groupby columns",
    ),
]

###TESTS FOR QUERY VALIDATION###


@pytest.fixture(name="validation_visitor", scope="class")
def setup_validation(request):
    """Set up test data."""
    catalog = Catalog()
    catalog.add_private_table(
        "private",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "X": ColumnDescriptor(ColumnType.DECIMAL),
            "D": ColumnDescriptor(ColumnType.DATE),
            "T": ColumnDescriptor(ColumnType.TIMESTAMP),
        },
    )
    catalog.add_public_table(
        "public", spark_schema_to_analytics_columns(GET_PUBLIC().schema)
    )
    catalog.add_public_table(
        "groupby_column_a",
        spark_schema_to_analytics_columns(GET_GROUPBY_COLUMN_A().schema),
    )
    catalog.add_public_table(
        "groupby_column_b",
        spark_schema_to_analytics_columns(GET_GROUPBY_COLUMN_B().schema),
    )
    catalog.add_public_table(
        "groupby_non_existing_column",
        spark_schema_to_analytics_columns(GET_GROUPBY_NON_EXISTING_COLUMN().schema),
    )
    catalog.add_public_table(
        "groupby_column_wrong_type",
        spark_schema_to_analytics_columns(GET_GROUPBY_COLUMN_WRONG_TYPE().schema),
    )
    catalog.add_private_table(
        "groupby_one_column_private", {"A": ColumnDescriptor(ColumnType.VARCHAR)}
    )
    visitor = OutputSchemaVisitor(catalog)
    request.cls.visitor = visitor


@pytest.mark.usefixtures("validation_visitor")
class TestValidation:
    """Test Validation with Visitor."""

    visitor: OutputSchemaVisitor

    @pytest.mark.parametrize(
        "query_expr,expected_error_msg", OUTPUT_SCHEMA_INVALID_QUERY_TESTS
    )
    def test_invalid_query_expr(
        self, query_expr: QueryExpr, expected_error_msg: str
    ) -> None:
        """Check that appropriate exceptions are raised on invalid queries."""
        with pytest.raises(ValueError, match=expected_error_msg):
            query_expr.accept(self.visitor)

    @pytest.mark.parametrize(
        "groupby_keys,exception_type,expected_error_msg",
        [
            (
                KeySet.from_dict({"A": [0, 1]}),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
            (
                KeySet.from_dict({"X": [0, 1]}),
                ValueError,
                (
                    "Groupby column 'X' has type 'INTEGER', but the column with the"
                    " same name in the input data has type 'DECIMAL' instead."
                ),
            ),
            (
                KeySet.from_dict({"Y": ["0"]}),
                KeyError,
                "Groupby column 'Y' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_NON_EXISTING_COLUMN),
                KeyError,
                "Groupby column 'yay' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_COLUMN_WRONG_TYPE),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
        ],
    )
    def test_invalid_group_by_count(
        self,
        groupby_keys: KeySet,
        exception_type: Type[Exception],
        expected_error_msg: str,
    ) -> None:
        """Test invalid measurement QueryExpr."""
        with pytest.raises(exception_type, match=expected_error_msg):
            GroupByCount(PrivateSource("private"), groupby_keys).accept(self.visitor)

    @pytest.mark.parametrize(
        "groupby_keys,exception_type,expected_error_msg",
        [
            (
                KeySet.from_dict({"B": [0, 1]}),
                ValueError,
                "Column to aggregate must be a non-grouped column, not 'B'",
            ),
            (
                KeySet.from_dict({"A": [0, 1]}),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
            (
                KeySet.from_dict({"Y": ["0"]}),
                KeyError,
                "Groupby column 'Y' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_COLUMN_B),
                ValueError,
                "Column to aggregate must be a non-grouped column, not 'B'",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_NON_EXISTING_COLUMN),
                KeyError,
                "Groupby column 'yay' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_COLUMN_WRONG_TYPE),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
        ],
    )
    def test_invalid_group_by_aggregations(
        self,
        groupby_keys: KeySet,
        exception_type: Type[Exception],
        expected_error_msg: str,
    ) -> None:
        """Test invalid measurement QueryExpr."""
        for DataClass in [
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ]:
            with pytest.raises(exception_type, match=expected_error_msg):
                DataClass(PrivateSource("private"), groupby_keys, "B", 1.0, 5.0).accept(
                    self.visitor
                )
        with pytest.raises(exception_type, match=expected_error_msg):
            GroupByQuantile(
                PrivateSource("private"), groupby_keys, "B", 0.5, 1.0, 5.0
            ).accept(self.visitor)


###QUERY VALIDATION WITH NULLS###
@pytest.fixture(name="test_data_nulls", scope="class")
def setup_visitor_with_nulls(request) -> None:
    """Set up test data."""
    catalog = Catalog()
    catalog.add_private_table(
        "private",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "X": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_inf=True, allow_nan=True
            ),
            "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
            "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
            "NOTNULL": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
        },
    )
    catalog.add_public_table(
        "public",
        {
            "A": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "A+B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        },
    )
    catalog.add_public_table(
        "groupby_column_a", {"A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)}
    )
    catalog.add_public_table(
        "groupby_column_b", {"B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)}
    )
    catalog.add_public_table(
        "groupby_nonexistent_column", {"yay": ColumnDescriptor(ColumnType.INTEGER)}
    )
    catalog.add_public_table(
        "groupby_column_wrong_type", {"A": ColumnDescriptor(ColumnType.INTEGER)}
    )
    catalog.add_private_table(
        "groupby_one_column_private",
        {"A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)},
    )
    visitor = OutputSchemaVisitor(catalog)
    request.cls.visitor = visitor


@pytest.mark.usefixtures("test_data_nulls")
class TestValidationWithNulls:
    """Test Validation with Nulls."""

    visitor: OutputSchemaVisitor

    @pytest.mark.parametrize(
        "query_expr,expected_error_msg", OUTPUT_SCHEMA_INVALID_QUERY_TESTS
    )
    def test_invalid_query_expr_null(
        self, query_expr: QueryExpr, expected_error_msg: str
    ) -> None:
        """Check that appropriate exceptions are raised on invalid queries."""
        with pytest.raises(ValueError, match=expected_error_msg):
            query_expr.accept(self.visitor)

    @pytest.mark.parametrize(
        "groupby_keys,exception_type,expected_error_msg",
        [
            (
                KeySet.from_dict({"A": [0, 1]}),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
            (
                KeySet.from_dict({"Y": ["0"]}),
                KeyError,
                "Groupby column 'Y' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_NON_EXISTING_COLUMN),
                KeyError,
                "Groupby column 'yay' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_COLUMN_WRONG_TYPE),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
        ],
    )
    def test_invalid_group_by_count_null(
        self,
        groupby_keys: KeySet,
        exception_type: Type[Exception],
        expected_error_msg: str,
    ) -> None:
        """Test invalid measurement QueryExpr."""
        with pytest.raises(exception_type, match=expected_error_msg):
            GroupByCount(PrivateSource("private"), groupby_keys).accept(self.visitor)

    @pytest.mark.parametrize(
        "groupby_keys,exception_type,expected_error_msg",
        [
            (
                KeySet.from_dict({"B": [0, 1]}),
                ValueError,
                "Column to aggregate must be a non-grouped column, not 'B'",
            ),
            (
                KeySet.from_dict({"A": [0, 1]}),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
            (
                KeySet.from_dict({"Y": ["0"]}),
                KeyError,
                "Groupby column 'Y' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_COLUMN_B),
                ValueError,
                "Column to aggregate must be a non-grouped column, not 'B'",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_NON_EXISTING_COLUMN),
                KeyError,
                "Groupby column 'yay' is not in the input schema.",
            ),
            (
                KeySet(dataframe=GET_GROUPBY_COLUMN_WRONG_TYPE),
                ValueError,
                (
                    "Groupby column 'A' has type 'INTEGER', but the column "
                    "with the same name in the input data has type 'VARCHAR' instead."
                ),
            ),
        ],
    )
    def test_invalid_group_by_aggregations_null(
        self,
        groupby_keys: KeySet,
        exception_type: Type[Exception],
        expected_error_msg: str,
    ) -> None:
        """Test invalid measurement QueryExpr."""
        for DataClass in [
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ]:
            with pytest.raises(exception_type, match=expected_error_msg):
                DataClass(PrivateSource("private"), groupby_keys, "B", 1.0, 5.0).accept(
                    self.visitor
                )
        with pytest.raises(exception_type, match=expected_error_msg):
            GroupByQuantile(
                PrivateSource("private"), groupby_keys, "B", 0.5, 1.0, 5.0
            ).accept(self.visitor)

    def test_visit_private_source(self) -> None:
        """Test visit_private_source."""
        query = PrivateSource("private")
        schema = self.visitor.visit_private_source(query)
        assert (
            schema
            == self.visitor._catalog.tables[  # pylint: disable=protected-access
                "private"
            ].schema
        )

    @pytest.mark.parametrize(
        "column_mapper,expected_schema",
        [
            (
                {"A": "AAA"},
                Schema(
                    {
                        "AAA": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
            (
                {"A": "AAA", "B": "BrandNewColumnName"},
                Schema(
                    {
                        "AAA": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "BrandNewColumnName": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=True
                        ),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
            (
                {"X": "Friendly Decimal", "NOTNULL": "still not null"},
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "Friendly Decimal": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "still not null": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
        ],
    )
    def test_visit_rename(
        self, column_mapper: Dict[str, str], expected_schema: Schema
    ) -> None:
        """Test visit_rename."""
        query = Rename(child=PrivateSource("private"), column_mapper=column_mapper)
        schema = self.visitor.visit_rename(query)
        assert schema == expected_schema

    @pytest.mark.parametrize("condition", ["B > X", "X < 500", "NOTNULL < 30"])
    def test_visit_filter(self, condition: str) -> None:
        """Test visit_filter."""
        query = Filter(child=PrivateSource("private"), condition=condition)
        schema = self.visitor.visit_filter(query)
        assert (
            schema
            == self.visitor._catalog.tables[  # pylint: disable=protected-access
                "private"
            ].schema
        )

    @pytest.mark.parametrize(
        "columns,expected_schema",
        [
            (
                ["A"],
                Schema({"A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)}),
            ),
            (
                ["NOTNULL", "B"],
                Schema(
                    {
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                    }
                ),
            ),
        ],
    )
    def test_visit_select(self, columns: List[str], expected_schema: Schema) -> None:
        """Test visit_select."""
        query = Select(child=PrivateSource("private"), columns=columns)
        schema = self.visitor.visit_select(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {"NEW": 1234},
                    schema_new_columns=Schema(
                        {"NEW": ColumnDescriptor(ColumnType.INTEGER, allow_null=False)}
                    ),
                    augment=True,
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                        "NEW": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                    }
                ),
            ),
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: dict.fromkeys(
                        list(row.keys()) + ["NEW"], list(row.values()) + [1234]
                    ),
                    schema_new_columns=Schema(
                        {"NEW": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)}
                    ),
                    augment=True,
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                        "NEW": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                    }
                ),
            ),
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {f"_{key}_": val for key, val in row.items()},
                    schema_new_columns=Schema(
                        {
                            "_A_": ColumnDescriptor(
                                ColumnType.VARCHAR, allow_null=True
                            ),
                            "_B_": ColumnDescriptor(
                                ColumnType.INTEGER, allow_null=True
                            ),
                            "_X_": ColumnDescriptor(
                                ColumnType.DECIMAL,
                                allow_null=True,
                                allow_nan=True,
                                allow_inf=True,
                            ),
                            "_D_": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                            "_T_": ColumnDescriptor(
                                ColumnType.TIMESTAMP, allow_null=True
                            ),
                            "_NOTNULL_": ColumnDescriptor(
                                ColumnType.INTEGER, allow_null=False
                            ),
                        }
                    ),
                    augment=False,
                ),
                Schema(
                    {
                        "_A_": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "_B_": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "_X_": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "_D_": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "_T_": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "_NOTNULL_": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=True
                        ),
                    }
                ),
            ),
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {"ABC": "abc"},
                    schema_new_columns=Schema(
                        {"ABC": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)}
                    ),
                    augment=False,
                ),
                Schema({"ABC": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)}),
            ),
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {"ABC": "abc"},
                    schema_new_columns=Schema(
                        # This has allow_null=False, but  the output schema
                        # should have allow_null=True
                        {"ABC": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False)}
                    ),
                    augment=False,
                ),
                Schema({"ABC": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)}),
            ),
        ],
    )
    def test_visit_map(self, query: Map, expected_schema: Schema) -> None:
        """Test visit_map."""
        schema = self.visitor.visit_map(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda row: [{"i": i} for i in range(len(row["A"] + 1))],
                    schema_new_columns=Schema(
                        {"i": ColumnDescriptor(ColumnType.INTEGER, allow_null=False)}
                    ),
                    augment=True,
                    max_num_rows=10,
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                        "i": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                    }
                ),
            ),
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda row: [{"i": i} for i in range(len(row["A"] + 1))],
                    schema_new_columns=Schema(
                        {"i": ColumnDescriptor(ColumnType.INTEGER, allow_null=False)}
                    ),
                    augment=False,
                    max_num_rows=10,
                ),
                Schema({"i": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)}),
            ),
        ],
    )
    def test_visit_flat_map(self, query: FlatMap, expected_schema: Schema) -> None:
        """Test visit_flat_map."""
        schema = self.visitor.visit_flat_map(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("groupby_one_column_private"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(10),
                    truncation_strategy_right=TruncationStrategy.DropExcess(10),
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            )
        ],
    )
    def test_visit_join_private(
        self, query: JoinPrivate, expected_schema: Schema
    ) -> None:
        """Test visit_join_private."""
        schema = self.visitor.visit_join_private(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                JoinPublic(
                    child=PrivateSource("private"), public_table="groupby_column_a"
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            )
        ],
    )
    def test_visit_join_public(
        self, query: JoinPublic, expected_schema: Schema
    ) -> None:
        """Test visit_join_public."""
        schema = self.visitor.visit_join_public(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                ReplaceNullAndNan(child=PrivateSource("private"), replace_with={}),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=False),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=False),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
            (
                ReplaceNullAndNan(
                    child=PrivateSource("private"), replace_with={"A": "", "B": 0}
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
            (
                ReplaceNullAndNan(
                    child=PrivateSource("private"),
                    replace_with={
                        "A": "this_was_null",
                        "B": 987,
                        "X": -123.45,
                        "D": datetime.date(2000, 1, 1),
                        "T": datetime.datetime(2020, 1, 1),
                    },
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=False),
                        "T": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=False),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
        ],
    )
    def test_visit_replace_null_and_nan(
        self, query: ReplaceNullAndNan, expected_schema: Schema
    ) -> None:
        """Test visit_replace_null_and_nan."""
        schema = self.visitor.visit_replace_null_and_nan(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                DropNullAndNan(child=PrivateSource("private"), columns=[]),
                Schema(
                    {
                        "A": ColumnDescriptor(
                            ColumnType.VARCHAR,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "B": ColumnDescriptor(
                            ColumnType.INTEGER,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(
                            ColumnType.DATE,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "T": ColumnDescriptor(
                            ColumnType.TIMESTAMP,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                    }
                ),
            ),
            (
                DropNullAndNan(child=PrivateSource("private"), columns=["A", "X", "T"]),
                Schema(
                    {
                        "A": ColumnDescriptor(
                            ColumnType.VARCHAR,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=True,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(
                            ColumnType.TIMESTAMP,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
        ],
    )
    def test_visit_drop_null_and_nan(
        self, query: DropNullAndNan, expected_schema: Schema
    ) -> None:
        """Test visit_drop_null_and_nan."""
        schema = self.visitor.visit_drop_null_and_nan(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                DropInfinity(child=PrivateSource("private"), columns=[]),
                Schema(
                    {
                        "A": ColumnDescriptor(
                            ColumnType.VARCHAR,
                            allow_null=True,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "B": ColumnDescriptor(
                            ColumnType.INTEGER,
                            allow_null=True,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=False,
                        ),
                        "D": ColumnDescriptor(
                            ColumnType.DATE,
                            allow_null=True,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "T": ColumnDescriptor(
                            ColumnType.TIMESTAMP,
                            allow_null=True,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER,
                            allow_null=False,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                    }
                ),
            ),
            (
                DropInfinity(child=PrivateSource("private"), columns=["X"]),
                Schema(
                    {
                        "A": ColumnDescriptor(
                            ColumnType.VARCHAR,
                            allow_null=True,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "X": ColumnDescriptor(
                            ColumnType.DECIMAL,
                            allow_null=True,
                            allow_nan=True,
                            allow_inf=False,
                        ),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "T": ColumnDescriptor(
                            ColumnType.TIMESTAMP,
                            allow_null=True,
                            allow_nan=False,
                            allow_inf=False,
                        ),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
        ],
    )
    def test_visit_drop_infinity(
        self, query: DropInfinity, expected_schema: Schema
    ) -> None:
        """Test visit_drop_infinity."""
        schema = self.visitor.visit_drop_infinity(query)
        assert schema == expected_schema

    @pytest.mark.parametrize(
        "query,expected_schema",
        [
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["a1", "a2"]}),
                    output_column="count",
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "count": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                    }
                ),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict(
                        {"A": ["a1", "a2"], "NOTNULL": [1, 2]}
                    ),
                    columns_to_count=["NOTNULL"],
                    output_column="count_distinct",
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "NOTNULL": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                        "count_distinct": ColumnDescriptor(
                            ColumnType.INTEGER, allow_null=False
                        ),
                    }
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict(
                        {"D": [datetime.date(1980, 1, 1), datetime.date(2000, 1, 1)]}
                    ),
                    measure_column="NOTNULL",
                    quantile=0.5,
                    low=-100,
                    high=100,
                    output_column="quantile",
                ),
                Schema(
                    {
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "quantile": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=False
                        ),
                    }
                ),
            ),
            (
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict(
                        {"D": [datetime.date(1980, 1, 1), datetime.date(2000, 1, 1)]}
                    ),
                    measure_column="NOTNULL",
                    low=-100,
                    high=100,
                    output_column="sum",
                ),
                Schema(
                    {
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "sum": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                    }
                ),
            ),
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [1, 2, 3]}),
                    measure_column="NOTNULL",
                    low=-100,
                    high=100,
                    output_column="avg",
                ),
                Schema(
                    {
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "avg": ColumnDescriptor(ColumnType.DECIMAL, allow_null=False),
                    }
                ),
            ),
            (
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["a1", "a2"], "B": [1, 2, 3]}),
                    measure_column="NOTNULL",
                    low=-100,
                    high=100,
                    output_column="variance",
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                        "variance": ColumnDescriptor(
                            ColumnType.DECIMAL, allow_null=False
                        ),
                    }
                ),
            ),
            (
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict(
                        {"A": ["a1", "a2"], "D": [datetime.date(1999, 12, 31)]}
                    ),
                    measure_column="NOTNULL",
                    low=-100,
                    high=100,
                    output_column="stdev",
                ),
                Schema(
                    {
                        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                        "D": ColumnDescriptor(ColumnType.DATE, allow_null=True),
                        "stdev": ColumnDescriptor(ColumnType.DECIMAL, allow_null=False),
                    }
                ),
            ),
        ],
    )
    def test_visit_groupby_queries(
        self, query: QueryExpr, expected_schema: Schema
    ) -> None:
        """Test visit_groupby_*."""
        schema = query.accept(self.visitor)
        assert schema == expected_schema
