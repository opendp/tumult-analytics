"""Defines a visitor for determining the output schemas of query expressions."""

# <placeholder: boilerplate>

from typing import KeysView, List, Optional, Union, cast

from pyspark.sql import SparkSession

from tmlt.analytics._catalog import Catalog, PrivateTable, PrivateView, PublicTable
from tmlt.analytics._schema import (
    Schema,
    analytics_to_spark_schema,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics.query_expr import (
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    QueryExprVisitor,
)


def _output_schema_for_join(
    left_schema: Schema, right_schema: Schema, join_columns: Optional[List[str]]
) -> Schema:
    """Return the resulting schema from joining two tables.

    Args:
        left_schema: Schema for the left table.
        right_schema: Schema for the right table.
        join_columns: The set of columns to join on.
    """
    if left_schema.grouping_column is None:
        grouping_column = right_schema.grouping_column
    elif right_schema.grouping_column is None:
        grouping_column = left_schema.grouping_column
    elif left_schema.grouping_column == right_schema.grouping_column:
        grouping_column = left_schema.grouping_column
    else:
        raise ValueError(
            "Both tables having different grouping columns is not supported"
        )
    common_columns = set(left_schema) & set(right_schema)

    join_columns = (
        join_columns
        if join_columns
        else sorted(common_columns, key=list(left_schema).index)
    )

    if not set(join_columns) <= common_columns:
        raise ValueError("Join columns must be common to both DataFrames.")

    for column in join_columns:
        if left_schema[column] != right_schema[column]:
            raise ValueError(
                "Join columns must have identical types on both "
                f"DataFrames. {left_schema[column]} and "
                f"{right_schema[column]} are incompatible."
            )

    output_schema = {
        **{column: left_schema[column] for column in join_columns},
        **{
            column + ("_left" if column in common_columns else ""): left_schema[column]
            for column in left_schema
            if column not in join_columns
        },
        **{
            column
            + ("_right" if column in common_columns else ""): right_schema[column]
            for column in right_schema
            if column not in join_columns
        },
    }
    return Schema(output_schema, grouping_column=grouping_column)


def _validate_groupby(
    query: Union[
        GroupByBoundedAverage,
        GroupByBoundedSTDEV,
        GroupByBoundedSum,
        GroupByBoundedVariance,
        GroupByCount,
        GroupByCountDistinct,
        GroupByQuantile,
    ],
    catalog: Catalog,
    output_schema_visitor: "OutputSchemaVisitor",
) -> Schema:
    """Validate groupby aggregate query.

    Args:
        query: Query expression to be validated.
        catalog: The catalog of the output schema visitor.
        output_schema_visitor: A visitor to get the output schema of an expression.

    Returns:
        Output schema of current QueryExpr
    """
    input_schema = query.child.accept(output_schema_visitor)

    groupby_columns: KeysView[str]
    schema: Schema
    # TODO(#1707): remove _public_id handling
    if query.groupby_keys._public_id is not None:  # pylint: disable=protected-access
        public_id = query.groupby_keys._public_id  # pylint: disable=protected-access
        try:
            public_table = catalog.tables[public_id]
        except KeyError:
            raise KeyError(f"Could not find source with ID '{public_id}'")
        if not isinstance(public_table, PublicTable):
            raise ValueError(
                f"Attempted a groupby on table '{public_id}', but it is "
                "not a public table."
            )
        groupby_columns = cast(KeysView[str], public_table.schema.keys())
        schema = public_table.schema
    else:
        groupby_columns = cast(KeysView[str], query.groupby_keys.schema().keys())
        schema = query.groupby_keys.schema()

    for column_name, column_type in schema.items():
        try:
            input_column_type = input_schema[column_name]
        except KeyError:
            raise KeyError(
                f"Groupby column '{column_name}' is not in the input schema."
            )
        if column_type != input_column_type:
            raise ValueError(
                f"Groupby column '{column_name}' has type '{column_type}', but "
                "the column with the same name in the input data has type "
                f"'{input_column_type}' instead."
            )

    grouping_column = input_schema.grouping_column
    if grouping_column is not None and grouping_column not in groupby_columns:
        raise ValueError(
            f"Column produced by grouping transformation '{grouping_column}' "
            f"is not in groupby columns {list(groupby_columns)}."
        )
    if (
        not isinstance(query, (GroupByCount, GroupByCountDistinct))
        and query.measure_column in groupby_columns
    ):
        raise ValueError(
            "Column to aggregate must be a non-grouped column, not "
            f"'{query.measure_column}'."
        )

    if isinstance(query, (GroupByCount, GroupByCountDistinct)):
        output_column_type = "INTEGER"
    elif isinstance(query, GroupByQuantile):
        if input_schema[query.measure_column] not in ["INTEGER", "DECIMAL"]:
            raise ValueError(
                f"Quantile query's measure column '{query.measure_column}' has "
                f"invalid type '{input_schema[query.measure_column]}'. "
                "Expected types: 'INTEGER' or 'DECIMAL'."
            )
        output_column_type = "DECIMAL"
    elif isinstance(
        query,
        (
            GroupByBoundedSum,
            GroupByBoundedSTDEV,
            GroupByBoundedAverage,
            GroupByBoundedVariance,
        ),
    ):
        if input_schema[query.measure_column] not in ["INTEGER", "DECIMAL"]:
            raise ValueError(
                f"{type(query).__name__} query's measure column "
                f"'{query.measure_column}' has invalid type "
                f"'{input_schema[query.measure_column]}'. "
                "Expected types: 'INTEGER' or 'DECIMAL'."
            )
        output_column_type = (
            input_schema[query.measure_column]
            if isinstance(query, GroupByBoundedSum)
            else "DECIMAL"
        )
    else:
        raise AssertionError(
            "Unexpected QueryExpr type. This should not happen and is"
            "probably a bug; please let us know so we can fix it!"
        )
    output_schema = Schema(
        {
            **{column: input_schema[column] for column in groupby_columns},
            **{query.output_column: output_column_type},
        },
        grouping_column=None,
    )
    return output_schema


class OutputSchemaVisitor(QueryExprVisitor):
    """A visitor to get the output schema of a query expression."""

    def __init__(self, catalog: Catalog):
        """Visitor constructor.

        Args:
            catalog: The catalog defining schemas and relations between tables.
        """
        self._catalog = catalog

    def visit_private_source(self, expr):
        """Return the resulting schema from evaluating a PrivateSource.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = PrivateSource("private")
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'B': 'INTEGER'})
        """
        if expr.source_id not in self._catalog.tables:
            raise ValueError(f"Query references invalid source '{expr.source_id}'.")
        table = self._catalog.tables[expr.source_id]
        if not isinstance(table, (PrivateTable, PrivateView)):
            raise ValueError(
                f"Attempted query on '{expr.source_id}'. "
                f"'{expr.source_id}' is not a private table."
            )
        return table.schema

    def visit_rename(self, expr):
        """Returns the resulting schema from evaluating a Rename.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import PrivateSource, Rename

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = Rename(PrivateSource("private"), {"B": "C"})
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'C': 'INTEGER'})
        """
        input_schema = expr.child.accept(self)
        grouping_column = input_schema.grouping_column
        nonexistent_columns = set(expr.column_mapper) - set(input_schema)
        if nonexistent_columns:
            raise ValueError(
                f"Non existent columns {nonexistent_columns} in Rename query."
            )
        for old, new in expr.column_mapper.items():
            if new in input_schema and new != old:
                raise ValueError(
                    f"Cannot rename '{old}' to '{new}'. Column '{new}' already exists."
                )
            if old == grouping_column:
                grouping_column = new
        return Schema(
            {
                expr.column_mapper.get(column, column): input_schema[column]
                for column in input_schema
            },
            grouping_column=grouping_column,
        )

    def visit_filter(self, expr):
        """Returns the resulting schema from evaluating a Filter.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import Filter, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = Filter(PrivateSource("private"), 'B > 10')
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'B': 'INTEGER'})
        """
        input_schema = expr.child.accept(self)
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame(
            [], schema=analytics_to_spark_schema(input_schema)
        )
        try:
            test_df.filter(expr.predicate)
        except:
            raise ValueError(
                f"Invalid filter expression: '{expr.predicate}' in Filter query."
            )
        return input_schema

    def visit_select(self, expr):
        """Returns the resulting schema from evaluating a Select.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import PrivateSource, Select

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = Select(PrivateSource("private"), ["A"])
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR'})
        """
        input_schema = expr.child.accept(self)
        grouping_column = input_schema.grouping_column
        nonexistent_columns = set(expr.columns) - set(input_schema)
        if nonexistent_columns:
            raise ValueError(
                f"Non existent columns {nonexistent_columns} in Select query."
            )
        if grouping_column is not None and grouping_column not in expr.columns:
            raise ValueError(
                f"grouping column {grouping_column} must be included in Select query"
            )
        return Schema(
            {column: input_schema[column] for column in expr.columns},
            grouping_column=grouping_column,
        )

    def visit_map(self, expr):
        """Returns the resulting schema from evaluating a Map.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import Map, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query1 = Map( # Augment = False example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: {"C": row["B"] + 1, "D": "A"},
            ...     schema_new_columns=Schema(
            ...         {"C": ColumnType.INTEGER, "D": ColumnType.VARCHAR}
            ...     ),
            ...     augment=False,
            ... )
            >>> query1.accept(output_schema_visitor)
            Schema({'C': 'INTEGER', 'D': 'VARCHAR'})
            >>> query2 = Map( # Augment = True example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: {"C": row["B"] + 1, "D": "A"},
            ...     schema_new_columns=Schema(
            ...         {"C": ColumnType.INTEGER, "D": ColumnType.VARCHAR}
            ...     ),
            ...     augment=True,
            ... )
            >>> query2.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'B': 'INTEGER', 'C': 'INTEGER', 'D': 'VARCHAR'})
        """
        input_schema = expr.child.accept(self)
        if expr.augment:
            return Schema(
                {**input_schema, **expr.schema_new_columns},
                grouping_column=input_schema.grouping_column,
            )
        elif input_schema.grouping_column:
            raise ValueError(
                "Need to set augment=True to ensure that the grouping column "
                "is available for groupby."
            )
        return expr.schema_new_columns

    def visit_flat_map(self, expr):
        # pylint: disable=line-too-long
        """Returns the resulting schema from evaluating a FlatMap.

        ..
            >>> from tmlt.analytics._schema import ColumnType, Schema
            >>> from tmlt.analytics.query_expr import FlatMap, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query1 = FlatMap( # Augment = False example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: [{"C": row["B"]}, {"C": row["B"] + 1}],
            ...     max_num_rows=2,
            ...     schema_new_columns=Schema({"C": ColumnType.INTEGER}),
            ...     augment=False,
            ... )
            >>> query1.accept(output_schema_visitor)
            Schema({'C': 'INTEGER'})
            >>> query2 = FlatMap( # Augment = True example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: [{"C": row["B"]}, {"C": row["B"] + 1}],
            ...     max_num_rows=2,
            ...     schema_new_columns=Schema({"C": ColumnType.INTEGER}),
            ...     augment=True,
            ... )
            >>> query2.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'B': 'INTEGER', 'C': 'INTEGER'})
            >>> query3 = FlatMap( # Grouping example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: [{"C": row["B"]}, {"C": row["B"] + 1}],
            ...     max_num_rows=2,
            ...     schema_new_columns=Schema(
            ...         {"C": ColumnType.INTEGER}, grouping_column="C",
            ...     ),
            ...     augment=True,
            ... )
            >>> query3.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'B': 'INTEGER', 'C': 'INTEGER'}, grouping_column='C')
        """
        input_schema = expr.child.accept(self)
        if expr.schema_new_columns.grouping_column is None:
            grouping_column = input_schema.grouping_column
        else:
            if input_schema.grouping_column:
                raise ValueError(
                    "Multiple grouping transformations are used in this query. "
                    "Only one grouping transformation is allowed."
                )
            (grouping_column,) = expr.schema_new_columns

        if expr.augment:
            return Schema(
                {**input_schema, **expr.schema_new_columns},
                grouping_column=grouping_column,
            )
        elif input_schema.grouping_column is not None:
            raise ValueError(
                "Need to set augment=True to ensure that the grouping column "
                "is available for groupby."
            )
        return expr.schema_new_columns

    def visit_join_private(self, expr):
        # pylint: disable=line-too-long
        """Returns the resulting schema from evaluating a JoinPrivate.

        The ordering of output columns are:

        1. The join columns
        2. Columns that are only in the left table
        3. Columns that are only in the right table
        4. Columns that are in both tables, but not included in the join columns. These
           columns are included with _left and _right suffixes.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import (
            ...     JoinPrivate, PrivateSource
            ... )
            >>> from tmlt.analytics.truncation_strategy import TruncationStrategy

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="left_source",
            ...     col_types={
            ...         "left_only": ColumnType.DECIMAL,
            ...         "common1": ColumnType.INTEGER,
            ...         "common2": ColumnType.VARCHAR,
            ...         "common3": ColumnType.INTEGER
            ...     },
            ...     stability=1,
            ... )
            >>> catalog.add_private_view(
            ...     source_id="right_source",
            ...     col_types={
            ...         "common1": ColumnType.INTEGER,
            ...         "common2": ColumnType.VARCHAR,
            ...         "common3": ColumnType.INTEGER,
            ...         "right_only": ColumnType.VARCHAR,
            ...    },
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> # join_columns default behavior is ["common1", "common2", "common3"]
            >>> query1 = JoinPrivate(
            ...     child=PrivateSource("left_source"),
            ...     right_operand_expr=PrivateSource("right_source"),
            ...     truncation_strategy_left=TruncationStrategy.DropExcess(1),
            ...     truncation_strategy_right=TruncationStrategy.DropExcess(1),
            ... )
            >>> query1.accept(output_schema_visitor)
            Schema({'common1': 'INTEGER', 'common2': 'VARCHAR', 'common3': 'INTEGER', 'left_only': 'DECIMAL', 'right_only': 'VARCHAR'})
            >>> query2 = JoinPrivate(
            ...     child=PrivateSource("left_source"),
            ...     right_operand_expr=PrivateSource("right_source"),
            ...     truncation_strategy_left=TruncationStrategy.DropExcess(1),
            ...     truncation_strategy_right=TruncationStrategy.DropExcess(1),
            ...     join_columns=["common3"],
            ... )
            >>> query2.accept(output_schema_visitor)
            Schema({'common3': 'INTEGER', 'left_only': 'DECIMAL', 'common1_left': 'INTEGER', 'common2_left': 'VARCHAR', 'common1_right': 'INTEGER', 'common2_right': 'VARCHAR', 'right_only': 'VARCHAR'})
        """
        return _output_schema_for_join(
            left_schema=expr.child.accept(self),
            right_schema=expr.right_operand_expr.accept(self),
            join_columns=expr.join_columns,
        )

    def visit_join_public(self, expr):
        """Returns the resulting schema from evaluating a JoinPublic.

        Has analogous behavior to :meth:`OutputSchemaVisitor.visit_join_private`,
        where the private table is the left table.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import JoinPublic, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> catalog.add_public_source(
            ...     "public", {"B": ColumnType.INTEGER, "C": ColumnType.DECIMAL}
            ... )
            >>> query = JoinPublic(
            ...    child=PrivateSource("private"), public_table="public"
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'B': 'INTEGER', 'A': 'VARCHAR', 'C': 'DECIMAL'})
        """
        input_schema = expr.child.accept(self)
        if isinstance(expr.public_table, str):
            public_table = self._catalog.tables[expr.public_table]
            if not isinstance(public_table, PublicTable):
                raise ValueError(
                    f"Attempted JoinPublic on '{expr.public_table}' table. "
                    f"'{expr.public_table}' is not a public table."
                )
            right_schema = public_table.schema
        else:
            right_schema = Schema(
                spark_schema_to_analytics_columns(expr.public_table.schema)
            )
        return _output_schema_for_join(
            left_schema=input_schema,
            right_schema=right_schema,
            join_columns=expr.join_columns,
        )

    def visit_groupby_count(self, expr):
        """Returns the resulting schema from evaluating a GroupByCount.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByCount(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     output_column="count",
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'count': 'INTEGER'})
        """
        return _validate_groupby(expr, self._catalog, self)

    def visit_groupby_count_distinct(self, expr):
        """Returns the resulting schema from evaluating a GroupByCountDistinct.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByCountDistinct(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     output_column="count_distinct",
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'count_distinct': 'INTEGER'})
        """
        return _validate_groupby(expr, self._catalog, self)

    def visit_groupby_quantile(self, expr):
        """Returns the resulting schema from evaluating a GroupByQuantile.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByQuantile(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     measure_column="B",
            ...     quantile=0.5,
            ...     low=0,
            ...     high=10,
            ...     output_column="quantile",
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'quantile': 'DECIMAL'})
        """
        return _validate_groupby(expr, self._catalog, self)

    def visit_groupby_bounded_sum(self, expr):
        """Returns the resulting schema from evaluating a GroupByBoundedSum.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByBoundedSum(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     measure_column="B",
            ...     low=0,
            ...     high=10,
            ...     output_column="sum",
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'sum': 'INTEGER'})
        """
        return _validate_groupby(expr, self._catalog, self)

    def visit_groupby_bounded_average(self, expr):
        """Returns the resulting schema from evaluating a GroupByBoundedAverage.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByBoundedAverage(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     measure_column="B",
            ...     low=0,
            ...     high=10,
            ...     output_column="average",
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'average': 'DECIMAL'})
        """
        return _validate_groupby(expr, self._catalog, self)

    def visit_groupby_bounded_variance(self, expr):
        """Returns the resulting schema from evaluating a GroupByBoundedVariance.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByBoundedAverage(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     measure_column="B",
            ...     low=0,
            ...     high=10,
            ...     output_column="variance",
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'variance': 'DECIMAL'})
        """
        return _validate_groupby(expr, self._catalog, self)

    def visit_groupby_bounded_stdev(self, expr):
        """Returns the resulting schema from evaluating a GroupByBoundedSTDEV.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_source(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ...     stability=1,
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByBoundedSTDEV(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     measure_column="B",
            ...     low=0,
            ...     high=10,
            ...     output_column="stdev",
            ... )
            >>> query.accept(output_schema_visitor)
            Schema({'A': 'VARCHAR', 'stdev': 'DECIMAL'})
        """
        return _validate_groupby(expr, self._catalog, self)
