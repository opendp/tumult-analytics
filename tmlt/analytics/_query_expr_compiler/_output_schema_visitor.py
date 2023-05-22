"""Defines a visitor for determining the output schemas of query expressions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from copy import deepcopy
from typing import KeysView, List, Optional, Union, cast

from pyspark.sql import SparkSession

from tmlt.analytics._catalog import Catalog, PrivateTable, PublicTable
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_py_types,
    analytics_to_spark_schema,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics.constraints import MaxGroupsPerID, MaxRowsPerGroupPerID
from tmlt.analytics.query_expr import (
    DropInfinity,
    DropNullAndNan,
    EnforceConstraint,
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
    QueryExprVisitor,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
)


def _output_schema_for_join(
    left_schema: Schema,
    right_schema: Schema,
    join_columns: Optional[List[str]],
    join_id_space: Optional[str] = None,
) -> Schema:
    """Return the resulting schema from joining two tables.

    It is assumed that if either schema has an ID column, the one from
    left_schema should be used. This is because the appropriate behavior here
    depends on the type of join being performed, so checks for compatibility of
    ID columns must happen outside this function.

    Args:
        left_schema: Schema for the left table.
        right_schema: Schema for the right table.
        join_columns: The set of columns to join on.
        join_id_space: The ID space of the resulting join.
    """
    if left_schema.grouping_column is None:
        grouping_column = right_schema.grouping_column
    elif right_schema.grouping_column is None:
        grouping_column = left_schema.grouping_column
    elif left_schema.grouping_column == right_schema.grouping_column:
        grouping_column = left_schema.grouping_column
    else:
        raise ValueError(
            "Joining tables which both have grouping columns is only supported "
            "if they have the same grouping column"
        )
    common_columns = set(left_schema) & set(right_schema)
    if join_columns is None and not common_columns:
        raise ValueError("Tables have no common columns to join on")
    if join_columns is not None and not join_columns:
        # This error case should be caught when constructing the query
        # expression, so it should never get here.
        raise AssertionError(
            "Empty list of join columns provided. This is probably a bug; "
            "please let us know about it so we can fix it!"
        )

    join_columns = (
        join_columns
        if join_columns
        else sorted(common_columns, key=list(left_schema).index)
    )

    if not set(join_columns) <= common_columns:
        raise ValueError("Join columns must be common to both tables")

    for column in join_columns:
        if left_schema[column].column_type != right_schema[column].column_type:
            raise ValueError(
                "Join columns must have identical types on both tables, "
                f"but column '{column}' does not: {left_schema[column]} and "
                f"{right_schema[column]} are incompatible"
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
    return Schema(
        output_schema,
        grouping_column=grouping_column,
        id_column=left_schema.id_column,
        id_space=join_id_space,
    )


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
    output_schema_visitor: "OutputSchemaVisitor",
) -> Schema:
    """Validate groupby aggregate query.

    Args:
        query: Query expression to be validated.
        output_schema_visitor: A visitor to get the output schema of an expression.

    Returns:
        Output schema of current QueryExpr
    """
    input_schema = query.child.accept(output_schema_visitor)

    groupby_columns: KeysView[str] = cast(
        KeysView[str], query.groupby_keys.schema().keys()
    )
    schema: Schema = query.groupby_keys.schema()

    for column_name, column_desc in schema.items():
        try:
            input_column_desc = input_schema[column_name]
        except KeyError as e:
            raise KeyError(
                f"Groupby column '{column_name}' is not in the input schema."
            ) from e
        if column_desc.column_type != input_column_desc.column_type:
            raise ValueError(
                f"Groupby column '{column_name}' has type"
                f" '{column_desc.column_type.name}', but the column with the same name"
                f" in the input data has type '{input_column_desc.column_type.name}'"
                " instead."
            )

    grouping_column = input_schema.grouping_column
    if grouping_column is not None and grouping_column not in groupby_columns:
        raise ValueError(
            f"Column '{grouping_column}' produced by grouping transformation "
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
        output_column_type = ColumnType.INTEGER
    elif isinstance(query, GroupByQuantile):
        if input_schema[query.measure_column].column_type not in [
            ColumnType.INTEGER,
            ColumnType.DECIMAL,
        ]:
            raise ValueError(
                f"Quantile query's measure column '{query.measure_column}' has invalid"
                f" type '{input_schema[query.measure_column].column_type.name}'."
                " Expected types: 'INTEGER' or 'DECIMAL'."
            )
        output_column_type = ColumnType.DECIMAL
    elif isinstance(
        query,
        (
            GroupByBoundedSum,
            GroupByBoundedSTDEV,
            GroupByBoundedAverage,
            GroupByBoundedVariance,
        ),
    ):
        if input_schema[query.measure_column].column_type not in [
            ColumnType.INTEGER,
            ColumnType.DECIMAL,
        ]:
            raise ValueError(
                f"{type(query).__name__} query's measure column "
                f"'{query.measure_column}' has invalid type "
                f"'{input_schema[query.measure_column].column_type.name}'. "
                "Expected types: 'INTEGER' or 'DECIMAL'."
            )
        output_column_type = (
            input_schema[query.measure_column].column_type
            if isinstance(query, GroupByBoundedSum)
            else ColumnType.DECIMAL
        )
    else:
        raise AssertionError(
            "Unexpected QueryExpr type. This should not happen and is"
            "probably a bug; please let us know so we can fix it!"
        )
    output_schema = Schema(
        {
            **{column: input_schema[column] for column in groupby_columns},
            **{
                query.output_column: ColumnDescriptor(
                    output_column_type, allow_null=False
                )
            },
        },
        grouping_column=None,
        id_column=None,
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

    def visit_private_source(self, expr: PrivateSource) -> Schema:
        """Return the resulting schema from evaluating a PrivateSource.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = PrivateSource("private")
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'B': 'INTEGER'}
        """
        if expr.source_id not in self._catalog.tables:
            raise ValueError(f"Query references nonexistent table '{expr.source_id}'")
        table = self._catalog.tables[expr.source_id]
        if not isinstance(table, PrivateTable):
            raise ValueError(
                f"Attempted query on table '{expr.source_id}', which is "
                "not a private table"
            )
        return table.schema

    def visit_rename(self, expr: Rename) -> Schema:
        """Returns the resulting schema from evaluating a Rename.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import PrivateSource, Rename

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = Rename(PrivateSource("private"), {"B": "C"})
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'C': 'INTEGER'}
        """
        input_schema = expr.child.accept(self)
        grouping_column = input_schema.grouping_column
        id_column = input_schema.id_column
        id_space = input_schema.id_space
        nonexistent_columns = set(expr.column_mapper) - set(input_schema)
        if nonexistent_columns:
            raise ValueError(
                f"Nonexistent columns {nonexistent_columns} in rename query"
            )
        for old, new in expr.column_mapper.items():
            if new in input_schema and new != old:
                raise ValueError(
                    f"Cannot rename '{old}' to '{new}': column '{new}' already exists"
                )
            if old == grouping_column:
                grouping_column = new
            if old == id_column:
                id_column = new

        return Schema(
            {
                expr.column_mapper.get(column, column): input_schema[column]
                for column in input_schema
            },
            grouping_column=grouping_column,
            id_column=id_column,
            id_space=id_space,
        )

    def visit_filter(self, expr: Filter) -> Schema:
        """Returns the resulting schema from evaluating a Filter.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import Filter, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = Filter(PrivateSource("private"), 'B > 10')
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'B': 'INTEGER'}
        """
        input_schema = expr.child.accept(self)
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame(
            [], schema=analytics_to_spark_schema(input_schema)
        )
        try:
            test_df.filter(expr.condition)
        except Exception as e:
            raise ValueError(f"Invalid filter condition '{expr.condition}': {e}") from e
        return input_schema

    def visit_select(self, expr: Select) -> Schema:
        """Returns the resulting schema from evaluating a Select.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import PrivateSource, Select

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = Select(PrivateSource("private"), ["A"])
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR'}
        """
        input_schema = expr.child.accept(self)

        grouping_column = input_schema.grouping_column
        id_column = input_schema.id_column
        if grouping_column is not None and grouping_column not in expr.columns:
            raise ValueError(
                f"Grouping column '{grouping_column}' may not "
                "be dropped by select query"
            )
        if id_column is not None and id_column not in expr.columns:
            raise ValueError(
                f"ID column '{id_column}' may not be dropped by select query"
            )

        nonexistent_columns = set(expr.columns) - set(input_schema)
        if nonexistent_columns:
            raise ValueError(
                f"Nonexistent columns {nonexistent_columns} in select query."
            )

        return Schema(
            {column: input_schema[column] for column in expr.columns},
            grouping_column=grouping_column,
            id_column=id_column,
            id_space=input_schema.id_space,
        )

    def visit_map(self, expr: Map) -> Schema:
        """Returns the resulting schema from evaluating a Map.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import Map, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={
            ...         "A": ColumnType.VARCHAR,
            ...         "B": ColumnType.INTEGER,
            ...     },
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
            >>> query1.accept(output_schema_visitor).column_types
            {'C': 'INTEGER', 'D': 'VARCHAR'}
            >>> query2 = Map( # Augment = True example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: {"C": row["B"] + 1, "D": "A"},
            ...     schema_new_columns=Schema(
            ...         {"C": ColumnType.INTEGER, "D": ColumnType.VARCHAR}
            ...     ),
            ...     augment=True,
            ... )
            >>> query2.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'B': 'INTEGER', 'C': 'INTEGER', 'D': 'VARCHAR'}
        """
        input_schema = expr.child.accept(self)
        # Make a deep copy -  that way we don't modify the schema that the
        # user provided
        new_columns = deepcopy(expr.schema_new_columns)
        # Any column created by Map could contain a null value
        for name in list(new_columns.keys()):
            new_columns[name].allow_null = True
        if expr.augment:
            return Schema(
                {**input_schema, **new_columns},
                grouping_column=input_schema.grouping_column,
                id_column=input_schema.id_column,
                id_space=input_schema.id_space,
            )
        elif input_schema.grouping_column:
            raise ValueError(
                "Map must set augment=True to ensure that "
                f"grouping column '{input_schema.grouping_column}' is not lost."
            )
        elif input_schema.id_column:
            raise ValueError(
                "Map must set augment=True to ensure that "
                f"ID column '{input_schema.id_column}' is not lost."
            )
        return new_columns

    def visit_flat_map(self, expr: FlatMap) -> Schema:
        """Returns the resulting schema from evaluating a FlatMap.

        ..
            >>> from tmlt.analytics._schema import ColumnType, Schema
            >>> from tmlt.analytics.query_expr import FlatMap, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query1 = FlatMap( # Augment = False example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: [{"C": row["B"]}, {"C": row["B"] + 1}],
            ...     schema_new_columns=Schema({"C": ColumnType.INTEGER}),
            ...     augment=False,
            ...     max_num_rows=2,
            ... )
            >>> query1.accept(output_schema_visitor).column_types
            {'C': 'INTEGER'}
            >>> query2 = FlatMap( # Augment = True example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: [{"C": row["B"]}, {"C": row["B"] + 1}],
            ...     schema_new_columns=Schema({"C": ColumnType.INTEGER}),
            ...     augment=True,
            ...     max_num_rows=2,
            ... )
            >>> query2.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'B': 'INTEGER', 'C': 'INTEGER'}
            >>> query3 = FlatMap( # Grouping example
            ...     child=PrivateSource("private"),
            ...     f=lambda row: [{"C": row["B"]}, {"C": row["B"] + 1}],
            ...     schema_new_columns=Schema(
            ...         {"C": ColumnType.INTEGER}, grouping_column="C",
            ...     ),
            ...     augment=True,
            ...     max_num_rows=2,
            ... )
            >>> query3.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'B': 'INTEGER', 'C': 'INTEGER'}
        """
        input_schema = expr.child.accept(self)
        if expr.schema_new_columns.grouping_column is not None:
            if input_schema.grouping_column:
                raise ValueError(
                    "Multiple grouping transformations are used in this query. "
                    "Only one grouping transformation is allowed."
                )
            if input_schema.id_column:
                raise ValueError(
                    "Grouping flat map cannot be used on tables with "
                    "the AddRowsWithID protected change"
                )
            grouping_column = expr.schema_new_columns.grouping_column
        else:
            grouping_column = input_schema.grouping_column

        # Make a deep copy - that way we don't modify the schema
        # that the user provided
        new_columns = deepcopy(expr.schema_new_columns)
        # Any column created by the FlatMap could contain a null value
        for name in list(new_columns.keys()):
            new_columns[name].allow_null = True
        if expr.augment:
            return Schema(
                {**input_schema, **new_columns},
                grouping_column=grouping_column,
                id_column=input_schema.id_column,
                id_space=input_schema.id_space,
            )
        elif input_schema.grouping_column:
            raise ValueError(
                "Flat map must set augment=True to ensure that "
                f"grouping column '{input_schema.grouping_column}' is not lost"
            )
        elif input_schema.id_column:
            raise ValueError(
                "Flat map must set augment=True to ensure that "
                f"ID column '{input_schema.id_column}' is not lost"
            )

        return new_columns

    def visit_join_private(self, expr: JoinPrivate) -> Schema:
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
            >>> catalog.add_private_table(
            ...     source_id="left_source",
            ...     col_types={
            ...         "left_only": ColumnType.DECIMAL,
            ...         "common1": ColumnType.INTEGER,
            ...         "common2": ColumnType.VARCHAR,
            ...         "common3": ColumnType.INTEGER
            ...     },
            ... )
            >>> catalog.add_private_table(
            ...     source_id="right_source",
            ...     col_types={
            ...         "common1": ColumnType.INTEGER,
            ...         "common2": ColumnType.VARCHAR,
            ...         "common3": ColumnType.INTEGER,
            ...         "right_only": ColumnType.VARCHAR,
            ...    },
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> # join_columns default behavior is ["common1", "common2", "common3"]
            >>> query1 = JoinPrivate(
            ...     child=PrivateSource("left_source"),
            ...     right_operand_expr=PrivateSource("right_source"),
            ...     truncation_strategy_left=TruncationStrategy.DropExcess(1),
            ...     truncation_strategy_right=TruncationStrategy.DropExcess(1),
            ... )
            >>> query1.accept(output_schema_visitor).column_types
            {'common1': 'INTEGER', 'common2': 'VARCHAR', 'common3': 'INTEGER', 'left_only': 'DECIMAL', 'right_only': 'VARCHAR'}
            >>> query2 = JoinPrivate(
            ...     child=PrivateSource("left_source"),
            ...     right_operand_expr=PrivateSource("right_source"),
            ...     truncation_strategy_left=TruncationStrategy.DropExcess(1),
            ...     truncation_strategy_right=TruncationStrategy.DropExcess(1),
            ...     join_columns=["common3"],
            ... )
            >>> query2.accept(output_schema_visitor).column_types
            {'common3': 'INTEGER', 'left_only': 'DECIMAL', 'common1_left': 'INTEGER', 'common2_left': 'VARCHAR', 'common1_right': 'INTEGER', 'common2_right': 'VARCHAR', 'right_only': 'VARCHAR'}
        """
        left_schema = expr.child.accept(self)
        right_schema = expr.right_operand_expr.accept(self)
        if left_schema.id_column != right_schema.id_column:
            if left_schema.id_column is None or right_schema.id_column is None:
                raise ValueError(
                    "Private joins can only be performed between two tables "
                    "with the same type of protected change"
                )
            raise ValueError(
                "Private joins between tables with the AddRowsWithID "
                "protected change are only possible when the ID columns of "
                "the two tables have the same name"
            )
        if (
            left_schema.id_space
            and right_schema.id_space
            and left_schema.id_space != right_schema.id_space
        ):
            raise ValueError(
                "Private joins between tables with the AddRowsWithID protected change"
                " are only possible when both tables are in the same ID space"
            )
        join_id_space: Optional[str] = None
        if left_schema.id_space and right_schema.id_space:
            join_id_space = left_schema.id_space
        return _output_schema_for_join(
            left_schema=left_schema,
            right_schema=right_schema,
            join_columns=expr.join_columns,
            join_id_space=join_id_space,
        )

    def visit_join_public(self, expr: JoinPublic) -> Schema:
        """Returns the resulting schema from evaluating a JoinPublic.

        Has analogous behavior to :meth:`OutputSchemaVisitor.visit_join_private`,
        where the private table is the left table.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.query_expr import JoinPublic, PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> catalog.add_public_table(
            ...     "public", {"B": ColumnType.INTEGER, "C": ColumnType.DECIMAL}
            ... )
            >>> query = JoinPublic(
            ...    child=PrivateSource("private"), public_table="public"
            ... )
            >>> query.accept(output_schema_visitor).column_types
            {'B': 'INTEGER', 'A': 'VARCHAR', 'C': 'DECIMAL'}
        """
        input_schema = expr.child.accept(self)
        if isinstance(expr.public_table, str):
            public_table = self._catalog.tables[expr.public_table]
            if not isinstance(public_table, PublicTable):
                raise ValueError(
                    f"Attempted public join on table '{expr.public_table}', "
                    "which is not a public table"
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
            join_id_space=input_schema.id_space,
        )

    def visit_replace_null_and_nan(self, expr: ReplaceNullAndNan) -> Schema:
        """Returns the resulting schema from evaluating a ReplaceNullAndNan."""
        input_schema = expr.child.accept(self)

        if (
            input_schema.grouping_column
            and input_schema.grouping_column in expr.replace_with
        ):
            raise ValueError(
                "Cannot replace null values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column"
            )
        if input_schema.id_column and input_schema.id_column in expr.replace_with:
            raise ValueError(
                f"Cannot replace null values in column '{input_schema.id_column}', "
                "as it is an ID column"
            )

        if len(expr.replace_with) != 0:
            pytypes = analytics_to_py_types(input_schema)
            for col, val in expr.replace_with.items():
                if col not in input_schema.keys():
                    raise ValueError(
                        f"Column '{col}' does not exist in this table, "
                        f"available columns are {list(input_schema.keys())}"
                    )
                if not isinstance(val, pytypes[col]):
                    # it's ok to use an int as a float
                    # so don't raise an error in that case
                    if not (isinstance(val, int) and pytypes[col] == float):
                        raise ValueError(
                            f"Column '{col}' cannot have nulls replaced with "
                            f"{repr(val)}, as that value's type does not match the "
                            f"column type {input_schema[col].column_type.name}"
                        )

        columns_to_change = list(dict(expr.replace_with).keys())
        if len(columns_to_change) == 0:
            columns_to_change = [
                col
                for col in input_schema.column_descs.keys()
                if (input_schema[col].allow_null or input_schema[col].allow_nan)
                and not (col == input_schema.grouping_column)
            ]

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=(cd.allow_null and not name in columns_to_change),
                    allow_nan=(cd.allow_nan and not name in columns_to_change),
                    allow_inf=cd.allow_inf,
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def visit_replace_infinity(self, expr: ReplaceInfinity) -> Schema:
        """Returns the resulting schema from evaluating a ReplaceInfinity."""
        input_schema = expr.child.accept(self)

        if (
            input_schema.grouping_column
            and input_schema.grouping_column in expr.replace_with
        ):
            raise ValueError(
                "Cannot replace infinite values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column"
            )
        # Float-valued columns cannot be ID columns, but include this to be safe.
        if input_schema.id_column and input_schema.id_column in expr.replace_with:
            raise ValueError(
                f"Cannot replace infinite values in column '{input_schema.id_column}', "
                "as it is an ID column"
            )

        columns_to_change = list(expr.replace_with.keys())
        if len(columns_to_change) == 0:
            columns_to_change = [
                col
                for col in input_schema.column_descs.keys()
                if input_schema[col].column_type == ColumnType.DECIMAL
            ]
        else:
            for name in expr.replace_with:
                if name not in input_schema.keys():
                    raise ValueError(
                        f"Column '{name}' does not exist in this table, "
                        f"available columns are {list(input_schema.keys())}"
                    )
                if input_schema[name].column_type != ColumnType.DECIMAL:
                    raise ValueError(
                        f"Column '{name}' has a replacement value provided, but it is "
                        f"of type {input_schema[name].column_type.name} (not "
                        f"{ColumnType.DECIMAL.name}) and so cannot "
                        "contain infinite values"
                    )
        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=cd.allow_null,
                    allow_nan=cd.allow_nan,
                    allow_inf=(cd.allow_inf and not name in columns_to_change),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def visit_drop_null_and_nan(self, expr: DropNullAndNan) -> Schema:
        """Returns the resulting schema from evaluating a DropNullAndNan."""
        input_schema = expr.child.accept(self)

        if (
            input_schema.grouping_column
            and input_schema.grouping_column in expr.columns
        ):
            raise ValueError(
                f"Cannot drop null values in column '{input_schema.grouping_column}', "
                "as it is a grouping column"
            )
        if input_schema.id_column and input_schema.id_column in expr.columns:
            raise ValueError(
                f"Cannot drop null values in column '{input_schema.id_column}', "
                "as it is an ID column"
            )

        columns = expr.columns.copy()
        if len(columns) == 0:
            columns = [
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan)
                and not name == input_schema.grouping_column
            ]
        else:
            for name in columns:
                if name not in input_schema.keys():
                    raise ValueError(
                        f"Column '{name}' does not exist in this table, "
                        f"available columns are {list(input_schema.keys())}"
                    )

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=(cd.allow_null and not name in columns),
                    allow_nan=(cd.allow_nan and not name in columns),
                    allow_inf=(cd.allow_inf),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def visit_drop_infinity(self, expr: DropInfinity) -> Schema:
        """Returns the resulting schema from evaluating a DropInfinity."""
        input_schema = expr.child.accept(self)

        if (
            input_schema.grouping_column
            and input_schema.grouping_column in expr.columns
        ):
            raise ValueError(
                "Cannot drop infinite values in column "
                f"'{input_schema.grouping_column}', as it is a grouping column"
            )
        # Float-valued columns cannot be ID columns, but include this to be safe.
        if input_schema.id_column and input_schema.id_column in expr.columns:
            raise ValueError(
                f"Cannot drop infinite values in column '{input_schema.id_column}', "
                "as it is an ID column"
            )

        columns = expr.columns.copy()
        if len(columns) == 0:
            columns = [
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_inf) and not name == input_schema.grouping_column
            ]
        else:
            for name in columns:
                if name not in input_schema.keys():
                    raise ValueError(
                        f"Column '{name}' does not exist in this table, "
                        f"available columns are {list(input_schema.keys())}"
                    )
                if input_schema[name].column_type != ColumnType.DECIMAL:
                    raise ValueError(
                        f"Column '{name}' was given as a column to drop "
                        "infinite values from, but it is of type"
                        f"{input_schema[name].column_type.name} (not "
                        f"{ColumnType.DECIMAL.name}) and so cannot "
                        "contain infinite values"
                    )

        return Schema(
            {
                name: ColumnDescriptor(
                    column_type=cd.column_type,
                    allow_null=cd.allow_null,
                    allow_nan=cd.allow_nan,
                    allow_inf=(cd.allow_inf and not name in columns),
                )
                for name, cd in input_schema.column_descs.items()
            },
            grouping_column=input_schema.grouping_column,
            id_column=input_schema.id_column,
            id_space=input_schema.id_space,
        )

    def visit_enforce_constraint(self, expr: EnforceConstraint) -> Schema:
        """Returns the resulting schema from evaluating an EnforceConstraint."""
        input_schema = expr.child.accept(self)
        constraint = expr.constraint

        if not input_schema.id_column:
            raise ValueError(
                f"Constraint {expr.constraint} can only be applied to tables"
                " with the AddRowsWithID protected change"
            )
        if isinstance(constraint, (MaxGroupsPerID, MaxRowsPerGroupPerID)):
            grouping_column = constraint.grouping_column
            if grouping_column not in input_schema:
                raise ValueError(
                    f"The grouping column of constraint {constraint}"
                    " does not exist in this table; available columns"
                    f" are: {', '.join(input_schema.keys())}"
                )
            if grouping_column == input_schema.id_column:
                raise ValueError(
                    f"The grouping column of constraint {constraint} cannot be"
                    " the ID column of the table it is applied to"
                )

        # No current constraints modify the schema. If that changes in the
        # future, the logic for it may have to be pushed into the Constraint
        # type (like how constraint._enforce() works), but for now this works.
        return input_schema

    def visit_groupby_count(self, expr: GroupByCount) -> Schema:
        """Returns the resulting schema from evaluating a GroupByCount.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByCount(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     output_column="count",
            ... )
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'count': 'INTEGER'}
        """
        return _validate_groupby(expr, self)

    def visit_groupby_count_distinct(self, expr: GroupByCountDistinct) -> Schema:
        """Returns the resulting schema from evaluating a GroupByCountDistinct.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
            ... )
            >>> output_schema_visitor = OutputSchemaVisitor(catalog)
            >>> query = GroupByCountDistinct(
            ...     child=PrivateSource("private"),
            ...     groupby_keys=KeySet.from_dict({"A": ["a1", "a2", "a3"]}),
            ...     output_column="count_distinct",
            ... )
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'count_distinct': 'INTEGER'}
        """
        return _validate_groupby(expr, self)

    def visit_groupby_quantile(self, expr: GroupByQuantile) -> Schema:
        """Returns the resulting schema from evaluating a GroupByQuantile.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
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
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'quantile': 'DECIMAL'}
        """
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_sum(self, expr: GroupByBoundedSum) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedSum.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
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
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'sum': 'INTEGER'}
        """
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_average(self, expr: GroupByBoundedAverage) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedAverage.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
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
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'average': 'DECIMAL'}
        """
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_variance(self, expr: GroupByBoundedVariance) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedVariance.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
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
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'variance': 'DECIMAL'}
        """
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_stdev(self, expr: GroupByBoundedSTDEV) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedSTDEV.

        ..
            >>> from tmlt.analytics._schema import ColumnType
            >>> from tmlt.analytics.keyset import KeySet
            >>> from tmlt.analytics.query_expr import PrivateSource

        Example:
            >>> catalog = Catalog()
            >>> catalog.add_private_table(
            ...     source_id="private",
            ...     col_types={"A": ColumnType.VARCHAR, "B": ColumnType.INTEGER},
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
            >>> query.accept(output_schema_visitor).column_types
            {'A': 'VARCHAR', 'stdev': 'DECIMAL'}
        """
        return _validate_groupby(expr, self)
