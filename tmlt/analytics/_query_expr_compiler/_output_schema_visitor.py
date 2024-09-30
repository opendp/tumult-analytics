"""Defines a visitor for determining the output schemas of query expressions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from collections.abc import Collection
from dataclasses import replace
from typing import Optional, Tuple, Union

from pyspark.sql import SparkSession
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.utils.join import domain_after_join

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._catalog import Catalog, PrivateTable, PublicTable
from tmlt.analytics._query_expr import (
    DropInfinity,
    DropNullAndNan,
    EnforceConstraint,
    Filter,
    FlatMap,
    FlatMapByID,
    GetBounds,
    GetGroups,
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
    SuppressAggregates,
)
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_py_types,
    analytics_to_spark_columns_descriptor,
    analytics_to_spark_schema,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics.constraints import MaxGroupsPerID, MaxRowsPerGroupPerID
from tmlt.analytics.keyset import KeySet


def _output_schema_for_join(
    left_schema: Schema,
    right_schema: Schema,
    join_columns: Optional[Tuple[str, ...]],
    join_id_space: Optional[str] = None,
    how: str = "inner",
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
        how: The type of join to perform. Default is "inner".
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
        raise AnalyticsInternalError("Empty list of join columns provided.")

    join_columns = (
        join_columns
        if join_columns
        else tuple(sorted(common_columns, key=list(left_schema).index))
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

    join_column_schemas = {column: left_schema[column] for column in join_columns}
    output_schema = {
        **join_column_schemas,
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
    # Use Core's join utilities for determining whether a column can be null
    # TODO: This could potentially be used more in this function
    output_domain = domain_after_join(
        left_domain=SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(left_schema)
        ),
        right_domain=SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(right_schema)
        ),
        on=list(join_columns),
        how=how,
        nulls_are_equal=True,
    )
    for column in output_schema:
        col_schema = output_schema[column]
        output_schema[column] = ColumnDescriptor(
            column_type=col_schema.column_type,
            allow_null=output_domain.schema[column].allow_null,
            allow_nan=col_schema.allow_nan,
            allow_inf=col_schema.allow_inf,
        )
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
        GetBounds,
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

    if isinstance(query.groupby_keys, KeySet):
        # Checks that the KeySet is valid
        schema = query.groupby_keys.schema()
        groupby_columns: Collection[str] = schema.keys()

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
                    f" '{column_desc.column_type.name}', but the column with the same "
                    f"name in the input data has type "
                    f"'{input_column_desc.column_type.name}' instead."
                )
    elif isinstance(query.groupby_keys, tuple):
        # Checks that the listed groupby columns exist in the schema
        for col in query.groupby_keys:
            if col not in input_schema:
                raise ValueError(f"Groupby column '{col}' is not in the input schema.")
        groupby_columns = query.groupby_keys
    else:
        raise AnalyticsInternalError(
            f"Unexpected groupby_keys type: {type(query.groupby_keys)}."
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
    elif isinstance(query, GetBounds):
        # Measure column type check not needed, since we check it early in
        # OutputSchemaVisitor.visit_get_bounds
        output_column_type = input_schema[query.measure_column].column_type
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
    if isinstance(query, GetBounds):
        output_schema = Schema(
            {
                **{column: input_schema[column] for column in groupby_columns},
                **{
                    query.lower_bound_column: ColumnDescriptor(
                        output_column_type, allow_null=False
                    )
                },
                **{
                    query.upper_bound_column: ColumnDescriptor(
                        output_column_type, allow_null=False
                    )
                },
            },
            grouping_column=None,
            id_column=None,
        )
    else:
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
        """Return the resulting schema from evaluating a PrivateSource."""
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
        """Returns the resulting schema from evaluating a Rename."""
        input_schema = expr.child.accept(self)
        grouping_column = input_schema.grouping_column
        id_column = input_schema.id_column
        id_space = input_schema.id_space
        nonexistent_columns = set(expr.column_mapper) - set(input_schema)
        if nonexistent_columns:
            raise ValueError(
                f"Nonexistent columns in rename query: {nonexistent_columns}"
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
        """Returns the resulting schema from evaluating a Filter."""
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
        """Returns the resulting schema from evaluating a Select."""
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
                f"Nonexistent columns in select query: {nonexistent_columns}"
            )

        return Schema(
            {column: input_schema[column] for column in expr.columns},
            grouping_column=grouping_column,
            id_column=id_column,
            id_space=input_schema.id_space,
        )

    def visit_map(self, expr: Map) -> Schema:
        """Returns the resulting schema from evaluating a Map."""
        input_schema = expr.child.accept(self)
        new_columns = expr.schema_new_columns.column_descs
        # Any column created by Map could contain a null value
        for name in list(new_columns.keys()):
            new_columns[name] = replace(new_columns[name], allow_null=True)

        if expr.augment:
            overlapping_columns = set(input_schema.keys()) & set(new_columns.keys())
            if overlapping_columns:
                raise ValueError(
                    "New columns in augmenting map must not overwrite "
                    "existing columns, but found new columns that "
                    f"already exist: {', '.join(overlapping_columns)}"
                )
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
        return Schema(
            new_columns,
            grouping_column=expr.schema_new_columns.grouping_column,
            id_column=expr.schema_new_columns.id_column,
            id_space=expr.schema_new_columns.id_space,
        )

    def visit_flat_map(self, expr: FlatMap) -> Schema:
        """Returns the resulting schema from evaluating a FlatMap."""
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
                    "the AddRowsWithID protected change."
                )
            grouping_column = expr.schema_new_columns.grouping_column
        else:
            grouping_column = input_schema.grouping_column

        new_columns = expr.schema_new_columns.column_descs
        # Any column created by the FlatMap could contain a null value
        for name in list(new_columns.keys()):
            new_columns[name] = replace(new_columns[name], allow_null=True)
        if expr.augment:
            overlapping_columns = set(input_schema.keys()) & set(new_columns.keys())
            if overlapping_columns:
                raise ValueError(
                    "New columns in augmenting map must not overwrite "
                    "existing columns, but found new columns that "
                    f"already exist: {', '.join(overlapping_columns)}"
                )
            return Schema(
                {**input_schema, **new_columns},
                grouping_column=grouping_column,
                id_column=input_schema.id_column,
                id_space=input_schema.id_space,
            )
        elif input_schema.grouping_column:
            raise ValueError(
                "Flat map must set augment=True to ensure that "
                f"grouping column '{input_schema.grouping_column}' is not lost."
            )
        elif input_schema.id_column:
            raise ValueError(
                "Flat map must set augment=True to ensure that "
                f"ID column '{input_schema.id_column}' is not lost."
            )

        return Schema(
            new_columns,
            grouping_column=grouping_column,
            id_column=expr.schema_new_columns.id_column,
            id_space=expr.schema_new_columns.id_space,
        )

    def visit_flat_map_by_id(self, expr: FlatMapByID) -> Schema:
        """Returns the resulting schema from evaluating a FlatMapByID."""
        input_schema = expr.child.accept(self)
        id_column = input_schema.id_column
        new_columns = expr.schema_new_columns.column_descs

        if not id_column:
            raise ValueError(
                "Flat-map-by-ID may only be used on tables with ID columns."
            )
        if input_schema.grouping_column:
            raise AnalyticsInternalError(
                "Encountered table with both an ID column and a grouping column."
            )
        if id_column in new_columns:
            raise ValueError(
                "Flat-map-by-ID mapping function output cannot include ID column."
            )

        for name in list(new_columns.keys()):
            new_columns[name] = replace(new_columns[name], allow_null=True)
        return Schema(
            {id_column: input_schema[id_column], **new_columns},
            grouping_column=None,
            id_column=id_column,
            id_space=input_schema.id_space,
        )

    def visit_join_private(self, expr: JoinPrivate) -> Schema:
        """Returns the resulting schema from evaluating a JoinPrivate.

        The ordering of output columns are:

        1. The join columns
        2. Columns that are only in the left table
        3. Columns that are only in the right table
        4. Columns that are in both tables, but not included in the join columns. These
           columns are included with _left and _right suffixes.
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
            how=expr.how,
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
                f"'{input_schema.grouping_column}', as it is a grouping column."
            )
        if input_schema.id_column and input_schema.id_column in expr.replace_with:
            raise ValueError(
                f"Cannot replace null values in column '{input_schema.id_column}', "
                "as it is an ID column."
            )
        if input_schema.id_column and (len(expr.replace_with) == 0):
            raise RuntimeWarning(
                f"Replacing null values in the ID column '{input_schema.id_column}' "
                "is not allowed, so the ID column may still contain null values."
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
                    # it's okay to use an int as a float
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
                and not (col in [input_schema.grouping_column, input_schema.id_column])
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
                "as it is an ID column."
            )
        if input_schema.id_column and len(expr.columns) == 0:
            raise RuntimeWarning(
                f"Replacing null values in the ID column '{input_schema.id_column}' "
                "is not allowed, so the ID column may still contain null values."
            )
        columns = expr.columns
        if len(columns) == 0:
            columns = tuple(
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_null or cd.allow_nan)
                and not name in [input_schema.grouping_column, input_schema.id_column]
            )
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

        columns = expr.columns
        if len(columns) == 0:
            columns = tuple(
                name
                for name, cd in input_schema.column_descs.items()
                if (cd.allow_inf) and not name == input_schema.grouping_column
            )
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

    def visit_get_groups(self, expr: GetGroups) -> Schema:
        """Returns the resulting schema from GetGroups."""
        input_schema = expr.child.accept(self)

        if expr.columns:
            nonexistent_columns = set(expr.columns) - set(input_schema)
            if nonexistent_columns:
                raise ValueError(
                    f"Nonexistent columns in get_groups query: {nonexistent_columns}"
                )
            input_schema = Schema(
                {column: input_schema[column] for column in expr.columns}
            )

        else:
            input_schema = Schema(
                {
                    column: input_schema[column]
                    for column in input_schema
                    if column != input_schema.id_column
                }
            )

        return input_schema

    def visit_get_bounds(self, expr: GetBounds) -> Schema:
        """Returns the resulting schema from GetBounds."""
        input_schema = expr.child.accept(self)

        if expr.measure_column not in set(input_schema):
            raise ValueError(
                f"Cannot get bounds for column '{expr.measure_column}', which "
                "does not exist"
            )

        column = input_schema[expr.measure_column]
        if column.column_type not in [
            ColumnType.INTEGER,
            ColumnType.DECIMAL,
        ]:
            raise ValueError(
                f"Cannot get bounds for column '{expr.measure_column}',"
                f" which is of type {column.column_type.name}; only columns of"
                f" numerical type are supported."
            )

        # Check if we're trying to get the bounds of the ID column.
        if input_schema.id_column and (input_schema.id_column == expr.measure_column):
            raise ValueError(
                "get_bounds cannot be used on the privacy ID column"
                f" ({input_schema.id_column}) of a table with the AddRowsWithID"
                " protected change."
            )
        return _validate_groupby(expr, self)

    def visit_groupby_count(self, expr: GroupByCount) -> Schema:
        """Returns the resulting schema from evaluating a GroupByCount."""
        return _validate_groupby(expr, self)

    def visit_groupby_count_distinct(self, expr: GroupByCountDistinct) -> Schema:
        """Returns the resulting schema from evaluating a GroupByCountDistinct."""
        return _validate_groupby(expr, self)

    def visit_groupby_quantile(self, expr: GroupByQuantile) -> Schema:
        """Returns the resulting schema from evaluating a GroupByQuantile."""
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_sum(self, expr: GroupByBoundedSum) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedSum."""
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_average(self, expr: GroupByBoundedAverage) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedAverage."""
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_variance(self, expr: GroupByBoundedVariance) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedVariance."""
        return _validate_groupby(expr, self)

    def visit_groupby_bounded_stdev(self, expr: GroupByBoundedSTDEV) -> Schema:
        """Returns the resulting schema from evaluating a GroupByBoundedSTDEV."""
        return _validate_groupby(expr, self)

    def visit_suppress_aggregates(self, expr: SuppressAggregates) -> Schema:
        """Returns the resulting schema from evaluating a SuppressAggregates."""
        return expr.child.accept(self)
