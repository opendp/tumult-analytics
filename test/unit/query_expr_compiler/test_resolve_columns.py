"""Tests for resolving automatically selected query columns."""

# SPDX-License-Identifier: Apache-2.0

import datetime
from contextlib import nullcontext

import pytest
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics import AnalyticsInternalError, KeySet
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr import (
    AnalyticsDefault,
    AutoDropInfinity,
    AutoDropNullAndNan,
    AutoGetGroups,
    AutoGroupByCountDistinct,
    AutoJoinPrivate,
    AutoReplaceInfinity,
    AutoReplaceNullAndNan,
    DropInfinity,
    DropNullAndNan,
    GetGroups,
    GroupByBoundedSum,
    GroupByCountDistinct,
    JoinPrivate,
    PrivateSource,
    QueryExpr,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
)
from tmlt.analytics._query_expr_compiler import QueryExprCompiler
from tmlt.analytics._query_expr_compiler._rewrite_rules import (
    CompilationInfo,
    resolve_columns,
    rewrite,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, FrozenDict, Schema
from tmlt.analytics.constraints import Constraint, MaxRowsPerGroupPerID, MaxRowsPerID

NULLABLE_INTEGER_SCHEMA = Schema(
    {"value": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)}
)
GROUPING_SCHEMA = Schema(
    {"group": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)},
    grouping_column="group",
)
ID_SCHEMA = Schema(
    {
        "id": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        "value": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True, allow_inf=True),
        "label": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
    },
    id_column="id",
    id_space="ids",
)
ID_SCHEMA_WITHOUT_NULLS = Schema(
    {
        **ID_SCHEMA,
        "value": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
        "label": ColumnType.VARCHAR,
    },
    id_column="id",
    id_space="ids",
)
ID_SCHEMA_WITHOUT_INFINITY = Schema(
    {**ID_SCHEMA, "value": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True)},
    id_column="id",
    id_space="ids",
)


def test_unresolved_lookups_raise():
    catalog = Catalog()
    query = Rename(
        AutoReplaceNullAndNan(PrivateSource("private")),
        FrozenDict.from_dict({"value": "renamed"}),
    )
    with pytest.raises(AnalyticsInternalError, match="Columns must be resolved"):
        query.schema(catalog)
    with pytest.raises(AnalyticsInternalError, match="Columns must be resolved"):
        query.constraints(catalog)


@parametrize(
    Case("replace_nulls_type_defaults")(
        input_schemas={
            "private": Schema(
                {t.name: ColumnDescriptor(t, allow_null=True) for t in ColumnType}
                | {"not_null": ColumnType.INTEGER}
            )
        },
        query=AutoReplaceNullAndNan(PrivateSource("private")),
        expected_query=ReplaceNullAndNan(
            PrivateSource("private"),
            FrozenDict.from_dict(
                {
                    "INTEGER": 0,
                    "DECIMAL": 0.0,
                    "VARCHAR": "",
                    "DATE": datetime.date.fromtimestamp(0),
                    "TIMESTAMP": datetime.datetime.fromtimestamp(0),
                }
            ),
        ),
        expected_schema=Schema(
            {t.name: t for t in ColumnType} | {"not_null": ColumnType.INTEGER}
        ),
    ),
    Case("replace_nulls_preserves_id_and_drops_group_limit")(
        input_schemas={"private": ID_SCHEMA},
        input_constraints={
            "private": {MaxRowsPerID(10), MaxRowsPerGroupPerID("value", 2)}
        },
        query=AutoReplaceNullAndNan(PrivateSource("private")),
        expected_query=ReplaceNullAndNan(
            PrivateSource("private"),
            FrozenDict.from_dict({"value": AnalyticsDefault.DECIMAL, "label": ""}),
        ),
        expected_schema=ID_SCHEMA_WITHOUT_NULLS,
        expected_constraints={MaxRowsPerID(10)},
        warning_match="ID column 'id'",
    ),
    Case("replace_nulls_grouping_column_excluded")(
        input_schemas={"private": GROUPING_SCHEMA},
        query=AutoReplaceNullAndNan(PrivateSource("private")),
        expected_query=ReplaceNullAndNan(
            PrivateSource("private"), FrozenDict.from_dict({})
        ),
        expected_schema=GROUPING_SCHEMA,
    ),
    Case("drop_nulls_preserves_id_and_constraints")(
        input_schemas={"private": ID_SCHEMA},
        input_constraints={
            "private": {MaxRowsPerID(10), MaxRowsPerGroupPerID("value", 2)}
        },
        query=AutoDropNullAndNan(PrivateSource("private")),
        expected_query=DropNullAndNan(PrivateSource("private"), ("value", "label")),
        expected_schema=ID_SCHEMA_WITHOUT_NULLS,
        expected_constraints={MaxRowsPerID(10), MaxRowsPerGroupPerID("value", 2)},
        warning_match="ID column 'id'",
    ),
    Case("drop_infinity")(
        input_schemas={"private": ID_SCHEMA},
        input_constraints={
            "private": {MaxRowsPerID(10), MaxRowsPerGroupPerID("value", 2)}
        },
        query=AutoDropInfinity(PrivateSource("private")),
        expected_query=DropInfinity(PrivateSource("private"), ("value",)),
        expected_schema=ID_SCHEMA_WITHOUT_INFINITY,
        expected_constraints={MaxRowsPerID(10), MaxRowsPerGroupPerID("value", 2)},
    ),
    Case("replace_infinity_drops_group_limit")(
        input_schemas={"private": ID_SCHEMA},
        input_constraints={
            "private": {MaxRowsPerID(10), MaxRowsPerGroupPerID("value", 2)}
        },
        query=AutoReplaceInfinity(PrivateSource("private")),
        expected_query=ReplaceInfinity(
            PrivateSource("private"),
            FrozenDict.from_dict(
                {"value": (AnalyticsDefault.DECIMAL, AnalyticsDefault.DECIMAL)}
            ),
        ),
        expected_schema=ID_SCHEMA_WITHOUT_INFINITY,
        expected_constraints={MaxRowsPerID(10)},
    ),
    Case("join_resolves_both_children")(
        input_schemas={"private": NULLABLE_INTEGER_SCHEMA},
        query=AutoJoinPrivate(
            AutoReplaceNullAndNan(PrivateSource("private")),
            AutoReplaceNullAndNan(PrivateSource("private")),
        ),
        expected_query=JoinPrivate(
            ReplaceNullAndNan(
                PrivateSource("private"), FrozenDict.from_dict({"value": 0})
            ),
            ReplaceNullAndNan(
                PrivateSource("private"), FrozenDict.from_dict({"value": 0})
            ),
            join_columns=("value",),
        ),
        expected_schema=Schema({"value": ColumnType.INTEGER}),
    ),
    Case("get_groups_excludes_id")(
        input_schemas={"private": ID_SCHEMA},
        query=AutoGetGroups(PrivateSource("private")),
        expected_query=GetGroups(PrivateSource("private"), ("value", "label")),
        expected_schema=Schema(
            {"value": ID_SCHEMA["value"], "label": ID_SCHEMA["label"]}
        ),
    ),
    Case("count_distinct_includes_id")(
        input_schemas={"private": ID_SCHEMA},
        query=AutoGroupByCountDistinct(PrivateSource("private"), KeySet.from_dict({})),
        expected_query=GroupByCountDistinct(
            PrivateSource("private"),
            KeySet.from_dict({}),
            ("id", "value", "label"),
            core_mechanism=NoiseMechanism.GEOMETRIC,
        ),
        expected_schema=Schema({"count_distinct": ColumnType.INTEGER}),
    ),
    Case("bottom_up_resolution")(
        input_schemas={"private": NULLABLE_INTEGER_SCHEMA},
        query=AutoReplaceNullAndNan(
            Rename(
                AutoReplaceNullAndNan(PrivateSource("private")),
                FrozenDict.from_dict({"value": "renamed"}),
            )
        ),
        # The inner replacement has already removed nulls, so the outer one is empty.
        expected_query=ReplaceNullAndNan(
            Rename(
                ReplaceNullAndNan(
                    PrivateSource("private"), FrozenDict.from_dict({"value": 0})
                ),
                FrozenDict.from_dict({"value": "renamed"}),
            ),
            FrozenDict.from_dict({}),
        ),
        expected_schema=Schema({"renamed": ColumnType.INTEGER}),
    ),
    Case("resolution_before_aggregation_rules")(
        input_schemas={"private": NULLABLE_INTEGER_SCHEMA},
        query=GroupByBoundedSum(
            AutoReplaceNullAndNan(PrivateSource("private")),
            KeySet.from_dict({}),
            "value",
            low=0,
            high=10,
        ),
        output_measure=RhoZCDP(),
        expected_query=GroupByBoundedSum(
            ReplaceNullAndNan(
                PrivateSource("private"), FrozenDict.from_dict({"value": 0})
            ),
            KeySet.from_dict({}),
            "value",
            low=0,
            high=10,
            core_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
        ),
        expected_schema=Schema({"sum": ColumnType.INTEGER}),
    ),
)
def test_resolve_columns(
    input_schemas: dict[str, Schema],
    input_constraints: dict[str, set[Constraint]] | None,
    query: QueryExpr,
    output_measure: PureDP | RhoZCDP | None,
    expected_query: QueryExpr,
    expected_schema: Schema,
    expected_constraints: set[Constraint] | None,
    warning_match: str | None,
) -> None:
    catalog = Catalog()
    for table_name, schema in input_schemas.items():
        catalog.add_private_table(
            table_name,
            schema,
            constraints=(input_constraints or {}).get(table_name, set()),
            grouping_column=schema.grouping_column,
            id_column=schema.id_column,
            id_space=schema.id_space,
        )
    measure = output_measure if output_measure is not None else PureDP()
    info = CompilationInfo(measure, catalog)
    compiler = QueryExprCompiler(measure)

    def warning_context():
        if warning_match is not None:
            return pytest.warns(match=warning_match)
        return nullcontext()

    with warning_context():
        resolved = rewrite(info, query)
    assert resolved == expected_query
    assert resolve_columns(info)(resolved) == resolved

    with warning_context():
        assert compiler.query_schema(query, catalog) == expected_schema

    if not query.is_measurement():
        with warning_context():
            assert compiler.query_constraints(query, catalog) == (
                expected_constraints or set()
            )
