"""Tests for rewrite rules."""
from dataclasses import dataclass, replace
from typing import Any, Union

import pytest
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics import KeySet
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
    DropInfinity,
    GetBounds,
    DropNullAndNan,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    PrivateSource,
    QueryExpr,
    QueryExprVisitor,
    ReplaceInfinity,
    SingleChildQueryExpr,
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)
from tmlt.analytics._query_expr_compiler._rewrite_rules import (
    CompilationInfo,
    add_special_value_handling,
    select_noise_mechanism,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, FrozenDict, Schema

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025


@pytest.fixture(scope="module", name="catalog")
def fixture_catalog():
    """Setup tests."""
    c = Catalog()
    c.add_private_table(
        "private",
        {
            "string_col": ColumnDescriptor(ColumnType.VARCHAR),
            "int_col": ColumnDescriptor(ColumnType.INTEGER),
            "float_col": ColumnDescriptor(ColumnType.DECIMAL),
        },
    )
    return c


BASE_EXPR = PrivateSource("private")

AGG_CLASSES = {
    "count": (GroupByCount, CountMechanism),
    "count_distinct": (GroupByCountDistinct, CountDistinctMechanism),
    "average": (GroupByBoundedAverage, AverageMechanism),
    "sum": (GroupByBoundedSum, SumMechanism),
    "stdev": (GroupByBoundedSTDEV, StdevMechanism),
    "variance": (GroupByBoundedVariance, VarianceMechanism),
}


@parametrize(
    [
        Case()(
            query_mechanism=mech,
            output_measure=meas,
            expected_mechanism="GEOMETRIC",
        )
        for mech in ["DEFAULT", "LAPLACE"]
        for meas in [PureDP(), ApproxDP()]
    ]
    + [
        Case()(
            query_mechanism=mech,
            output_measure=RhoZCDP(),
            expected_mechanism="DISCRETE_GAUSSIAN",
        )
        for mech in ["DEFAULT", "GAUSSIAN"]
    ]
    + [
        Case()(
            query_mechanism="LAPLACE",
            output_measure=RhoZCDP(),
            expected_mechanism="GEOMETRIC",
        )
    ],
)
@parametrize(
    [
        Case()(agg="count"),
        Case()(agg="count_distinct"),
    ]
)
def test_noise_selection_counts(
    catalog: Catalog,
    agg: str,
    query_mechanism: str,
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    expected_mechanism: str,
) -> None:
    """Test noise selection for GroupByCount{Distinct,} query expressions."""
    (AggExpr, AggMech) = AGG_CLASSES[agg]
    expr = AggExpr(
        child=BASE_EXPR,
        groupby_keys=KeySet.from_dict({}),
        mechanism=AggMech[query_mechanism],
    )
    info = CompilationInfo(output_measure=output_measure, catalog=catalog)
    got_expr = select_noise_mechanism(info)(expr)
    assert got_expr == replace(expr, core_mechanism=NoiseMechanism[expected_mechanism])


@parametrize(
    [
        Case()(
            query_mechanism=mech,
            output_measure=meas,
            measure_column="int_col",
            expected_mechanism="GEOMETRIC",
        )
        for mech in ["DEFAULT", "LAPLACE"]
        for meas in [PureDP(), ApproxDP()]
    ]
    + [
        Case()(
            query_mechanism=mech,
            output_measure=meas,
            measure_column="float_col",
            expected_mechanism="LAPLACE",
        )
        for mech in ["DEFAULT", "LAPLACE"]
        for meas in [PureDP(), ApproxDP()]
    ]
    + [
        Case()(
            query_mechanism=mech,
            output_measure=RhoZCDP(),
            measure_column="int_col",
            expected_mechanism="DISCRETE_GAUSSIAN",
        )
        for mech in ["DEFAULT", "GAUSSIAN"]
    ]
    + [
        Case()(
            query_mechanism=mech,
            output_measure=RhoZCDP(),
            measure_column="float_col",
            expected_mechanism="GAUSSIAN",
        )
        for mech in ["DEFAULT", "GAUSSIAN"]
    ]
    + [
        Case()(
            query_mechanism="LAPLACE",
            output_measure=RhoZCDP(),
            measure_column="int_col",
            expected_mechanism="GEOMETRIC",
        )
    ]
    + [
        Case()(
            query_mechanism="LAPLACE",
            output_measure=RhoZCDP(),
            measure_column="float_col",
            expected_mechanism="LAPLACE",
        )
    ],
)
@parametrize(
    [
        Case()(agg="sum"),
        Case()(agg="average"),
        Case()(agg="stdev"),
        Case()(agg="variance"),
    ]
)
def test_noise_selection_numeric_aggregations(
    catalog: Catalog,
    agg: str,
    query_mechanism: str,
    measure_column: str,
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    expected_mechanism: str,
) -> None:
    """Test noise selection for GroupByBoundedAverage query exprs."""
    (AggExpr, AggMech) = AGG_CLASSES[agg]
    expr = AggExpr(
        child=BASE_EXPR,
        measure_column=measure_column,
        low=0,
        high=1,
        mechanism=AggMech[query_mechanism],
        groupby_keys=KeySet.from_dict({}),
    )
    info = CompilationInfo(output_measure=output_measure, catalog=catalog)
    got_expr = select_noise_mechanism(info)(expr)
    assert got_expr == replace(expr, core_mechanism=NoiseMechanism[expected_mechanism])


@parametrize(
    [
        Case()(
            query_mechanism=mech, output_measure=meas, expected_mechanism="GEOMETRIC"
        )
        for mech in ["DEFAULT", "LAPLACE"]
        for meas in [PureDP(), ApproxDP()]
    ]
    + [
        Case()(
            query_mechanism=mech,
            output_measure=RhoZCDP(),
            expected_mechanism="DISCRETE_GAUSSIAN",
        )
        for mech in ["DEFAULT", "GAUSSIAN"]
    ]
    + [
        Case()(
            query_mechanism="LAPLACE",
            output_measure=meas,
            expected_mechanism="GEOMETRIC",
        )
        for meas in [PureDP(), ApproxDP(), RhoZCDP()]
    ],
)
def test_noise_selection_suppress_aggregates(
    catalog: Catalog,
    query_mechanism: str,
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    expected_mechanism: str,
) -> None:
    """Test noise selection for GroupByCount query expressions."""
    expr = SuppressAggregates(
        child=GroupByCount(
            child=BASE_EXPR,
            groupby_keys=KeySet.from_dict({}),
            mechanism=CountMechanism[query_mechanism],
        ),
        column="count",
        threshold=42,
    )
    info = CompilationInfo(output_measure=output_measure, catalog=catalog)
    got_expr = select_noise_mechanism(info)(expr)
    assert got_expr == replace(
        expr,
        child=replace(expr.child, core_mechanism=NoiseMechanism[expected_mechanism]),
    )


@parametrize(
    [
        Case()(
            expr=GroupByCount(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                mechanism=CountMechanism.GAUSSIAN,
            )
        ),
        Case()(
            expr=GroupByCountDistinct(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                mechanism=CountDistinctMechanism.GAUSSIAN,
            )
        ),
        Case()(
            expr=GroupByBoundedAverage(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="int_col",
                low=0,
                high=1,
                mechanism=AverageMechanism.GAUSSIAN,
            )
        ),
        Case()(
            expr=GroupByBoundedSum(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="int_col",
                low=0,
                high=1,
                mechanism=SumMechanism.GAUSSIAN,
            )
        ),
        Case()(
            expr=GroupByBoundedSTDEV(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="int_col",
                low=0,
                high=1,
                mechanism=StdevMechanism.GAUSSIAN,
            )
        ),
        Case()(
            expr=GroupByBoundedVariance(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="int_col",
                low=0,
                high=1,
                mechanism=VarianceMechanism.GAUSSIAN,
            )
        ),
        Case()(
            expr=SuppressAggregates(
                child=GroupByCount(
                    child=PrivateSource("blah"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.GAUSSIAN,
                ),
                column="count",
                threshold=42,
            )
        ),
    ],
)
@parametrize(
    [
        Case()(output_measure=PureDP()),
        Case()(output_measure=ApproxDP()),
    ]
)
def test_noise_selection_invalid_noise(
    catalog: Catalog, expr: QueryExpr, output_measure: Union[PureDP, ApproxDP]
) -> None:
    info = CompilationInfo(output_measure=output_measure, catalog=catalog)
    with pytest.raises(
        ValueError,
        match=(
            "Gaussian noise is only supported when using a RhoZCDP budget. "
            "Use Laplace noise instead, or switch to RhoZCDP."
        ),
    ):
        select_noise_mechanism(info)(expr)


@dataclass(frozen=True)
class SomeKindOfPostProcessing(SingleChildQueryExpr):
    """A fake post-processing QueryExpr."""

    field: int
    """A field, because why not."""

    def schema(self, catalog: Catalog) -> Schema:
        """Just propagate the schema from the child."""
        return self.child.schema(catalog)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """This should not be called."""
        raise NotImplementedError()


def test_recursive_noise_selection(catalog: Catalog) -> None:
    """Checks that noise selection works for new post-processing QueryExprs."""
    expr = SomeKindOfPostProcessing(
        child=SomeKindOfPostProcessing(
            child=GroupByBoundedAverage(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="int_col",
                low=0,
                high=1,
                mechanism=AverageMechanism.DEFAULT,
            ),
            field=42,
        ),
        field=17,
    )
    expected_expr = SomeKindOfPostProcessing(
        child=SomeKindOfPostProcessing(
            child=GroupByBoundedAverage(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="int_col",
                low=0,
                high=1,
                mechanism=AverageMechanism.DEFAULT,
                core_mechanism=NoiseMechanism.GEOMETRIC,
            ),
            field=42,
        ),
        field=17,
    )
    info = CompilationInfo(output_measure=ApproxDP(), catalog=catalog)
    got_expr = select_noise_mechanism(info)(expr)
    assert got_expr == expected_expr


@parametrize(
    [
        Case()(agg="count"),
        Case()(agg="count_distinct"),
    ]
)
@parametrize(
    [
        Case()(col_desc=ColumnDescriptor(ColumnType.INTEGER, allow_null=False)),
        Case()(col_desc=ColumnDescriptor(ColumnType.INTEGER, allow_null=True)),
        Case()(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=False, allow_nan=False, allow_inf=False
            )
        ),
        Case()(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            )
        ),
    ]
)
def test_special_value_handling_count_unaffected(
    agg: str,
    col_desc: ColumnDescriptor,
) -> None:
    (AggExpr, AggMech) = AGG_CLASSES[agg]
    expr = AggExpr(
        child=BASE_EXPR,
        groupby_keys=KeySet.from_dict({}),
        mechanism=AggMech["DEFAULT"],
    )
    catalog = Catalog()
    catalog.add_private_table("private", {"col": col_desc})
    info = CompilationInfo(output_measure=PureDP(), catalog=catalog)
    got_expr = add_special_value_handling(info)(expr)
    assert got_expr == expr


@parametrize(
    [
        # Columns with no special values should be unaffected
        Case(f"no_op_null_{col_type}")(
            col_desc=ColumnDescriptor(
                col_type, allow_null=False, allow_nan=False, allow_inf=False
            ),
            new_child=BASE_EXPR,
        )
        for col_type in [
            ColumnType.INTEGER,
            ColumnType.DECIMAL,
            ColumnType.DATE,
            ColumnType.TIMESTAMP,
        ]
    ]
    + [
        # NaNs and infinities do not matter for non-floats
        Case(f"no_op_nan_inf_{col_type}")(
            col_desc=ColumnDescriptor(
                col_type, allow_null=False, allow_nan=True, allow_inf=True
            ),
            new_child=BASE_EXPR,
        )
        for col_type in [ColumnType.INTEGER, ColumnType.DATE, ColumnType.TIMESTAMP]
    ]
    + [
        # Nulls must be dropped if needed
        Case(f"drop_null_{col_type}")(
            col_desc=ColumnDescriptor(
                col_type, allow_null=True, allow_nan=False, allow_inf=False
            ),
            new_child=DropNullAndNan(child=BASE_EXPR, columns=("col",)),
        )
        for col_type in [
            ColumnType.INTEGER,
            ColumnType.DECIMAL,
            ColumnType.DATE,
            ColumnType.TIMESTAMP,
        ]
    ]
    + [
        # NaNs must also be dropped added if needed
        Case("drop_nan")(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=False, allow_nan=True, allow_inf=False
            ),
            new_child=DropNullAndNan(child=BASE_EXPR, columns=("col",)),
        ),
        # Only one pass is enough to drop both nulls and NaNs
        Case("drop_both")(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=False
            ),
            new_child=DropNullAndNan(child=BASE_EXPR, columns=("col",)),
        ),
        # If not handled, infinities must be clamped to the clamping bounds
        Case("clamp_inf")(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=False, allow_nan=False, allow_inf=True
            ),
            new_child=ReplaceInfinity(
                child=BASE_EXPR, replace_with=FrozenDict.from_dict({"col": (0, 1)})
            ),
        ),
        # Handling both kinds of special values at once. This would fail if the two
        # value handling exprs are in the wrong order; this is not ideal, but ah well.
        Case("drop_nan_clamp_inf")(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
            new_child=ReplaceInfinity(
                child=DropNullAndNan(child=BASE_EXPR, columns=("col",)),
                replace_with=FrozenDict.from_dict({"col": (0, 1)}),
            ),
        )
    ]
)
@parametrize(
    [
        Case()(agg="sum"),
        Case()(agg="average"),
        Case()(agg="stdev"),
        Case()(agg="variance"),
    ]
)
def test_special_value_handling_numeric_aggregations(
    agg: str,
    col_desc: ColumnDescriptor,
    new_child: QueryExpr,
) -> None:
    (AggExpr, AggMech) = AGG_CLASSES[agg]
    expr = AggExpr(
        child=BASE_EXPR,
        measure_column="col",
        low=0,
        high=1,
        groupby_keys=KeySet.from_dict({}),
        mechanism=AggMech["DEFAULT"],
    )
    catalog = Catalog()
    catalog.add_private_table("private", {"col": col_desc})
    info = CompilationInfo(output_measure=PureDP(), catalog=catalog)
    got_expr = add_special_value_handling(info)(expr)
    assert got_expr == replace(
        expr,
        child=new_child,
    )

@parametrize(
    [
        # Columns with no special values should be unaffected
        Case(f"no-op-{col_type}")(
            col_desc=ColumnDescriptor(
                col_type, allow_null=False, allow_nan=False, allow_inf=False
            ),
            new_child=BASE_EXPR,
        )
        for col_type in [
            ColumnType.INTEGER,
            ColumnType.DECIMAL,
            ColumnType.DATE,
            ColumnType.TIMESTAMP,
        ]
    ]
    + [
        # NaNs and infinities do not matter for non-floats
        Case(f"no-op-nan-inf-{col_type}")(
            col_desc=ColumnDescriptor(
                col_type, allow_null=False, allow_nan=True, allow_inf=True
            ),
            new_child=BASE_EXPR,
        )
        for col_type in [ColumnType.INTEGER, ColumnType.DATE, ColumnType.TIMESTAMP]
    ]
    + [
        # Nulls must be dropped if needed
        Case(f"drop-nulls-{col_type}")(
            col_desc=ColumnDescriptor(
                col_type, allow_null=True, allow_nan=False, allow_inf=False
            ),
            new_child=DropNullAndNan(child=BASE_EXPR, columns=("col",)),
        )
        for col_type in [
            ColumnType.INTEGER,
            ColumnType.DECIMAL,
            ColumnType.DATE,
            ColumnType.TIMESTAMP,
        ]
    ]
    + [
        # NaNs must also be dropped added if needed
        Case("drop-nan")(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=False, allow_nan=True, allow_inf=False
            ),
            new_child=DropNullAndNan(child=BASE_EXPR, columns=("col",)),
        ),
        # If not handled, infinities must be clamped to the clamping bounds
        Case("drop-inf")(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=False, allow_nan=False, allow_inf=True
            ),
            new_child=DropInfinity( child=BASE_EXPR, columns=("col",)),
        ),
        # And both kinds of special values must be handled
        Case("drop-nan-and-inf")(
            col_desc=ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
            new_child=DropInfinity(
                child=DropNullAndNan(child=BASE_EXPR, columns=("col",)),
                columns=("col",)),
        )
    ]
)
def test_special_value_handling_get_bounds(
    col_desc: ColumnDescriptor,
    new_child: QueryExpr,
) -> None:
    expr = GetBounds(
        child=BASE_EXPR,
        measure_column="col",
        groupby_keys=KeySet.from_dict({}),
        lower_bound_column="lower",
        upper_bound_column="upper",
    )
    catalog = Catalog()
    catalog.add_private_table("private", {"col": col_desc})
    info = CompilationInfo(output_measure=PureDP(), catalog=catalog)
    got_expr = add_special_value_handling(info)(expr)
    assert got_expr == replace(
        expr,
        child=new_child,
    )
