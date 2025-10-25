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
    GroupByBoundedAverage,
    GroupByBoundedStdev,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    PrivateSource,
    QueryExpr,
    QueryExprVisitor,
    SingleChildQueryExpr,
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)
from tmlt.analytics._query_expr_compiler._rewrite_rules import (
    CompilationInfo,
    select_noise_mechanism,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema

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
    "stdev": (GroupByBoundedStdev, StdevMechanism),
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
            expr=GroupByBoundedStdev(
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
