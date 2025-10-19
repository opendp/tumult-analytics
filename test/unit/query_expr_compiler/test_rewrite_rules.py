"""Tests for rewrite rules."""
from dataclasses import replace
from typing import Optional, Union

import pytest
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP

from tmlt.analytics import KeySet
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    PrivateSource,
    QueryExpr,
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)
from tmlt.analytics._query_expr_compiler._rewrite_rules import CompilationInfo, rewrite
from tmlt.analytics._schema import ColumnDescriptor, ColumnType

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025


@pytest.fixture(name="test_data", scope="class")
def setup_test(request):
    """Setup tests."""
    catalog = Catalog()
    catalog.add_private_table(
        "private",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "X": ColumnDescriptor(ColumnType.DECIMAL),
        },
    )
    request.cls.catalog = catalog


BASE_EXPR = PrivateSource("private")


@pytest.mark.usefixtures("test_data")
class TestRewriteRules:
    """Tests for rewrite rules."""

    catalog: Catalog
    base_expr: QueryExpr

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,expected_mechanism",
        [
            (mech, meas, NoiseMechanism.GEOMETRIC)
            for mech in [CountMechanism.DEFAULT, CountMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (mech, RhoZCDP(), NoiseMechanism.DISCRETE_GAUSSIAN)
            for mech in [CountMechanism.DEFAULT, CountMechanism.GAUSSIAN]
        ]
        + [
            (CountMechanism.LAPLACE, meas, NoiseMechanism.GEOMETRIC)
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ],
    )
    def test_noise_selection_for_count(
        self,
        query_mechanism: CountMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test noise selection for GroupByCount query expressions."""
        expr = GroupByCount(
            child=BASE_EXPR,
            groupby_keys=KeySet.from_dict({}),
            mechanism=query_mechanism,
        )
        info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
        got_expr = rewrite(info, expr)
        assert got_expr == replace(expr, core_mechanism=expected_mechanism)

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,expected_mechanism",
        [
            (mech, meas, NoiseMechanism.GEOMETRIC)
            for mech in [CountDistinctMechanism.DEFAULT, CountDistinctMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (mech, RhoZCDP(), NoiseMechanism.DISCRETE_GAUSSIAN)
            for mech in [
                CountDistinctMechanism.DEFAULT,
                CountDistinctMechanism.GAUSSIAN,
            ]
        ]
        + [
            (CountDistinctMechanism.LAPLACE, meas, NoiseMechanism.GEOMETRIC)
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ],
    )
    def test_noise_selection_for_count_distinct(
        self,
        query_mechanism: CountDistinctMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test noise selection for GroupByCountDistinct query expressions."""
        expr = GroupByCountDistinct(
            child=BASE_EXPR,
            groupby_keys=KeySet.from_dict({}),
            mechanism=query_mechanism,
        )
        info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
        got_expr = rewrite(info, expr)
        assert got_expr == replace(expr, core_mechanism=expected_mechanism)

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (mech, meas, SparkIntegerColumnDescriptor(), NoiseMechanism.GEOMETRIC)
            for mech in [AverageMechanism.DEFAULT, AverageMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (mech, meas, SparkFloatColumnDescriptor(), NoiseMechanism.LAPLACE)
            for mech in [AverageMechanism.DEFAULT, AverageMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (
                mech,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            )
            for mech in [AverageMechanism.DEFAULT, AverageMechanism.GAUSSIAN]
        ]
        + [
            (mech, RhoZCDP(), SparkFloatColumnDescriptor(), NoiseMechanism.GAUSSIAN)
            for mech in [AverageMechanism.DEFAULT, AverageMechanism.GAUSSIAN]
        ]
        + [
            (
                AverageMechanism.LAPLACE,
                meas,
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ]
        + [
            (
                AverageMechanism.LAPLACE,
                meas,
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ],
    )
    def test_noise_selection_for_average(
        self,
        query_mechanism: AverageMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test noise selection for GroupByBoundedAverage query exprs."""
        if isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        elif isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        else:
            raise AssertionError("Unknown measure column type")
        expr = GroupByBoundedAverage(
            child=BASE_EXPR,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
        got_expr = rewrite(info, expr)
        assert got_expr == replace(expr, core_mechanism=expected_mechanism)

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (mech, meas, SparkIntegerColumnDescriptor(), NoiseMechanism.GEOMETRIC)
            for mech in [SumMechanism.DEFAULT, SumMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (mech, meas, SparkFloatColumnDescriptor(), NoiseMechanism.LAPLACE)
            for mech in [SumMechanism.DEFAULT, SumMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (
                mech,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            )
            for mech in [SumMechanism.DEFAULT, SumMechanism.GAUSSIAN]
        ]
        + [
            (mech, RhoZCDP(), SparkFloatColumnDescriptor(), NoiseMechanism.GAUSSIAN)
            for mech in [SumMechanism.DEFAULT, SumMechanism.GAUSSIAN]
        ]
        + [
            (
                SumMechanism.LAPLACE,
                meas,
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ]
        + [
            (
                SumMechanism.LAPLACE,
                meas,
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ],
    )
    def test_noise_selection_for_sum(
        self,
        query_mechanism: SumMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test noise selection for GroupByBoundedSum query exprs."""
        if isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        elif isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        else:
            raise AssertionError("Unknown measure column type")
        expr = GroupByBoundedSum(
            child=BASE_EXPR,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
        got_expr = rewrite(info, expr)
        assert got_expr == replace(expr, core_mechanism=expected_mechanism)

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (mech, meas, SparkIntegerColumnDescriptor(), NoiseMechanism.GEOMETRIC)
            for mech in [VarianceMechanism.DEFAULT, VarianceMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (mech, meas, SparkFloatColumnDescriptor(), NoiseMechanism.LAPLACE)
            for mech in [VarianceMechanism.DEFAULT, VarianceMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (
                mech,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            )
            for mech in [VarianceMechanism.DEFAULT, VarianceMechanism.GAUSSIAN]
        ]
        + [
            (mech, RhoZCDP(), SparkFloatColumnDescriptor(), NoiseMechanism.GAUSSIAN)
            for mech in [VarianceMechanism.DEFAULT, VarianceMechanism.GAUSSIAN]
        ]
        + [
            (
                VarianceMechanism.LAPLACE,
                meas,
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ]
        + [
            (
                VarianceMechanism.LAPLACE,
                meas,
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ],
    )
    def test_noise_selection_for_variance(
        self,
        query_mechanism: VarianceMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test noise selection for GroupByBoundedVariance query exprs."""
        if isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        elif isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        else:
            raise AssertionError("Unknown measure column type")
        expr = GroupByBoundedVariance(
            child=BASE_EXPR,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
        got_expr = rewrite(info, expr)
        assert got_expr == replace(expr, core_mechanism=expected_mechanism)

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (mech, meas, SparkIntegerColumnDescriptor(), NoiseMechanism.GEOMETRIC)
            for mech in [StdevMechanism.DEFAULT, StdevMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (mech, meas, SparkFloatColumnDescriptor(), NoiseMechanism.LAPLACE)
            for mech in [StdevMechanism.DEFAULT, StdevMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (
                mech,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            )
            for mech in [StdevMechanism.DEFAULT, StdevMechanism.GAUSSIAN]
        ]
        + [
            (mech, RhoZCDP(), SparkFloatColumnDescriptor(), NoiseMechanism.GAUSSIAN)
            for mech in [StdevMechanism.DEFAULT, StdevMechanism.GAUSSIAN]
        ]
        + [
            (
                StdevMechanism.LAPLACE,
                meas,
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ]
        + [
            (
                StdevMechanism.LAPLACE,
                meas,
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            )
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ],
    )
    def test_noise_selection_for_stdev(
        self,
        query_mechanism: StdevMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test noise selection for GroupByBoundedSTDEV query exprs."""
        if isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        elif isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        else:
            raise AssertionError("Unknown measure column type")
        expr = GroupByBoundedSTDEV(
            child=BASE_EXPR,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
        got_expr = rewrite(info, expr)
        assert got_expr == replace(expr, core_mechanism=expected_mechanism)

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,expected_mechanism",
        [
            (mech, meas, NoiseMechanism.GEOMETRIC)
            for mech in [CountMechanism.DEFAULT, CountMechanism.LAPLACE]
            for meas in [PureDP(), ApproxDP()]
        ]
        + [
            (mech, RhoZCDP(), NoiseMechanism.DISCRETE_GAUSSIAN)
            for mech in [CountMechanism.DEFAULT, CountMechanism.GAUSSIAN]
        ]
        + [
            (CountMechanism.LAPLACE, meas, NoiseMechanism.GEOMETRIC)
            for meas in [PureDP(), ApproxDP(), RhoZCDP()]
        ],
    )
    def test_noise_selection_suppress_aggregates(
        self,
        query_mechanism: CountMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test noise selection for GroupByCount query expressions."""
        expr = SuppressAggregates(
            child=GroupByCount(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                mechanism=query_mechanism,
            ),
            column="count",
            threshold=42,
        )
        info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
        got_expr = rewrite(info, expr)
        assert got_expr == replace(
            expr, child=replace(expr.child, core_mechanism=expected_mechanism)
        )

    @pytest.mark.parametrize(
        "expr",
        [
            GroupByCount(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                mechanism=CountMechanism.GAUSSIAN,
            ),
            GroupByCountDistinct(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                mechanism=CountDistinctMechanism.GAUSSIAN,
            ),
            GroupByBoundedAverage(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="A",
                low=0,
                high=1,
                mechanism=AverageMechanism.GAUSSIAN,
            ),
            GroupByBoundedSum(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="A",
                low=0,
                high=1,
                mechanism=SumMechanism.GAUSSIAN,
            ),
            GroupByBoundedSTDEV(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="A",
                low=0,
                high=1,
                mechanism=StdevMechanism.GAUSSIAN,
            ),
            GroupByBoundedVariance(
                child=BASE_EXPR,
                groupby_keys=KeySet.from_dict({}),
                measure_column="A",
                low=0,
                high=1,
                mechanism=VarianceMechanism.GAUSSIAN,
            ),
            SuppressAggregates(
                child=GroupByCount(
                    child=PrivateSource("blah"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.GAUSSIAN,
                ),
                column="count",
                threshold=42,
            ),
        ],
    )
    def test_noise_selection_invalid_noise(self, expr: QueryExpr) -> None:
        for output_measure in (PureDP(), ApproxDP()):
            info = CompilationInfo(output_measure=output_measure, catalog=self.catalog)
            with pytest.raises(
                ValueError,
                match=(
                    "Gaussian noise is only supported when using a RhoZCDP budget. "
                    "Use Laplace noise instead, or switch to RhoZCDP."
                ),
            ):
                rewrite(info, expr)
