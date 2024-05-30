"""Tests for MeasurementVisitor."""
from test.conftest import assert_frame_equal_with_sort, create_empty_input
from typing import List, Optional, Union
from unittest.mock import patch

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.chaining import ChainTM
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.type_utils import assert_never

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._noise_info import NoiseInfo, _NoiseMechanism
from tmlt.analytics._query_expr_compiler._base_measurement_visitor import (
    _get_query_bounds,
)
from tmlt.analytics._query_expr_compiler._measurement_visitor import MeasurementVisitor
from tmlt.analytics._query_expr_compiler._output_schema_visitor import (
    OutputSchemaVisitor,
)
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget, RhoZCDPBudget
from tmlt.analytics.query_expr import (
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
)
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
)
from tmlt.analytics.query_expr import ReplaceInfinity as ReplaceInfExpr
from tmlt.analytics.query_expr import ReplaceNullAndNan
from tmlt.analytics.query_expr import Select as SelectExpr
from tmlt.analytics.query_expr import (
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)
from tmlt.analytics.truncation_strategy import TruncationStrategy

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024


def chain_to_list(t: ChainTT) -> List[Transformation]:
    """Turns a ChainTT's tree into a list."""
    left: List[Transformation]
    if not isinstance(t.transformation1, ChainTT):
        left = [t.transformation1]
    else:
        left = chain_to_list(t.transformation1)
    right: List[Transformation]
    if not isinstance(t.transformation2, ChainTT):
        right = [t.transformation2]
    else:
        right = chain_to_list(t.transformation2)
    return left + right


### Tests just for _get_query_bounds. ###


@pytest.mark.parametrize("lower,upper", [(0, 1), (-123456, 0), (7899000, 9999999)])
def test_average(lower: float, upper: float) -> None:
    """Test _get_query_bounds on Average query expr, with lower!=upper."""
    average = GroupByBoundedAverage(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=lower,
        high=upper,
    )
    (low, high) = _get_query_bounds(average)
    assert low == ExactNumber.from_float(lower, round_up=True)
    assert high == ExactNumber.from_float(upper, round_up=False)


@pytest.mark.parametrize("lower,upper", [(0, 1), (-123456, 0), (7899000, 9999999)])
def test_stdev(lower: float, upper: float) -> None:
    """Test _get_query_bounds on STDEV query expr, with lower!=upper."""
    stdev = GroupByBoundedSTDEV(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=lower,
        high=upper,
    )
    (low, high) = _get_query_bounds(stdev)
    assert low == ExactNumber.from_float(lower, round_up=True)
    assert high == ExactNumber.from_float(upper, round_up=False)


@pytest.mark.parametrize("lower,upper", [(0, 1), (-123456, 0), (7899000, 9999999)])
def test_sum(lower: float, upper: float) -> None:
    """Test _get_query_bounds on Sum query expr, with lower!=upper."""
    sum_query = GroupByBoundedSum(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=lower,
        high=upper,
    )
    (low, high) = _get_query_bounds(sum_query)
    assert low == ExactNumber.from_float(lower, round_up=True)
    assert high == ExactNumber.from_float(upper, round_up=False)


@pytest.mark.parametrize("lower,upper", [(0, 1), (-123456, 0), (7899000, 9999999)])
def test_variance(lower: float, upper: float) -> None:
    """Test _get_query_bounds on Variance query expr, with lower!=upper."""
    variance = GroupByBoundedVariance(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=lower,
        high=upper,
    )
    (low, high) = _get_query_bounds(variance)
    assert low == ExactNumber.from_float(lower, round_up=True)
    assert high == ExactNumber.from_float(upper, round_up=False)


@pytest.mark.parametrize("lower,upper", [(0, 1), (-123456, 0), (7899000, 9999999)])
def test_quantile(lower: float, upper: float) -> None:
    """Test _get_query_bounds on Quantile query expr, with lower!=upper."""
    quantile = GroupByQuantile(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=lower,
        high=upper,
        quantile=0.5,
    )
    (low, high) = _get_query_bounds(quantile)
    assert low == ExactNumber.from_float(lower, round_up=True)
    assert high == ExactNumber.from_float(upper, round_up=True)


###Prepare Data for Tests###


@pytest.fixture(name="test_data", scope="class")
def prepare_visitor(spark, request):
    """Setup tests."""
    input_domain = DictDomain(
        {
            NamedTable("private"): SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkIntegerColumnDescriptor(),
                    "X": SparkFloatColumnDescriptor(),
                    "null": SparkFloatColumnDescriptor(allow_null=True),
                    "nan": SparkFloatColumnDescriptor(allow_nan=True),
                    "inf": SparkFloatColumnDescriptor(allow_inf=True),
                    "null_and_nan": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True
                    ),
                    "null_and_inf": SparkFloatColumnDescriptor(
                        allow_null=True, allow_inf=True
                    ),
                    "nan_and_inf": SparkFloatColumnDescriptor(
                        allow_nan=True, allow_inf=True
                    ),
                    "null_and_nan_and_inf": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                }
            ),
            NamedTable("private_2"): SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "C": SparkIntegerColumnDescriptor(),
                }
            ),
        }
    )

    input_metric = DictMetric(
        {
            NamedTable("private"): SymmetricDifference(),
            NamedTable("private_2"): SymmetricDifference(),
        }
    )

    public_sources = {
        "public": spark.createDataFrame(
            pd.DataFrame({"A": ["zero", "one"], "B": [0, 1]}),
            schema=StructType(
                [
                    StructField("A", StringType(), False),
                    StructField("B", LongType(), False),
                ]
            ),
        )
    }
    request.cls.base_query = PrivateSource(source_id="private")

    catalog = Catalog()
    catalog.add_private_table(
        "private",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "X": ColumnDescriptor(ColumnType.DECIMAL),
            "null": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
            "nan": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True),
            "inf": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
            "null_and_nan": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True
            ),
            "null_and_inf": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_inf=True
            ),
            "nan_and_inf": ColumnDescriptor(
                ColumnType.DECIMAL, allow_nan=True, allow_inf=True
            ),
            "null_and_nan_and_inf": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
        },
    )
    catalog.add_private_table(
        "private_2",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "C": ColumnDescriptor(ColumnType.INTEGER),
        },
    )
    catalog.add_public_table(
        "public",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    )
    request.cls.catalog = catalog

    budget = PureDPBudget(10)
    stability = {
        NamedTable("private"): ExactNumber(3).expr,
        NamedTable("private_2"): ExactNumber(3).expr,
    }
    request.cls.visitor = MeasurementVisitor(
        privacy_budget=budget,
        stability=stability,
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=PureDP(),
        default_mechanism=NoiseMechanism.LAPLACE,
        public_sources=public_sources,
        catalog=catalog,
        table_constraints={t: [] for t in stability},
    )
    # for the methods which alter the output measure of a visitor.
    request.cls.pick_noise_visitor = MeasurementVisitor(
        privacy_budget=budget,
        stability=stability,
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=PureDP(),
        default_mechanism=NoiseMechanism.LAPLACE,
        public_sources=public_sources,
        catalog=catalog,
        table_constraints={t: [] for t in stability},
    )


@pytest.mark.usefixtures("test_data")
class TestMeasurementVisitor:
    """Tests for Measurement Visitor."""

    visitor: MeasurementVisitor
    pick_noise_visitor: MeasurementVisitor
    catalog: Catalog
    base_query: QueryExpr

    def run_with_empty_data_and_check_schema(
        self, query: QueryExpr, output_measure: Union[PureDP, RhoZCDP]
    ):
        """Run a query and check the schema of the result."""
        expected_column_types = query.accept(
            OutputSchemaVisitor(self.catalog)
        ).column_types
        self.visitor.output_measure = output_measure
        measurement, _ = query.accept(self.visitor)
        empty_data = create_empty_input(measurement.input_domain)
        result = measurement(empty_data)
        actual_column_types = Schema(
            spark_schema_to_analytics_columns(result.schema)
        ).column_types
        assert actual_column_types == expected_column_types

    def check_noise_info(
        self,
        query: QueryExpr,
        output_measure: Union[PureDP, RhoZCDP],
        expected_noise_info: NoiseInfo,
    ):
        """Check the noise info for a query."""
        self.pick_noise_visitor.output_measure = output_measure
        _, noise_info = query.accept(self.pick_noise_visitor)
        assert noise_info == expected_noise_info

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,expected_mechanism",
        [
            (CountMechanism.DEFAULT, PureDP(), NoiseMechanism.GEOMETRIC),
            (CountMechanism.DEFAULT, RhoZCDP(), NoiseMechanism.DISCRETE_GAUSSIAN),
            (CountMechanism.LAPLACE, PureDP(), NoiseMechanism.GEOMETRIC),
            (CountMechanism.LAPLACE, RhoZCDP(), NoiseMechanism.GEOMETRIC),
            (CountMechanism.GAUSSIAN, PureDP(), NoiseMechanism.DISCRETE_GAUSSIAN),
            (CountMechanism.GAUSSIAN, RhoZCDP(), NoiseMechanism.DISCRETE_GAUSSIAN),
        ],
    )
    def test_pick_noise_for_count(
        self,
        query_mechanism: CountMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test _pick_noise_for_count for GroupByCount query expressions."""
        query = GroupByCount(
            child=self.base_query,
            groupby_keys=KeySet.from_dict({}),
            mechanism=query_mechanism,
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        got_mechanism = self.pick_noise_visitor._pick_noise_for_count(query)
        # pylint: enable=protected-access
        assert got_mechanism == expected_mechanism

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,expected_mechanism",
        [
            (CountDistinctMechanism.DEFAULT, PureDP(), NoiseMechanism.GEOMETRIC),
            (
                CountDistinctMechanism.DEFAULT,
                RhoZCDP(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (CountDistinctMechanism.LAPLACE, PureDP(), NoiseMechanism.GEOMETRIC),
            (CountDistinctMechanism.LAPLACE, RhoZCDP(), NoiseMechanism.GEOMETRIC),
            (
                CountDistinctMechanism.GAUSSIAN,
                PureDP(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                CountDistinctMechanism.GAUSSIAN,
                RhoZCDP(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
        ],
    )
    def test_pick_noise_for_count_distinct(
        self,
        query_mechanism: CountDistinctMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test _pick_noise_for_count for GroupByCountDistinct query expressions."""
        query = GroupByCountDistinct(
            child=self.base_query,
            groupby_keys=KeySet.from_dict({}),
            mechanism=query_mechanism,
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        got_mechanism = self.pick_noise_visitor._pick_noise_for_count(query)
        # pylint: enable=protected-access
        assert got_mechanism == expected_mechanism

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (
                AverageMechanism.DEFAULT,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                AverageMechanism.DEFAULT,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                AverageMechanism.DEFAULT,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                AverageMechanism.DEFAULT,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
            (
                AverageMechanism.LAPLACE,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                AverageMechanism.LAPLACE,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                AverageMechanism.LAPLACE,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                AverageMechanism.LAPLACE,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (AverageMechanism.GAUSSIAN, PureDP(), SparkIntegerColumnDescriptor(), None),
            (AverageMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                AverageMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                AverageMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
        ],
    )
    def test_pick_noise_for_average(
        self,
        query_mechanism: AverageMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test _pick_noise_for_non_count for GroupByBoundedAverage query exprs."""
        if isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        elif isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        else:
            raise AssertionError("Unknown measure column type")
        query = GroupByBoundedAverage(
            child=self.base_query,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(query)
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(
                ValueError,
                match=(
                    "Gaussian noise is not supported under PureDP. "
                    "Please use RhoZCDP or another measure."
                ),
            ):
                self.pick_noise_visitor._pick_noise_for_non_count(query)
        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (
                SumMechanism.DEFAULT,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                SumMechanism.DEFAULT,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                SumMechanism.DEFAULT,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                SumMechanism.DEFAULT,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
            (
                SumMechanism.LAPLACE,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                SumMechanism.LAPLACE,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                SumMechanism.LAPLACE,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                SumMechanism.LAPLACE,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (SumMechanism.GAUSSIAN, PureDP(), SparkIntegerColumnDescriptor(), None),
            (SumMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                SumMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                SumMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
        ],
    )
    def test_pick_noise_for_sum(
        self,
        query_mechanism: SumMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test _pick_noise_for_non_count for GroupByBoundedSum query exprs."""
        if isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        elif isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        else:
            raise AssertionError("Unknown measure column type")
        query = GroupByBoundedSum(
            child=self.base_query,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(query)
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(
                ValueError,
                match=(
                    "Gaussian noise is not supported under PureDP. "
                    "Please use RhoZCDP or another measure."
                ),
            ):
                self.pick_noise_visitor._pick_noise_for_non_count(query)
        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (
                VarianceMechanism.DEFAULT,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                VarianceMechanism.DEFAULT,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                VarianceMechanism.DEFAULT,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                VarianceMechanism.DEFAULT,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
            (
                VarianceMechanism.LAPLACE,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                VarianceMechanism.LAPLACE,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                VarianceMechanism.LAPLACE,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                VarianceMechanism.LAPLACE,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                VarianceMechanism.GAUSSIAN,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                None,
            ),
            (VarianceMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                VarianceMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                VarianceMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
        ],
    )
    def test_pick_noise_for_variance(
        self,
        query_mechanism: VarianceMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test _pick_noise_for_non_count for GroupByBoundedVariance query exprs."""
        if isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        elif isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        else:
            raise AssertionError("Unknown measure column type")
        query = GroupByBoundedVariance(
            child=self.base_query,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(query)
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(
                ValueError,
                match=(
                    "Gaussian noise is not supported under PureDP. "
                    "Please use RhoZCDP or another measure."
                ),
            ):
                self.pick_noise_visitor._pick_noise_for_non_count(query)
        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "query_mechanism,output_measure,measure_column_type,expected_mechanism",
        [
            (
                StdevMechanism.DEFAULT,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                StdevMechanism.DEFAULT,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                StdevMechanism.DEFAULT,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                StdevMechanism.DEFAULT,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
            (
                StdevMechanism.LAPLACE,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                StdevMechanism.LAPLACE,
                PureDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (
                StdevMechanism.LAPLACE,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                StdevMechanism.LAPLACE,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.LAPLACE,
            ),
            (StdevMechanism.GAUSSIAN, PureDP(), SparkIntegerColumnDescriptor(), None),
            (StdevMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                StdevMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                StdevMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkFloatColumnDescriptor(),
                NoiseMechanism.GAUSSIAN,
            ),
        ],
    )
    def test_pick_noise_for_stdev(
        self,
        query_mechanism: StdevMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column_type: SparkColumnDescriptor,
        # if expected_mechanism is None, this combination is not supported
        expected_mechanism: Optional[NoiseMechanism],
    ) -> None:
        """Test _pick_noise_for_non_count for GroupByBoundedSTDEV query exprs."""
        if isinstance(measure_column_type, SparkFloatColumnDescriptor):
            measure_column = "X"
        elif isinstance(measure_column_type, SparkIntegerColumnDescriptor):
            measure_column = "B"
        else:
            raise AssertionError("Unknown measure column type")
        query = GroupByBoundedSTDEV(
            child=self.base_query,
            measure_column=measure_column,
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(query)
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(
                ValueError,
                match=(
                    "Gaussian noise is not supported under PureDP. "
                    "Please use RhoZCDP or another measure."
                ),
            ):
                self.pick_noise_visitor._pick_noise_for_non_count(query)
        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "mechanism",
        [
            (AverageMechanism.LAPLACE),
            (StdevMechanism.LAPLACE),
            (SumMechanism.LAPLACE),
            (VarianceMechanism.LAPLACE),
        ],
    )
    def test_pick_noise_invalid_column(
        self,
        mechanism: Union[
            AverageMechanism, StdevMechanism, SumMechanism, VarianceMechanism
        ],
    ) -> None:
        """Test _pick_noise_for_non_count with a non-numeric column.

        This only tests Laplace noise.
        """
        query: Union[
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ]
        if isinstance(mechanism, AverageMechanism):
            query = GroupByBoundedAverage(
                child=self.base_query,
                measure_column="A",
                low=0,
                high=1,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, StdevMechanism):
            query = GroupByBoundedSTDEV(
                child=self.base_query,
                measure_column="A",
                low=0,
                high=1,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, SumMechanism):
            query = GroupByBoundedSum(
                child=self.base_query,
                measure_column="A",
                low=0,
                high=1,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, VarianceMechanism):
            query = GroupByBoundedVariance(
                child=self.base_query,
                measure_column="A",
                low=0,
                high=1,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        else:
            assert_never(mechanism)
        with pytest.raises(
            AssertionError, match="Query's measure column should be numeric."
        ):
            # pylint: disable=protected-access
            self.visitor._pick_noise_for_non_count(query)
            # pylint: enable=protected-access

    def test_validate_measurement(self):
        """Test _validate_measurement."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement:
            mock_measurement.privacy_function.return_value = self.visitor.budget.value
            mid_stability = ExactNumber(2).expr
            # This should finish without raising an error
            # pylint: disable=protected-access
            self.visitor._validate_measurement(mock_measurement, mid_stability)

            # Change it so that the privacy function returns something else
            mock_measurement.privacy_function.return_value = ExactNumber(3).expr
            with pytest.raises(
                AssertionError,
                match="Privacy function does not match per-query privacy budget.",
            ):
                self.visitor._validate_measurement(mock_measurement, mid_stability)
            # pylint: enable=protected-access

    def _check_measurement(self, measurement: Measurement):
        """Check the basic attributes of a measurement (for all query exprs).

        The measurement almost certainly looks like this:
        ``child_transformation | mock_measurement``
        so extensive testing of the latter is likely to be counterproductive.
        """
        assert isinstance(measurement, ChainTM)

        assert measurement.transformation.input_domain == self.visitor.input_domain
        assert measurement.transformation.input_metric == self.visitor.input_metric
        assert isinstance(
            measurement.transformation.output_domain, SparkDataFrameDomain
        )
        assert (
            measurement.transformation.output_domain
            == measurement.measurement.input_domain
        )
        assert isinstance(
            measurement.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )
        assert (
            measurement.transformation.output_metric
            == measurement.measurement.input_metric
        )

    @pytest.mark.parametrize(
        "query,output_measure,noise_info",
        [
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.DEFAULT,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.3,
                        }
                    ]
                ),
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    mechanism=CountMechanism.LAPLACE,
                    output_column="count",
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.3,
                        }
                    ]
                ),
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=CountMechanism.GAUSSIAN,
                    output_column="custom_count_column",
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 0.449999999999999,
                        }
                    ]
                ),
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.DEFAULT,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 0.449999999999999,
                        }
                    ]
                ),
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.LAPLACE,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.6708203932499359,
                        }
                    ]
                ),
            ),
        ],
    )
    def test_visit_groupby_count(
        self,
        query: GroupByCount,
        output_measure: Union[PureDP, RhoZCDP],
        noise_info: NoiseInfo,
    ) -> None:
        """Test visit_groupby_count."""
        self.run_with_empty_data_and_check_schema(query, output_measure)
        self.check_noise_info(query, output_measure, noise_info)

    @pytest.mark.parametrize(
        "query,output_measure,noise_info",
        [
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountDistinctMechanism.DEFAULT,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.3,
                        }
                    ]
                ),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    mechanism=CountDistinctMechanism.LAPLACE,
                    output_column="count",
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.3,
                        }
                    ]
                ),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    columns_to_count=["A"],
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.3,
                        }
                    ]
                ),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=CountDistinctMechanism.GAUSSIAN,
                    output_column="custom_count_column",
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 0.449999999999999,
                        }
                    ]
                ),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountDistinctMechanism.DEFAULT,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 0.449999999999999,
                        }
                    ]
                ),
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountDistinctMechanism.LAPLACE,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.6708203932499359,
                        }
                    ]
                ),
            ),
        ],
    )
    def test_visit_groupby_count_distinct(
        self,
        query: GroupByCountDistinct,
        output_measure: Union[PureDP, RhoZCDP],
        noise_info: NoiseInfo,
    ) -> None:
        """Test visit_groupby_count_distinct."""
        self.run_with_empty_data_and_check_schema(query, output_measure)
        self.check_noise_info(query, output_measure, noise_info)

    @pytest.mark.parametrize(
        "query,output_measure,noise_info",
        [
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    low=-100,
                    high=100,
                    output_column="custom_output_column",
                    measure_column="B",
                    quantile=0.1,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                            "noise_parameter": 3.3333333333333326,
                        }
                    ]
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    low=-100,
                    high=100,
                    output_column="custom_output_column",
                    measure_column="null_and_nan",
                    quantile=0.1,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                            "noise_parameter": 3.3333333333333326,
                        }
                    ]
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    output_column="quantile",
                    low=123.345,
                    high=987.65,
                    quantile=0.25,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                            "noise_parameter": 3.3333333333333326,
                        }
                    ]
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="null_and_inf",
                    output_column="quantile",
                    low=123.345,
                    high=987.65,
                    quantile=0.25,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                            "noise_parameter": 3.3333333333333326,
                        }
                    ]
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    quantile=0.5,
                    measure_column="X",
                    low=0,
                    high=1,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                            "noise_parameter": 2.9814239699997196,
                        }
                    ]
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    quantile=0.5,
                    measure_column="nan_and_inf",
                    low=0,
                    high=1,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                            "noise_parameter": 2.9814239699997196,
                        }
                    ]
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    quantile=0.5,
                    measure_column="inf",
                    low=0,
                    high=1,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.EXPONENTIAL,
                            "noise_parameter": 2.9814239699997196,
                        }
                    ]
                ),
            ),
        ],
    )
    def test_visit_groupby_quantile(
        self,
        query: GroupByQuantile,
        output_measure: Union[PureDP, RhoZCDP],
        noise_info: NoiseInfo,
    ) -> None:
        """Test visit_groupby_quantile."""
        self.run_with_empty_data_and_check_schema(query, output_measure)
        self.check_noise_info(query, output_measure, noise_info)

    @pytest.mark.parametrize(
        "query,output_measure,noise_info",
        [
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    low=-100,
                    high=100,
                    mechanism=AverageMechanism.DEFAULT,
                    output_column="custom_output_column",
                    measure_column="B",
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 60,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.6,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=AverageMechanism.DEFAULT,
                    output_column="sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 259.2915,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 0.6,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=AverageMechanism.LAPLACE,
                    output_column="different_sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 259.2915,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 0.6,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=AverageMechanism.DEFAULT,
                    measure_column="B",
                    low=0,
                    high=1,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 0.899999999999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 0.899999999999999,
                        },
                    ]
                ),
            ),
        ],
    )
    def test_visit_groupby_bounded_average(
        self,
        query: GroupByBoundedAverage,
        output_measure: Union[PureDP, RhoZCDP],
        noise_info: NoiseInfo,
    ) -> None:
        """Test visit_groupby_bounded_average."""
        self.run_with_empty_data_and_check_schema(query, output_measure)
        self.check_noise_info(query, output_measure, noise_info)

    @pytest.mark.parametrize(
        "query,output_measure,noise_info",
        [
            (
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    low=-100,
                    high=100,
                    mechanism=SumMechanism.DEFAULT,
                    output_column="custom_output_column",
                    measure_column="B",
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 30,
                        }
                    ]
                ),
            ),
            (
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=SumMechanism.DEFAULT,
                    output_column="sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 296.2949999999999,
                        }
                    ]
                ),
            ),
            (
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=SumMechanism.LAPLACE,
                    output_column="different_sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 296.2949999999999,
                        }
                    ]
                ),
            ),
            (
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=SumMechanism.DEFAULT,
                    measure_column="B",
                    low=0,
                    high=1,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 0.449999999999999,
                        }
                    ]
                ),
            ),
        ],
    )
    def test_visit_groupby_bounded_sum(
        self,
        query: GroupByBoundedSum,
        output_measure: Union[PureDP, RhoZCDP],
        noise_info: NoiseInfo,
    ) -> None:
        """Test visit_groupby_bounded_sum."""
        self.run_with_empty_data_and_check_schema(query, output_measure)
        self.check_noise_info(query, output_measure, noise_info)

    @pytest.mark.parametrize(
        "query,output_measure,noise_info",
        [
            (
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    low=-100,
                    high=100,
                    mechanism=VarianceMechanism.DEFAULT,
                    output_column="custom_output_column",
                    measure_column="B",
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 90,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 4500,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.899999999999999,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=VarianceMechanism.DEFAULT,
                    output_column="sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 388.9372499999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 432107.34006374987,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 0.899999999999999,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=VarianceMechanism.LAPLACE,
                    output_column="different_sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 388.9372499999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 432107.34006374987,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 0.899999999999999,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=VarianceMechanism.DEFAULT,
                    measure_column="B",
                    low=0,
                    high=1,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 1.349999999999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 1.349999999999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 1.349999999999999,
                        },
                    ]
                ),
            ),
        ],
    )
    def test_visit_groupby_bounded_variance(
        self,
        query: GroupByBoundedVariance,
        output_measure: Union[PureDP, RhoZCDP],
        noise_info: NoiseInfo,
    ) -> None:
        """Test visit_groupby_bounded_variance."""
        self.run_with_empty_data_and_check_schema(query, output_measure)
        self.check_noise_info(query, output_measure, noise_info)

    @pytest.mark.parametrize(
        "query,output_measure,noise_info",
        [
            (
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    low=-100,
                    high=100,
                    mechanism=StdevMechanism.DEFAULT,
                    output_column="custom_output_column",
                    measure_column="B",
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 90,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 4500,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.GEOMETRIC,
                            "noise_parameter": 0.899999999999999,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=StdevMechanism.DEFAULT,
                    output_column="sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 388.9372499999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 432107.34006374987,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 0.899999999999999,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    measure_column="X",
                    mechanism=StdevMechanism.LAPLACE,
                    output_column="different_sum",
                    low=123.345,
                    high=987.65,
                ),
                PureDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 388.9372499999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 432107.34006374987,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.LAPLACE,
                            "noise_parameter": 0.899999999999999,
                        },
                    ]
                ),
            ),
            (
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=StdevMechanism.DEFAULT,
                    measure_column="B",
                    low=0,
                    high=1,
                ),
                RhoZCDP(),
                NoiseInfo(
                    [
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 1.349999999999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 1.349999999999999,
                        },
                        {
                            "noise_mechanism": _NoiseMechanism.DISCRETE_GAUSSIAN,
                            "noise_parameter": 1.349999999999999,
                        },
                    ]
                ),
            ),
        ],
    )
    def test_visit_groupby_bounded_stdev(
        self,
        query: GroupByBoundedSTDEV,
        output_measure: Union[PureDP, RhoZCDP],
        noise_info: NoiseInfo,
    ) -> None:
        """Test visit_groupby_bounded_stdev."""
        self.run_with_empty_data_and_check_schema(query, output_measure)
        self.check_noise_info(query, output_measure, noise_info)

    @pytest.mark.parametrize(
        "query",
        [
            (PrivateSource("private")),
            (Rename(child=PrivateSource("private"), column_mapper={"A": "A2"})),
            (Filter(child=PrivateSource("private"), condition="B > 2")),
            (SelectExpr(child=PrivateSource("private"), columns=["A"])),
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {"C": "c" + str(row["B"])},
                    schema_new_columns=Schema({"C": "VARCHAR"}),
                    augment=True,
                )
            ),
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda row: [{"i": n for n in range(row["B"] + 1)}],
                    schema_new_columns=Schema({"i": "DECIMAL"}),
                    augment=False,
                    max_rows=11,
                )
            ),
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropExcess(3),
                )
            ),
            (JoinPublic(child=PrivateSource("private"), public_table="public")),
            (ReplaceNullAndNan(child=PrivateSource("private"))),
            (ReplaceInfExpr(child=PrivateSource("private"))),
            (DropNullAndNan(child=PrivateSource("private"))),
            (DropInfExpr(child=PrivateSource("private"))),
        ],
    )
    def test_visit_transformations(self, query: QueryExpr):
        """Test that visiting transformations returns an error."""
        with pytest.raises(NotImplementedError):
            query.accept(self.visitor)

    @pytest.mark.parametrize(
        "query",
        [
            SuppressAggregates(
                child=GroupByCount(
                    groupby_keys=KeySet.from_dict({}),
                    child=PrivateSource("private"),
                    output_column="count",
                ),
                column="count",
                threshold=5.5,
            ),
            SuppressAggregates(
                child=GroupByCount(
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    child=PrivateSource("private"),
                    output_column="count",
                ),
                column="count",
                threshold=-10,
            ),
        ],
    )
    def test_visit_suppress_aggregates(self, query: SuppressAggregates) -> None:
        """Test visit_suppress_aggregates."""
        output_measures: List[Union[PureDP, RhoZCDP]] = [PureDP(), RhoZCDP()]
        for output_measure in output_measures:
            self.run_with_empty_data_and_check_schema(query, output_measure)

    @pytest.mark.parametrize(
        "query,output_measure,budget,expected_result",
        [
            (query, output_measure, budget, expected_result)
            for (query, expected_result) in [
                (
                    SuppressAggregates(
                        child=GroupByCount(
                            child=PrivateSource("private_2"),
                            groupby_keys=KeySet.from_dict(
                                {"A": ["a0", "a1", "a2", "a3"]}
                            ),
                            output_column="count",
                        ),
                        column="count",
                        threshold=0,
                    ),
                    pd.DataFrame(
                        [
                            ["a0", 0],
                            ["a1", 1],
                            ["a2", 2],
                            ["a3", 3],
                        ],
                        columns=["A", "count"],
                    ),
                ),
                (
                    SuppressAggregates(
                        child=GroupByCount(
                            child=PrivateSource("private_2"),
                            groupby_keys=KeySet.from_dict(
                                {"A": ["a0", "a1", "a2", "a3"]},
                            ),
                            output_column="custom_count_name",
                        ),
                        column="custom_count_name",
                        threshold=3,
                    ),
                    pd.DataFrame(
                        [
                            ["a3", 3],
                        ],
                        columns=["A", "custom_count_name"],
                    ),
                ),
            ]
            for output_measure, budget in [
                (PureDP(), PureDPBudget(float("inf"))),
                (RhoZCDP(), RhoZCDPBudget(float("inf"))),
            ]
        ],
    )
    def test_suppress_aggregates_correctness(
        self,
        spark: SparkSession,
        query: SuppressAggregates,
        output_measure: Union[PureDP, RhoZCDP],
        budget: Union[PureDPBudget, RhoZCDPBudget],
        expected_result: pd.DataFrame,
    ) -> None:
        """Test that measurement from visit_suppress_aggregates gets correct results."""
        input_data = create_empty_input(self.visitor.input_domain)
        input_data[NamedTable("private_2")] = spark.createDataFrame(
            pd.DataFrame(
                [
                    ["a1", 1],
                    ["a2", 1],
                    ["a2", 2],
                    ["a3", 1],
                    ["a3", 2],
                    ["a3", 3],
                ],
                columns=["A", "C"],
            ),
        )
        self.visitor.output_measure = output_measure
        self.visitor.budget = budget
        self.visitor.adjusted_budget = budget
        measurement, _ = query.accept(self.visitor)
        got = measurement(input_data)
        assert_frame_equal_with_sort(got.toPandas(), expected_result)
