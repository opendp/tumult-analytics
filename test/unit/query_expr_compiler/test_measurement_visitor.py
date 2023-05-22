"""Tests for MeasurementVisitor."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=no-self-use

from typing import List, Optional, Union
from unittest.mock import patch

import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, StringType, StructField, StructType
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
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
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.type_utils import assert_never

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._measurement_visitor import (
    MeasurementVisitor,
    _get_query_bounds,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics._table_reference import lookup_domain, lookup_metric
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget
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
from tmlt.analytics.query_expr import StdevMechanism, SumMechanism, VarianceMechanism
from tmlt.analytics.truncation_strategy import TruncationStrategy

from ...conftest import assert_frame_equal_with_sort


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
            (AverageMechanism.DEFAULT, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
            (
                AverageMechanism.GAUSSIAN,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (AverageMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                AverageMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (AverageMechanism.GAUSSIAN, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
        query = GroupByBoundedAverage(
            child=self.base_query,
            measure_column="",
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(NotImplementedError):
                self.pick_noise_visitor._pick_noise_for_non_count(
                    query, measure_column_type
                )
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
            (SumMechanism.DEFAULT, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
            (
                SumMechanism.GAUSSIAN,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (SumMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                SumMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (SumMechanism.GAUSSIAN, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
        query = GroupByBoundedSum(
            child=self.base_query,
            measure_column="",
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(NotImplementedError):
                self.pick_noise_visitor._pick_noise_for_non_count(
                    query, measure_column_type
                )
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
            (VarianceMechanism.DEFAULT, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (VarianceMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                VarianceMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (VarianceMechanism.GAUSSIAN, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
        query = GroupByBoundedVariance(
            child=self.base_query,
            measure_column="",
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(NotImplementedError):
                self.pick_noise_visitor._pick_noise_for_non_count(
                    query, measure_column_type
                )
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
            (StdevMechanism.DEFAULT, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
            (
                StdevMechanism.GAUSSIAN,
                PureDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (StdevMechanism.GAUSSIAN, PureDP(), SparkFloatColumnDescriptor(), None),
            (
                StdevMechanism.GAUSSIAN,
                RhoZCDP(),
                SparkIntegerColumnDescriptor(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (StdevMechanism.GAUSSIAN, RhoZCDP(), SparkFloatColumnDescriptor(), None),
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
        query = GroupByBoundedSTDEV(
            child=self.base_query,
            measure_column="",
            low=0,
            high=1,
            mechanism=query_mechanism,
            groupby_keys=KeySet.from_dict({}),
        )
        self.pick_noise_visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.pick_noise_visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            assert got_mechanism == expected_mechanism
        else:
            with pytest.raises(NotImplementedError):
                self.pick_noise_visitor._pick_noise_for_non_count(
                    query, measure_column_type
                )
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
                measure_column="",
                low=0,
                high=1,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, StdevMechanism):
            query = GroupByBoundedSTDEV(
                child=self.base_query,
                measure_column="",
                low=0,
                high=1,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, SumMechanism):
            query = GroupByBoundedSum(
                child=self.base_query,
                measure_column="",
                low=0,
                high=1,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, VarianceMechanism):
            query = GroupByBoundedVariance(
                child=self.base_query,
                measure_column="",
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
            self.visitor._pick_noise_for_non_count(query, SparkStringColumnDescriptor())
            # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "input_metric,mechanism,expected_output_metric",
        [
            (HammingDistance(), NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
            (HammingDistance(), NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
            (
                HammingDistance(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
                RootSumOfSquared(SymmetricDifference()),
            ),
            (
                SymmetricDifference(),
                NoiseMechanism.LAPLACE,
                SumOf(SymmetricDifference()),
            ),
            (
                SymmetricDifference(),
                NoiseMechanism.GEOMETRIC,
                SumOf(SymmetricDifference()),
            ),
            (
                SymmetricDifference(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
                RootSumOfSquared(SymmetricDifference()),
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(SymmetricDifference())),
                NoiseMechanism.LAPLACE,
                SumOf(SymmetricDifference()),
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(SymmetricDifference())),
                NoiseMechanism.GEOMETRIC,
                SumOf(SymmetricDifference()),
            ),
            (
                IfGroupedBy(
                    column="A", inner_metric=RootSumOfSquared(SymmetricDifference())
                ),
                NoiseMechanism.DISCRETE_GAUSSIAN,
                RootSumOfSquared(SymmetricDifference()),
            ),
        ],
    )
    def test_build_groupby(
        self,
        input_metric: Union[HammingDistance, SymmetricDifference, IfGroupedBy],
        mechanism: NoiseMechanism,
        expected_output_metric: Union[RootSumOfSquared, SumOf],
    ) -> None:
        """Test _build_groupby (without a _public_id)."""
        input_domain = SparkDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
            }
        )
        keyset = KeySet.from_dict({"A": ["zero", "one"], "B": [0, 1]})
        # pylint: disable=protected-access
        got = self.visitor._build_groupby(
            input_domain=input_domain,
            input_metric=input_metric,
            groupby_keys=keyset,
            mechanism=mechanism,
        )
        # pylint: enable=protected-access
        assert got.input_domain == input_domain
        assert got.input_metric == input_metric
        assert_frame_equal_with_sort(
            got.group_keys.toPandas(), keyset.dataframe().toPandas()
        )
        expected_output_domain = SparkGroupedDataFrameDomain(
            schema=input_domain.schema, group_keys=keyset.dataframe()
        )
        assert isinstance(got.output_domain, SparkGroupedDataFrameDomain)
        assert got.output_domain.schema == expected_output_domain.schema
        assert_frame_equal_with_sort(
            got.output_domain.group_keys.toPandas(),
            expected_output_domain.group_keys.toPandas(),
        )
        assert got.output_metric == expected_output_metric

    @pytest.mark.parametrize(
        "query,expected_mid_stability,expected_mechanism",
        [
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="B",
                    low=-100,
                    high=100,
                    mechanism=AverageMechanism.LAPLACE,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.GEOMETRIC,
            ),
            (
                GroupByBoundedAverage(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="nan",
                    low=-100,
                    high=100,
                    mechanism=AverageMechanism.LAPLACE,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.LAPLACE,
            ),
            (
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="X",
                    low=-100,
                    high=100,
                    mechanism=SumMechanism.LAPLACE,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.LAPLACE,
            ),
            (
                GroupByBoundedSum(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="null",
                    low=-100,
                    high=100,
                    mechanism=SumMechanism.LAPLACE,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.LAPLACE,
            ),
            (
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="B",
                    low=-100,
                    high=100,
                    mechanism=VarianceMechanism.GAUSSIAN,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                GroupByBoundedVariance(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="inf",
                    low=-100,
                    high=100,
                    mechanism=VarianceMechanism.LAPLACE,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.LAPLACE,
            ),
            (
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="B",
                    low=-100,
                    high=100,
                    mechanism=StdevMechanism.DEFAULT,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.GEOMETRIC,
            ),
            (
                GroupByBoundedSTDEV(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero", "one"]}),
                    measure_column="null_and_nan_and_inf",
                    low=-100,
                    high=100,
                    mechanism=StdevMechanism.DEFAULT,
                ),
                ExactNumber(3).expr,
                NoiseMechanism.LAPLACE,
            ),
        ],
    )
    def test_build_common(
        self,
        query: Union[
            GroupByBoundedAverage,
            GroupByBoundedSTDEV,
            GroupByBoundedSum,
            GroupByBoundedVariance,
        ],
        expected_mid_stability: sp.Expr,
        expected_mechanism: NoiseMechanism,
    ):
        """Test _build_common."""
        info = self.visitor._build_common(query)  # pylint: disable=protected-access
        transformation: ChainTT
        assert isinstance(info.transformation, ChainTT)
        transformation = info.transformation

        assert transformation.input_domain == self.visitor.input_domain
        assert transformation.input_metric == self.visitor.input_metric

        assert info.mechanism == expected_mechanism
        assert info.mid_stability == expected_mid_stability
        assert info.lower_bound == ExactNumber.from_float(query.low, round_up=True)
        assert info.upper_bound == ExactNumber.from_float(query.high, round_up=False)

        assert isinstance(info.groupby, GroupBy)
        table_domain = transformation.output_domain
        assert isinstance(table_domain, SparkDataFrameDomain)
        expected_groupby_domain = SparkGroupedDataFrameDomain(
            schema=dict(table_domain.schema), group_keys=query.groupby_keys.dataframe()
        )
        assert isinstance(info.groupby.output_domain, SparkGroupedDataFrameDomain)
        assert info.groupby.output_domain.schema == expected_groupby_domain.schema
        assert_frame_equal_with_sort(
            info.groupby.output_domain.group_keys.toPandas(),
            expected_groupby_domain.group_keys.toPandas(),
        )

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
            mock_measurement.privacy_function.return_value = ExactNumber(-10).expr
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

    def check_mock_groupby_call(
        self,
        mock_groupby,
        transformation: Transformation,
        keys: KeySet,
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Check that the mock groupby was called with the right arguments."""
        groupby_df: DataFrame = keys.dataframe()
        mock_groupby.assert_called_with(
            input_domain=transformation.output_domain,
            input_metric=transformation.output_metric,
            use_l2=(expected_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN),
            group_keys=groupby_df,
        )

    def _setup_mock_measurement(
        self,
        mock_measurement,
        child_query: QueryExpr,
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Initialize a mock measurement."""
        # pylint: disable=protected-access
        transformation, reference, _ = self.visitor._visit_child_transformation(
            child_query, expected_mechanism
        )
        # pylint: enable=protected-access
        mock_measurement.input_domain = lookup_domain(
            transformation.output_domain, reference
        )
        mock_measurement.input_metric = lookup_metric(
            transformation.output_metric, reference
        )
        mock_measurement.privacy_function.return_value = self.visitor.budget.value

    @pytest.mark.parametrize(
        "query,output_measure,expected_mechanism",
        [
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.DEFAULT,
                ),
                PureDP(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    mechanism=CountMechanism.LAPLACE,
                    output_column="count",
                ),
                PureDP(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=CountMechanism.GAUSSIAN,
                    output_column="custom_count_column",
                ),
                RhoZCDP(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.DEFAULT,
                ),
                RhoZCDP(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                GroupByCount(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountMechanism.LAPLACE,
                ),
                RhoZCDP(),
                NoiseMechanism.GEOMETRIC,
            ),
        ],
    )
    def test_visit_groupby_count(
        self,
        query: GroupByCount,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test visit_groupby_count."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
            autospec=True,
        ) as mock_groupby, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement, patch(
            "tmlt.analytics._query_expr_compiler."
            + "_measurement_visitor.create_count_measurement",
            autospec=True,
        ) as mock_create_count:
            self.visitor.output_measure = output_measure
            self._setup_mock_measurement(
                mock_measurement, query.child, expected_mechanism
            )
            mock_create_count.return_value = mock_measurement

            measurement = self.visitor.visit_groupby_count(query)
            assert isinstance(measurement, ChainTM)

            self._check_measurement(measurement)
            self.check_mock_groupby_call(
                mock_groupby,
                measurement.transformation,
                query.groupby_keys,
                expected_mechanism,
            )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )
            mock_create_count.assert_called_with(
                input_domain=measurement.transformation.output_domain,
                input_metric=measurement.transformation.output_metric,
                noise_mechanism=expected_mechanism,
                d_in=mid_stability,
                d_out=self.visitor.budget.value,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                count_column=query.output_column,
            )

    @pytest.mark.parametrize(
        "query,output_measure,expected_mechanism",
        [
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountDistinctMechanism.DEFAULT,
                ),
                PureDP(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"B": [0, 1]}),
                    mechanism=CountDistinctMechanism.LAPLACE,
                    output_column="count",
                ),
                PureDP(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    columns_to_count=["A"],
                ),
                PureDP(),
                NoiseMechanism.GEOMETRIC,
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    mechanism=CountDistinctMechanism.GAUSSIAN,
                    output_column="custom_count_column",
                ),
                RhoZCDP(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountDistinctMechanism.DEFAULT,
                ),
                RhoZCDP(),
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
            (
                GroupByCountDistinct(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({}),
                    mechanism=CountDistinctMechanism.LAPLACE,
                ),
                RhoZCDP(),
                NoiseMechanism.GEOMETRIC,
            ),
        ],
    )
    def test_visit_groupby_count_distinct(
        self,
        query: GroupByCountDistinct,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test visit_groupby_count_distinct."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
            autospec=True,
        ) as mock_groupby, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement, patch(
            "tmlt.analytics._query_expr_compiler."
            + "_measurement_visitor.create_count_distinct_measurement",
            autospec=True,
        ) as mock_create_count_distinct:
            self.visitor.output_measure = output_measure

            mock_create_count_distinct.return_value = mock_measurement
            self._setup_mock_measurement(
                mock_measurement, query.child, expected_mechanism
            )
            if query.columns_to_count:
                select_columns = query.columns_to_count + list(
                    query.groupby_keys.schema().keys()
                )
                mock_measurement.input_domain = SparkDataFrameDomain(
                    {
                        c: d
                        for c, d in mock_measurement.input_domain.schema.items()
                        if c in select_columns
                    }
                )

            measurement = self.visitor.visit_groupby_count_distinct(query)

            self._check_measurement(measurement)
            assert isinstance(measurement, ChainTM)
            assert measurement.measurement == mock_measurement
            if query.columns_to_count:
                assert isinstance(measurement.transformation, ChainTT)
                assert isinstance(
                    measurement.transformation.transformation2, SelectTransformation
                )
                assert (
                    measurement.transformation.transformation2.columns == select_columns
                )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )

            assert measurement.measurement == mock_measurement
            self.check_mock_groupby_call(
                mock_groupby,
                measurement.transformation,
                query.groupby_keys,
                expected_mechanism,
            )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )

            mock_create_count_distinct.assert_called_with(
                input_domain=measurement.transformation.output_domain,
                input_metric=measurement.transformation.output_metric,
                noise_mechanism=expected_mechanism,
                d_in=mid_stability,
                d_out=self.visitor.budget.value,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                count_column=query.output_column,
            )

    @pytest.mark.parametrize(
        "query,output_measure,expected_new_child",
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
                None,
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
                DropNullAndNan(PrivateSource("private"), ["null_and_nan"]),
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
                None,
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
                ReplaceInfExpr(
                    DropNullAndNan(PrivateSource("private"), ["null_and_inf"]),
                    {"null_and_inf": (123.345, 987.65)},
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
                None,
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
                ReplaceInfExpr(
                    DropNullAndNan(PrivateSource("private"), ["nan_and_inf"]),
                    {"nan_and_inf": (0, 1)},
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
                ReplaceInfExpr(PrivateSource("private"), {"inf": (0, 1)}),
            ),
        ],
    )
    def test_visit_groupby_quantile(
        self,
        query: GroupByQuantile,
        output_measure: Union[PureDP, RhoZCDP],
        expected_new_child: Optional[QueryExpr],
    ) -> None:
        """Test visit_groupby_quantile."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
            autospec=True,
        ) as mock_groupby, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement, patch(
            "tmlt.analytics._query_expr_compiler."
            + "_measurement_visitor.create_quantile_measurement",
            autospec=True,
        ) as mock_create_quantile:
            self.visitor.output_measure = output_measure

            expected_mechanism = self.visitor.default_mechanism
            expected_child_query: QueryExpr
            if expected_new_child is not None:
                expected_child_query = expected_new_child
            else:
                expected_child_query = query.child
            self._setup_mock_measurement(
                mock_measurement, expected_child_query, expected_mechanism
            )
            mock_create_quantile.return_value = mock_measurement

            measurement = self.visitor.visit_groupby_quantile(query)

            self._check_measurement(measurement)
            assert isinstance(measurement, ChainTM)
            assert measurement.measurement == mock_measurement
            self.check_mock_groupby_call(
                mock_groupby,
                measurement.transformation,
                query.groupby_keys,
                expected_mechanism,
            )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )
            mock_create_quantile.assert_called_with(
                input_domain=measurement.transformation.output_domain,
                input_metric=measurement.transformation.output_metric,
                measure_column=query.measure_column,
                quantile=query.quantile,
                lower=query.low,
                upper=query.high,
                d_in=mid_stability,
                d_out=self.visitor.budget.value,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                quantile_column=query.output_column,
            )

    @pytest.mark.parametrize(
        "query,output_measure,expected_mechanism",
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
                NoiseMechanism.GEOMETRIC,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
        ],
    )
    def test_visit_groupby_bounded_sum(
        self,
        query: GroupByBoundedSum,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test visit_groupby_bounded_sum."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
            autospec=True,
        ) as mock_groupby, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement, patch(
            "tmlt.analytics._query_expr_compiler."
            + "_measurement_visitor.create_sum_measurement",
            autospec=True,
        ) as mock_create_sum:
            self.visitor.output_measure = output_measure

            self._setup_mock_measurement(
                mock_measurement, query.child, expected_mechanism
            )
            mock_create_sum.return_value = mock_measurement

            measurement = self.visitor.visit_groupby_bounded_sum(query)

            self._check_measurement(measurement)
            assert isinstance(measurement, ChainTM)
            assert measurement.measurement == mock_measurement
            self.check_mock_groupby_call(
                mock_groupby,
                measurement.transformation,
                query.groupby_keys,
                expected_mechanism,
            )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )
            lower, upper = _get_query_bounds(query)
            mock_create_sum.assert_called_with(
                input_domain=measurement.transformation.output_domain,
                input_metric=measurement.transformation.output_metric,
                measure_column=query.measure_column,
                lower=lower,
                upper=upper,
                noise_mechanism=expected_mechanism,
                d_in=mid_stability,
                d_out=self.visitor.budget.value,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                sum_column=query.output_column,
            )

    @pytest.mark.parametrize(
        "query,output_measure,expected_mechanism",
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
                NoiseMechanism.GEOMETRIC,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
        ],
    )
    def test_visit_groupby_bounded_average(
        self,
        query: GroupByBoundedAverage,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test visit_groupby_bounded_average."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
            autospec=True,
        ) as mock_groupby, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement, patch(
            "tmlt.analytics._query_expr_compiler."
            + "_measurement_visitor.create_average_measurement",
            autospec=True,
        ) as mock_create_average:
            self.visitor.output_measure = output_measure

            self._setup_mock_measurement(
                mock_measurement, query.child, expected_mechanism
            )
            mock_create_average.return_value = mock_measurement

            measurement = self.visitor.visit_groupby_bounded_average(query)

            self._check_measurement(measurement)
            assert isinstance(measurement, ChainTM)
            assert measurement.measurement == mock_measurement
            self.check_mock_groupby_call(
                mock_groupby,
                measurement.transformation,
                query.groupby_keys,
                expected_mechanism,
            )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )
            lower, upper = _get_query_bounds(query)
            mock_create_average.assert_called_with(
                input_domain=measurement.transformation.output_domain,
                input_metric=measurement.transformation.output_metric,
                measure_column=query.measure_column,
                lower=lower,
                upper=upper,
                noise_mechanism=expected_mechanism,
                d_in=mid_stability,
                d_out=self.visitor.budget.value,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                average_column=query.output_column,
            )

    @pytest.mark.parametrize(
        "query,output_measure,expected_mechanism",
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
                NoiseMechanism.GEOMETRIC,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
        ],
    )
    def test_visit_groupby_bounded_variance(
        self,
        query: GroupByBoundedVariance,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test visit_groupby_bounded_variance."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
            autospec=True,
        ) as mock_groupby, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement, patch(
            "tmlt.analytics._query_expr_compiler."
            + "_measurement_visitor.create_variance_measurement",
            autospec=True,
        ) as mock_create_variance:
            self.visitor.output_measure = output_measure

            self._setup_mock_measurement(
                mock_measurement, query.child, expected_mechanism
            )
            mock_create_variance.return_value = mock_measurement

            measurement = self.visitor.visit_groupby_bounded_variance(query)

            self._check_measurement(measurement)
            assert isinstance(measurement, ChainTM)
            assert measurement.measurement == mock_measurement
            self.check_mock_groupby_call(
                mock_groupby,
                measurement.transformation,
                query.groupby_keys,
                expected_mechanism,
            )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )
            lower, upper = _get_query_bounds(query)
            mock_create_variance.assert_called_with(
                input_domain=measurement.transformation.output_domain,
                input_metric=measurement.transformation.output_metric,
                measure_column=query.measure_column,
                lower=lower,
                upper=upper,
                noise_mechanism=expected_mechanism,
                d_in=mid_stability,
                d_out=self.visitor.budget.value,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                variance_column=query.output_column,
            )

    @pytest.mark.parametrize(
        "query,output_measure,expected_mechanism",
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
                NoiseMechanism.GEOMETRIC,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.LAPLACE,
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
                NoiseMechanism.DISCRETE_GAUSSIAN,
            ),
        ],
    )
    def test_visit_groupby_bounded_stdev(
        self,
        query: GroupByBoundedSTDEV,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test visit_groupby_bounded_stdev."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
            autospec=True,
        ) as mock_groupby, patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement, patch(
            "tmlt.analytics._query_expr_compiler."
            + "_measurement_visitor.create_standard_deviation_measurement",
            autospec=True,
        ) as mock_create_stdev:
            self.visitor.output_measure = output_measure
            self.visitor.output_measure = output_measure

            self._setup_mock_measurement(
                mock_measurement, query.child, expected_mechanism
            )
            mock_create_stdev.return_value = mock_measurement

            measurement = self.visitor.visit_groupby_bounded_stdev(query)

            self._check_measurement(measurement)
            assert isinstance(measurement, ChainTM)
            assert measurement.measurement == mock_measurement
            self.check_mock_groupby_call(
                mock_groupby,
                measurement.transformation,
                query.groupby_keys,
                expected_mechanism,
            )

            mid_stability = measurement.transformation.stability_function(
                self.visitor.stability
            )
            lower, upper = _get_query_bounds(query)
            mock_create_stdev.assert_called_with(
                input_domain=measurement.transformation.output_domain,
                input_metric=measurement.transformation.output_metric,
                measure_column=query.measure_column,
                lower=lower,
                upper=upper,
                noise_mechanism=expected_mechanism,
                d_in=mid_stability,
                d_out=self.visitor.budget.value,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                standard_deviation_column=query.output_column,
            )

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
                    max_num_rows=11,
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
