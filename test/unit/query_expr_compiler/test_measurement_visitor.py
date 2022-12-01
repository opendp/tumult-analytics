# type: ignore[attr-defined]
"""Tests for MeasurementVisitor."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
# pylint: disable=protected-access, pointless-string-statement, no-member, no-self-use


from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._measurement_visitor import (
    MeasurementVisitor,
    _get_query_bounds,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.keyset import KeySet
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
from tmlt.core.transformations.dictionary import GetValue
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.nan import DropNaNs, DropNulls
from tmlt.core.transformations.spark_transformations.nan import (
    ReplaceInfs as ReplaceInfTransformation,
)
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.type_utils import assert_never

from ...conftest import (  # pylint: disable=no-name-in-module
    assert_frame_equal_with_sort,
)


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


"""Tests just for _get_query_bounds."""


@pytest.mark.parametrize("bound", [(0), (-123456), (7899000)])
def test_equal_upper_and_lower_average(bound: float) -> None:
    """Test _get_query_bounds on Average query expr, with lower=upper."""
    average = GroupByBoundedAverage(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=bound,
        high=bound,
    )
    (low, high) = _get_query_bounds(average)
    assert low == high
    expected = ExactNumber.from_float(bound, round_up=True)
    assert low == expected
    assert high == expected


@pytest.mark.parametrize("bound", [(0), (-123456), (7899000)])
def test_equal_upper_and_lower_stdev(bound: float) -> None:
    """Test _get_query_bounds on STDEV query expr, with lower=upper."""
    stdev = GroupByBoundedSTDEV(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=bound,
        high=bound,
    )
    (low, high) = _get_query_bounds(stdev)
    assert low == high
    expected = ExactNumber.from_float(bound, round_up=True)
    assert low == expected
    assert high == expected


@pytest.mark.parametrize("bound", [(0), (-123456), (7899000)])
def test_equal_upper_and_lower_sum(bound: float) -> None:
    """Test _get_query_bounds on Sum query expr, with lower=upper."""
    sum_query = GroupByBoundedSum(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=bound,
        high=bound,
    )
    (low, high) = _get_query_bounds(sum_query)
    assert low == high
    expected = ExactNumber.from_float(bound, round_up=True)
    assert low == expected
    assert high == expected


@pytest.mark.parametrize("bound", [(0), (-123456), (7899000)])
def test_equal_upper_and_lower_variance(bound: float) -> None:
    """Test _get_query_bounds on Variance query expr, with lower=upper."""
    variance = GroupByBoundedVariance(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=bound,
        high=bound,
    )
    (low, high) = _get_query_bounds(variance)
    assert low == high
    expected = ExactNumber.from_float(bound, round_up=True)
    assert low == expected
    assert high == expected


@pytest.mark.parametrize("bound", [(0), (-123456), (7899000)])
def test_equal_upper_and_lower_quantile(bound: float) -> None:
    """Test _get_query_bounds on Quantile query expr, with lower=upper."""
    quantile = GroupByQuantile(
        child=PrivateSource("private"),
        groupby_keys=KeySet.from_dict({}),
        measure_column="",
        low=bound,
        high=bound,
        quantile=0.5,
    )
    (low, high) = _get_query_bounds(quantile)
    assert low == high
    expected = ExactNumber.from_float(bound, round_up=True)
    assert low == expected
    assert high == expected


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
            "private": SparkDataFrameDomain(
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
            "private_2": SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "C": SparkIntegerColumnDescriptor(),
                }
            ),
        }
    )

    input_metric = DictMetric(
        {"private": SymmetricDifference(), "private_2": SymmetricDifference()}
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
    catalog.add_private_source(
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
        stability=3,
    )
    catalog.add_private_view(
        "private_2",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "C": ColumnDescriptor(ColumnType.INTEGER),
        },
        stability=3,
    )
    catalog.add_public_source(
        "public",
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
        },
    )
    request.cls.catalog = catalog

    budget = ExactNumber(10).expr
    stability = {"private": ExactNumber(3).expr, "private_2": ExactNumber(3).expr}
    request.cls.visitor = MeasurementVisitor(
        per_query_privacy_budget=budget,
        stability=stability,
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=PureDP(),
        default_mechanism=NoiseMechanism.LAPLACE,
        public_sources=public_sources,
        catalog=catalog,
    )
    # for the methods which alter the output measure of a visitor.
    request.cls.pick_noise_visitor = MeasurementVisitor(
        per_query_privacy_budget=budget,
        stability=stability,
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=PureDP(),
        default_mechanism=NoiseMechanism.LAPLACE,
        public_sources=public_sources,
        catalog=catalog,
    )


@pytest.mark.usefixtures("test_data")
class TestMeasurementVisitor:
    """Tests for Measurement Visitor."""

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
            high=0,
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
            high=0,
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
            high=0,
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
            high=0,
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
                high=0,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, StdevMechanism):
            query = GroupByBoundedSTDEV(
                child=self.base_query,
                measure_column="",
                low=0,
                high=0,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, SumMechanism):
            query = GroupByBoundedSum(
                child=self.base_query,
                measure_column="",
                low=0,
                high=0,
                mechanism=mechanism,
                groupby_keys=KeySet.from_dict({}),
            )
        elif isinstance(mechanism, VarianceMechanism):
            query = GroupByBoundedVariance(
                child=self.base_query,
                measure_column="",
                low=0,
                high=0,
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
        "query,expected_mid_stability,expected_mechanism,expected_new_child",
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
                None,
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
                DropNullAndNan(child=PrivateSource("private"), columns=["nan"]),
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
                None,
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
                DropNullAndNan(child=PrivateSource("private"), columns=["null"]),
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
                None,
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
                ReplaceInfExpr(
                    child=PrivateSource("private"), replace_with={"inf": (-100, 100)}
                ),
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
                None,
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
                ReplaceInfExpr(
                    child=DropNullAndNan(
                        child=PrivateSource("private"), columns=["null_and_nan_and_inf"]
                    ),
                    replace_with={"null_and_nan_and_inf": (-100, 100)},
                ),
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
        expected_new_child: Optional[QueryExpr],
    ):
        """Test _build_common."""
        info = self.visitor._build_common(query)  # pylint: disable=protected-access
        get_value_transformation: GetValue
        if expected_new_child is None:
            assert isinstance(info.transformation, GetValue)
            get_value_transformation = info.transformation
        else:
            expected_null_nan_columns: List[str] = []
            expected_inf_replace: Dict[str, Tuple[float, float]] = {}
            if isinstance(expected_new_child, ReplaceInfExpr):
                expected_inf_replace = expected_new_child.replace_with
                if isinstance(expected_new_child.child, DropNullAndNan):
                    expected_null_nan_columns = expected_new_child.child.columns
            if isinstance(expected_new_child, DropNullAndNan):
                expected_null_nan_columns = expected_new_child.columns
            assert isinstance(info.transformation, ChainTT)
            transformations = chain_to_list(info.transformation)
            assert isinstance(transformations[0], GetValue)
            get_value_transformation = transformations[0]
            got_null_nan_columns: List[str] = []
            for t in transformations[1:]:
                assert isinstance(t, (ReplaceInfTransformation, DropNaNs, DropNulls))
                if isinstance(t, ReplaceInfTransformation):
                    assert sorted(list(t.replace_map.keys())) == sorted(
                        list(expected_inf_replace)
                    )
                    for k, v in t.replace_map.items():
                        assert v == expected_inf_replace[k]
                elif isinstance(t, DropNaNs):
                    got_null_nan_columns += t.columns
                elif isinstance(t, DropNulls):
                    got_null_nan_columns += t.columns
            assert sorted(list(set(got_null_nan_columns))) == sorted(
                expected_null_nan_columns
            )
        assert get_value_transformation.key == "private"
        assert get_value_transformation.input_domain == self.visitor.input_domain
        assert get_value_transformation.input_metric == self.visitor.input_metric
        assert (
            get_value_transformation.output_domain
            == self.visitor.input_domain["private"]
        )
        assert (
            get_value_transformation.output_metric
            == self.visitor.input_metric["private"]
        )

        assert info.mechanism == expected_mechanism
        assert info.mid_stability == expected_mid_stability
        assert info.lower_bound == ExactNumber.from_float(query.low, round_up=True)
        assert info.upper_bound == ExactNumber.from_float(query.high, round_up=False)

        assert isinstance(info.groupby, GroupBy)
        assert isinstance(self.visitor.input_domain["private"], SparkDataFrameDomain)
        expected_groupby_domain = SparkGroupedDataFrameDomain(
            schema=self.visitor.input_domain["private"].schema.copy(),
            group_keys=query.groupby_keys.dataframe(),
        )
        if expected_new_child is not None:
            new_transform: QueryExpr = expected_new_child
            while isinstance(new_transform, (ReplaceInfExpr, DropNullAndNan)):
                if isinstance(new_transform, ReplaceInfExpr):
                    replace_inf_expr: ReplaceInfExpr = new_transform
                    schema = expected_groupby_domain.schema
                    for col in replace_inf_expr.replace_with:
                        if isinstance(schema[col], SparkFloatColumnDescriptor):
                            schema[col] = SparkFloatColumnDescriptor(allow_inf=False)
                    new_transform = replace_inf_expr.child
                elif isinstance(new_transform, DropNullAndNan):
                    drop_null_and_nan_expr: DropNullAndNan = new_transform
                    schema = expected_groupby_domain.schema
                    for col in new_transform.columns:
                        if isinstance(schema[col], SparkFloatColumnDescriptor):
                            schema[col] = SparkFloatColumnDescriptor(
                                allow_null=False, allow_nan=False
                            )
                        else:
                            schema[col] = SparkIntegerColumnDescriptor(allow_null=False)
                    new_transform = drop_null_and_nan_expr.child
                expected_groupby_domain = SparkGroupedDataFrameDomain(
                    schema=schema, group_keys=query.groupby_keys.dataframe()
                )
        assert isinstance(info.groupby.output_domain, SparkGroupedDataFrameDomain)
        assert info.groupby.output_domain.schema == expected_groupby_domain.schema
        assert_frame_equal_with_sort(
            info.groupby.output_domain.group_keys.toPandas(),
            expected_groupby_domain.group_keys.toPandas(),
        )
        assert info.new_child == expected_new_child

    def test_validate_measurement(self):
        """Test _validate_measurement."""
        with patch(
            "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
            autospec=True,
        ) as mock_measurement:
            mock_measurement.privacy_function.return_value = self.visitor.budget
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

    def _check_measurement(
        self, measurement: Measurement, child_is_base_query: bool = True
    ):
        """Check the basic attributes of a measurement (for all query exprs).

        If ``child_is_base_query`` is true, it is assumed that the child
        query was ``PrivateSource("private")``.

        The measurement almost certainly looks like this:
        ``child_transformation | mock_measurement``
        so extensive testing of the latter is likely to be counterproductive.
        """
        assert isinstance(measurement, ChainTM)

        if child_is_base_query:
            assert isinstance(measurement.transformation, GetValue)
            assert measurement.transformation.key == "private"

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
        transformation = self.visitor._visit_child_transformation(
            child_query, expected_mechanism
        )
        # pylint: enable=protected-access
        mock_measurement.input_domain = transformation.output_domain
        mock_measurement.input_metric = transformation.output_metric
        mock_measurement.privacy_function.return_value = self.visitor.budget

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

            self._check_measurement(measurement, query.child == self.base_query)
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
                d_out=self.visitor.budget,
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

            mock_measurement.privacy_function.return_value = self.visitor.budget
            mock_create_count_distinct.return_value = mock_measurement
            # Sometimes a CountDistinct query needs to know which columns to select
            query_needs_columns_selected = (
                query.columns_to_count is not None and len(query.columns_to_count) > 0
            )

            expected_child_query: QueryExpr
            expected_child_query = query.child
            # If it is expected that a query will need a
            # specific set of columns selected,
            # make that expected Select query now
            # (since the function modifies the query)
            if query_needs_columns_selected:
                # Build the list of columns that we're expecting to group by
                groupby_columns: List[str] = list(query.groupby_keys.schema().keys())
                # There is literally no way this cannot be true,
                # but if you leave this out then MyPy will complain.
                assert query.columns_to_count is not None
                expected_child_query = SelectExpr(
                    child=query.child, columns=query.columns_to_count + groupby_columns
                )
            self._setup_mock_measurement(
                mock_measurement, expected_child_query, expected_mechanism
            )

            measurement = self.visitor.visit_groupby_count_distinct(query)

            self._check_measurement(
                measurement,
                (query.child == self.base_query and not query_needs_columns_selected),
            )
            assert isinstance(measurement, ChainTM)
            assert measurement.measurement == mock_measurement
            if query_needs_columns_selected:
                assert isinstance(measurement.transformation, ChainTT)
                if query.child == self.base_query:
                    assert isinstance(
                        measurement.transformation.transformation1, GetValue
                    )
                    get_value = measurement.transformation.transformation1
                    assert isinstance(get_value, GetValue)
                    assert get_value.key == "private"
                assert isinstance(
                    measurement.transformation.transformation2, SelectTransformation
                )
                select_transformation = measurement.transformation.transformation2
                assert isinstance(select_transformation, SelectTransformation)
                assert isinstance(expected_child_query, SelectExpr)
                assert select_transformation.columns == expected_child_query.columns

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
                d_out=self.visitor.budget,
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
                    high=0,
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
                    high=0,
                ),
                RhoZCDP(),
                ReplaceInfExpr(
                    DropNullAndNan(PrivateSource("private"), ["nan_and_inf"]),
                    {"nan_and_inf": (0, 0)},
                ),
            ),
            (
                GroupByQuantile(
                    child=PrivateSource("private"),
                    groupby_keys=KeySet.from_dict({"A": ["zero"]}),
                    quantile=0.5,
                    measure_column="inf",
                    low=0,
                    high=0,
                ),
                RhoZCDP(),
                ReplaceInfExpr(PrivateSource("private"), {"inf": (0, 0)}),
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

            self._check_measurement(
                measurement,
                query.child == self.base_query and expected_new_child is None,
            )
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
                d_out=self.visitor.budget,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                quantile_column=query.output_column,
            )

            # Check for the drop transformations, if expected
            if expected_new_child is not None:
                expected_null_nan_columns: List[str] = []
                expected_inf_replace: Dict[str, Tuple[float, float]] = {}
                if isinstance(expected_new_child, ReplaceInfExpr):
                    expected_inf_replace = expected_new_child.replace_with
                    if isinstance(expected_new_child.child, DropNullAndNan):
                        expected_null_nan_columns = expected_new_child.child.columns
                elif isinstance(expected_new_child, DropNullAndNan):
                    expected_null_nan_columns = expected_new_child.columns
                assert isinstance(measurement.transformation, ChainTT)
                transformations = chain_to_list(measurement.transformation)
                assert isinstance(transformations[0], GetValue)
                got_null_nan_columns: List[str] = []
                for t in transformations[1:]:
                    assert isinstance(
                        t, (ReplaceInfTransformation, DropNaNs, DropNulls)
                    )
                    if isinstance(t, ReplaceInfTransformation):
                        assert sorted(list(t.replace_map.keys())) == sorted(
                            list(expected_inf_replace.keys())
                        )
                        for k, v in t.replace_map.items():
                            assert v == expected_inf_replace[k]
                    elif isinstance(t, DropNaNs):
                        got_null_nan_columns += t.columns
                    elif isinstance(t, DropNulls):
                        got_null_nan_columns += t.columns
                assert sorted(list(set(got_null_nan_columns))) == sorted(
                    expected_null_nan_columns
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
                    high=0,
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

            self._check_measurement(measurement, query.child == self.base_query)
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
                d_out=self.visitor.budget,
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
                    high=0,
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

            self._check_measurement(measurement, query.child == self.base_query)
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
                d_out=self.visitor.budget,
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
                    high=0,
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

            self._check_measurement(measurement, query.child == self.base_query)
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
                d_out=self.visitor.budget,
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
                    high=0,
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

            self._check_measurement(measurement, query.child == self.base_query)
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
                d_out=self.visitor.budget,
                output_measure=self.visitor.output_measure,
                groupby_transformation=mock_groupby.return_value,
                standard_deviation_column=query.output_column,
            )

    @pytest.mark.parametrize(
        "query",
        [
            (PrivateSource("private")),
            (Rename(child=PrivateSource("private"), column_mapper={"A": "A2"})),
            (Filter(child=PrivateSource("private"), predicate="B > 2")),
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
                    max_num_rows=11,
                    schema_new_columns=Schema({"i": "DECIMAL"}),
                    augment=False,
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
