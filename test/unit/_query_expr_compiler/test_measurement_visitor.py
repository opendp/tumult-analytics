"""Tests for MeasurementVisitor."""

# <placeholder: boilerplate>

import dataclasses
from typing import List, Optional, Union
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import sympy as sp
from parameterized import parameterized
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
    AnalyticsDefault,
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
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
from tmlt.core.transformations.spark_transformations.nan import (
    ReplaceInfs,
    ReplaceNaNs,
    ReplaceNulls,
)
from tmlt.core.transformations.spark_transformations.select import (
    Select as SelectTransformation,
)
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import PySparkTest
from tmlt.core.utils.type_utils import assert_never


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


class TestGetQueryBounds(TestCase):
    """Tests just for _get_query_bounds."""

    @parameterized.expand([(0,), (-123456,), (7899000,)])
    def test_equal_upper_and_lower_average(self, bound: float) -> None:
        """Test _get_query_bounds on Average query expr, with lower=upper."""
        average = GroupByBoundedAverage(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=bound,
            high=bound,
        )
        (low, high) = _get_query_bounds(average)
        self.assertEqual(low, high)
        expected = ExactNumber.from_float(bound, round_up=True)
        self.assertEqual(low, expected)
        self.assertEqual(high, expected)

    @parameterized.expand([(0,), (-123456,), (7899000,)])
    def test_equal_upper_and_lower_stdev(self, bound: float) -> None:
        """Test _get_query_bounds on STDEV query expr, with lower=upper."""
        stdev = GroupByBoundedSTDEV(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=bound,
            high=bound,
        )
        (low, high) = _get_query_bounds(stdev)
        self.assertEqual(low, high)
        expected = ExactNumber.from_float(bound, round_up=True)
        self.assertEqual(low, expected)
        self.assertEqual(high, expected)

    @parameterized.expand([(0,), (-123456,), (7899000,)])
    def test_equal_upper_and_lower_sum(self, bound: float) -> None:
        """Test _get_query_bounds on Sum query expr, with lower=upper."""
        sum_query = GroupByBoundedSum(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=bound,
            high=bound,
        )
        (low, high) = _get_query_bounds(sum_query)
        self.assertEqual(low, high)
        expected = ExactNumber.from_float(bound, round_up=True)
        self.assertEqual(low, expected)
        self.assertEqual(high, expected)

    @parameterized.expand([(0,), (-123456,), (7899000,)])
    def test_equal_upper_and_lower_variance(self, bound: float) -> None:
        """Test _get_query_bounds on Variance query expr, with lower=upper."""
        variance = GroupByBoundedVariance(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=bound,
            high=bound,
        )
        (low, high) = _get_query_bounds(variance)
        self.assertEqual(low, high)
        expected = ExactNumber.from_float(bound, round_up=True)
        self.assertEqual(low, expected)
        self.assertEqual(high, expected)

    @parameterized.expand([(0,), (-123456,), (7899000,)])
    def test_equal_upper_and_lower_quantile(self, bound: float) -> None:
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
        self.assertEqual(low, high)
        expected = ExactNumber.from_float(bound, round_up=True)
        self.assertEqual(low, expected)
        self.assertEqual(high, expected)

    @parameterized.expand([(0, 1), (-123456, 0), (7899000, 9999999)])
    def test_average(self, lower: float, upper: float) -> None:
        """Test _get_query_bounds on Average query expr, with lower!=upper."""
        average = GroupByBoundedAverage(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=lower,
            high=upper,
        )
        (low, high) = _get_query_bounds(average)
        self.assertEqual(low, ExactNumber.from_float(lower, round_up=True))
        self.assertEqual(high, ExactNumber.from_float(upper, round_up=False))

    @parameterized.expand([(0, 1), (-123456, 0), (7899000, 9999999)])
    def test_stdev(self, lower: float, upper: float) -> None:
        """Test _get_query_bounds on STDEV query expr, with lower!=upper."""
        stdev = GroupByBoundedSTDEV(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=lower,
            high=upper,
        )
        (low, high) = _get_query_bounds(stdev)
        self.assertEqual(low, ExactNumber.from_float(lower, round_up=True))
        self.assertEqual(high, ExactNumber.from_float(upper, round_up=False))

    @parameterized.expand([(0, 1), (-123456, 0), (7899000, 9999999)])
    def test_sum(self, lower: float, upper: float) -> None:
        """Test _get_query_bounds on Sum query expr, with lower!=upper."""
        sum_query = GroupByBoundedSum(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=lower,
            high=upper,
        )
        (low, high) = _get_query_bounds(sum_query)
        self.assertEqual(low, ExactNumber.from_float(lower, round_up=True))
        self.assertEqual(high, ExactNumber.from_float(upper, round_up=False))

    @parameterized.expand([(0, 1), (-123456, 0), (7899000, 9999999)])
    def test_variance(self, lower: float, upper: float) -> None:
        """Test _get_query_bounds on Variance query expr, with lower!=upper."""
        variance = GroupByBoundedVariance(
            child=PrivateSource("private"),
            groupby_keys=KeySet.from_dict({}),
            measure_column="",
            low=lower,
            high=upper,
        )
        (low, high) = _get_query_bounds(variance)
        self.assertEqual(low, ExactNumber.from_float(lower, round_up=True))
        self.assertEqual(high, ExactNumber.from_float(upper, round_up=False))

    @parameterized.expand([(0, 1), (-123456, 0), (7899000, 9999999)])
    def test_quantile(self, lower: float, upper: float) -> None:
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
        self.assertEqual(low, ExactNumber.from_float(lower, round_up=True))
        self.assertEqual(high, ExactNumber.from_float(upper, round_up=True))


class TestMeasurementVisitor(PySparkTest):
    """Test MeasurementVisitor."""

    def setUp(self) -> None:
        """Setup tests."""
        input_domain = DictDomain(
            {
                "private": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "X": SparkFloatColumnDescriptor(),
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
            "public": self.spark.createDataFrame(
                pd.DataFrame({"A": ["zero", "one"], "B": [0, 1]}),
                schema=StructType(
                    [
                        StructField("A", StringType(), False),
                        StructField("B", LongType(), False),
                    ]
                ),
            )
        }
        self.base_query = PrivateSource(source_id="private")

        self.catalog = Catalog()
        self.catalog.add_private_source(
            "private",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR),
                "B": ColumnDescriptor(ColumnType.INTEGER),
                "X": ColumnDescriptor(ColumnType.DECIMAL),
            },
            stability=3,
        )
        self.catalog.add_private_view(
            "private_2",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR),
                "C": ColumnDescriptor(ColumnType.INTEGER),
            },
            stability=3,
        )
        self.catalog.add_public_source(
            "public",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR),
                "B": ColumnDescriptor(ColumnType.INTEGER),
            },
        )

        budget = ExactNumber(10).expr
        stability = {"private": ExactNumber(3).expr, "private_2": ExactNumber(3).expr}
        self.visitor = MeasurementVisitor(
            per_query_privacy_budget=budget,
            stability=stability,
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=PureDP(),
            default_mechanism=NoiseMechanism.LAPLACE,
            public_sources=public_sources,
            catalog=self.catalog,
        )

    @parameterized.expand(
        [
            (CountMechanism.DEFAULT, PureDP(), NoiseMechanism.GEOMETRIC),
            (CountMechanism.DEFAULT, RhoZCDP(), NoiseMechanism.DISCRETE_GAUSSIAN),
            (CountMechanism.LAPLACE, PureDP(), NoiseMechanism.GEOMETRIC),
            (CountMechanism.LAPLACE, RhoZCDP(), NoiseMechanism.GEOMETRIC),
            (CountMechanism.GAUSSIAN, PureDP(), NoiseMechanism.DISCRETE_GAUSSIAN),
            (CountMechanism.GAUSSIAN, RhoZCDP(), NoiseMechanism.DISCRETE_GAUSSIAN),
        ]
    )
    def test_pick_noise_for_count(
        self,
        query_mechanism: CountMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test _pick_noise_for_count for GroupByCount query expressions."""
        query = GroupByCount(
            child=self.base_query, groupby_keys=KeySet({}), mechanism=query_mechanism
        )
        self.visitor.output_measure = output_measure
        # pylint: disable=protected-access
        got_mechanism = self.visitor._pick_noise_for_count(query)
        # pylint: enable=protected-access
        self.assertEqual(got_mechanism, expected_mechanism)

    @parameterized.expand(
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
        ]
    )
    def test_pick_noise_for_count_distinct(
        self,
        query_mechanism: CountDistinctMechanism,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Test _pick_noise_for_count for GroupByCountDistinct query expressions."""
        query = GroupByCountDistinct(
            child=self.base_query, groupby_keys=KeySet({}), mechanism=query_mechanism
        )
        self.visitor.output_measure = output_measure
        # pylint: disable=protected-access
        got_mechanism = self.visitor._pick_noise_for_count(query)
        # pylint: enable=protected-access
        self.assertEqual(got_mechanism, expected_mechanism)

    @parameterized.expand(
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
        ]
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
            groupby_keys=KeySet({}),
        )
        self.visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            self.assertEqual(got_mechanism, expected_mechanism)
        else:
            with self.assertRaises(NotImplementedError):
                self.visitor._pick_noise_for_non_count(query, measure_column_type)
        # pylint: enable=protected-access

    @parameterized.expand(
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
        ]
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
            groupby_keys=KeySet({}),
        )
        self.visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            self.assertEqual(got_mechanism, expected_mechanism)
        else:
            with self.assertRaises(NotImplementedError):
                self.visitor._pick_noise_for_non_count(query, measure_column_type)
        # pylint: enable=protected-access

    @parameterized.expand(
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
        ]
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
            groupby_keys=KeySet({}),
        )
        self.visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            self.assertEqual(got_mechanism, expected_mechanism)
        else:
            with self.assertRaises(NotImplementedError):
                self.visitor._pick_noise_for_non_count(query, measure_column_type)
        # pylint: enable=protected-access

    @parameterized.expand(
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
        ]
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
            groupby_keys=KeySet({}),
        )
        self.visitor.output_measure = output_measure
        # pylint: disable=protected-access
        if expected_mechanism is not None:
            got_mechanism = self.visitor._pick_noise_for_non_count(
                query, measure_column_type
            )
            self.assertEqual(got_mechanism, expected_mechanism)
        else:
            with self.assertRaises(NotImplementedError):
                self.visitor._pick_noise_for_non_count(query, measure_column_type)
        # pylint: enable=protected-access

    @parameterized.expand(
        [
            (AverageMechanism.LAPLACE,),
            (StdevMechanism.LAPLACE,),
            (SumMechanism.LAPLACE,),
            (VarianceMechanism.LAPLACE,),
        ]
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
                groupby_keys=KeySet({}),
            )
        elif isinstance(mechanism, StdevMechanism):
            query = GroupByBoundedSTDEV(
                child=self.base_query,
                measure_column="",
                low=0,
                high=0,
                mechanism=mechanism,
                groupby_keys=KeySet({}),
            )
        elif isinstance(mechanism, SumMechanism):
            query = GroupByBoundedSum(
                child=self.base_query,
                measure_column="",
                low=0,
                high=0,
                mechanism=mechanism,
                groupby_keys=KeySet({}),
            )
        elif isinstance(mechanism, VarianceMechanism):
            query = GroupByBoundedVariance(
                child=self.base_query,
                measure_column="",
                low=0,
                high=0,
                mechanism=mechanism,
                groupby_keys=KeySet({}),
            )
        else:
            assert_never(mechanism)
        with self.assertRaisesRegex(
            AssertionError, "Query's measure column should be numeric."
        ):
            # pylint: disable=protected-access
            self.visitor._pick_noise_for_non_count(query, SparkStringColumnDescriptor())
            # pylint: enable=protected-access

    @parameterized.expand(
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
        ]
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
        self.assertEqual(got.input_domain, input_domain)
        self.assertEqual(got.input_metric, input_metric)
        self.assert_frame_equal_with_sort(
            got.group_keys.toPandas(), keyset.dataframe().toPandas()
        )
        expected_output_domain = SparkGroupedDataFrameDomain(
            schema=input_domain.schema, group_keys=keyset.dataframe()
        )
        self.assertIsInstance(got.output_domain, SparkGroupedDataFrameDomain)
        assert isinstance(got.output_domain, SparkGroupedDataFrameDomain)
        self.assertEqual(got.output_domain.schema, expected_output_domain.schema)
        self.assert_frame_equal_with_sort(
            got.output_domain.group_keys.toPandas(),
            expected_output_domain.group_keys.toPandas(),
        )
        self.assertEqual(got.output_metric, expected_output_metric)

    @parameterized.expand(
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
        ]
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
        self.assertIsInstance(info.transformation, GetValue)
        assert isinstance(info.transformation, GetValue)
        self.assertEqual(info.transformation.key, "private")
        self.assertEqual(info.transformation.input_domain, self.visitor.input_domain)
        self.assertEqual(info.transformation.input_metric, self.visitor.input_metric)
        self.assertEqual(
            info.transformation.output_domain, self.visitor.input_domain["private"]
        )
        self.assertEqual(
            info.transformation.output_metric, self.visitor.input_metric["private"]
        )

        self.assertEqual(info.mechanism, expected_mechanism)
        self.assertEqual(info.mid_stability, expected_mid_stability)
        self.assertEqual(
            info.lower_bound, ExactNumber.from_float(query.low, round_up=True)
        )
        self.assertEqual(
            info.upper_bound, ExactNumber.from_float(query.high, round_up=False)
        )

        self.assertIsInstance(info.groupby, GroupBy)
        assert isinstance(self.visitor.input_domain["private"], SparkDataFrameDomain)
        expected_groupby_domain = SparkGroupedDataFrameDomain(
            schema=self.visitor.input_domain["private"].schema,
            group_keys=query.groupby_keys.dataframe(),
        )
        self.assertIsInstance(info.groupby.output_domain, SparkGroupedDataFrameDomain)
        assert isinstance(info.groupby.output_domain, SparkGroupedDataFrameDomain)
        self.assertEqual(
            info.groupby.output_domain.schema, expected_groupby_domain.schema
        )
        self.assert_frame_equal_with_sort(
            info.groupby.output_domain.group_keys.toPandas(),
            expected_groupby_domain.group_keys.toPandas(),
        )

    @patch("tmlt.core.measurements.base.Measurement", autospec=True)
    def test_validate_measurement(self, mock_measurement):
        """Test _validate_measurement."""
        mock_measurement.privacy_function.return_value = self.visitor.budget
        mid_stability = ExactNumber(2).expr
        # This should finish without raising an error
        # pylint: disable=protected-access
        self.visitor._validate_measurement(mock_measurement, mid_stability)

        # Change it so that the privacy function returns something else
        mock_measurement.privacy_function.return_value = ExactNumber(-10).expr
        with self.assertRaisesRegex(
            AssertionError, "Privacy function does not match per-query privacy budget."
        ):
            self.visitor._validate_measurement(mock_measurement, mid_stability)
        # pylint: enable=protected-access

    def check_measurement(
        self, measurement: Measurement, child_is_base_query: bool = True
    ):
        """Check the basic attributes of a measurement (for all query exprs).

        If `child_is_base_query` is true, it is assumed that the child
        query was `PrivateSource("private")`.

        The measurement almost certainly looks like this:
        `child_transformation | mock_measurement`
        so extensive testing of the latter is likely to be counterproductive.
        """
        self.assertIsInstance(measurement, ChainTM)
        assert isinstance(measurement, ChainTM)

        if child_is_base_query:
            self.assertIsInstance(measurement.transformation, GetValue)
            assert isinstance(measurement.transformation, GetValue)
            self.assertEqual(measurement.transformation.key, "private")

        self.assertEqual(
            measurement.transformation.input_domain, self.visitor.input_domain
        )
        self.assertEqual(
            measurement.transformation.input_metric, self.visitor.input_metric
        )
        self.assertIsInstance(
            measurement.transformation.output_domain, SparkDataFrameDomain
        )
        self.assertEqual(
            measurement.transformation.output_domain,
            measurement.measurement.input_domain,
        )
        self.assertIsInstance(
            measurement.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )
        self.assertEqual(
            measurement.transformation.output_metric,
            measurement.measurement.input_metric,
        )

    @staticmethod
    def check_mock_groupby_call(
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

    def setup_mock_measurement(
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

    @parameterized.expand(
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
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_count_measurement",
        autospec=True,
    )
    def test_visit_groupby_count(
        self,
        query: GroupByCount,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
        mock_create_count,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test visit_groupby_count."""
        self.visitor.output_measure = output_measure
        self.setup_mock_measurement(mock_measurement, query.child, expected_mechanism)
        mock_create_count.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_count(query)

        self.check_measurement(measurement, query.child == self.base_query)
        assert isinstance(measurement, ChainTM)
        self.assertEqual(measurement.measurement, mock_measurement)
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

    @parameterized.expand(
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
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_count_distinct_measurement",
        autospec=True,
    )
    def test_visit_groupby_count_distinct(
        self,
        query: GroupByCountDistinct,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
        mock_create_count_distinct,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test visit_groupby_count_distinct."""
        self.visitor.output_measure = output_measure

        mock_measurement.privacy_function.return_value = self.visitor.budget
        mock_create_count_distinct.return_value = mock_measurement
        # Sometimes a CountDistinct query needs to know which columns to select
        query_needs_columns_selected = (
            query.columns_to_count is not None and len(query.columns_to_count) > 0
        )

        expected_child_query: QueryExpr
        expected_child_query = query.child
        # If it is expected that a query will need a specific set of columns selected,
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
        self.setup_mock_measurement(
            mock_measurement, expected_child_query, expected_mechanism
        )

        measurement = self.visitor.visit_groupby_count_distinct(query)

        self.check_measurement(
            measurement,
            (query.child == self.base_query and not query_needs_columns_selected),
        )
        assert isinstance(measurement, ChainTM)
        self.assertEqual(measurement.measurement, mock_measurement)
        if query_needs_columns_selected:
            self.assertIsInstance(measurement.transformation, ChainTT)
            assert isinstance(measurement.transformation, ChainTT)
            if query.child == self.base_query:
                self.assertIsInstance(
                    measurement.transformation.transformation1, GetValue
                )
                get_value = measurement.transformation.transformation1
                assert isinstance(get_value, GetValue)
                self.assertEqual(get_value.key, "private")
            self.assertIsInstance(
                measurement.transformation.transformation2, SelectTransformation
            )
            select_transformation = measurement.transformation.transformation2
            assert isinstance(select_transformation, SelectTransformation)
            assert isinstance(expected_child_query, SelectExpr)
            self.assertEqual(
                select_transformation.columns, expected_child_query.columns
            )

        mid_stability = measurement.transformation.stability_function(
            self.visitor.stability
        )

        self.assertEqual(measurement.measurement, mock_measurement)
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

    @parameterized.expand(
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
            ),
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_quantile_measurement",
        autospec=True,
    )
    def test_visit_groupby_quantile(
        self,
        query: GroupByQuantile,
        output_measure: Union[PureDP, RhoZCDP],
        mock_create_quantile,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test visit_groupby_quantile."""
        self.visitor.output_measure = output_measure

        expected_mechanism = self.visitor.default_mechanism
        self.setup_mock_measurement(mock_measurement, query.child, expected_mechanism)
        mock_create_quantile.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_quantile(query)

        self.check_measurement(measurement, query.child == self.base_query)
        assert isinstance(measurement, ChainTM)
        self.assertEqual(measurement.measurement, mock_measurement)
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

    @parameterized.expand(
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
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_sum_measurement",
        autospec=True,
    )
    def test_visit_groupby_bounded_sum(
        self,
        query: GroupByBoundedSum,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
        mock_create_sum,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test visit_groupby_bounded_sum."""
        self.visitor.output_measure = output_measure

        self.setup_mock_measurement(mock_measurement, query.child, expected_mechanism)
        mock_create_sum.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_sum(query)

        self.check_measurement(measurement, query.child == self.base_query)
        assert isinstance(measurement, ChainTM)
        self.assertEqual(measurement.measurement, mock_measurement)
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

    @parameterized.expand(
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
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_average_measurement",
        autospec=True,
    )
    def test_visit_groupby_bounded_average(
        self,
        query: GroupByBoundedAverage,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
        mock_create_average,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test visit_groupby_bounded_average."""
        self.visitor.output_measure = output_measure

        self.setup_mock_measurement(mock_measurement, query.child, expected_mechanism)
        mock_create_average.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_average(query)

        self.check_measurement(measurement, query.child == self.base_query)
        assert isinstance(measurement, ChainTM)
        self.assertEqual(measurement.measurement, mock_measurement)
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

    @parameterized.expand(
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
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_variance_measurement",
        autospec=True,
    )
    def test_visit_groupby_bounded_variance(
        self,
        query: GroupByBoundedVariance,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
        mock_create_variance,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test visit_groupby_bounded_variance."""
        self.visitor.output_measure = output_measure

        self.setup_mock_measurement(mock_measurement, query.child, expected_mechanism)
        mock_create_variance.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_variance(query)

        self.check_measurement(measurement, query.child == self.base_query)
        assert isinstance(measurement, ChainTM)
        self.assertEqual(measurement.measurement, mock_measurement)
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

    @parameterized.expand(
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
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_standard_deviation_measurement",
        autospec=True,
    )
    def test_visit_groupby_bounded_stdev(
        self,
        query: GroupByBoundedSTDEV,
        output_measure: Union[PureDP, RhoZCDP],
        expected_mechanism: NoiseMechanism,
        mock_create_stdev,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test visit_groupby_bounded_stdev."""
        self.visitor.output_measure = output_measure

        self.setup_mock_measurement(mock_measurement, query.child, expected_mechanism)
        mock_create_stdev.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_stdev(query)

        self.check_measurement(measurement, query.child == self.base_query)
        assert isinstance(measurement, ChainTM)
        self.assertEqual(measurement.measurement, mock_measurement)
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

    @parameterized.expand(
        [
            (PrivateSource("private"),),
            (Rename(child=PrivateSource("private"), column_mapper={"A": "A2"}),),
            (Filter(child=PrivateSource("private"), predicate="B > 2"),),
            (SelectExpr(child=PrivateSource("private"), columns=["A"]),),
            (
                Map(
                    child=PrivateSource("private"),
                    f=lambda row: {"C": "c" + str(row["B"])},
                    schema_new_columns=Schema({"C": "VARCHAR"}),
                    augment=True,
                ),
            ),
            (
                FlatMap(
                    child=PrivateSource("private"),
                    f=lambda row: [{"i": n for n in range(row["B"] + 1)}],
                    max_num_rows=11,
                    schema_new_columns=Schema({"i": "DECIMAL"}),
                    augment=False,
                ),
            ),
            (
                JoinPrivate(
                    child=PrivateSource("private"),
                    right_operand_expr=PrivateSource("private_2"),
                    truncation_strategy_left=TruncationStrategy.DropExcess(3),
                    truncation_strategy_right=TruncationStrategy.DropExcess(3),
                ),
            ),
            (JoinPublic(child=PrivateSource("private"), public_table="public"),),
        ]
    )
    def test_visit_transformations(self, query: QueryExpr):
        """Test that visiting transformations returns an error."""
        with self.assertRaises(NotImplementedError):
            query.accept(self.visitor)


class TestMeasurementVisitorImplicitConversions(PySparkTest):
    """Test MeasurementVisitor."""

    def setUp(self) -> None:
        """Setup tests."""
        input_domain = DictDomain(
            {
                "private": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "X": SparkFloatColumnDescriptor(),
                        "has_nulls": SparkFloatColumnDescriptor(allow_null=True),
                        "has_nans": SparkFloatColumnDescriptor(allow_nan=True),
                        "has_infs": SparkFloatColumnDescriptor(allow_inf=True),
                        "null_and_nan": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True
                        ),
                        "null_and_inf": SparkFloatColumnDescriptor(
                            allow_null=True, allow_inf=True
                        ),
                        "nan_and_inf": SparkFloatColumnDescriptor(
                            allow_nan=True, allow_inf=True
                        ),
                        "R": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True, allow_inf=True
                        ),
                    }
                )
            }
        )
        input_metric = DictMetric({"private": SymmetricDifference()})
        self.base_query = PrivateSource(source_id="private")

        self.catalog = Catalog()
        self.catalog.add_private_source(
            "private",
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR),
                "X": ColumnDescriptor(ColumnType.DECIMAL),
                "has_nulls": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
                "has_nans": ColumnDescriptor(ColumnType.DECIMAL, allow_nan=True),
                "has_infs": ColumnDescriptor(ColumnType.DECIMAL, allow_inf=True),
                "null_and_nan": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True
                ),
                "null_and_inf": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_inf=True
                ),
                "nan_and_inf": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_nan=True, allow_inf=True
                ),
                "R": ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
                ),
            },
            stability=3,
        )

        budget = ExactNumber(10).expr
        stability = {"private": ExactNumber(3).expr}
        self.visitor = MeasurementVisitor(
            per_query_privacy_budget=budget,
            stability=stability,
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=PureDP(),
            default_mechanism=NoiseMechanism.LAPLACE,
            public_sources={},
            catalog=self.catalog,
        )
        self.keyset = KeySet.from_dict({"A": ["a0", "a1", "a2"]})

    def _check_sum_measurement(
        self,
        query: GroupByBoundedSum,
        mock_create_sum,
        mock_groupby,
        mock_measurement,
        got_measurement: Measurement,
        expected_mechanism: NoiseMechanism,
    ) -> None:
        """Check that the actual sum measurement was generated correctly."""
        self.assertIsInstance(got_measurement, ChainTM)
        assert isinstance(got_measurement, ChainTM)

        self.assertEqual(
            got_measurement.transformation.input_domain, self.visitor.input_domain
        )
        self.assertEqual(
            got_measurement.transformation.input_metric, self.visitor.input_metric
        )
        self.assertIsInstance(
            got_measurement.transformation.output_domain, SparkDataFrameDomain
        )
        self.assertEqual(
            got_measurement.transformation.output_domain,
            got_measurement.measurement.input_domain,
        )
        self.assertIsInstance(
            got_measurement.transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        )
        self.assertEqual(
            got_measurement.transformation.output_metric,
            got_measurement.measurement.input_metric,
        )
        # The leftmost transformation should be GetValue("private")
        transformation: Transformation = got_measurement.transformation
        while isinstance(transformation, ChainTT):
            transformation = transformation.transformation1
        self.assertIsInstance(transformation, GetValue)
        assert isinstance(transformation, GetValue)
        self.assertEqual(transformation.key, "private")

        groupby_df: DataFrame = query.groupby_keys.dataframe()
        mock_groupby.assert_called_with(
            input_domain=got_measurement.transformation.output_domain,
            input_metric=got_measurement.transformation.output_metric,
            use_l2=(expected_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN),
            group_keys=groupby_df,
        )

        self.assertEqual(got_measurement.measurement, mock_measurement)

        mid_stability = got_measurement.transformation.stability_function(
            self.visitor.stability
        )
        lower, upper = _get_query_bounds(query)
        mock_create_sum.assert_called_with(
            input_domain=got_measurement.transformation.output_domain,
            input_metric=got_measurement.transformation.output_metric,
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

    @parameterized.expand([(PureDP(),), (RhoZCDP(),)])
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_sum_measurement",
        autospec=True,
    )
    def test_sum_no_replacement(
        self,
        output_measure: Union[PureDP, RhoZCDP],
        mock_create_sum,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test GroupByBoundedSum with no expected replacements."""
        query = GroupByBoundedSum(
            child=self.base_query,
            measure_column="X",
            mechanism=SumMechanism.LAPLACE,
            output_column="sum",
            low=-100.0,
            high=100.0,
            groupby_keys=self.keyset,
        )
        self.visitor.output_measure = output_measure
        mock_measurement.input_domain = self.visitor.input_domain["private"]
        mock_measurement.input_metric = self.visitor.input_metric["private"]
        mock_measurement.privacy_function.return_value = self.visitor.budget
        mock_create_sum.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_sum(query)
        self._check_sum_measurement(
            query=query,
            mock_create_sum=mock_create_sum,
            mock_groupby=mock_groupby,
            mock_measurement=mock_measurement,
            got_measurement=measurement,
            expected_mechanism=NoiseMechanism.LAPLACE,
        )

    def _expected_input_domain(self, measure_column: str) -> SparkDataFrameDomain:
        """Get the expected input domain after 'fixing' measure_column."""
        df_domain = self.visitor.input_domain["private"]
        assert isinstance(df_domain, SparkDataFrameDomain)
        expected_columns = df_domain.schema.copy()
        expected_columns[measure_column] = dataclasses.replace(
            df_domain.schema[measure_column],
            allow_null=False,
            allow_nan=False,
            allow_inf=False,
        )
        return SparkDataFrameDomain(expected_columns)

    @parameterized.expand(
        [
            (PureDP(), "has_nulls"),
            (RhoZCDP(), "has_nulls"),
            (PureDP(), "has_nans"),
            (RhoZCDP(), "has_nans"),
            (PureDP(), "null_and_nan"),
            (RhoZCDP(), "null_and_nan"),
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_sum_measurement",
        autospec=True,
    )
    def test_sum_replace_null_or_nan(
        self,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column,
        mock_create_sum,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test GroupByBoundedSum replacing a nan and/or a null value."""
        self.visitor.output_measure = output_measure
        query = GroupByBoundedSum(
            child=self.base_query,
            measure_column=measure_column,
            mechanism=SumMechanism.LAPLACE,
            output_column="sum",
            low=-100.0,
            high=100.0,
            groupby_keys=self.keyset,
        )
        expected_input_domain = self._expected_input_domain(measure_column)
        mock_measurement.input_domain = expected_input_domain
        mock_measurement.input_metric = self.visitor.input_metric["private"]
        mock_measurement.privacy_function.return_value = self.visitor.budget
        mock_create_sum.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_sum(query)
        self._check_sum_measurement(
            query=query,
            mock_create_sum=mock_create_sum,
            mock_groupby=mock_groupby,
            mock_measurement=mock_measurement,
            got_measurement=measurement,
            expected_mechanism=NoiseMechanism.LAPLACE,
        )
        assert isinstance(measurement, ChainTM)
        self.assertIsInstance(measurement.transformation, ChainTT)
        assert isinstance(measurement.transformation, ChainTT)
        transforms = chain_to_list(measurement.transformation)
        self.assertEqual(len(transforms), 3)
        # The first one is a GetValue; we can skip it
        replace_null_transform = transforms[1]
        self.assertIsInstance(replace_null_transform, ReplaceNulls)
        assert isinstance(replace_null_transform, ReplaceNulls)
        self.assertEqual(
            replace_null_transform.replace_map,
            {measure_column: AnalyticsDefault.DECIMAL},
        )
        replace_nan_transform = transforms[2]
        self.assertIsInstance(replace_nan_transform, ReplaceNaNs)
        assert isinstance(replace_nan_transform, ReplaceNaNs)
        self.assertEqual(replace_nan_transform.output_domain, expected_input_domain)
        self.assertEqual(
            replace_nan_transform.replace_map,
            {measure_column: AnalyticsDefault.DECIMAL},
        )

    @parameterized.expand([(PureDP(), "has_infs"), (RhoZCDP(), "has_infs")])
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_sum_measurement",
        autospec=True,
    )
    def test_sum_replace_infs(
        self,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column,
        mock_create_sum,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test GroupByBoundedSum replacing a nan and/or a null value."""
        self.visitor.output_measure = output_measure
        query = GroupByBoundedSum(
            child=self.base_query,
            measure_column=measure_column,
            mechanism=SumMechanism.LAPLACE,
            output_column="sum",
            low=-100.0,
            high=100.0,
            groupby_keys=self.keyset,
        )
        expected_input_domain = self._expected_input_domain(measure_column)
        mock_measurement.input_domain = expected_input_domain
        mock_measurement.input_metric = self.visitor.input_metric["private"]
        mock_measurement.privacy_function.return_value = self.visitor.budget
        mock_create_sum.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_sum(query)
        self._check_sum_measurement(
            query=query,
            mock_create_sum=mock_create_sum,
            mock_groupby=mock_groupby,
            mock_measurement=mock_measurement,
            got_measurement=measurement,
            expected_mechanism=NoiseMechanism.LAPLACE,
        )
        assert isinstance(measurement, ChainTM)
        self.assertIsInstance(measurement.transformation, ChainTT)
        assert isinstance(measurement.transformation, ChainTT)
        replace_infs_transform = measurement.transformation.transformation2
        self.assertIsInstance(replace_infs_transform, ReplaceInfs)
        assert isinstance(replace_infs_transform, ReplaceInfs)
        self.assertEqual(
            replace_infs_transform.replace_map, {measure_column: (-100.0, 100.0)}
        )
        self.assertEqual(replace_infs_transform.output_domain, expected_input_domain)

    @parameterized.expand(
        [
            (PureDP(), "null_and_inf"),
            (RhoZCDP(), "null_and_inf"),
            (PureDP(), "nan_and_inf"),
            (RhoZCDP(), "nan_and_inf"),
            (PureDP(), "R"),
            (RhoZCDP(), "R"),
        ]
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.Measurement",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler._measurement_visitor.GroupBy",
        autospec=True,
    )
    @patch(
        "tmlt.analytics._query_expr_compiler."
        "_measurement_visitor.create_sum_measurement",
        autospec=True,
    )
    def test_sum_replace_all(
        self,
        output_measure: Union[PureDP, RhoZCDP],
        measure_column,
        mock_create_sum,
        mock_groupby,
        mock_measurement,
    ) -> None:
        """Test GroupByBoundedSum replacing a nan and/or a null value."""
        self.visitor.output_measure = output_measure
        query = GroupByBoundedSum(
            child=self.base_query,
            measure_column=measure_column,
            mechanism=SumMechanism.LAPLACE,
            output_column="sum",
            low=-100.0,
            high=100.0,
            groupby_keys=self.keyset,
        )
        expected_input_domain = self._expected_input_domain(measure_column)
        mock_measurement.input_domain = expected_input_domain
        mock_measurement.input_metric = self.visitor.input_metric["private"]
        mock_measurement.privacy_function.return_value = self.visitor.budget
        mock_create_sum.return_value = mock_measurement

        measurement = self.visitor.visit_groupby_bounded_sum(query)
        self._check_sum_measurement(
            query=query,
            mock_create_sum=mock_create_sum,
            mock_groupby=mock_groupby,
            mock_measurement=mock_measurement,
            got_measurement=measurement,
            expected_mechanism=NoiseMechanism.LAPLACE,
        )
        assert isinstance(measurement, ChainTM)

        self.assertIsInstance(measurement.transformation, ChainTT)
        assert isinstance(measurement.transformation, ChainTT)
        transforms = chain_to_list(measurement.transformation)
        self.assertEqual(len(transforms), 4)
        # The first transform is a GetValue;
        # _check_sum_measurement already checked it for us
        replace_null_transform = transforms[1]
        self.assertIsInstance(replace_null_transform, ReplaceNulls)
        assert isinstance(replace_null_transform, ReplaceNulls)
        self.assertEqual(
            replace_null_transform.replace_map,
            {measure_column: AnalyticsDefault.DECIMAL},
        )
        replace_nan_transform = transforms[2]
        self.assertIsInstance(replace_nan_transform, ReplaceNaNs)
        assert isinstance(replace_nan_transform, ReplaceNaNs)
        self.assertEqual(
            replace_nan_transform.replace_map,
            {measure_column: AnalyticsDefault.DECIMAL},
        )
        replace_inf_transform = transforms[3]
        self.assertIsInstance(replace_inf_transform, ReplaceInfs)
        assert isinstance(replace_inf_transform, ReplaceInfs)
        self.assertEqual(replace_inf_transform.output_domain, expected_input_domain)
        self.assertEqual(
            replace_inf_transform.replace_map, {measure_column: (-100.0, 100.0)}
        )

    def _recursive_visit(self, transformation: Transformation) -> str:
        if not isinstance(transformation, ChainTT):
            return str(transformation)
        left = self._recursive_visit(transformation.transformation1)
        right = self._recursive_visit(transformation.transformation2)
        left = "\t\n".join(left.split("\n"))
        right = "\t\n".join(right.split("\n"))
        return f"Left: {left}\nRight:{right}"
