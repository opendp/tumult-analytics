"""Unit tests for Session."""

# <placeholder: boilerplate>

import re
from typing import Any, List, Tuple, Type, Union
from unittest.mock import ANY, Mock, patch

import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from typeguard import check_type

from tmlt.analytics._query_expr_compiler import QueryExprCompiler
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import PrivateSource, QueryExpr
from tmlt.analytics.session import Session
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountant,
    PrivacyAccountantState,
    SequentialComposition,
    SequentialQueryable,
)
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import (
    DictMetric,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.dictionary import GetValue
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import PySparkTest, create_mock_transformation


def _privacy_budget_to_exact_number(
    budget: Union[PureDPBudget, RhoZCDPBudget]
) -> ExactNumber:
    """Turn a privacy budget into an Exact Number."""
    if isinstance(budget, PureDPBudget):
        return ExactNumber(budget.epsilon)
    if isinstance(budget, RhoZCDPBudget):
        return ExactNumber(budget.rho)
    raise AssertionError("This should be unreachable")


class TestSession(PySparkTest):
    """Tests for :class:`~tmlt.analytics.session.Session`."""

    def setUp(self) -> None:
        """Set up test data."""
        self.sdf = self.spark.createDataFrame(
            pd.DataFrame(
                [["0", 0, 0], ["0", 0, 1], ["0", 1, 2], ["1", 0, 3]],
                columns=["A", "B", "X"],
            )
        )
        self.sdf_col_types = Schema(
            {
                "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                "X": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            }
        )
        self.validated_col_types = {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
            "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
            "X": ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
        }
        self.sdf_input_domain = DictDomain(
            {
                "private": SparkDataFrameDomain(
                    analytics_to_spark_columns_descriptor(Schema(self.sdf_col_types))
                )
            }
        )
        self.validated_input_domain = DictDomain(
            {
                "private": SparkDataFrameDomain(
                    analytics_to_spark_columns_descriptor(
                        Schema(self.validated_col_types)
                    )
                )
            }
        )

        self.join_df = self.spark.createDataFrame(
            pd.DataFrame([["0", 0], ["0", 1], ["1", 1], ["1", 2]], columns=["A", "A+B"])
        )

        self.private_schema = {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "X": ColumnDescriptor(ColumnType.INTEGER),
        }
        self.public_schema = {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "A+B": ColumnDescriptor(ColumnType.INTEGER),
        }

    @parameterized.expand(
        [
            (ExactNumber(10), PureDP(), PureDPBudget(10)),
            (ExactNumber(10), RhoZCDP(), RhoZCDPBudget(10)),
        ]
    )
    @patch("tmlt.analytics.session.QueryExprCompiler", autospec=True)
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_remaining_privacy_budget(
        self,
        budget_value,
        output_measure,
        expected_budget,
        mock_accountant,
        mock_compiler,
    ):
        """Test that remaining_privacy_budget returns the right type of budget."""
        self._setup_accountant(
            mock_accountant, privacy_budget=budget_value, d_in=ExactNumber(1)
        )
        mock_accountant.output_measure = output_measure

        mock_compiler.output_measure = output_measure

        session = Session(mock_accountant, {}, mock_compiler)
        privacy_budget = session.remaining_privacy_budget
        self.assertEqual(type(expected_budget), type(privacy_budget))
        if isinstance(expected_budget, PureDPBudget):
            self.assertEqual(budget_value, ExactNumber(expected_budget.epsilon))
        elif isinstance(expected_budget, RhoZCDPBudget):
            self.assertEqual(budget_value, ExactNumber(expected_budget.rho))
        else:
            self.fail(f"Unexpected budget type: found {type(expected_budget)}")

    @parameterized.expand(
        [
            (PureDPBudget(float("inf")), PureDP(), True),
            (PureDPBudget(float("inf")), PureDP(), False),
        ]
    )
    @patch("tmlt.analytics.session.SequentialComposition", autospec=True)
    @patch.object(Session, "__init__", autospec=True, return_value=None)
    def test_from_dataframe(
        self,
        budget: Union[PureDPBudget, RhoZCDPBudget],
        expected_output_measure: Union[PureDP, RhoZCDP],
        validate: bool,
        mock_session_init,
        mock_composition_init,
    ):
        """Tests that :func:`Session.from_dataframe` works correctly."""
        mock_composition_init.return_value = Mock(
            spec_set=SequentialComposition,
            return_value=Mock(
                spec_set=SequentialComposition, output_measure=expected_output_measure
            ),
        )
        mock_composition_init.return_value.privacy_budget = (
            _privacy_budget_to_exact_number(budget)
        )
        expected_input_domain = self.sdf_input_domain
        if validate:
            expected_input_domain = self.validated_input_domain

        Session.from_dataframe(
            privacy_budget=budget,
            source_id="private",
            dataframe=self.sdf,
            stability=23,
            validate=validate,
        )

        mock_composition_init.assert_called_with(
            input_domain=expected_input_domain,
            input_metric=DictMetric({"private": SymmetricDifference()}),
            d_in={"private": 23},
            privacy_budget=sp.oo,
            output_measure=expected_output_measure,
        )
        mock_composition_init.return_value.assert_called()
        self.assert_frame_equal_with_sort(
            mock_composition_init.return_value.mock_calls[0][1][0][
                "private"
            ].toPandas(),
            self.sdf.toPandas(),
        )
        mock_session_init.assert_called_with(
            self=ANY, accountant=ANY, public_sources=dict(), compiler=ANY
        )

    @patch("tmlt.analytics.session.QueryExprCompiler")
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_add_public_dataframe(self, mock_accountant, mock_compiler):
        """Tests that :func:`add_public_dataframe` works correctly."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = self.sdf_input_domain
        mock_accountant.d_in = {"private": ExactNumber(1)}
        mock_compiler.output_measure = PureDP()
        session = Session(
            accountant=mock_accountant, public_sources=dict(), compiler=mock_compiler
        )
        session.add_public_dataframe(source_id="public", dataframe=self.join_df)
        assert "public" in session.public_source_dataframes
        self.assert_frame_equal_with_sort(
            session.public_source_dataframes["public"].toPandas(),
            self.join_df.toPandas(),
        )
        expected_schema = StructType(
            [
                StructField("A", StringType(), nullable=False),
                StructField("A+B", LongType(), nullable=False),
            ]
        )
        actual_schema = session.public_source_dataframes["public"].schema
        self.assertEqual(actual_schema, expected_schema)

    @parameterized.expand([(sp.Integer(1),), (sp.sqrt(sp.Integer(2)),)])
    @patch.object(QueryExprCompiler, "build_transformation", autospec=True)
    @patch(
        "tmlt.core.measurements.interactive_measurements.PrivacyAccountant",
        autospec=True,
    )
    def test_create_view(self, d_in, mock_accountant, mock_compiler_transform):
        """Creating views without caching works"""
        mock_accountant.output_measure = PureDP()
        # Use RootSumOfSquared since SymmetricDifference doesn't allow non-ints. Wrap
        # that in IfGroupedBy since RootSumOfSquared on its own is not valid in many
        # places in the framework.
        mock_accountant.input_metric = DictMetric(
            {"private": IfGroupedBy("A", RootSumOfSquared(SymmetricDifference()))}
        )
        mock_accountant.input_domain = self.sdf_input_domain
        mock_accountant.d_in = {"private": ExactNumber(d_in)}
        view_transformation = create_mock_transformation(
            input_domain=self.sdf_input_domain,
            input_metric=DictMetric(
                {"private": IfGroupedBy("A", RootSumOfSquared(SymmetricDifference()))}
            ),
            output_domain=self.sdf_input_domain["private"],
            output_metric=SymmetricDifference(),
            stability_function_implemented=True,
            stability_function_return_value=ExactNumber(13),
        )
        mock_compiler_transform.return_value = view_transformation

        session = Session(accountant=mock_accountant, public_sources=dict())
        session.create_view(
            query_expr=PrivateSource("private"),
            source_id="identity_transformation",
            cache=False,
        )

        mock_compiler_transform.assert_called_with(
            self=ANY,
            query=PrivateSource("private"),
            input_domain=mock_accountant.input_domain,
            input_metric=mock_accountant.input_metric,
            public_sources={},
            catalog=ANY,
        )

    @parameterized.expand([(sp.Integer(1),), (sp.sqrt(sp.Integer(2)),)])
    @patch.object(QueryExprCompiler, "__call__", autospec=True)
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_evaluate_puredp_session(self, d_in, mock_accountant, mock_compiler):
        """Tests that :func:`evaluate` calls the right things given a puredp session."""
        self._setup_accountant_and_compiler(d_in, mock_accountant, mock_compiler)
        mock_accountant.privacy_budget = ExactNumber(10)
        session = Session(accountant=mock_accountant, public_sources=dict())
        answer = session.evaluate(
            query_expr=PrivateSource("private"), privacy_budget=PureDPBudget(10)
        )
        self._assert_test_evaluate_correctness(
            session, mock_accountant, mock_compiler, 10
        )
        check_type("answer", answer, DataFrame)

    @parameterized.expand([(sp.Integer(1),), (sp.sqrt(sp.Integer(2)),)])
    @patch.object(QueryExprCompiler, "__call__", autospec=True)
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_evaluate_with_zero_budget(self, d_in, mock_accountant, mock_compiler):
        """Confirm that calling evaluate with a 'budget' of 0 fails."""
        self._setup_accountant_and_compiler(d_in, mock_accountant, mock_compiler)
        mock_accountant.privacy_budget = ExactNumber(10)
        session = Session(accountant=mock_accountant, public_sources=dict())
        with self.assertRaisesRegex(
            ValueError, "You need a non-zero privacy budget to evaluate a query."
        ):
            session.evaluate(
                query_expr=PrivateSource("private"), privacy_budget=PureDPBudget(0)
            )

        # set output measures to RhoZCDP
        mock_accountant.output_measure = RhoZCDP()
        mock_compiler.output_measure = RhoZCDP()
        with self.assertRaisesRegex(
            ValueError, "You need a non-zero privacy budget to evaluate a query."
        ):
            session.evaluate(
                query_expr=PrivateSource("private"), privacy_budget=RhoZCDPBudget(0)
            )

    @parameterized.expand([(sp.Integer(1),), (sp.sqrt(sp.Integer(2)),)])
    @patch.object(QueryExprCompiler, "__call__", autospec=True)
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_evaluate_zcdp_session_puredp_query(
        self, d_in, mock_accountant, mock_compiler
    ):
        """Confirm that using a puredp query on a zcdp accountant raises an error."""
        self._setup_accountant_and_compiler(d_in, mock_accountant, mock_compiler)
        mock_accountant.privacy_budget = ExactNumber(10)
        # Set the output measures manually
        mock_accountant.output_measure = RhoZCDP()
        mock_compiler.output_measure = RhoZCDP()
        session = Session(accountant=mock_accountant, public_sources=dict())
        with self.assertRaisesRegex(
            ValueError,
            "Your requested privacy budget type must match the type of the privacy"
            " budget your Session was created with.",
        ):
            session.evaluate(
                query_expr=PrivateSource("private"), privacy_budget=PureDPBudget(10)
            )

    @parameterized.expand([(sp.Integer(1),), (sp.sqrt(sp.Integer(2)),)])
    @patch.object(QueryExprCompiler, "__call__", autospec=True)
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_evaluate_puredp_session_zcdp_query(
        self, d_in, mock_accountant, mock_compiler
    ):
        """Confirm that using a zcdp query on a puredp accountant raises an error."""
        self._setup_accountant_and_compiler(d_in, mock_accountant, mock_compiler)
        mock_accountant.privacy_budget = ExactNumber(10)
        session = Session(accountant=mock_accountant, public_sources=dict())
        with self.assertRaisesRegex(
            ValueError,
            "Your requested privacy budget type must match the type of the privacy"
            " budget your Session was created with.",
        ):
            session.evaluate(
                query_expr=PrivateSource("private"), privacy_budget=RhoZCDPBudget(10)
            )

    @parameterized.expand([(sp.Integer(1),), (sp.sqrt(sp.Integer(2)),)])
    @patch.object(QueryExprCompiler, "__call__", autospec=True)
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_evaluate_zcdp_session(self, d_in, mock_accountant, mock_compiler):
        """Tests that :func:`evaluate` calls the right things given a zcdp session."""
        self._setup_accountant_and_compiler(d_in, mock_accountant, mock_compiler)
        mock_accountant.privacy_budget = ExactNumber(5)
        # Set the output measures manually
        mock_accountant.output_measure = RhoZCDP()
        mock_compiler.output_measure = RhoZCDP()
        session = Session(accountant=mock_accountant, public_sources=dict())
        answer = session.evaluate(
            query_expr=PrivateSource("private"), privacy_budget=RhoZCDPBudget(5)
        )
        self._assert_test_evaluate_correctness(
            session, mock_accountant, mock_compiler, 5
        )
        check_type("answer", answer, DataFrame)

    def _setup_accountant(
        self, mock_accountant, d_in=None, privacy_budget=None
    ) -> None:
        """Initialize only a mock accountant."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = self.sdf_input_domain
        if d_in is not None:
            mock_accountant.d_in = {"private": d_in}
        else:
            mock_accountant.d_in = {"private": ExactNumber(1)}
        if privacy_budget is not None:
            mock_accountant.privacy_budget = privacy_budget
        else:
            mock_accountant.privacy_budget = ExactNumber(10)

    def _setup_accountant_and_compiler(self, d_in, mock_accountant, mock_compiler):
        """Initialize the mocks for testing :func:`evaluate`."""
        mock_accountant.output_measure = PureDP()
        # Use RootSumOfSquared since SymmetricDifference doesn't allow non-ints. Wrap
        # that in IfGroupedBy since RootSumOFSquared on its own is not valid in many
        # places in the framework.
        mock_accountant.input_metric = DictMetric(
            {"private": IfGroupedBy("A", RootSumOfSquared(SymmetricDifference()))}
        )
        mock_accountant.input_domain = self.sdf_input_domain
        mock_accountant.d_in = {"private": d_in}
        # The accountant's measure method will return a list
        # containing 1 empty dataframe
        mock_accountant.measure.return_value = [
            self.spark.createDataFrame(
                self.spark.sparkContext.emptyRDD(), StructType([])
            )
        ]
        mock_compiler.output_measure = PureDP()
        mock_compiler.return_value = Mock(spec_set=Measurement)

    def _assert_test_evaluate_correctness(
        self, session, mock_accountant, mock_compiler, budget
    ):
        """Confirm that :func:`evaluate` worked correctly."""
        self.assertIn("private", session.private_sources)
        self.assertEqual(session.get_schema("private"), self.sdf_col_types)

        mock_compiler.assert_called_with(
            self=ANY,
            queries=[PrivateSource("private")],
            stability=mock_accountant.d_in,
            input_domain=mock_accountant.input_domain,
            input_metric=mock_accountant.input_metric,
            privacy_budget=sp.Integer(budget),
            public_sources={},
            catalog=ANY,
        )

        mock_accountant.measure.assert_called_with(
            mock_compiler.return_value, d_out=ExactNumber(budget)
        )

    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_partition_and_create(self, mock_accountant):
        """Tests that :func:`partition_and_create` calls the right things."""
        self._setup_accountant(mock_accountant, privacy_budget=ExactNumber(10))
        mock_accountant.split.return_value = [
            Mock(
                spec_set=PrivacyAccountant,
                input_metric=DictMetric({"part0": SymmetricDifference()}),
                input_domain=self.sdf_input_domain,
                output_measure=PureDP(),
            ),
            Mock(
                spec_set=PrivacyAccountant,
                input_metric=DictMetric({"part1": SymmetricDifference()}),
                input_domain=self.sdf_input_domain,
                output_measure=PureDP(),
            ),
        ]

        session = Session(accountant=mock_accountant, public_sources=dict())

        new_sessions = session.partition_and_create(
            source_id="private",
            privacy_budget=PureDPBudget(10),
            attr_name="A",
            splits={"part0": "0", "part1": "1"},
        )

        partition_query = mock_accountant.mock_calls[-1][1][0]
        self.assertIsInstance(partition_query, Transformation)
        assert isinstance(partition_query, Transformation)
        self.assertIsInstance(partition_query, ChainTT)
        assert isinstance(partition_query, ChainTT)

        self.assertIsInstance(partition_query.transformation1, GetValue)
        assert isinstance(partition_query.transformation1, GetValue)
        self.assertEqual(
            partition_query.transformation1.input_domain, mock_accountant.input_domain
        )
        self.assertEqual(
            partition_query.transformation1.input_metric, mock_accountant.input_metric
        )
        self.assertEqual(partition_query.transformation1.key, "private")
        self.assertIsInstance(partition_query.transformation2, PartitionByKeys)
        assert isinstance(partition_query.transformation2, PartitionByKeys)
        self.assertEqual(
            partition_query.transformation2.input_domain,
            self.sdf_input_domain["private"],
        )
        self.assertEqual(
            partition_query.transformation2.input_metric, SymmetricDifference()
        )
        self.assertEqual(
            partition_query.transformation2.output_metric, SumOf(SymmetricDifference())
        )
        self.assertEqual(partition_query.transformation2.keys, ["A"])
        self.assertEqual(partition_query.transformation2.list_values, [("0",), ("1",)])

        mock_accountant.split.assert_called_with(
            partition_query, privacy_budget=ExactNumber(10)
        )

        self.assertIsInstance(new_sessions, dict)
        assert isinstance(new_sessions, dict)
        for new_session_name, new_session in new_sessions.items():
            self.assertIsInstance(new_session_name, str)
            self.assertIsInstance(new_session, Session)

    def test_supported_spark_types(self):
        """Session works with supported Spark data types."""
        alltypes_sdf = self.spark.createDataFrame(
            pd.DataFrame(
                [[1.2, 3.4, 17, 42, "blah"]], columns=["A", "B", "C", "D", "E"]
            ),
            schema=StructType(
                [
                    StructField("A", FloatType()),
                    StructField("B", DoubleType()),
                    StructField("C", IntegerType()),
                    StructField("D", LongType()),
                    StructField("E", StringType()),
                ]
            ),
        )
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(1), source_id="private", dataframe=alltypes_sdf
        )
        session.add_public_dataframe(source_id="public", dataframe=alltypes_sdf)

        sum_a_query = QueryBuilder("private").sum("A", low=0, high=2)
        session.evaluate(sum_a_query, privacy_budget=PureDPBudget(1))

    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_stop(self, mock_accountant):
        """Test that after session.stop(), session returns the right error"""
        self._setup_accountant(mock_accountant)

        def retire_side_effect():
            mock_accountant.state = PrivacyAccountantState.RETIRED

        mock_accountant.retire.side_effect = retire_side_effect
        session = Session(accountant=mock_accountant, public_sources=dict())

        session.stop()

        with self.assertRaisesRegex(
            RuntimeError,
            "This session is no longer active, and no new queries can be performed",
        ):
            session.evaluate(PrivateSource(source_id="private"), PureDPBudget(1))

        with self.assertRaisesRegex(
            RuntimeError,
            "This session is no longer active, and no new queries can be performed",
        ):
            session.create_view(
                query_expr=PrivateSource(source_id="private"),
                source_id="new_view",
                cache=False,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "This session is no longer active, and no new queries can be performed",
        ):
            session.delete_view("private")

        with self.assertRaisesRegex(
            RuntimeError,
            "This session is no longer active, and no new queries can be performed",
        ):
            session.partition_and_create(
                "private",
                privacy_budget=PureDPBudget(1),
                attr_name="A",
                splits={"part0": 0, "part1": 1},
            )


class TestInvalidSession(PySparkTest):
    """Unit tests for invalid session."""

    spark: SparkSession

    def setUp(self) -> None:
        """Set up test data."""
        self.pdf = pd.DataFrame(
            [["0", 0, 0.0], ["0", 0, 1.0], ["0", 1, 2.0], ["1", 0, 3.0]],
            columns=["A", "B", "X"],
        )
        self.sdf = self.spark.createDataFrame(self.pdf)
        self.sdf_col_types = {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "X": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
        }
        self.sdf_input_domain = SparkDataFrameDomain(
            analytics_to_spark_columns_descriptor(Schema(self.sdf_col_types))
        )

        self.schema = {
            "A": ColumnDescriptor(ColumnType.VARCHAR),
            "B": ColumnDescriptor(ColumnType.INTEGER),
            "C": ColumnDescriptor(ColumnType.DECIMAL),
        }

    def _setup_accountant(self, mock_accountant) -> None:
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": sp.Integer(1)}

    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_compiler_initialization(self, mock_accountant):
        """session errors if compiler is not a QueryExprCompiler."""
        with self.assertRaisesRegex(
            TypeError,
            r"type of compiler must be one of "
            r"\(QueryExprCompiler, NoneType\); got list instead",
        ):
            self._setup_accountant(mock_accountant)
            Session(accountant=mock_accountant, public_sources=dict(), compiler=[])

    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_dataframe_initialization(self, mock_accountant):
        """session raises error on invalid dataframe type"""
        # Private
        with self.assertRaisesRegex(
            TypeError,
            'type of argument "dataframe" must be pyspark.sql.dataframe.DataFrame; '
            "got pandas.core.frame.DataFrame instead",
        ):
            Session.from_dataframe(
                privacy_budget=PureDPBudget(1), source_id="private", dataframe=self.pdf
            )
        # Public
        self._setup_accountant(mock_accountant)

        session = Session(accountant=mock_accountant, public_sources=dict())
        with self.assertRaisesRegex(
            TypeError,
            'type of argument "dataframe" must be pyspark.sql.dataframe.DataFrame; '
            "got pandas.core.frame.DataFrame instead",
        ):
            session.add_public_dataframe(source_id="public", dataframe=self.pdf)

    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_data_properties(self, mock_accountant):
        """session raises error on invalid data properties"""
        self._setup_accountant(mock_accountant)
        session = Session(
            accountant=mock_accountant,
            public_sources={
                "public": self.spark.createDataFrame(
                    pd.DataFrame({"A": ["a1", "a2"], "B": [1, 2]})
                )
            },
        )

        # source_id not existent
        with self.assertRaises(KeyError):
            session.get_schema("view")
        with self.assertRaises(KeyError):
            session.get_grouping_column("view")

        # public source_id doesn't have a grouping_column
        source_id = "public"
        with self.assertRaisesRegex(
            ValueError,
            f"'{source_id}' does not have a grouping column, "
            "because it is not a private table.",
        ):
            session.get_grouping_column(source_id)

    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_column_name(self, mock_accountant) -> None:
        """Builder raises an error if a column is named "".

        Columns named "" (empty string) are not allowed.
        """
        self._setup_accountant(mock_accountant)
        with self.assertRaisesRegex(
            ValueError,
            re.escape('This DataFrame contains a column named "" (the empty string)'),
        ):
            Session.from_dataframe(
                privacy_budget=PureDPBudget(1),
                source_id="private",
                dataframe=self.spark.createDataFrame(
                    pd.DataFrame({"A": ["a0", "a1"], "": [0, 1]})
                ),
            )
        session = Session(accountant=mock_accountant, public_sources={})
        with self.assertRaisesRegex(
            ValueError,
            re.escape('This DataFrame contains a column named "" (the empty string)'),
        ):
            session.add_public_dataframe(
                source_id="my_public_data",
                dataframe=self.spark.createDataFrame(
                    pd.DataFrame(
                        [["0", 0, 0.0], ["1", 1, 1.1], ["2", 2, 2.2]],
                        columns=["A", "B", ""],
                    )
                ),
            )

    @parameterized.expand(
        [
            (2, TypeError, 'type of argument "source_id" must be str; got int instead'),
            ("@str", ValueError, "source_id must be a valid Python identifier."),
        ]
    )
    @patch("tmlt.analytics._query_expr_compiler.QueryExprCompiler")
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_source_id(
        self,
        source_id: str,
        exception_type: Type[Exception],
        expected_error_msg: str,
        mock_accountant,
        mock_compiler,
    ):
        """session raises error on invalid source_id."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": sp.Integer(1)}
        mock_compiler.output_measure = PureDP()

        #### from spark dataframe ####
        # Private
        with self.assertRaisesRegex(exception_type, expected_error_msg):
            Session.from_dataframe(
                privacy_budget=PureDPBudget(1), source_id=source_id, dataframe=self.sdf
            )
        # Public
        session = Session(
            accountant=mock_accountant, public_sources=dict(), compiler=mock_compiler
        )
        with self.assertRaisesRegex(exception_type, expected_error_msg):
            session.add_public_dataframe(source_id, dataframe=self.sdf)
        # create_view
        with self.assertRaisesRegex(exception_type, expected_error_msg):
            session.create_view(PrivateSource("private"), source_id, cache=False)

    @patch("tmlt.analytics._query_expr_compiler.QueryExprCompiler")
    @patch("tmlt.analytics.session.PrivacyAccountant")
    def test_invalid_public_source(self, mock_accountant, mock_compiler):
        """Session raises an error adding a public source with duplicate source_id."""
        mock_accountant.launch().output_measure = PureDP()
        mock_accountant.launch().input_metric = DictMetric(
            {"private": SymmetricDifference()}
        )
        mock_accountant.launch().input_domain = DictDomain(
            {"private": self.sdf_input_domain}
        )
        mock_accountant.launch().d_in = {"private": sp.Integer(1)}
        mock_compiler.output_measure = PureDP()

        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(1), source_id="private", dataframe=self.sdf
        )

        # This should work
        session.add_public_dataframe("public_df", dataframe=self.sdf)

        # But this should not
        with self.assertRaisesRegex(
            ValueError,
            "This session already has a public source with the source_id public_df",
        ):
            session.add_public_dataframe("public_df", dataframe=self.sdf)

    @parameterized.expand([(["filter private A == 0"],), ([PrivateSource("private")],)])
    @patch("tmlt.analytics.session.QueryExprCompiler")
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_queries_evaluate(
        self, query_expr: Any, mock_accountant, mock_compiler
    ):
        """evaluate raises error on invalid queries."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": sp.Integer(1)}
        mock_compiler.output_measure = PureDP()

        session = Session(
            accountant=mock_accountant, public_sources=dict(), compiler=mock_compiler
        )
        with self.assertRaisesRegex(
            TypeError,
            "type of query_expr must be tmlt.analytics.query_expr.QueryExpr; got list"
            " instead",
        ):
            session.evaluate(query_expr, privacy_budget=PureDPBudget(float("inf")))

    @parameterized.expand(
        [
            (
                "filter private A == 0",
                TypeError,
                'type of argument "query_expr" must be one of '
                r"\(QueryExpr, QueryBuilder\); got str instead",
            )
        ]
    )
    @patch("tmlt.analytics.session.QueryExprCompiler")
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_queries_create(
        self,
        query_expr: QueryExpr,
        exception_type: Type[Exception],
        expected_error_msg: str,
        mock_accountant,
        mock_compiler,
    ):
        """create functions raise error on invalid input queries."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": sp.Integer(1)}
        mock_compiler.output_measure = PureDP()

        session = Session(
            accountant=mock_accountant, public_sources=dict(), compiler=mock_compiler
        )
        with self.assertRaisesRegex(exception_type, expected_error_msg):
            session.create_view(query_expr, source_id="view", cache=True)

    @patch("tmlt.analytics.session.QueryExprCompiler")
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_attr_name(self, mock_accountant, mock_compiler):
        """Tests that invalid column name for attr_name errors."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": sp.Integer(1)}
        mock_compiler.output_measure = PureDP()

        mock_compiler.build_transformation.return_value = (
            Mock(
                spec_set=Transformation,
                output_domain=self.sdf_input_domain,
                output_metric=SymmetricDifference(),
            ),
            sp.Integer(1),
        )

        session = Session(
            accountant=mock_accountant, public_sources=dict(), compiler=mock_compiler
        )

        expected_schema = spark_schema_to_analytics_columns(self.sdf.schema)
        # We expect a transformation that will disallow NaNs on floats and infs
        expected_schema["X"].allow_nan = False
        expected_schema["X"].allow_inf = False

        with self.assertRaisesRegex(
            KeyError,
            re.escape(
                "'T' not present in transformed dataframe's columns; "
                "schema of transformed dataframe is "
                f"{expected_schema}"
            ),
        ):
            session.partition_and_create(
                "private",
                privacy_budget=PureDPBudget(1),
                attr_name="T",
                splits={"private0": "0", "private1": "1"},
            )

    @patch("tmlt.analytics.session.QueryExprCompiler")
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_invalid_splits_name(self, mock_accountant, mock_compiler):
        """Tests that invalid splits name errors."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": sp.Integer(1)}
        mock_compiler.output_measure = PureDP()

        mock_compiler.build_transformation.return_value = (
            Mock(
                spec_set=Transformation,
                output_domain=self.sdf_input_domain,
                output_metric=SymmetricDifference(),
            ),
            sp.Integer(1),
        )

        session = Session(
            accountant=mock_accountant, public_sources=dict(), compiler=mock_compiler
        )

        with self.assertRaisesRegex(
            ValueError,
            "The string passed as split name must be a valid Python identifier",
        ):
            session.partition_and_create(
                "private",
                privacy_budget=PureDPBudget(1),
                attr_name="A",
                splits={" ": 0, "space present": 1, "2startsWithNumber": 2},
            )

    @patch("tmlt.analytics.session.QueryExprCompiler")
    @patch("tmlt.core.measurements.interactive_measurements.PrivacyAccountant")
    def test_splits_value_type(self, mock_accountant, mock_compiler):
        """Tests error when given invalid splits value type on partition."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric({"private": SymmetricDifference()})
        mock_accountant.input_domain = DictDomain({"private": self.sdf_input_domain})
        mock_accountant.d_in = {"private": sp.Integer(1)}
        mock_compiler.output_measure = PureDP()

        mock_compiler.build_transformation.return_value = (
            Mock(
                spec_set=Transformation,
                output_domain=self.sdf_input_domain,
                output_metric=SymmetricDifference(),
            ),
            sp.Integer(1),
        )

        session = Session(
            accountant=mock_accountant, public_sources=dict(), compiler=mock_compiler
        )

        with self.assertRaisesRegex(
            TypeError,
            "'A' column is of type 'StringType'; 'StringType' "
            "column not compatible with splits value type 'int'",
        ):
            session.partition_and_create(
                "private",
                privacy_budget=PureDPBudget(1),
                attr_name="A",
                splits={"private0": 0, "private1": 1},
            )

    def test_session_raises_error_on_unsupported_spark_column_types(self):
        """Session raises error when initialized with unsupported column types."""
        sdf = self.spark.createDataFrame(
            [], schema=StructType([StructField("A", BooleanType())])
        )
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported Spark data type: Tumult Analytics does not yet support the"
            " Spark data types for the following columns",
        ):
            Session.from_dataframe(
                privacy_budget=PureDPBudget(1), source_id="private", dataframe=sdf
            )

    @parameterized.expand([(None,), (float("nan"),)])
    def test_session_raises_error_on_dataframe_null_nans(self, value: Any):
        """Session raises error when initialized with nans/nulls in DataFrame."""
        sdf = self.spark.createDataFrame(
            [(value,)], schema=StructType([StructField("A", DoubleType())])
        )
        with self.assertRaisesRegex(
            ValueError, "This DataFrame contains a null or nan value"
        ):
            Session.from_dataframe(
                privacy_budget=PureDPBudget(1), source_id="private", dataframe=sdf
            )


class TestSessionBuilder(PySparkTest):
    """Tests for :class:`~tmlt.analytics.session.Session.Builder`."""

    def setUp(self):
        """Setup for tests."""
        df1 = self.spark.createDataFrame(
            [(1, 2, "A"), (3, 4, "B")], schema=["A", "B", "C"]
        )
        df2 = self.spark.createDataFrame(
            [("X", "A"), ("Y", "B"), ("Z", "B")], schema=["K", "C"]
        )

        self.dataframes = {"df1": df1, "df2": df2}

    @parameterized.expand(
        [
            (
                Session.Builder(),
                "Privacy budget must be specified.",
            ),  # No Privacy Budget
            (
                Session.Builder().with_privacy_budget(PureDPBudget(10)),
                "At least one private source must be provided.",
            ),  # No Private Sources
        ]
    )
    def test_invalid_build(self, builder: Session.Builder, error_msg: str):
        """Tests that builds raise relevant errors when builder is not configured."""
        with self.assertRaisesRegex(ValueError, error_msg):
            builder.build()

    def test_invalid_stability(self):
        """Tests that private source cannot be added with an invalid stability."""
        with self.assertRaisesRegex(ValueError, "Stability must be a positive integer"):
            Session.Builder().with_private_dataframe(
                source_id="df1", dataframe=self.dataframes["df1"], stability=0
            )
        with self.assertRaisesRegex(ValueError, "Stability must be a positive integer"):
            Session.Builder().with_private_dataframe(
                source_id="df1", dataframe=self.dataframes["df1"], stability=-1
            )

    @parameterized.expand([(PureDPBudget(1),), (RhoZCDPBudget(1),)])
    def test_invalid_to_add_budget_twice(self, initial_budget: PrivacyBudget):
        """Test that you can't call `with_privacy_budget()` twice."""
        builder = Session.Builder().with_privacy_budget(initial_budget)
        with self.assertRaisesRegex(
            ValueError, "This Builder already has a privacy budget"
        ):
            builder.with_privacy_budget(PureDPBudget(1))
        with self.assertRaisesRegex(
            ValueError, "This Builder already has a privacy budget"
        ):
            builder.with_privacy_budget(RhoZCDPBudget(1))

    def test_duplicate_source_id(self):
        """Tests that a repeated source id raises appropriate error."""
        builder = Session.Builder().with_private_dataframe(
            source_id="A", dataframe=self.dataframes["df1"], stability=1
        )
        with self.assertRaisesRegex(ValueError, "Duplicate source id: 'A'"):
            builder.with_private_dataframe(
                source_id="A", dataframe=self.dataframes["df2"], stability=2
            )
        with self.assertRaisesRegex(ValueError, "Duplicate source id: 'A'"):
            builder.with_public_dataframe(
                source_id="A", dataframe=self.dataframes["df2"]
            )

    @parameterized.expand(
        [
            (
                Session.Builder().with_privacy_budget(PureDPBudget(10)),
                sp.Integer(10),
                PureDP(),
                [("df1", 1)],
                [],
            ),
            (
                Session.Builder().with_privacy_budget(PureDPBudget(1.5)),
                sp.Rational("1.5"),
                PureDP(),
                [("df1", 1)],
                [],
            ),
            (
                Session.Builder().with_privacy_budget(RhoZCDPBudget(0)),
                sp.Integer(0),
                RhoZCDP(),
                [("df1", 4)],
                ["df2"],
            ),
            (
                Session.Builder().with_privacy_budget(RhoZCDPBudget(float("inf"))),
                sp.oo,
                RhoZCDP(),
                [("df1", 4), ("df2", 5)],
                [],
            ),
        ]
    )
    @patch.object(Session, "__init__", autospec=True, return_value=None)
    def test_build_works_correctly(
        self,
        builder: Session.Builder,
        expected_sympy_budget: sp.Expr,
        expected_output_measure: Measure,
        private_dataframes: List[Tuple[str, int]],
        public_dataframes: List[str],
        mock_session_init,
    ):
        """Tests that building a Session works correctly."""

        # Set up the builder.
        expected_private_sources, expected_public_sources = dict(), dict()
        expected_stabilities = dict()
        for source_id, stability in private_dataframes:
            builder = builder.with_private_dataframe(
                source_id=source_id,
                dataframe=self.dataframes[source_id],
                stability=stability,
            )
            expected_private_sources[source_id] = self.dataframes[source_id]
            expected_stabilities[source_id] = stability

        for source_id in public_dataframes:
            builder = builder.with_public_dataframe(
                source_id=source_id, dataframe=self.dataframes[source_id]
            )
            expected_public_sources[source_id] = self.dataframes[source_id]

        # Build the session and verify that it worked.
        builder.build()

        session_init_kwargs = mock_session_init.call_args[1]
        assert all(
            key in session_init_kwargs
            for key in ["accountant", "public_sources", "compiler"]
        )
        accountant = session_init_kwargs["accountant"]
        assert isinstance(accountant, PrivacyAccountant)
        self.assertTrue(
            accountant.privacy_budget
            == expected_sympy_budget
            == accountant.privacy_budget
        )
        self.assertEqual(accountant.output_measure, expected_output_measure)
        for source_id in expected_private_sources:
            # pylint: disable=protected-access
            assert accountant._queryable is not None
            assert isinstance(accountant._queryable, SequentialQueryable)
            self.assert_frame_equal_with_sort(
                accountant._queryable._data[source_id].toPandas(),
                expected_private_sources[source_id].toPandas(),
            )
            # pylint: enable=protected-access
        self.assertEqual(accountant.d_in, expected_stabilities)

        public_sources = session_init_kwargs["public_sources"]
        self.assertEqual(public_sources.keys(), expected_public_sources.keys())
        for key in public_sources:
            self.assert_frame_equal_with_sort(
                public_sources[key].toPandas(), expected_public_sources[key].toPandas()
            )

        compiler = session_init_kwargs["compiler"]
        assert isinstance(compiler, QueryExprCompiler)
        self.assertEqual(compiler.output_measure, expected_output_measure)

    def test_builder_with_dataframe_nonnullable(self):
        """with_dataframe methods mark all columns in schema nonnullable."""
        builder = Session.Builder()
        builder = builder.with_private_dataframe(
            source_id="private_df",
            dataframe=self.spark.createDataFrame(
                [(1,)], schema=StructType([StructField("A", LongType(), nullable=True)])
            ),
        )
        builder = builder.with_public_dataframe(
            source_id="public_df",
            dataframe=self.spark.createDataFrame(
                [(1,)], schema=StructType([StructField("A", LongType(), nullable=True)])
            ),
        )
        actual_private_schema = (
            builder._private_sources[  # pylint: disable=protected-access
                "private_df"
            ].dataframe.schema
        )
        actual_public_schema = (
            builder._public_sources[  # pylint: disable=protected-access
                "public_df"
            ].schema
        )
        expected_schema = StructType([StructField("A", LongType(), nullable=False)])
        self.assertEqual(actual_private_schema, expected_schema)
        self.assertEqual(actual_public_schema, expected_schema)
