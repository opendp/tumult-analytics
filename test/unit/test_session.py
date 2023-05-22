"""Unit tests for Session."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=no-self-use, unidiomatic-typecheck

import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import ANY, Mock, patch

import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import DataFrame
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
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountant,
    PrivacyAccountantState,
    SequentialComposition,
    SequentialQueryable,
)
from tmlt.core.measures import ApproxDP, Measure, PureDP, RhoZCDP
from tmlt.core.metrics import AddRemoveKeys as CoreAddRemoveKeys
from tmlt.core.metrics import (
    DictMetric,
    IfGroupedBy,
    Metric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.exact_number import ExactNumber
from typeguard import check_type

from tmlt.analytics._neighboring_relation import (
    AddRemoveKeys,
    AddRemoveRows,
    AddRemoveRowsAcrossGroups,
    Conjunction,
    NeighboringRelation,
)
from tmlt.analytics._query_expr_compiler import QueryExprCompiler
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics._table_identifier import NamedTable, TableCollection
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics.constraints import MaxRowsPerID
from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)
from tmlt.analytics.protected_change import (
    AddMaxRows,
    AddMaxRowsInMaxGroups,
    AddOneRow,
    AddRowsWithID,
)
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.query_expr import PrivateSource, QueryExpr
from tmlt.analytics.session import Session

from ..conftest import assert_frame_equal_with_sort, create_mock_transformation


def _privacy_budget_to_exact_number(
    budget: Union[PureDPBudget, RhoZCDPBudget]
) -> ExactNumber:
    """Turn a privacy budget into an Exact Number."""
    if isinstance(budget, (PureDPBudget, RhoZCDPBudget)):
        return budget.value
    raise AssertionError("This should be unreachable")


@pytest.fixture(name="test_data", scope="class")
def setup_test_data(spark, request) -> None:
    """Set up test data."""
    sdf = spark.createDataFrame(
        pd.DataFrame(
            [["0", 0, 0], ["0", 0, 1], ["0", 1, 2], ["1", 0, 3]],
            columns=["A", "B", "X"],
        )
    )
    request.cls.sdf = sdf
    sdf_col_types = Schema(
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "X": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        }
    )
    request.cls.sdf_col_types = sdf_col_types

    sdf_input_domain = DictDomain(
        {
            NamedTable("private"): SparkDataFrameDomain(
                analytics_to_spark_columns_descriptor(Schema(sdf_col_types))
            )
        }
    )
    request.cls.sdf_input_domain = sdf_input_domain

    join_df = spark.createDataFrame(
        pd.DataFrame([["0", 0], ["0", 1], ["1", 1], ["1", 2]], columns=["A", "A+B"])
    )

    request.cls.join_df = join_df

    join_df_col_types = Schema(
        {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
            "A+B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        }
    )
    request.cls.join_df_col_types = join_df_col_types

    join_df_input_domain = DictDomain(
        {
            NamedTable("join_private"): SparkDataFrameDomain(
                analytics_to_spark_columns_descriptor(Schema(join_df_col_types))
            )
        }
    )
    request.cls.join_df_input_domain = join_df_input_domain

    private_schema = {
        "A": ColumnDescriptor(ColumnType.VARCHAR),
        "B": ColumnDescriptor(ColumnType.INTEGER),
        "X": ColumnDescriptor(ColumnType.INTEGER),
    }
    request.cls.private_schema = private_schema

    public_schema = {
        "A": ColumnDescriptor(ColumnType.VARCHAR),
        "A+B": ColumnDescriptor(ColumnType.INTEGER),
    }

    request.cls.public_schema = public_schema

    combined_input_domain = DictDomain(
        {
            NamedTable("private"): SparkDataFrameDomain(
                analytics_to_spark_columns_descriptor(Schema(sdf_col_types))
            ),
            NamedTable("join_private"): SparkDataFrameDomain(
                analytics_to_spark_columns_descriptor(Schema(join_df_col_types))
            ),
        }
    )
    request.cls.combined_input_domain = combined_input_domain


@pytest.mark.usefixtures("test_data")
class TestSession:
    """Tests for :class:`~tmlt.analytics.session.Session`."""

    sdf: DataFrame
    sdf_col_types: Schema
    sdf_input_domain: DictDomain
    join_df: DataFrame
    join_col_types: Schema
    join_input_domain: DictDomain
    private_schema: Dict[str, ColumnDescriptor]
    public_schema: Dict[str, ColumnDescriptor]
    combined_input_domain: DictDomain

    @pytest.mark.parametrize(
        "budget_value,output_measure,expected_budget",
        [
            pytest.param(ExactNumber(10), PureDP(), PureDPBudget(10), id="puredp"),
            pytest.param(ExactNumber(10), RhoZCDP(), RhoZCDPBudget(10), id="zcdp"),
        ],
    )
    def test_remaining_privacy_budget(
        self, budget_value, output_measure, expected_budget
    ):
        """Test that remaining_privacy_budget returns the right type of budget."""
        with patch(
            "tmlt.analytics.session.QueryExprCompiler", autospec=True
        ) as mock_compiler, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(
                mock_accountant, privacy_budget=budget_value, d_in=ExactNumber(1)
            )
            mock_accountant.output_measure = output_measure

            mock_compiler.output_measure = output_measure

            session = Session(mock_accountant, {}, mock_compiler)
            privacy_budget = session.remaining_privacy_budget
            assert type(expected_budget) == type(privacy_budget)
            if isinstance(expected_budget, PureDPBudget):
                assert budget_value == ExactNumber(expected_budget.epsilon)
            elif isinstance(expected_budget, RhoZCDPBudget):
                assert budget_value == ExactNumber(expected_budget.rho)
            else:
                raise RuntimeError(
                    f"Unexpected budget type: found {type(expected_budget)}"
                )

    @pytest.mark.parametrize(
        "budget,expected_output_measure,expected_metric,from_dataframe_args",
        [
            pytest.param(
                PureDPBudget(float("inf")),
                PureDP(),
                SymmetricDifference(),
                {"stability": 21, "grouping_column": None, "protected_change": None},
                id="puredp-stability",
            ),
            pytest.param(
                PureDPBudget(float("inf")),
                PureDP(),
                SymmetricDifference(),
                {
                    "stability": None,
                    "grouping_column": None,
                    "protected_change": AddMaxRows(21),
                },
                id="puredp-protected_change",
            ),
            pytest.param(
                PureDPBudget(float("inf")),
                PureDP(),
                IfGroupedBy("X", SumOf(SymmetricDifference())),
                {"stability": 21, "grouping_column": "X", "protected_change": None},
                id="puredp-grouped-stability",
            ),
            pytest.param(
                RhoZCDPBudget(float("inf")),
                RhoZCDP(),
                IfGroupedBy("X", RootSumOfSquared(SymmetricDifference())),
                {"stability": 21, "grouping_column": "X", "protected_change": None},
                id="zcdp-grouped-stability",
            ),
            pytest.param(
                PureDPBudget(float("inf")),
                PureDP(),
                IfGroupedBy("X", SumOf(SymmetricDifference())),
                {
                    "stability": None,
                    "grouping_column": None,
                    "protected_change": AddMaxRowsInMaxGroups("X", 3, 7),
                },
                id="puredp-grouped-protected_change",
            ),
            pytest.param(
                RhoZCDPBudget(float("inf")),
                RhoZCDP(),
                IfGroupedBy("X", RootSumOfSquared(SymmetricDifference())),
                {
                    "stability": None,
                    "grouping_column": None,
                    "protected_change": AddMaxRowsInMaxGroups("X", 9, 7),
                },
                id="zcdp-grouped-protected_change",
            ),
        ],
    )
    def test_from_dataframe(
        self,
        budget: Union[PureDPBudget, RhoZCDPBudget],
        expected_output_measure: Union[PureDP, RhoZCDP],
        expected_metric: Metric,
        from_dataframe_args: Dict,
    ):
        """Tests that :func:`Session.from_dataframe` works with a grouping column."""
        with patch(
            "tmlt.analytics.session.SequentialComposition", autospec=True
        ) as mock_composition_init, patch.object(
            Session, "__init__", autospec=True, return_value=None
        ) as mock_session_init:
            mock_composition_init.return_value = Mock(
                spec_set=SequentialComposition,
                return_value=Mock(spec_set=SequentialComposition),
            )
            mock_composition_init.return_value.privacy_budget = (
                _privacy_budget_to_exact_number(budget)
            )
            mock_composition_init.return_value.d_in = {NamedTable("private"): 21}
            mock_composition_init.return_value.output_measure = expected_output_measure

            Session.from_dataframe(
                privacy_budget=budget,
                source_id="private",
                dataframe=self.sdf,
                **from_dataframe_args,
            )

            mock_composition_init.assert_called_with(
                input_domain=self.sdf_input_domain,
                input_metric=DictMetric({NamedTable("private"): expected_metric}),
                d_in={NamedTable("private"): 21},
                privacy_budget=sp.oo,
                output_measure=expected_output_measure,
            )
            mock_composition_init.return_value.assert_called()
            assert_frame_equal_with_sort(
                mock_composition_init.return_value.mock_calls[0][1][0][
                    NamedTable("private")
                ].toPandas(),
                self.sdf.toPandas(),
            )
            mock_session_init.assert_called_with(
                self=ANY, accountant=ANY, public_sources={}, compiler=ANY
            )

    @pytest.mark.parametrize(
        "budget,expected_output_measure,expected_metric,from_dataframe_args",
        [
            pytest.param(
                PureDPBudget(float("inf")),
                PureDP(),
                DictMetric(
                    {
                        TableCollection("default_id_space"): CoreAddRemoveKeys(
                            {NamedTable("private"): "A"}
                        )
                    }
                ),
                {
                    "stability": None,
                    "grouping_column": None,
                    "protected_change": AddRowsWithID("A"),
                },
                id="puredp-addrowswithID-protected_change",
            )
        ],
    )
    def test_from_dataframe_add_remove_keys(
        self,
        budget: Union[PureDPBudget, RhoZCDPBudget],
        expected_output_measure: Union[PureDP, RhoZCDP],
        expected_metric: Metric,
        from_dataframe_args: Dict,
    ) -> None:
        """Test Session.from_dataframe for AddRemoveKeys.

        AddRemoveKeys doesn't create a DictMetric because it's special.
        """
        with patch(
            "tmlt.analytics.session.SequentialComposition", autospec=True
        ) as mock_composition_init, patch.object(
            Session, "__init__", autospec=True, return_value=None
        ) as mock_session_init:
            mock_composition_init.return_value = Mock(
                spec_set=SequentialComposition,
                return_value=Mock(spec_set=SequentialComposition),
            )
            mock_composition_init.return_value.privacy_budget = (
                _privacy_budget_to_exact_number(budget)
            )
            expected_d_in = {TableCollection("default_id_space"): 1}
            mock_composition_init.return_value.d_in = expected_d_in
            mock_composition_init.return_value.output_measure = expected_output_measure

            Session.from_dataframe(
                privacy_budget=budget,
                source_id="private",
                dataframe=self.sdf,
                **from_dataframe_args,
            )

            expected_input_domain = DictDomain(
                {TableCollection("default_id_space"): self.sdf_input_domain}
            )

            mock_composition_init.assert_called_with(
                input_domain=expected_input_domain,
                input_metric=expected_metric,
                d_in=expected_d_in,
                privacy_budget=sp.oo,
                output_measure=expected_output_measure,
            )
            mock_composition_init.return_value.assert_called()
            assert_frame_equal_with_sort(
                mock_composition_init.return_value.mock_calls[0][1][0][
                    TableCollection("default_id_space")
                ][NamedTable("private")].toPandas(),
                self.sdf.toPandas(),
            )
            mock_session_init.assert_called_with(
                self=ANY, accountant=ANY, public_sources={}, compiler=ANY
            )

    @pytest.mark.parametrize(
        "budget,relation,expected_metric,expected_output_measure",
        [
            pytest.param(
                PureDPBudget(float("inf")),
                AddRemoveRows(table="private", n=6),
                DictMetric(
                    key_to_metric={NamedTable("private"): SymmetricDifference()}
                ),
                PureDP(),
                id="addremoverows_session",
            ),
            pytest.param(
                PureDPBudget(float("inf")),
                AddRemoveRowsAcrossGroups(
                    table="private", grouping_column="X", max_groups=3, per_group=2
                ),
                DictMetric(
                    key_to_metric={
                        NamedTable("private"): IfGroupedBy(
                            column="X", inner_metric=SumOf(SymmetricDifference())
                        )
                    }
                ),
                PureDP(),
                id="acrossgroupspuredp_session",
            ),
            pytest.param(
                RhoZCDPBudget(float("inf")),
                AddRemoveRowsAcrossGroups(
                    table="private", grouping_column="X", max_groups=4, per_group=3
                ),
                DictMetric(
                    key_to_metric={
                        NamedTable("private"): IfGroupedBy(
                            column="X",
                            inner_metric=RootSumOfSquared(SymmetricDifference()),
                        )
                    }
                ),
                RhoZCDP(),
                id="acrossgroupsrhozcdp_session",
            ),
        ],
    )
    def test_from_neighboring_relation_single(
        self,
        budget: Union[PureDPBudget, RhoZCDPBudget],
        relation: NeighboringRelation,
        expected_metric: DictMetric,
        expected_output_measure: Union[PureDP, RhoZCDP],
    ):
        """Tests that :func:`Session._from_neighboring_relation` works as expected
        with a single relation.
        """

        sess = Session._from_neighboring_relation(  # pylint: disable=protected-access
            privacy_budget=budget,
            private_sources={"private": self.sdf},
            relation=relation,
        )
        # pylint: disable=protected-access
        assert sess._input_domain == self.sdf_input_domain
        assert sess._input_metric == expected_metric
        assert sess._accountant.d_in == {NamedTable("private"): 6}
        assert sess._accountant.privacy_budget == sp.oo
        assert sess._accountant.output_measure == expected_output_measure
        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "budget,relation,expected_metric,expected_output_measure",
        [
            pytest.param(
                PureDPBudget(float("inf")),
                AddRemoveKeys("private", {"private": "A"}, max_keys=5),
                DictMetric(
                    {
                        TableCollection("private"): CoreAddRemoveKeys(
                            {NamedTable("private"): "A"}
                        )
                    }
                ),
                PureDP(),
                id="addremovekeys_puredp_session",
            )
        ],
    )
    def test_from_neighboring_relation_add_remove_keys(
        self,
        budget: Union[PureDPBudget, RhoZCDPBudget],
        relation: NeighboringRelation,
        expected_metric: DictMetric,
        expected_output_measure: Union[PureDP, RhoZCDP],
    ):
        """Tests that :func:`Session._from_neighboring_relation` works as expected
        with a single AddRemoveKeys relation.
        """

        sess = Session._from_neighboring_relation(  # pylint: disable=protected-access
            privacy_budget=budget,
            private_sources={"private": self.sdf},
            relation=relation,
        )
        # pylint: disable=protected-access
        assert sess._input_domain == DictDomain(
            {TableCollection("private"): self.sdf_input_domain}
        )
        assert sess._input_metric == expected_metric
        assert sess._accountant.d_in == {TableCollection("private"): 5}
        assert sess._accountant.privacy_budget == sp.oo
        assert sess._accountant.output_measure == expected_output_measure
        # pylint: enable=protected-access

    @pytest.mark.parametrize(
        "budget,relation,expected_metric,expected_output_measure",
        [
            pytest.param(
                PureDPBudget(float("inf")),
                Conjunction(
                    AddRemoveRows(table="private", n=6),
                    AddRemoveRowsAcrossGroups(
                        table="join_private",
                        grouping_column="A+B",
                        max_groups=3,
                        per_group=3,
                    ),
                ),
                DictMetric(
                    key_to_metric={
                        NamedTable("join_private"): IfGroupedBy(
                            column="A+B", inner_metric=SumOf(SymmetricDifference())
                        ),
                        NamedTable("private"): SymmetricDifference(),
                    }
                ),
                PureDP(),
                id="conjunction_session",
            )
        ],
    )
    def test_from_neighboring_relation_conjunction(
        self,
        budget: Union[PureDPBudget, RhoZCDPBudget],
        relation: NeighboringRelation,
        expected_metric: DictMetric,
        expected_output_measure: Union[PureDP, RhoZCDP],
    ):
        """Tests that :func:`Session._from_neighboring_relation` works as expected
        when passed a conjunction.
        """
        sess = Session._from_neighboring_relation(  # pylint: disable=protected-access
            privacy_budget=budget,
            private_sources={"private": self.sdf, "join_private": self.join_df},
            relation=relation,
        )

        # pylint: disable=protected-access
        assert sess._input_domain == self.combined_input_domain
        assert sess._input_metric == expected_metric
        assert sess._accountant.d_in == {
            NamedTable("private"): 6,
            NamedTable("join_private"): 9,
        }
        assert sess._accountant.privacy_budget == sp.oo
        assert sess._accountant.output_measure == expected_output_measure
        # pylint: enable=protected-access

    def test_add_public_dataframe(self):
        """Tests that :func:`add_public_dataframe` works correctly."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics.session.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = self.sdf_input_domain
            mock_accountant.d_in = {NamedTable("private"): ExactNumber(1)}
            mock_compiler.output_measure = PureDP()
            session = Session(
                accountant=mock_accountant, public_sources={}, compiler=mock_compiler
            )
            session.add_public_dataframe(source_id="public", dataframe=self.join_df)
            assert "public" in session.public_source_dataframes
            assert_frame_equal_with_sort(
                session.public_source_dataframes["public"].toPandas(),
                self.join_df.toPandas(),
            )
            expected_schema = self.join_df.schema
            actual_schema = session.public_source_dataframes["public"].schema
            assert actual_schema == expected_schema

    @pytest.mark.parametrize("d_in", [(sp.Integer(1)), (sp.sqrt(sp.Integer(2)))])
    def test_create_view(self, d_in):
        """Creating views without caching works."""
        with patch.object(
            QueryExprCompiler, "build_transformation", autospec=True
        ) as mock_compiler_transform, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant",
            autospec=True,
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            # Use RootSumOfSquared since SymmetricDifference
            # doesn't allow non-ints. Wrap
            # that in IfGroupedBy since RootSumOfSquared on its own is not valid in many
            # places in the framework.
            mock_accountant.input_metric = DictMetric(
                {
                    NamedTable("private"): IfGroupedBy(
                        "A", RootSumOfSquared(SymmetricDifference())
                    )
                }
            )
            mock_accountant.input_domain = self.sdf_input_domain
            mock_accountant.d_in = {NamedTable("private"): ExactNumber(d_in)}
            view_transformation = create_mock_transformation(
                input_domain=self.sdf_input_domain,
                input_metric=DictMetric(
                    {
                        NamedTable("private"): IfGroupedBy(
                            "A", RootSumOfSquared(SymmetricDifference())
                        )
                    }
                ),
                output_domain=self.sdf_input_domain,
                output_metric=DictMetric(
                    {
                        NamedTable("private"): IfGroupedBy(
                            "A", RootSumOfSquared(SymmetricDifference())
                        )
                    }
                ),
                stability_function_implemented=True,
                stability_function_return_value=ExactNumber(13),
            )
            mock_compiler_transform.return_value = (
                view_transformation,
                TableReference(path=[NamedTable("private")]),
                [],
            )
            session = Session(accountant=mock_accountant, public_sources={})
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
                table_constraints=ANY,
            )

    @pytest.mark.parametrize("d_in", [(sp.Integer(1)), (sp.sqrt(sp.Integer(2)))])
    def test_evaluate_puredp_session(self, spark, d_in):
        """Tests that :func:`evaluate` calls the right things given a puredp session."""
        with patch.object(
            QueryExprCompiler, "__call__", autospec=True
        ) as mock_compiler, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_and_compiler(
                spark, d_in, mock_accountant, mock_compiler
            )
            mock_accountant.privacy_budget = ExactNumber(10)
            session = Session(accountant=mock_accountant, public_sources={})
            answer = session.evaluate(
                query_expr=PrivateSource("private"), privacy_budget=PureDPBudget(10)
            )
            self._assert_test_evaluate_correctness(
                session, mock_accountant, mock_compiler, PureDPBudget(10)
            )
            check_type("answer", answer, DataFrame)

    @pytest.mark.parametrize("d_in", [(sp.Integer(1)), (sp.sqrt(sp.Integer(2)))])
    def test_evaluate_puredp_session_approxdp_query(self, spark, d_in):
        """Confirm that using an approxdp query on a puredp accountant raises an
        error."""
        with patch.object(
            QueryExprCompiler, "__call__", autospec=True
        ) as mock_compiler, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_and_compiler(
                spark, d_in, mock_accountant, mock_compiler
            )
            mock_accountant.privacy_budget = ExactNumber(10)
            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(
                ValueError,
                match=(
                    "Your requested privacy budget type must match the type of the"
                    " privacy budget your Session was created with."
                ),
            ):
                session.evaluate(
                    query_expr=PrivateSource("private"),
                    privacy_budget=ApproxDPBudget(10, 0.5),
                )

    @pytest.mark.parametrize("d_in", [(sp.Integer(1)), (sp.sqrt(sp.Integer(2)))])
    def test_evaluate_with_zero_budget(self, spark, d_in):
        """Confirm that calling evaluate with a 'budget' of 0 fails."""
        with patch.object(
            QueryExprCompiler, "__call__", autospec=True
        ) as mock_compiler, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_and_compiler(
                spark, d_in, mock_accountant, mock_compiler
            )
            mock_accountant.privacy_budget = ExactNumber(10)
            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(
                ValueError,
                match="You need a non-zero privacy budget to evaluate a query.",
            ):
                session.evaluate(
                    query_expr=PrivateSource("private"), privacy_budget=PureDPBudget(0)
                )

            # set output measures to RhoZCDP
            mock_accountant.output_measure = RhoZCDP()
            mock_compiler.output_measure = RhoZCDP()
            with pytest.raises(
                ValueError,
                match="You need a non-zero privacy budget to evaluate a query.",
            ):
                session.evaluate(
                    query_expr=PrivateSource("private"), privacy_budget=RhoZCDPBudget(0)
                )

    @pytest.mark.parametrize("d_in", [(sp.Integer(1)), (sp.sqrt(sp.Integer(2)))])
    def test_evaluate_zcdp_session_puredp_query(self, spark, d_in):
        """Confirm that using a puredp query on a zcdp accountant raises an error."""
        with patch.object(
            QueryExprCompiler, "__call__", autospec=True
        ) as mock_compiler, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_and_compiler(
                spark, d_in, mock_accountant, mock_compiler
            )
            mock_accountant.privacy_budget = ExactNumber(10)
            # Set the output measures manually
            mock_accountant.output_measure = RhoZCDP()
            mock_compiler.output_measure = RhoZCDP()
            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(
                ValueError,
                match=(
                    "Your requested privacy budget type must match the type of the"
                    " privacy budget your Session was created with."
                ),
            ):
                session.evaluate(
                    query_expr=PrivateSource("private"), privacy_budget=PureDPBudget(10)
                )

    @pytest.mark.parametrize("d_in", [(sp.Integer(1)), (sp.sqrt(sp.Integer(2)))])
    def test_evaluate_puredp_session_zcdp_query(self, spark, d_in):
        """Confirm that using a zcdp query on a puredp accountant raises an error."""
        with patch.object(
            QueryExprCompiler, "__call__", autospec=True
        ) as mock_compiler, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_and_compiler(
                spark, d_in, mock_accountant, mock_compiler
            )
            mock_accountant.privacy_budget = ExactNumber(10)
            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(
                ValueError,
                match=(
                    "Your requested privacy budget type must match the type of the"
                    " privacy budget your Session was created with."
                ),
            ):
                session.evaluate(
                    query_expr=PrivateSource("private"),
                    privacy_budget=RhoZCDPBudget(10),
                )

    @pytest.mark.parametrize("d_in", [(sp.Integer(1)), (sp.sqrt(sp.Integer(2)))])
    def test_evaluate_zcdp_session(self, spark, d_in):
        """Tests that :func:`evaluate` calls the right things given a zcdp session."""
        with patch.object(
            QueryExprCompiler, "__call__", autospec=True
        ) as mock_compiler, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_and_compiler(
                spark, d_in, mock_accountant, mock_compiler
            )
            mock_accountant.privacy_budget = ExactNumber(5)
            # Set the output measures manually
            mock_accountant.output_measure = RhoZCDP()
            mock_compiler.output_measure = RhoZCDP()
            session = Session(accountant=mock_accountant, public_sources={})
            answer = session.evaluate(
                query_expr=PrivateSource("private"), privacy_budget=RhoZCDPBudget(5)
            )
            self._assert_test_evaluate_correctness(
                session, mock_accountant, mock_compiler, RhoZCDPBudget(5)
            )
            check_type("answer", answer, DataFrame)

    def _setup_accountant(
        self, mock_accountant, d_in=None, privacy_budget=None
    ) -> None:
        """Initialize only a mock accountant."""
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric(
            {NamedTable("private"): SymmetricDifference()}
        )
        mock_accountant.input_domain = self.sdf_input_domain
        if d_in is not None:
            mock_accountant.d_in = {NamedTable("private"): d_in}
        else:
            mock_accountant.d_in = {NamedTable("private"): ExactNumber(1)}
        if privacy_budget is not None:
            mock_accountant.privacy_budget = privacy_budget
        else:
            mock_accountant.privacy_budget = ExactNumber(10)

    def _setup_accountant_with_id(self, mock_accountant, privacy_budget=None) -> None:
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric(
            key_to_metric={
                TableCollection(name="identifier_A"): CoreAddRemoveKeys(
                    df_to_key_column={NamedTable(name="private"): "A"}
                )
            }
        )
        mock_accountant.input_domain = DictDomain(
            key_to_domain={
                TableCollection(name="identifier_A"): DictDomain(
                    key_to_domain={
                        NamedTable(name="private"): SparkDataFrameDomain(
                            schema={
                                "A": SparkStringColumnDescriptor(allow_null=True),
                                "B": SparkIntegerColumnDescriptor(
                                    allow_null=True, size=64
                                ),
                                "X": SparkIntegerColumnDescriptor(
                                    allow_null=True, size=64
                                ),
                            }
                        )
                    }
                )
            }
        )
        mock_accountant.d_in = {TableCollection(name="identifier_A"): 1}
        if privacy_budget is not None:
            mock_accountant.privacy_budget = privacy_budget
        else:
            mock_accountant.privacy_budget = ExactNumber(10)

    def _setup_accountant_and_compiler(
        self, spark, d_in, mock_accountant, mock_compiler
    ):
        """Initialize the mocks for testing :func:`evaluate`."""
        mock_accountant.output_measure = PureDP()
        # Use RootSumOfSquared since SymmetricDifference doesn't allow non-ints. Wrap
        # that in IfGroupedBy since RootSumOFSquared on its own is not valid in many
        # places in the framework.
        mock_accountant.input_metric = DictMetric(
            {
                NamedTable("private"): IfGroupedBy(
                    "A", RootSumOfSquared(SymmetricDifference())
                )
            }
        )
        mock_accountant.input_domain = self.sdf_input_domain
        mock_accountant.d_in = {NamedTable("private"): d_in}
        # The accountant's measure method will return a list
        # containing 1 empty dataframe
        mock_accountant.measure.return_value = [
            spark.createDataFrame(spark.sparkContext.emptyRDD(), StructType([]))
        ]
        mock_compiler.output_measure = PureDP()
        mock_compiler.return_value = Mock(spec_set=Measurement)

    def _assert_test_evaluate_correctness(
        self, session, mock_accountant, mock_compiler, privacy_budget
    ):
        """Confirm that :func:`evaluate` worked correctly."""
        assert "private" in session.private_sources
        assert session.get_schema("private") == self.sdf_col_types

        mock_compiler.assert_called_with(
            self=ANY,
            queries=[PrivateSource("private")],
            stability=mock_accountant.d_in,
            input_domain=mock_accountant.input_domain,
            input_metric=mock_accountant.input_metric,
            privacy_budget=privacy_budget,
            public_sources={},
            catalog=ANY,
            table_constraints={t: [] for t in mock_accountant.d_in.keys()},
        )

        mock_accountant.measure.assert_called_with(
            mock_compiler.return_value, d_out=privacy_budget.value
        )

    def test_partition_and_create(self):
        """Tests that :func:`partition_and_create` calls the right things."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
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

            session = Session(accountant=mock_accountant, public_sources={})

        # Test that you need to provide splits
        with pytest.raises(
            ValueError,
            match=re.escape(
                "You must provide a dictionary mapping split names (new source_ids) to"
                " values on which to partition"
            ),
        ):
            session.partition_and_create(
                source_id="private", privacy_budget=PureDPBudget(10), column="A"
            )

        new_sessions = session.partition_and_create(
            source_id="private",
            privacy_budget=PureDPBudget(10),
            column="A",
            splits={"part0": "0", "part1": "1"},
        )

        partition_query = mock_accountant.mock_calls[-1][1][0]
        assert isinstance(partition_query, ChainTT)

        assert isinstance(partition_query.transformation2, PartitionByKeys)
        assert (
            partition_query.transformation2.input_domain
            == self.sdf_input_domain[NamedTable("private")]
        )
        assert partition_query.transformation2.input_metric == SymmetricDifference()
        assert partition_query.transformation2.output_metric == SumOf(
            SymmetricDifference()
        )
        assert partition_query.transformation2.keys == ["A"]
        assert partition_query.transformation2.list_values == [("0",), ("1",)]

        mock_accountant.split.assert_called_with(
            partition_query, privacy_budget=ExactNumber(10)
        )

        assert isinstance(new_sessions, dict)
        for new_session_name, new_session in new_sessions.items():
            assert isinstance(new_session_name, str)
            assert isinstance(new_session, Session)

    def test_describe(self, spark):
        """Test that :func:`_describe` works correctly."""
        with patch("builtins.print") as mock_print, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant, privacy_budget=ExactNumber(10))
            mock_accountant.state = PrivacyAccountantState.ACTIVE

            public_df_1 = spark.createDataFrame(
                pd.DataFrame([["blah", 1], ["blah", 2]], columns=["A", "B"])
            )
            public_df_2 = spark.createDataFrame(
                pd.DataFrame(
                    {
                        "X": [1.1, 2.2, 3.3],
                        "very_long_column_name": ["blah", "blah", "blah"],
                    }
                )
            )
            session = Session(
                accountant=mock_accountant,
                public_sources={"public1": public_df_1, "public2": public_df_2},
            )
            # pylint: disable=line-too-long
            expected = f"""The session has a remaining privacy budget of {PureDPBudget(10)}.
The following private tables are available:
Table 'private' (no constraints):
\tColumns:
\t\t- 'A'  VARCHAR
\t\t- 'B'  INTEGER
\t\t- 'X'  INTEGER
The following public tables are available:
Public table 'public1':
\tColumns:
\t\t- 'A'  VARCHAR
\t\t- 'B'  INTEGER
Public table 'public2':
\tColumns:
\t\t- 'X'                      DECIMAL
\t\t- 'very_long_column_name'  VARCHAR"""
            # pylint: enable=line-too-long
            session.describe()
            mock_print.assert_called_with(expected)

    def test_describe_with_constraints(self, spark):
        """Test :func:`_describe` with a table with constraints."""
        with patch("builtins.print") as mock_print, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant, privacy_budget=ExactNumber(10))
            mock_accountant.state = PrivacyAccountantState.ACTIVE

            public_df_1 = spark.createDataFrame(
                pd.DataFrame([["blah", 1], ["blah", 2]], columns=["A", "B"])
            )
            public_df_2 = spark.createDataFrame(
                pd.DataFrame(
                    {
                        "X": [1.1, 2.2, 3.3],
                        "very_long_column_name": ["blah", "blah", "blah"],
                    }
                )
            )

            session = Session(
                accountant=mock_accountant,
                public_sources={"public1": public_df_1, "public2": public_df_2},
            )

            # pylint: disable=protected-access
            session._table_constraints[NamedTable("private")] = [MaxRowsPerID(5)]
            # pylint: enable=protected-access
            # pylint: disable=line-too-long
            expected = f"""The session has a remaining privacy budget of {PureDPBudget(10)}.
The following private tables are available:
Table 'private':
\tColumns:
\t\t- 'A'  VARCHAR
\t\t- 'B'  INTEGER
\t\t- 'X'  INTEGER
\tConstraints:
\t\t- MaxRowsPerID(max=5)
The following public tables are available:
Public table 'public1':
\tColumns:
\t\t- 'A'  VARCHAR
\t\t- 'B'  INTEGER
Public table 'public2':
\tColumns:
\t\t- 'X'                      DECIMAL
\t\t- 'very_long_column_name'  VARCHAR"""
            session.describe()
            # pylint: enable=line-too-long
            mock_print.assert_called_with(expected)

    def test_describe_with_id_column(self, spark):
        """Test :func:`_describe` with a table with an ID column."""

        with patch("builtins.print") as mock_print, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_with_id(
                mock_accountant, privacy_budget=ExactNumber(10)
            )
            mock_accountant.state = PrivacyAccountantState.ACTIVE

            public_df_1 = spark.createDataFrame(
                pd.DataFrame([["blah", 1], ["blah", 2]], columns=["A", "B"])
            )
            public_df_2 = spark.createDataFrame(
                pd.DataFrame(
                    {
                        "X": [1.1, 2.2, 3.3],
                        "very_long_column_name": ["blah", "blah", "blah"],
                    }
                )
            )

            session = Session(
                accountant=mock_accountant,
                public_sources={"public1": public_df_1, "public2": public_df_2},
            )
            # pylint: disable=line-too-long
            expected = f"""The session has a remaining privacy budget of {PureDPBudget(10)}.
The following private tables are available:
Table 'private' (no constraints):
\tColumns:
\t\t- 'A'  VARCHAR, ID column (in ID space identifier_A)
\t\t- 'B'  INTEGER
\t\t- 'X'  INTEGER
The following public tables are available:
Public table 'public1':
\tColumns:
\t\t- 'A'  VARCHAR
\t\t- 'B'  INTEGER
Public table 'public2':
\tColumns:
\t\t- 'X'                      DECIMAL
\t\t- 'very_long_column_name'  VARCHAR"""
            # pylint: enable=line-too-long
            session.describe()
            mock_print.assert_called_with(expected)

    @pytest.mark.parametrize(
        "query,expected_output",
        [
            pytest.param(
                "private",
                """Columns:
\t- 'A'  VARCHAR
\t- 'B'  INTEGER
\t- 'X'  INTEGER""",
                id="table_name",
            ),
            pytest.param(
                PrivateSource("private"),
                """Columns:
\t- 'A'  VARCHAR
\t- 'B'  INTEGER
\t- 'X'  INTEGER""",
                id="private_source_query",
            ),
            pytest.param(
                QueryBuilder("private"),
                """Columns:
\t- 'A'  VARCHAR
\t- 'B'  INTEGER
\t- 'X'  INTEGER""",
                id="query_builder_private_source",
            ),
            pytest.param(
                QueryBuilder("private").drop_null_and_nan(["A", "B", "X"]),
                """Columns:
\t- 'A'  VARCHAR, not null
\t- 'B'  INTEGER, not null
\t- 'X'  INTEGER, not null""",
                id="query_builder_drop_null",
            ),
        ],
    )
    def test_describe_query(
        self, spark, query: Union[str, QueryBuilder, QueryExpr], expected_output: str
    ):
        """Test :func:`_describe` with a QueryExpr, QueryBuilder, or table name."""
        with patch("builtins.print") as mock_print, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant, privacy_budget=ExactNumber(10))
            mock_accountant.state = PrivacyAccountantState.ACTIVE

            public_df_1 = spark.createDataFrame(
                pd.DataFrame([["blah", 1], ["blah", 2]], columns=["A", "B"])
            )
            public_df_2 = spark.createDataFrame(pd.DataFrame({"X": [1.1, 2.2, 3.3]}))
            session = Session(
                accountant=mock_accountant,
                public_sources={"public1": public_df_1, "public2": public_df_2},
            )

            session.describe(query)
            mock_print.assert_called_with(expected_output)

    def test_supported_spark_types(self, spark):
        """Session works with supported Spark data types."""
        alltypes_sdf = spark.createDataFrame(
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
            privacy_budget=PureDPBudget(1),
            source_id="private",
            dataframe=alltypes_sdf,
            protected_change=AddOneRow(),
        )
        session.add_public_dataframe(source_id="public", dataframe=alltypes_sdf)

        sum_a_query = QueryBuilder("private").sum("A", low=0, high=2)
        session.evaluate(sum_a_query, privacy_budget=PureDPBudget(1))

    def test_stop(self):
        """Test that after session.stop(), session returns the right error"""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant)

            def retire_side_effect():
                mock_accountant.state = PrivacyAccountantState.RETIRED

            mock_accountant.retire.side_effect = retire_side_effect
            session = Session(accountant=mock_accountant, public_sources={})

            session.stop()

            with pytest.raises(
                RuntimeError,
                match=(
                    "This session is no longer active, and no new queries can be"
                    " performed"
                ),
            ):
                count_query = QueryBuilder("private").count()
                session.evaluate(count_query, PureDPBudget(1))

            with pytest.raises(
                RuntimeError,
                match=(
                    "This session is no longer active, and no new queries can be"
                    " performed"
                ),
            ):
                session.create_view(
                    query_expr=PrivateSource(source_id="private"),
                    source_id="new_view",
                    cache=False,
                )

            with pytest.raises(
                RuntimeError,
                match=(
                    "This session is no longer active, and no new queries can be"
                    " performed"
                ),
            ):
                session.delete_view("private")

            with pytest.raises(
                RuntimeError,
                match=(
                    "This session is no longer active, and no new queries can be"
                    " performed"
                ),
            ):
                session.partition_and_create(
                    "private",
                    privacy_budget=PureDPBudget(1),
                    column="A",
                    splits={"part0": 0, "part1": 1},
                )


@pytest.fixture(name="test_data_invalid", scope="class")
def setup_invalid_session_data(spark, request) -> None:
    """Set up test data for invalid session tests."""
    pdf = pd.DataFrame(
        [["0", 0, 0.0], ["0", 0, 1.0], ["0", 1, 2.0], ["1", 0, 3.0]],
        columns=["A", "B", "X"],
    )
    request.cls.pdf = pdf

    sdf = spark.createDataFrame(pdf)
    request.cls.sdf = sdf
    sdf_col_types = {
        "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
        "B": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
        "X": ColumnDescriptor(ColumnType.DECIMAL, allow_null=True),
    }
    request.cls.sdf_col_types = sdf_col_types

    sdf_input_domain = SparkDataFrameDomain(
        analytics_to_spark_columns_descriptor(Schema(sdf_col_types))
    )
    request.cls.sdf_input_domain = sdf_input_domain

    schema = {
        "A": ColumnDescriptor(ColumnType.VARCHAR),
        "B": ColumnDescriptor(ColumnType.INTEGER),
        "C": ColumnDescriptor(ColumnType.DECIMAL),
    }
    request.cls.schema = schema


@pytest.mark.usefixtures("test_data_invalid")
class TestInvalidSession:
    """Unit tests for invalid session."""

    pdf: pd.DataFrame
    sdf: DataFrame
    sdf_col_types: Dict[str, ColumnDescriptor]
    sdf_input_domain: SparkDataFrameDomain
    schema: Dict[str, ColumnDescriptor]

    def _setup_accountant(self, mock_accountant) -> None:
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric(
            {NamedTable("private"): SymmetricDifference()}
        )
        mock_accountant.input_domain = DictDomain(
            {NamedTable("private"): self.sdf_input_domain}
        )
        mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

    def test_invalid_compiler_initialization(self):
        """session errors if compiler is not a QueryExprCompiler."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            with pytest.raises(
                TypeError,
                match=r"type of compiler must be one of "
                r"\(QueryExprCompiler, NoneType\); got list instead",
            ):
                self._setup_accountant(mock_accountant)
                Session(mock_accountant, public_sources={}, compiler=[])  # type: ignore

    def test_invalid_dataframe_initialization(self):
        """session raises error on invalid dataframe type"""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            # Private
            with pytest.raises(
                TypeError,
                match=(
                    'type of argument "dataframe" must be'
                    " pyspark.sql.dataframe.DataFrame; got pandas.core.frame.DataFrame"
                    " instead"
                ),
            ):
                Session.from_dataframe(
                    privacy_budget=PureDPBudget(1),
                    source_id="private",
                    dataframe=self.pdf,
                    protected_change=AddOneRow(),
                )
            # Public
            self._setup_accountant(mock_accountant)

            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(
                TypeError,
                match=(
                    'type of argument "dataframe" must be'
                    " pyspark.sql.dataframe.DataFrame; got pandas.core.frame.DataFrame"
                    " instead"
                ),
            ):
                session.add_public_dataframe(source_id="public", dataframe=self.pdf)

    def test_invalid_data_properties(self, spark):
        """session raises error on invalid data properties"""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant)
            session = Session(
                accountant=mock_accountant,
                public_sources={
                    "public": spark.createDataFrame(
                        pd.DataFrame({"A": ["a1", "a2"], "B": [1, 2]})
                    )
                },
            )

            # source_id not existent
            with pytest.raises(KeyError):
                session.get_schema("view")
            with pytest.raises(KeyError):
                session.get_grouping_column("view")

            # public source_id doesn't have a grouping_column
            with pytest.raises(
                ValueError,
                match=(
                    "Table 'public' is a public table, which cannot "
                    "have a grouping column."
                ),
            ):
                session.get_grouping_column("public")

    def test_invalid_column_name(self, spark) -> None:
        """Builder raises an error if a column is named "".

        Columns named "" (empty string) are not allowed.
        """
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant)
            with pytest.raises(
                ValueError,
                match=re.escape(
                    'This DataFrame contains a column named "" (the empty string)'
                ),
            ):
                Session.from_dataframe(
                    privacy_budget=PureDPBudget(1),
                    source_id="private",
                    dataframe=spark.createDataFrame(
                        pd.DataFrame({"A": ["a0", "a1"], "": [0, 1]})
                    ),
                    protected_change=AddOneRow(),
                )
            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(
                ValueError,
                match=re.escape(
                    'This DataFrame contains a column named "" (the empty string)'
                ),
            ):
                session.add_public_dataframe(
                    source_id="my_public_data",
                    dataframe=spark.createDataFrame(
                        pd.DataFrame(
                            [["0", 0, 0.0], ["1", 1, 1.1], ["2", 2, 2.2]],
                            columns=["A", "B", ""],
                        )
                    ),
                )

    def test_invalid_grouping_column(self, spark) -> None:
        """Builder raises an error if table's grouping column is not in dataframe."""
        with pytest.raises(
            ValueError,
            match=(
                "^Grouping column 'not_a_column' does not exist in the input. Available"
                " columns: A, B, X$"
            ),
        ):
            Session.from_dataframe(
                PureDPBudget(1),
                "private",
                self.sdf,
                protected_change=AddMaxRowsInMaxGroups("not_a_column", 1, 1),
            )
        with pytest.raises(
            ValueError,
            match=(
                "^Grouping column 'not_a_column' does not exist in the input. Available"
                " columns: A, B, X$"
            ),
        ):
            Session.from_dataframe(
                PureDPBudget(1), "private", self.sdf, grouping_column="not_a_column"
            )

        float_df = spark.createDataFrame(pd.DataFrame({"A": [1, 2], "F": [0.1, 0.2]}))
        with pytest.raises(
            ValueError,
            match=(
                "^Grouping column 'F' is not of a type on which grouping is supported.*"
            ),
        ):
            Session.from_dataframe(
                PureDPBudget(1),
                "private",
                float_df,
                protected_change=AddMaxRowsInMaxGroups("F", 1, 1),
            )
        with pytest.raises(
            ValueError,
            match=(
                "^Grouping column 'F' is not of a type on which grouping is supported.*"
            ),
        ):
            Session.from_dataframe(
                PureDPBudget(1), "private", float_df, grouping_column="F"
            )

    def test_invalid_key_column(self) -> None:
        """Builder raises an error if table's key column is not in dataframe."""
        with pytest.raises(
            ValueError,
            match=(
                "^Key column 'not_a_column' does not exist in the input. Available"
                " columns: A, B, X$"
            ),
        ):
            Session.from_dataframe(
                PureDPBudget(1),
                "private",
                self.sdf,
                protected_change=AddRowsWithID("not_a_column", "random_id"),
            )

    @pytest.mark.parametrize(
        "source_id,exception_type,expected_error_msg",
        [
            (2, TypeError, 'type of argument "source_id" must be str; got int instead'),
            ("@str", ValueError, "source_id must be a valid Python identifier."),
        ],
    )
    def test_invalid_source_id(
        self, source_id: str, exception_type: Type[Exception], expected_error_msg: str
    ):
        """session raises error on invalid source_id."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics._query_expr_compiler.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}
            mock_compiler.output_measure = PureDP()

            #### from spark dataframe ####
            # Private
            with pytest.raises(exception_type, match=expected_error_msg):
                Session.from_dataframe(
                    privacy_budget=PureDPBudget(1),
                    source_id=source_id,
                    dataframe=self.sdf,
                    protected_change=AddOneRow(),
                )
            # Public
            session = Session(
                accountant=mock_accountant, public_sources={}, compiler=mock_compiler
            )
            with pytest.raises(exception_type, match=expected_error_msg):
                session.add_public_dataframe(source_id, dataframe=self.sdf)
            # create_view
            with pytest.raises(exception_type, match=expected_error_msg):
                session.create_view(PrivateSource("private"), source_id, cache=False)

    def test_invalid_public_source(self):
        """Session raises an error adding a public source with duplicate source_id."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics._query_expr_compiler.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}
            mock_compiler.output_measure = PureDP()

            session = Session(
                accountant=mock_accountant, compiler=mock_compiler, public_sources={}
            )

            # This should work
            session.add_public_dataframe("public_df", dataframe=self.sdf)

            # But this should not
            with pytest.raises(
                ValueError, match="This session already has a table named 'public_df'."
            ):
                session.add_public_dataframe("public_df", dataframe=self.sdf)

    @pytest.mark.parametrize(
        "query_expr", [(["filter private A == 0"]), ([PrivateSource("private")])]
    )
    def test_invalid_queries_evaluate(self, query_expr: Any):
        """evaluate raises error on invalid queries."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics._query_expr_compiler.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}
            mock_compiler.output_measure = PureDP()

            session = Session(
                accountant=mock_accountant, public_sources={}, compiler=mock_compiler
            )
            with pytest.raises(
                TypeError,
                match=(
                    "type of query_expr must be tmlt.analytics.query_expr.QueryExpr;"
                    " got list instead"
                ),
            ):
                session.evaluate(query_expr, privacy_budget=PureDPBudget(float("inf")))

    @pytest.mark.parametrize(
        "query_expr,exception_type,expected_error_msg",
        [
            (
                "filter private A == 0",
                TypeError,
                'type of argument "query_expr" must be one of '
                r"\(QueryExpr, QueryBuilder\); got str instead",
            )
        ],
    )
    def test_invalid_queries_create(
        self,
        query_expr: QueryExpr,
        exception_type: Type[Exception],
        expected_error_msg: str,
    ):
        """create functions raise error on invalid input queries."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics._query_expr_compiler.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}
            mock_compiler.output_measure = PureDP()

            session = Session(
                accountant=mock_accountant, public_sources={}, compiler=mock_compiler
            )
            with pytest.raises(exception_type, match=expected_error_msg):
                session.create_view(query_expr, source_id="view", cache=True)

    def test_invalid_column(self):
        """Tests that invalid column name for column errors."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics._query_expr_compiler.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}
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
                accountant=mock_accountant, public_sources={}, compiler=mock_compiler
            )

            expected_schema = spark_schema_to_analytics_columns(self.sdf.schema)
            # We expect a transformation that will disallow NaNs on floats and infs
            expected_schema["X"].allow_nan = False
            expected_schema["X"].allow_inf = False

            with pytest.raises(
                KeyError,
                match=re.escape(
                    "'T' not present in transformed dataframe's columns; "
                    "schema of transformed dataframe is "
                    f"{expected_schema}"
                ),
            ):
                session.partition_and_create(
                    "private",
                    privacy_budget=PureDPBudget(1),
                    column="T",
                    splits={"private0": "0", "private1": "1"},
                )

    def test_invalid_splits_name(self):
        """Tests that invalid splits name errors."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics._query_expr_compiler.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}
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
                accountant=mock_accountant, public_sources={}, compiler=mock_compiler
            )

            with pytest.raises(
                ValueError,
                match=(
                    "The string passed as split name must be a valid Python identifier"
                ),
            ):
                session.partition_and_create(
                    "private",
                    privacy_budget=PureDPBudget(1),
                    column="A",
                    splits={" ": 0, "space present": 1, "2startsWithNumber": 2},
                )

    def test_splits_value_type(self):
        """Tests error when given invalid splits value type on partition."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant, patch(
            "tmlt.analytics._query_expr_compiler.QueryExprCompiler"
        ) as mock_compiler:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}
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
                accountant=mock_accountant, public_sources={}, compiler=mock_compiler
            )

            with pytest.raises(
                TypeError,
                match=(
                    r"'A' column is of type 'StringType\(?\)?'; 'StringType\(?\)?' "
                    "column not compatible with splits value type 'int'"
                ),
            ):
                session.partition_and_create(
                    "private",
                    privacy_budget=PureDPBudget(1),
                    column="A",
                    splits={"private0": 0, "private1": 1},
                )

    def test_session_raises_error_on_unsupported_spark_column_types(self, spark):
        """Session raises error when initialized with unsupported column types."""
        sdf = spark.createDataFrame(
            [], schema=StructType([StructField("A", BooleanType())])
        )
        with pytest.raises(
            ValueError,
            match=(
                "Unsupported Spark data type: Tumult Analytics does not yet support the"
                " Spark data types for the following columns"
            ),
        ):
            Session.from_dataframe(
                privacy_budget=PureDPBudget(1),
                source_id="private",
                dataframe=sdf,
                protected_change=AddOneRow(),
            )

    @pytest.mark.parametrize("nullable", [(True), (False)])
    def test_keep_nullable_status(self, spark, nullable: bool):
        """Session uses the nullable status of input dataframes."""
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(1),
            source_id="private_df",
            dataframe=spark.createDataFrame(
                [(1.0,)],
                schema=StructType([StructField("A", DoubleType(), nullable=nullable)]),
            ),
            protected_change=AddOneRow(),
        )
        session.add_public_dataframe(
            source_id="public_df",
            dataframe=spark.createDataFrame(
                [(1.0,)],
                schema=StructType([StructField("A", DoubleType(), nullable=nullable)]),
            ),
        )
        expected_schema = Schema(
            {
                "A": ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=nullable,
                    allow_nan=True,
                    allow_inf=True,
                )
            }
        )
        assert session.get_schema("private_df") == expected_schema
        assert session.get_schema("public_df") == expected_schema


@pytest.fixture(name="session_builder_data", scope="class")
def setup_session_build_data(spark, request):
    """Setup for tests."""
    df1 = spark.createDataFrame([(1, 2, "A"), (3, 4, "B")], schema=["A", "B", "C"])
    df2 = spark.createDataFrame([("X", "A"), ("Y", "B"), ("Z", "B")], schema=["K", "C"])
    df3 = spark.createDataFrame([(1, 1, "A"), (2, 2, "B")], schema=["A", "Y", "Z"])

    request.cls.dataframes = {"df1": df1, "df2": df2, "df3": df3}


@pytest.mark.usefixtures("session_builder_data")
class TestSessionBuilder:
    """Tests for :class:`~tmlt.analytics.session.Session.Builder`."""

    dataframes: Dict[str, DataFrame]

    @pytest.mark.parametrize(
        "builder,error_msg",
        [
            (
                Session.Builder(),
                "Privacy budget must be specified.",
            ),  # No Privacy Budget
            (
                Session.Builder().with_privacy_budget(PureDPBudget(10)),
                "At least one private source must be provided.",
            ),  # No Private Sources
        ],
    )
    def test_invalid_build(self, builder: Session.Builder, error_msg: str):
        """Tests that builds raise relevant errors when builder is not configured."""
        with pytest.raises(ValueError, match=error_msg):
            builder.build()

    def test_invalid_stability(self):
        """Tests that private source cannot be added with an invalid stability."""
        with pytest.raises(ValueError, match="Stability must be a positive integer"):
            Session.Builder().with_private_dataframe(
                source_id="df1", dataframe=self.dataframes["df1"], stability=0
            )
        with pytest.raises(ValueError, match="Stability must be a positive integer"):
            Session.Builder().with_private_dataframe(
                source_id="df1", dataframe=self.dataframes["df1"], stability=-1
            )

    @pytest.mark.parametrize(
        "stability,grouping_column,error_msg",
        [
            (None, None, "Using a default for protected_change is deprecated"),
            (
                None,
                "A",
                "Providing a grouping_column parameter instead of a"
                " protected_change parameter is deprecated",
            ),
            (
                1,
                None,
                "Providing a stability instead of a protected_change is deprecated",
            ),
            (
                1,
                "A",
                "Providing a grouping_column parameter instead of a"
                " protected_change parameter is deprecated",
            ),
        ],
    )
    def test_stability_deprecation(
        self, stability: Optional[int], grouping_column: Optional[str], error_msg: str
    ):
        """Test that stability and grouping_column give deprecation warnings."""
        with pytest.warns(DeprecationWarning, match=error_msg):
            Session.Builder().with_private_dataframe(
                source_id="df1",
                dataframe=self.dataframes["df1"],
                stability=stability,
                grouping_column=grouping_column,
            )

    @pytest.mark.parametrize("initial_budget", [(PureDPBudget(1)), (RhoZCDPBudget(1))])
    def test_invalid_to_add_budget_twice(self, initial_budget: PrivacyBudget):
        """Test that you can't call ``with_privacy_budget()`` twice."""
        builder = Session.Builder().with_privacy_budget(initial_budget)
        with pytest.raises(
            ValueError, match="This Builder already has a privacy budget"
        ):
            builder.with_privacy_budget(PureDPBudget(1))
        with pytest.raises(
            ValueError, match="This Builder already has a privacy budget"
        ):
            builder.with_privacy_budget(RhoZCDPBudget(1))

    def test_duplicate_source_id(self):
        """Tests that a repeated source id raises appropriate error."""
        builder = Session.Builder().with_private_dataframe(
            source_id="A", dataframe=self.dataframes["df1"], stability=1
        )
        with pytest.raises(ValueError, match="Duplicate source id: 'A'"):
            builder.with_private_dataframe(
                source_id="A", dataframe=self.dataframes["df2"], stability=2
            )
        with pytest.raises(ValueError, match="Duplicate source id: 'A'"):
            builder.with_public_dataframe(
                source_id="A", dataframe=self.dataframes["df2"]
            )

    def test_build_invalid_identifier(self):
        """Tests that build fails if protected change does
        not have associated ID space."""
        builder = (
            Session.Builder()
            .with_private_dataframe(
                source_id="A",
                dataframe=self.dataframes["df1"],
                protected_change=AddRowsWithID("A", "random_id"),
            )
            .with_privacy_budget(PureDPBudget(1))
        )

        assert len(builder._id_spaces) == 0  # pylint: disable=protected-access

        with pytest.raises(
            ValueError,
            match=(
                "An AddRowsWithID protected change was specified without an "
                "associated identifier space"
            ),
        ):
            builder.build()

        builder.with_id_space("not_random_id")
        with pytest.raises(
            ValueError,
            match=(
                "An AddRowsWithID protected change was specified without an "
                "associated identifier space"
            ),
        ):
            builder.build()
        ### build should succeed when the identifier space is added
        builder = builder.with_id_space("random_id")
        with pytest.raises(ValueError, match="This Builder already has an ID space"):
            builder.with_id_space("random_id")
        assert len(builder._id_spaces) == 2  # pylint: disable=protected-access
        builder.build()

    def test_build_multiple_ids(self):
        """Tests that build succeeds with multiple ID spaces."""
        builder = (
            Session.Builder()
            .with_private_dataframe(
                source_id="private1",
                dataframe=self.dataframes["df1"],
                protected_change=AddRowsWithID("A", "id_space_1"),
            )
            .with_id_space("id_space_1")
        )
        builder.with_private_dataframe(
            source_id="private2",
            dataframe=self.dataframes["df2"],
            protected_change=AddRowsWithID("C", "id_space_2"),
        ).with_id_space("id_space_2")

        builder.with_private_dataframe(
            source_id="private3",
            dataframe=self.dataframes["df3"],
            protected_change=AddRowsWithID("Y", "id_space_1"),
        )

        builder.with_privacy_budget(PureDPBudget(1)).build()

    @pytest.mark.parametrize(
        "builder,expected_sympy_budget,expected_output_measure,"
        + "private_dataframes,public_dataframes",
        [
            (
                Session.Builder().with_privacy_budget(PureDPBudget(10)),
                sp.Integer(10),
                PureDP(),
                [("df1", 1)],
                [],
            ),
            (
                Session.Builder().with_privacy_budget(ApproxDPBudget(10, 0.5)),
                (sp.Integer(10), 0),
                ApproxDP(),
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
        ],
    )
    def test_build_works_correctly(
        self,
        builder: Session.Builder,
        expected_sympy_budget: sp.Expr,
        expected_output_measure: Measure,
        private_dataframes: List[Tuple[str, int]],
        public_dataframes: List[str],
    ):
        """Tests that building a Session works correctly."""
        # Set up the builder.
        expected_private_sources, expected_public_sources = {}, {}
        expected_stabilities = {}
        for source_id, stability in private_dataframes:
            builder = builder.with_private_dataframe(
                source_id=source_id,
                dataframe=self.dataframes[source_id],
                protected_change=AddMaxRows(stability),
            )
            expected_private_sources[NamedTable(source_id)] = self.dataframes[source_id]
            expected_stabilities[NamedTable(source_id)] = stability

        for source_id in public_dataframes:
            builder = builder.with_public_dataframe(
                source_id=source_id, dataframe=self.dataframes[source_id]
            )
            expected_public_sources[source_id] = self.dataframes[source_id]

        # Build the session and verify that it worked.
        session = builder.build()
        # pylint: disable=protected-access
        accountant = session._accountant
        assert isinstance(accountant, PrivacyAccountant)
        assert accountant.privacy_budget == expected_sympy_budget
        assert accountant.output_measure == expected_output_measure

        for table_id, private_source in expected_private_sources.items():
            assert accountant._queryable is not None
            assert isinstance(accountant._queryable, SequentialQueryable)
            assert_frame_equal_with_sort(
                accountant._queryable._data[table_id].toPandas(),
                private_source.toPandas(),
            )

        assert accountant.d_in == expected_stabilities

        public_sources = session._public_sources
        assert public_sources.keys() == expected_public_sources.keys()
        for key in public_sources:
            assert_frame_equal_with_sort(
                public_sources[key].toPandas(), expected_public_sources[key].toPandas()
            )

        compiler = session._compiler
        # pylint: enable=protected-access
        assert isinstance(compiler, QueryExprCompiler)
        assert compiler.output_measure == expected_output_measure

    @pytest.mark.parametrize("nullable", [(True), (False)])
    def test_builder_with_dataframe_keep_nullable_status(self, spark, nullable: bool):
        """with_dataframe methods use the nullable status of the dataframe."""
        builder = Session.Builder()
        builder = builder.with_private_dataframe(
            source_id="private_df",
            dataframe=spark.createDataFrame(
                [(1,)],
                schema=StructType([StructField("A", LongType(), nullable=nullable)]),
            ),
        )
        builder = builder.with_public_dataframe(
            source_id="public_df",
            dataframe=spark.createDataFrame(
                [(1,)],
                schema=StructType([StructField("A", LongType(), nullable=nullable)]),
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
        expected_schema = StructType([StructField("A", LongType(), nullable=nullable)])
        assert actual_private_schema == expected_schema
        assert actual_public_schema == expected_schema
