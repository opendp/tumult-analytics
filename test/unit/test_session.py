"""Unit tests for Session."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

# pylint: disable=no-self-use, unidiomatic-typecheck

import re
from typing import Any, Dict, List, Tuple, Type, Union
from unittest.mock import ANY, Mock, patch

import pandas as pd
import pytest
import sympy as sp
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
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.exact_number import ExactNumber

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
from tmlt.analytics.config import config
from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.keyset import KeySet
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
    ProtectedChange,
)
from tmlt.analytics.query_builder import GroupedQueryBuilder, QueryBuilder
from tmlt.analytics.query_expr import PrivateSource, QueryExpr
from tmlt.analytics.session import Session

from ..conftest import assert_frame_equal_with_sort

# Disable redefined-outer-name because spark is used to create dataframes as test
# inputs and within tests to check outputs and run queries.
# pylint: disable=redefined-outer-name


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
        "budget_values,output_measure,expected_budget",
        [
            pytest.param(ExactNumber(10), PureDP(), PureDPBudget(10), id="puredp"),
            pytest.param(ExactNumber(10), RhoZCDP(), RhoZCDPBudget(10), id="zcdp"),
            pytest.param(
                (ExactNumber(10), ExactNumber("0.5")),
                ApproxDP(),
                ApproxDPBudget(10, 0.5),
                id="approxdp",
            ),
        ],
    )
    def test_remaining_privacy_budget(
        self, budget_values, output_measure, expected_budget
    ):
        """Test that remaining_privacy_budget returns the right type of budget."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(
                mock_accountant, privacy_budget=budget_values, d_in=ExactNumber(1)
            )
            mock_accountant.output_measure = output_measure

            session = Session(mock_accountant, {})
            privacy_budget = session.remaining_privacy_budget
            # Check that the output privacy_budget is the correct Type
            assert type(expected_budget) == type(privacy_budget)

            # Check that the correct privacy budget values are returned
            if isinstance(privacy_budget, PureDPBudget):
                assert budget_values == ExactNumber(privacy_budget.epsilon)
            elif isinstance(privacy_budget, RhoZCDPBudget):
                assert budget_values == ExactNumber(privacy_budget.rho)
            elif isinstance(privacy_budget, ApproxDPBudget):
                assert budget_values == (
                    ExactNumber(privacy_budget.epsilon),
                    ExactNumber(str(privacy_budget.delta)),
                )
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
                {
                    "protected_change": AddMaxRows(21),
                },
                id="puredp-protected_change",
            ),
            pytest.param(
                PureDPBudget(float("inf")),
                PureDP(),
                IfGroupedBy("X", SumOf(SymmetricDifference())),
                {
                    "protected_change": AddMaxRowsInMaxGroups("X", 3, 7),
                },
                id="puredp-grouped-protected_change",
            ),
            pytest.param(
                RhoZCDPBudget(float("inf")),
                RhoZCDP(),
                IfGroupedBy("X", RootSumOfSquared(SymmetricDifference())),
                {
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
                self=ANY, accountant=ANY, public_sources={}
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
                self=ANY, accountant=ANY, public_sources={}
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
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = self.sdf_input_domain
            mock_accountant.d_in = {NamedTable("private"): ExactNumber(1)}
            session = Session(accountant=mock_accountant, public_sources={})
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

    # Checks that every privacy budget type has an error with zero budget.
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

            # set output measures to ApproxDP
            mock_accountant.output_measure = ApproxDP()
            mock_compiler.output_measure = ApproxDP()
            with pytest.raises(
                ValueError,
                match="You need a non-zero privacy budget to evaluate a query.",
            ):
                session.evaluate(
                    query_expr=PrivateSource("private"),
                    privacy_budget=ApproxDPBudget(0, 0),
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

    def test_partition_and_create(self):
        """Tests that :func:`partition_and_create` calls the right things."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant, privacy_budget=ExactNumber(10))
            input_spark_domain = self.sdf_input_domain.key_to_domain[
                NamedTable("private")
            ]
            mock_accountant.split.return_value = [
                Mock(
                    spec_set=PrivacyAccountant,
                    input_metric=DictMetric(
                        {NamedTable("part0"): SymmetricDifference()}
                    ),
                    input_domain=DictDomain({NamedTable("part0"): input_spark_domain}),
                    output_measure=PureDP(),
                ),
                Mock(
                    spec_set=PrivacyAccountant,
                    input_metric=DictMetric(
                        {NamedTable("part1"): SymmetricDifference()}
                    ),
                    input_domain=DictDomain({NamedTable("part1"): input_spark_domain}),
                    output_measure=PureDP(),
                ),
            ]

            session = Session(accountant=mock_accountant, public_sources={})

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

    @pytest.mark.parametrize(
        "protected_change",
        [
            (AddMaxRowsInMaxGroups("B", max_groups=1, max_rows_per_group=1)),
            (AddOneRow()),
        ],
    )
    @pytest.mark.parametrize(
        "columns,expected_df,privacy_budget,possible_df",
        [
            # Tests without infinite privacy budget
            # Note: The count of records with [0,0] and [0,1] is large enough that they
            # should almost always appear in the output; [1,3] shouldn't appear.
            (
                ["count"],
                pd.DataFrame({"count": [0]}),
                ApproxDPBudget(1, 1e-5),
                pd.DataFrame({"count": [0, 1]}),
            ),
            (
                ["B"],
                pd.DataFrame({"B": [0, 1]}),
                ApproxDPBudget(1, 1e-5),
                pd.DataFrame({"B": [0, 1, 3]}),
            ),
            (
                ["count", "B"],
                pd.DataFrame({"count": [0, 0], "B": [0, 1]}),
                ApproxDPBudget(1, 1e-5),
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
            ),
            (
                [],
                pd.DataFrame({"count": [0, 0], "B": [0, 1]}),
                ApproxDPBudget(1, 1e-5),
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
            ),
            (
                None,
                pd.DataFrame({"count": [0, 0], "B": [0, 1]}),
                ApproxDPBudget(1, 1e-5),
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
            ),
            # Tests with infinite privacy budget
            # Note: If either the epsilon is infinite or the delta is 1, the output
            # budget is infinite. Exact results should be returned.
            (
                ["count"],
                pd.DataFrame({"count": [0, 1]}),
                ApproxDPBudget(float("inf"), 1e-5),
                None,
            ),
            (["count"], pd.DataFrame({"count": [0, 1]}), ApproxDPBudget(1e-5, 1), None),
            (
                ["B"],
                pd.DataFrame({"B": [0, 1, 3]}),
                ApproxDPBudget(float("inf"), 1),
                None,
            ),
            (["B"], pd.DataFrame({"B": [0, 1, 3]}), ApproxDPBudget(1e-5, 1), None),
            (
                ["count", "B"],
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
                ApproxDPBudget(float("inf"), 1),
                None,
            ),
            (
                ["count", "B"],
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
                ApproxDPBudget(1e-5, 1),
                None,
            ),
            (
                [],
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
                ApproxDPBudget(float("inf"), 1),
                None,
            ),
            (
                [],
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
                ApproxDPBudget(1e-5, 1),
                None,
            ),
            (
                None,
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
                ApproxDPBudget(float("inf"), 1),
                None,
            ),
            (
                None,
                pd.DataFrame({"count": [0, 0, 1], "B": [0, 1, 3]}),
                ApproxDPBudget(1e-5, 1),
                None,
            ),
        ],
    )
    def test_get_groups_with_various_protected_change(
        self,
        spark,
        protected_change,
        columns: List[str],
        expected_df: pd.DataFrame,
        privacy_budget: PrivacyBudget,
        possible_df: Union[pd.DataFrame, None],
    ):
        """GetGroups works with AddMaxRowsInMaxGroups and AddOneRow protected change."""
        sdf = spark.createDataFrame(
            pd.DataFrame(
                [[0, 0] for _ in range(10000)]
                + [[0, 1] for _ in range(10000)]
                + [[1, 3]],
                columns=["count", "B"],
            )
        )

        session = Session.from_dataframe(
            privacy_budget=privacy_budget,
            source_id="private",
            dataframe=sdf,
            protected_change=protected_change,
        )
        query = QueryBuilder("private").get_groups(columns)
        actual_sdf = session.evaluate(query, session.remaining_privacy_budget)

        try:
            assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)
        except AssertionError:
            # Deals with the case where the DFs mismatched due to noise.
            assert_frame_equal_with_sort(actual_sdf.toPandas(), possible_df)

    @pytest.mark.parametrize(
        "protected_change",
        [
            (AddMaxRowsInMaxGroups("B", max_groups=1, max_rows_per_group=1)),
            (AddOneRow()),
        ],
    )
    @pytest.mark.parametrize(
        "privacy_budget",
        [
            # Large Alpha, Low Epsilon
            ApproxDPBudget(1e-5, 0.99999),
            # Large Epsilon, Low Alpha
            ApproxDPBudget(10000, 0.01),
            # Large Epsilon, Large Alpha
            ApproxDPBudget(10000, 0.99999),
        ],
    )
    def test_get_groups_with_high_budget(
        self,
        spark,
        protected_change,
        privacy_budget: PrivacyBudget,
    ):
        """Smoke test for GetGroups with large but not infinite budgets."""
        # This test is required because there was a bug where get_groups would return
        # an empty dataframe when the budget was large but not infinite.
        sdf = spark.createDataFrame(
            pd.DataFrame(
                [[0, 0] for _ in range(10000)]
                + [[0, 1] for _ in range(10000)]
                + [[1, 3]],
                columns=["count", "B"],
            )
        )

        session = Session.from_dataframe(
            privacy_budget=privacy_budget,
            source_id="private",
            dataframe=sdf,
            protected_change=protected_change,
        )
        query = QueryBuilder("private").get_groups()
        actual_sdf = session.evaluate(query, session.remaining_privacy_budget)

        # Checks that the result is non-empty
        assert len(actual_sdf.toPandas()) > 0

    def test_get_groups_with_add_rows_with_id(self, spark):
        """GetGroups with AddRowsWithID protected change works on non-ID column."""
        sdf = spark.createDataFrame(
            pd.DataFrame([[0, i] for i in range(10000)], columns=["count", "B"])
        )
        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(1, 1e-5),
            source_id="private",
            dataframe=sdf,
            protected_change=AddRowsWithID("B"),
        )
        query = QueryBuilder("private").enforce(MaxRowsPerID(1)).get_groups(["count"])
        expected_df = pd.DataFrame({"count": [0]})
        actual_sdf = session.evaluate(query, session.remaining_privacy_budget)
        assert_frame_equal_with_sort(actual_sdf.toPandas(), expected_df)

    @pytest.mark.parametrize("columns", [(["B"]), (["count", "B"])])
    def test_get_groups_on_id_column(self, spark, columns: List[str]):
        """Test that the GetGroups on ID column errors."""
        sdf = spark.createDataFrame(
            pd.DataFrame([[0, i] for i in range(10000)], columns=["count", "B"])
        )
        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(1, 1e-5),
            source_id="private",
            dataframe=sdf,
            protected_change=AddRowsWithID("B"),
        )
        match_str = re.escape(
            "get_groups cannot be used on the privacy ID column"
            " (B) of a table with the AddRowsWithID protected change."
        )
        with pytest.raises(
            RuntimeError,
            match=match_str,
        ):
            session.evaluate(
                QueryBuilder("private").enforce(MaxRowsPerID(1)).get_groups(columns),
                session.remaining_privacy_budget,
            )

    @pytest.mark.parametrize(
        "test_df,id_column,expected_columns",
        [
            (pd.DataFrame({"ID": [1, 2, 3], "VAR": [4, 5, 6]}), "ID", ["VAR"]),
            (
                pd.DataFrame(
                    {
                        "CUST_ID": [1, 2, 3],
                        "VARONE": [4, 5, 6],
                        "VARTWO": ["A", "B", "C"],
                    }
                ),
                "CUST_ID",
                ["VARONE", "VARTWO"],
            ),
        ],
    )
    def test_get_groups_defaults_to_non_id_columns(
        self, spark, test_df: pd.DataFrame, id_column: str, expected_columns: List[str]
    ):
        """Tests that get_groups applies to all non-ID cols if no cols are provided."""
        sdf = spark.createDataFrame(test_df)
        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(1, 1e-5),
            source_id="private",
            dataframe=sdf,
            protected_change=AddRowsWithID(id_column),
        )

        get_groups_query = QueryBuilder("private").enforce(MaxRowsPerID(1)).get_groups()

        end_df = session.evaluate(get_groups_query, session.remaining_privacy_budget)

        assert set(expected_columns) == set(end_df.columns)

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
            pytest.param(
                QueryBuilder("private").count(),
                """Columns:
\t- 'count'  INTEGER, not null""",
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

    @pytest.mark.parametrize(
        "constraints,expected_output",
        [
            ([MaxRowsPerID(5)], "\t\t- MaxRowsPerID(max=5)"),
            (
                [MaxRowsPerGroupPerID("B", 1), MaxGroupsPerID("X", 5)],
                "\t\t- MaxRowsPerGroupPerID(grouping_column='B', max=1)\n\t\t"
                "- MaxGroupsPerID(grouping_column='X', max=5)",
            ),
        ],
    )
    def test_describe_table_with_constraints(
        self, constraints: List[Constraint], expected_output: str
    ):
        """Test :func:`_describe` with a table with constraints."""
        with patch("builtins.print") as mock_print, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_with_id(
                mock_accountant, privacy_budget=ExactNumber(10)
            )
            mock_accountant.state = PrivacyAccountantState.ACTIVE
            session = Session(accountant=mock_accountant, public_sources={})
            # pylint: disable=protected-access
            session._table_constraints[NamedTable("private")] = constraints
            # pylint: enable=protected-access
            expected = (
                """Columns:
\t- 'A'  VARCHAR, ID column (in ID space identifier_A)
\t- 'B'  INTEGER
\t- 'X'  INTEGER
\tConstraints:\n"""
                + expected_output
            )
            session.describe("private")
            # pylint: enable=line-too-long
            mock_print.assert_called_with(expected)

    @pytest.mark.parametrize(
        "query,constraint_output,group_output",
        [
            (
                QueryBuilder("private").enforce(MaxRowsPerID(5)),
                "\t\t- MaxRowsPerID(max=5)",
                "",
            ),
            (
                QueryBuilder("private")
                .enforce(MaxGroupsPerID("X", 5))
                .enforce(MaxRowsPerGroupPerID("B", 1)),
                "\t\t- MaxGroupsPerID(grouping_column='X', max=5)\n\t\t"
                "- MaxRowsPerGroupPerID(grouping_column='B', max=1)",
                "",
            ),
            (
                QueryBuilder("private")
                .enforce(MaxRowsPerID(5))
                .groupby(KeySet.from_dict({})),
                "\t\t- MaxRowsPerID(max=5)",
                "",
            ),
            (
                QueryBuilder("private")
                .enforce(MaxRowsPerID(5))
                .groupby(KeySet.from_dict({"A": ["0", "1"]})),
                "\t\t- MaxRowsPerID(max=5)",
                "\nGrouped on columns 'A' (2 groups)",
            ),
        ],
    )
    def test_describe_query_with_constraints(
        self,
        query: Union[QueryBuilder, GroupedQueryBuilder],
        constraint_output: str,
        group_output: str,
    ):
        """Test :func:`_describe` with a query with constraints."""
        with patch("builtins.print") as mock_print, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant_with_id(
                mock_accountant, privacy_budget=ExactNumber(10)
            )
            mock_accountant.state = PrivacyAccountantState.ACTIVE
            session = Session(accountant=mock_accountant, public_sources={})
            expected = (
                """Columns:
\t- 'A'  VARCHAR, ID column (in ID space identifier_A)
\t- 'B'  INTEGER
\t- 'X'  INTEGER
\tConstraints:\n"""
                + constraint_output
                + group_output
            )
            session.describe(query)
            # pylint: enable=line-too-long
            mock_print.assert_called_with(expected)

    @pytest.mark.parametrize(
        "query,expected_output",
        [
            pytest.param(
                QueryBuilder("private").groupby(KeySet.from_dict({})),
                """Columns:
\t- 'A'  VARCHAR
\t- 'B'  INTEGER
\t- 'X'  INTEGER""",
                id="query_builder_groupby_empty",
            ),
            pytest.param(
                QueryBuilder("private").groupby(KeySet.from_dict({"A": ["0", "1"]})),
                """Columns:
\t- 'A'  VARCHAR
\t- 'B'  INTEGER
\t- 'X'  INTEGER
Grouped on columns 'A' (2 groups)""",
                id="query_builder_groupby_1_col",
            ),
            pytest.param(
                QueryBuilder("private").groupby(
                    KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]})
                ),
                """Columns:
\t- 'A'  VARCHAR
\t- 'B'  INTEGER
\t- 'X'  INTEGER
Grouped on columns 'A', 'B' (4 groups)""",
                id="query_builder_groupby_multi_col",
            ),
        ],
    )
    def test_describe_grouped_query(
        self, query: GroupedQueryBuilder, expected_output: str
    ):
        """Test :func:`_describe` with a QueryExpr, QueryBuilder, or table name."""
        with patch("builtins.print") as mock_print, patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            self._setup_accountant(mock_accountant, privacy_budget=ExactNumber(10))
            mock_accountant.state = PrivacyAccountantState.ACTIVE

            session = Session(accountant=mock_accountant, public_sources={})

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
                    splits={"part0": "0", "part1": "1"},
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
            (
                "@str",
                ValueError,
                "Names must be valid Python identifiers: they can only contain "
                "alphanumeric characters and underscores, and cannot begin with a "
                "number.",
            ),
        ],
    )
    def test_invalid_source_id(
        self, source_id: str, exception_type: Type[Exception], expected_error_msg: str
    ):
        """session raises error on invalid source_id."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

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
            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(exception_type, match=expected_error_msg):
                session.add_public_dataframe(source_id, dataframe=self.sdf)
            # create_view
            with pytest.raises(exception_type, match=expected_error_msg):
                session.create_view(PrivateSource("private"), source_id, cache=False)

    def test_invalid_public_source(self):
        """Session raises an error adding a public source with duplicate source_id."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

            session = Session(accountant=mock_accountant, public_sources={})

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
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

            session = Session(accountant=mock_accountant, public_sources={})
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
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

            session = Session(accountant=mock_accountant, public_sources={})
            with pytest.raises(exception_type, match=expected_error_msg):
                session.create_view(query_expr, source_id="view", cache=True)

    def test_invalid_column(self):
        """Tests that invalid column name for column errors."""
        with patch(
            "tmlt.core.measurements.interactive_measurements.PrivacyAccountant"
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

            session = Session(accountant=mock_accountant, public_sources={})

            expected_schema = spark_schema_to_analytics_columns(self.sdf.schema)
            # We expect a transformation that will disallow NaNs on floats and infs
            expected_schema["X"].allow_nan = False
            expected_schema["X"].allow_inf = False

            with pytest.raises(
                KeyError,
                match=re.escape(
                    "'T' not present in transformed DataFrame's columns; "
                    "schema of transformed DataFrame is "
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
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

            session = Session(accountant=mock_accountant, public_sources={})

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
        ) as mock_accountant:
            mock_accountant.output_measure = PureDP()
            mock_accountant.input_metric = DictMetric(
                {NamedTable("private"): SymmetricDifference()}
            )
            mock_accountant.input_domain = DictDomain(
                {NamedTable("private"): self.sdf_input_domain}
            )
            mock_accountant.privacy_budget = ExactNumber(1)
            mock_accountant.d_in = {NamedTable("private"): sp.Integer(1)}

            session = Session(accountant=mock_accountant, public_sources={})

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
                "This builder must have a privacy budget set",
            ),  # No Privacy Budget
            (
                Session.Builder().with_privacy_budget(PureDPBudget(10)),
                "At least one private dataframe must be specified",
            ),  # No Private Sources
        ],
    )
    def test_invalid_build(self, builder: Session.Builder, error_msg: str):
        """Tests that builds raise relevant errors when builder is not configured."""
        with pytest.raises(ValueError, match=error_msg):
            builder.build()

    @pytest.mark.parametrize("initial_budget", [(PureDPBudget(1)), (RhoZCDPBudget(1))])
    def test_invalid_to_add_budget_twice(self, initial_budget: PrivacyBudget):
        """Test that you can't call ``with_privacy_budget()`` twice."""
        builder = Session.Builder().with_privacy_budget(initial_budget)
        with pytest.raises(
            ValueError, match="This builder already has a privacy budget set"
        ):
            builder.with_privacy_budget(PureDPBudget(1))
        with pytest.raises(
            ValueError, match="This builder already has a privacy budget set"
        ):
            builder.with_privacy_budget(RhoZCDPBudget(1))

    def test_duplicate_source_id(self):
        """Tests that a repeated source id raises appropriate error."""
        builder = Session.Builder().with_private_dataframe(
            source_id="A",
            dataframe=self.dataframes["df1"],
            protected_change=AddOneRow(),
        )
        with pytest.raises(ValueError, match="Table 'A' already exists"):
            builder.with_private_dataframe(
                source_id="A",
                dataframe=self.dataframes["df2"],
                protected_change=AddOneRow(),
            )
        with pytest.raises(ValueError, match="Table 'A' already exists"):
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
            .with_private_dataframe(
                source_id="B",
                dataframe=self.dataframes["df2"],
                protected_change=AddOneRow(),
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
        with pytest.raises(ValueError, match="ID space 'random_id' already exists"):
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

    def test_build_with_id_and_only_one_df(self) -> None:
        """Test that build works with only one private df + AddRowsWithID.

        Specifically, if there is only one private dataframe and that dataframe
        uses the AddRowsWithID ProtectedChange, building should succeed.
        """
        builder = Session.Builder().with_private_dataframe(
            source_id="private1",
            dataframe=self.dataframes["df1"],
            protected_change=AddRowsWithID("A", "id_space_1"),
        )

        # This should not raise an error
        builder.with_privacy_budget(PureDPBudget(1)).build()

        # Explicitly providing the ID space should also work
        builder = (
            Session.Builder()
            .with_private_dataframe(
                source_id="private1",
                dataframe=self.dataframes["df1"],
                protected_change=AddRowsWithID("A", "id_space_1"),
            )
            .with_id_space("id_space_1")
        )
        builder.with_privacy_budget(PureDPBudget(1)).build()

        # It should also work if you don't name the ID space

        builder = Session.Builder().with_private_dataframe(
            source_id="private1",
            dataframe=self.dataframes["df1"],
            protected_change=AddRowsWithID("A"),
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
                (sp.Integer(10), sp.Rational("0.5")),
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
        assert session._output_measure == expected_output_measure

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
            protected_change=AddOneRow(),
        )
        builder = builder.with_public_dataframe(
            source_id="public_df",
            dataframe=spark.createDataFrame(
                [(1,)],
                schema=StructType([StructField("A", LongType(), nullable=nullable)]),
            ),
        )
        actual_private_schema = (
            builder._private_dataframes[  # pylint: disable=protected-access
                "private_df"
            ].dataframe.schema
        )
        actual_public_schema = (
            builder._public_dataframes[  # pylint: disable=protected-access
                "public_df"
            ].schema
        )
        expected_schema = StructType([StructField("A", LongType(), nullable=nullable)])
        assert actual_private_schema == expected_schema
        assert actual_public_schema == expected_schema


# Test Constants
TEST_DATA_SIMPLE = pd.DataFrame(
    {
        "id": ["A", "A", "A", "B"],
        "A": [0, 1, 1, 1],
    }
).merge(pd.DataFrame({"agg_col": [1, 2, 3, 4]}), how="cross")

spark = SparkSession.builder.getOrCreate()
TEST_DATA_SPARK = spark.createDataFrame(TEST_DATA_SIMPLE)

with config.features.auto_partition_selection.enabled():
    GROUP_COLS = ["id", "A"]
    test_groupby = QueryBuilder(source_id="testdf").groupby(GROUP_COLS)

    # Creates the Query Expr to be tested.
    AGG_QUERIES = [
        test_groupby.count(name="agg_col"),
        test_groupby.count_distinct(name="agg_col"),
        test_groupby.sum(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby.average(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby.variance(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby.stdev(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby.quantile(
            column="agg_col", name="agg_col", quantile=0.5, low=1, high=4
        ),
        # Adds a special test case for grouping on a string.
        QueryBuilder(source_id="testdf").groupby("id").count(name="agg_col"),
    ]

EXPECTED_DFS = [
    TEST_DATA_SIMPLE.groupby(GROUP_COLS).agg({"agg_col": "count"}).reset_index(),
    # Supporting distinct count requires non-default Pandas aggregations.
    TEST_DATA_SIMPLE.drop_duplicates()
    .groupby(GROUP_COLS)
    .agg({"agg_col": "count"})
    .reset_index(),
    TEST_DATA_SIMPLE.groupby(GROUP_COLS).agg({"agg_col": "sum"}).reset_index(),
    TEST_DATA_SIMPLE.groupby(GROUP_COLS).agg({"agg_col": "mean"}).reset_index(),
    TEST_DATA_SIMPLE.groupby(GROUP_COLS).agg({"agg_col": "var"}).reset_index(),
    TEST_DATA_SIMPLE.groupby(GROUP_COLS).agg({"agg_col": "std"}).reset_index(),
    # The quantile calculation doesn't evaluate exactly with inf budget.
    # The ouptut will be a set range. This tests the output is within the range.
    {
        ("A", 0): (2, 3),
        ("A", 1): (2, 3),
        ("B", 1): (2, 3),
    },
    # Deals with the special case of grouping on a string.
    TEST_DATA_SIMPLE.groupby("id").agg({"agg_col": "count"}).reset_index(),
]


@pytest.mark.parametrize(
    "input_data,dp_query,expected_df",
    [
        (TEST_DATA_SIMPLE, dp_query, expected_df)
        for dp_query, expected_df in zip(AGG_QUERIES, EXPECTED_DFS)
    ],
)
@pytest.mark.parametrize(
    "protected_change",
    [
        AddOneRow(),
        AddMaxRows(5),
    ],
)
def test_automatic_partitions(
    input_data: pd.DataFrame,
    dp_query: QueryExpr,
    expected_df: Union[pd.DataFrame, Dict[Tuple[str, int], Tuple[float, float]]],
    protected_change: ProtectedChange,
):
    """Tests that partition selection is automatically called with correct queries."""

    # Turning on experimental features for this test.
    with config.features.auto_partition_selection.enabled():
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame(input_data)

        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(float("inf"), 1),
            source_id="testdf",
            dataframe=test_df,
            protected_change=protected_change,
        )
        end_df = session.evaluate(
            dp_query, privacy_budget=ApproxDPBudget(float("inf"), 1)
        )
        end_pd_df = end_df.toPandas()

        if isinstance(expected_df, pd.DataFrame):
            assert_frame_equal_with_sort(end_pd_df, expected_df)
        # Else the expected_df is a range of values for a quantile query.
        else:
            for pair, values in expected_df.items():
                filter_df = end_pd_df[end_pd_df["id"] == pair[0]]
                filter_df = filter_df[filter_df["A"] == pair[1]].reset_index()
                test_val = filter_df.at[0, "agg_col"]
                assert values[0] <= test_val <= values[1]


with config.features.auto_partition_selection.enabled():
    test_groupby_id = (
        QueryBuilder(source_id="testdf").enforce(MaxRowsPerID(20)).groupby(GROUP_COLS)
    )

    # Creates the Query Expr to be tested.
    AGG_ID_QUERIES = [
        test_groupby_id.count(name="agg_col"),
        test_groupby_id.count_distinct(name="agg_col"),
        test_groupby_id.sum(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby_id.average(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby_id.variance(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby_id.stdev(column="agg_col", name="agg_col", low=1, high=4),
        test_groupby_id.quantile(
            column="agg_col", name="agg_col", quantile=0.5, low=1, high=4
        ),
    ]


@pytest.mark.parametrize(
    "input_data,dp_query,expected_df",
    [
        (TEST_DATA_SIMPLE, dp_query, expected_df)
        for dp_query, expected_df in zip(AGG_ID_QUERIES, EXPECTED_DFS)
    ],
)
def test_automatic_partitions_with_ids(
    input_data: pd.DataFrame,
    dp_query: QueryExpr,
    expected_df: Any,
):
    """Tests automatic partition selection with the AddRowsWithID protected change."""
    # Turning on experimental features for this test.
    with config.features.auto_partition_selection.enabled():
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame(input_data)

        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(float("inf"), 1),
            source_id="testdf",
            dataframe=test_df,
            protected_change=AddRowsWithID(id_column="id"),
        )
        end_df = session.evaluate(
            dp_query, privacy_budget=ApproxDPBudget(float("inf"), 1)
        )
        end_pd_df = end_df.toPandas()

        if isinstance(expected_df, pd.DataFrame):
            assert_frame_equal_with_sort(end_pd_df, expected_df)
        # Else the expected_df is a range of values for a quantile query.
        else:
            for pair, values in expected_df.items():
                filter_df = end_pd_df[end_pd_df["id"] == pair[0]]
                filter_df = filter_df[filter_df["A"] == pair[1]].reset_index()
                test_val = filter_df.at[0, "agg_col"]
                assert values[0] <= test_val <= values[1]


@pytest.mark.parametrize(
    "input_data,dp_query",
    [(TEST_DATA_SIMPLE, dp_query) for dp_query in AGG_QUERIES],
)
@pytest.mark.parametrize(
    "budget,expected_error",
    [
        (
            ApproxDPBudget(0, 0.1),
            "Automatic partition selection requires a positive "
            f"epsilon. The budget provided was {ApproxDPBudget(0,.1)}.",
        ),
        (
            ApproxDPBudget(1, 0),
            "Automatic partition selection requires a positive "
            f"delta. The budget provided was {ApproxDPBudget(1,0)}.",
        ),
    ],
)
def test_automatic_partition_selection_invalid_budget(
    input_data: pd.DataFrame,
    dp_query: QueryExpr,
    budget: ApproxDPBudget,
    expected_error: str,
):
    """Test that Automatic Partition Selection queries with an invalid budget error."""

    with config.features.auto_partition_selection.enabled():
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame(input_data)

        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(float("inf"), 1),
            source_id="testdf",
            dataframe=test_df,
            protected_change=AddOneRow(),
        )
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            session.evaluate(dp_query, privacy_budget=budget)


@pytest.mark.parametrize(
    "query_expr,expected_columns",
    list(
        zip(
            AGG_QUERIES,
            [
                GROUP_COLS,
                GROUP_COLS,
                GROUP_COLS,
                GROUP_COLS,
                GROUP_COLS,
                GROUP_COLS,
                GROUP_COLS,
                ["id"],
            ],
        )
    ),
)
def test_automatic_partition_null_keyset(query_expr: QueryExpr, expected_columns: List):
    """Tests that automatic partition selection with null keyset raises a warning and
    completes with an output dataframe with len(0) but the correct schema."""

    with config.features.auto_partition_selection.enabled():
        spark = SparkSession.builder.getOrCreate()
        # An empty DF ensures that automatic partition selection returns a null keyset.
        empty_df = pd.DataFrame({"id": [], "A": [], "agg_col": []})
        test_df = spark.createDataFrame(empty_df, schema=TEST_DATA_SPARK.schema)
        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(float("inf"), 1),
            source_id="testdf",
            dataframe=test_df,
            protected_change=AddOneRow(),
        )

        warning_str = (
            "This query tried to automatically determine a keyset, but "
            "a null dataframe was returned from the partition selection."
            "This may be because the dataset is empty or because the "
            " ApproxDPBudget used was too small."
        )
        with pytest.warns(UserWarning, match=warning_str):
            df_out = session.evaluate(
                query_expr, privacy_budget=ApproxDPBudget(5, 0.05)
            )
            for col in expected_columns:
                assert col in df_out.columns
