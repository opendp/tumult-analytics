"""Tests for constraint handling in TransformationVisitor."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import Dict, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from tmlt.core.metrics import SymmetricDifference

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._table_identifier import Identifier
from tmlt.analytics._transformation_utils import get_table_from_ref
from tmlt.analytics.constraints import (
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.query_expr import EnforceConstraint, PrivateSource

from .conftest import TestTransformationVisitor


class TestConstraints(TestTransformationVisitor):
    """Tests for constraint handling in the transformation visitor."""

    visitor: TransformationVisitor
    catalog: Catalog
    input_data: Dict[Identifier, Union[DataFrame, Dict[Identifier, DataFrame]]]
    dataframes: Dict[str, DataFrame]

    @staticmethod
    def _test_is_subset(input_df: pd.DataFrame, result_df: pd.DataFrame):
        """Check that the result is a subset of the input, accounting for duplicates."""
        input_row_counts = (
            input_df.groupby(list(input_df.columns)).size().to_frame("input")
        )
        result_row_counts = (
            result_df.groupby(list(result_df.columns)).size().to_frame("result")
        )
        counts = input_row_counts.join(result_row_counts, how="outer").fillna(0)
        assert (counts["result"] <= counts["input"]).all()

    @pytest.mark.parametrize("constraint_max", [1, 2, 3])
    def test_max_rows_per_id(self, constraint_max: int):
        """Test truncation with MaxRowsPerID."""
        constraint = MaxRowsPerID(constraint_max)
        query = EnforceConstraint(PrivateSource("ids_duplicates"), constraint)
        transformation, ref, constraints = query.accept(self.visitor)
        assert len(constraints) == 1
        assert constraints[0] == constraint

        input_df: pd.DataFrame = self.dataframes["ids_duplicates"].toPandas()
        result_df = self._get_result(transformation, ref)

        # Check that each ID doesn't appear more times than the constraint bound.
        rows_per_id = result_df.groupby("id")["id"].count()
        assert all(
            rows_per_id <= constraint_max
        ), f"MaxRowsPerID constraint violated, counts were:\n{str(rows_per_id)}"

        self._test_is_subset(input_df, result_df)

    @pytest.mark.parametrize(
        "grouping_col,constraint_max", [("St", 1), ("St", 2), ("St", 3)]
    )
    def test_max_groups_per_id(self, grouping_col: str, constraint_max: int):
        """Test truncation with MaxGroupsPerID."""
        constraint = MaxGroupsPerID(grouping_col, constraint_max)
        query = EnforceConstraint(PrivateSource("ids_duplicates"), constraint)
        transformation, ref, constraints = query.accept(self.visitor)
        assert len(constraints) == 1
        assert constraints[0] == constraint

        input_df: pd.DataFrame = self.dataframes["ids_duplicates"].toPandas()
        result_df = self._get_result(transformation, ref)

        # Check that each no ID has more groups associated with it than the
        # truncation bound.
        groups_per_id = result_df.groupby("id").nunique()[grouping_col]
        assert all(
            groups_per_id <= constraint_max
        ), f"MaxGroupsPerID constraint violated, counts were:\n{str(groups_per_id)}"

        self._test_is_subset(input_df, result_df)

    @pytest.mark.parametrize(
        "constraint_max,grouping_col", [(2, "St"), (1, "St"), (3, "St")]
    )
    def test_max_rows_per_group_per_id(self, constraint_max: int, grouping_col: str):
        """Test truncation with MaxRowsPerGroupPerID."""
        constraint = MaxRowsPerGroupPerID(grouping_col, constraint_max)
        query = EnforceConstraint(PrivateSource("ids_duplicates"), constraint)
        transformation, ref, constraints = query.accept(self.visitor)
        assert len(constraints) == 1
        assert constraints[0] == constraint

        input_df: pd.DataFrame = self.dataframes["ids_duplicates"].toPandas()
        result_df = self._get_result(transformation, ref)

        # Check that each (ID, grouping_column) pair doesn't appear more
        # times than the constraint bound.
        rows_per_group_per_id = result_df.value_counts(["id", grouping_col])
        assert all(
            rows_per_group_per_id <= constraint_max
        ), "MaxRowsPerGroupPerID constraint violated, counts were:\n" + str(
            rows_per_group_per_id
        )

        self._test_is_subset(input_df, result_df)

    @pytest.mark.parametrize("constraint_max", [1, 2, 3])
    def test_l1_update_metric(self, constraint_max: int):
        """Test L1 truncation with updating metric."""
        constraint = MaxRowsPerID(constraint_max)
        query = EnforceConstraint(
            PrivateSource("ids_duplicates"), constraint, options={"update_metric": True}
        )
        transformation, ref, constraints = query.accept(self.visitor)
        assert len(constraints) == 1
        assert constraints[0] == constraint
        assert (
            get_table_from_ref(transformation, ref).output_metric
            == SymmetricDifference()
        )

        input_df: pd.DataFrame = self.dataframes["ids_duplicates"].toPandas()
        result_df = self._get_result(transformation, ref)

        # Check that each ID doesn't appear more times than the constraint bound.
        rows_per_id = result_df.groupby("id")["id"].count()
        assert all(
            rows_per_id <= constraint_max
        ), f"MaxRowsPerID constraint violated, counts were:\n{str(rows_per_id)}"

        self._test_is_subset(input_df, result_df)

    @pytest.mark.parametrize(
        "group_max,row_max,grouping_col", [(1, 1, "St"), (1, 2, "St"), (2, 1, "St")]
    )
    def test_l0_linf_update_metric(
        self, group_max: int, row_max: int, grouping_col: str
    ):
        """Test L0 + L-inf truncation with updating metric."""
        query = EnforceConstraint(
            EnforceConstraint(
                PrivateSource("ids_duplicates"),
                MaxGroupsPerID(grouping_col, group_max),
                options={"update_metric": True},
            ),
            MaxRowsPerGroupPerID(grouping_col, row_max),
            options={"update_metric": True},
        )
        transformation, ref, constraints = query.accept(self.visitor)
        assert len(constraints) == 2
        assert (
            get_table_from_ref(transformation, ref).output_metric
            == SymmetricDifference()
        )

        input_df: pd.DataFrame = self.dataframes["ids_duplicates"].toPandas()
        result_df = self._get_result(transformation, ref)

        # Check that each no ID has more groups associated with it than the
        # truncation bound.
        groups_per_id = result_df.groupby("id").nunique()[grouping_col]
        assert all(
            groups_per_id <= group_max
        ), f"MaxGroupsPerID constraint violated, counts were:\n{str(groups_per_id)}"

        # Check that each (ID, grouping_column) pair doesn't appear more
        # times than the constraint bound.
        rows_per_group_per_id = result_df.value_counts(["id", grouping_col])
        assert all(
            rows_per_group_per_id <= row_max
        ), "MaxRowsPerGroupPerID constraint violated, counts were:\n" + str(
            rows_per_group_per_id
        )

        self._test_is_subset(input_df, result_df)
