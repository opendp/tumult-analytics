"""Tests for generic builders."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import pandas as pd
import pytest
from pyspark.sql.types import DoubleType, LongType, StructField, StructType

from tmlt.analytics._base_builder import (
    BaseBuilder,
    DataFrameMixin,
    ParameterMixin,
    PrivacyBudgetMixin,
)
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.protected_change import AddMaxRows, AddRowsWithID


class _Builder(PrivacyBudgetMixin, DataFrameMixin, ParameterMixin, BaseBuilder):
    def build(self):
        return (
            self._privacy_budget,
            self._private_dataframes,
            self._public_dataframes,
            self._id_spaces,
            self._parameters,
        )


def test_builder(spark):
    """BaseBuilder subclasses with multiple mixins work correctly."""
    df1 = spark.createDataFrame(pd.DataFrame({"A": [1]}))
    df2 = spark.createDataFrame(pd.DataFrame({"A": [2.0]}))
    df3 = spark.createDataFrame(pd.DataFrame({"A": [3]}))

    budget = PureDPBudget(1)
    privacy_budget, private_dfs, public_dfs, id_spaces, parameters = (
        _Builder()
        .with_privacy_budget(budget)
        .with_id_space("id1")
        .with_id_space("df3")  # Make sure id spaces don't conflict with dataframes
        .with_private_dataframe("df1", df1, AddRowsWithID("A"))
        .with_private_dataframe("df2", df2, AddMaxRows(5))
        .with_public_dataframe("df3", df3)
        .with_parameter("a", 1)
        .with_parameter("b", "x")
        .with_parameter("c", {1: 2, 3: 4})
        .build()
    )

    assert privacy_budget == budget

    assert set(private_dfs.keys()) == {"df1", "df2"}
    assert private_dfs["df1"][0].head()["A"] == 1
    assert private_dfs["df1"][1] == AddRowsWithID("A")
    assert private_dfs["df1"][0].schema == StructType(
        [StructField("A", LongType(), True)]
    )

    assert private_dfs["df2"][0].head()["A"] == 2.0
    assert private_dfs["df2"][1] == AddMaxRows(5)
    assert private_dfs["df2"][0].schema == StructType(
        [StructField("A", DoubleType(), True)]
    )

    assert set(public_dfs.keys()) == {"df3"}
    assert public_dfs["df3"].head()["A"] == 3
    assert public_dfs["df3"].schema == StructType([StructField("A", LongType(), True)])

    assert id_spaces == {"id1", "df3"}

    assert parameters["a"] == 1
    assert parameters["b"] == "x"
    assert parameters["c"] == {1: 2, 3: 4}


def test_multiple_builds(spark):
    """Builders work correctly when build is used multiple times."""
    private_df = spark.createDataFrame(pd.DataFrame({"A": [1]}))
    public_df = spark.createDataFrame(pd.DataFrame({"A": [2.0]}))
    builder1 = (
        _Builder()
        .with_privacy_budget(PureDPBudget(1))
        .with_private_dataframe("df1", private_df, AddRowsWithID("A"))
        .with_public_dataframe("df2", public_df)
        .with_id_space("id")
        .with_parameter("a", 1)
    )

    first_build = builder1.build()
    assert builder1.build() == first_build


def test_incomplete_builds(spark):
    """Builds fail when they do not contain a budget."""
    builder = _Builder()
    with pytest.raises(ValueError):
        builder.build()

    df = spark.createDataFrame(pd.DataFrame({"A": [1]}))
    builder.with_public_dataframe("df", df)
    with pytest.raises(ValueError):
        builder.build()

    builder.with_id_space("id")
    with pytest.raises(ValueError):
        builder.build()

    builder.with_parameter("a", 1)
    with pytest.raises(ValueError):
        builder.build()


def test_immutability(spark):
    """Modifying values returned by builders does not affect builder internals."""
    df1 = spark.createDataFrame(pd.DataFrame({"A": [1]}))
    df2 = spark.createDataFrame(pd.DataFrame({"A": [2.0]}))
    df3 = spark.createDataFrame(pd.DataFrame({"A": [3]}))

    budget = PureDPBudget(1)
    builder = (
        _Builder()
        .with_privacy_budget(budget)
        .with_id_space("id1")
        .with_id_space("df3")  # Make sure id spaces don't conflict with dataframes
        .with_private_dataframe("df1", df1, AddRowsWithID("A"))
        .with_private_dataframe("df2", df2, AddMaxRows(5))
        .with_public_dataframe("df3", df3)
        .with_parameter("a", 1)
        .with_parameter("b", "x")
        .with_parameter("c", {1: 2, 3: 4})
    )
    _, private_dfs, _, _, parameters = builder.build()

    assert builder._privacy_budget == budget

    private_df, _ = private_dfs["df1"]
    private_dfs["df1"] = (private_df, AddMaxRows(5))
    assert builder._private_dataframes["df1"][1] == AddRowsWithID("A")

    parameters["a"] = 2
    assert builder._parameters["a"] == 1
