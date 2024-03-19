"""Tests for builder mixins."""

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


class _PrivacyBudgetBuilder(PrivacyBudgetMixin, BaseBuilder):
    def build(self):
        return self._privacy_budget


def test_privacy_budget():
    """PrivacyBudgetMixin works correctly."""
    budget = PureDPBudget(1)
    assert _PrivacyBudgetBuilder().with_privacy_budget(budget).build() == budget


def test_privacy_budget_not_set():
    """PrivacyBudgetMixin raises an exception when budget is not set."""
    with pytest.raises(ValueError):
        _PrivacyBudgetBuilder().build()


def test_privacy_budget_multiple_set():
    """PrivacyBudgetMixin raises an exception when budget is set multiple times."""
    budget = PureDPBudget(1)
    builder = _PrivacyBudgetBuilder().with_privacy_budget(budget)
    with pytest.raises(ValueError):
        builder.with_privacy_budget(budget)


class _DataFrameBuilder(DataFrameMixin, BaseBuilder):
    def build(self):
        return self._private_dataframes, self._public_dataframes, self._id_spaces


def test_dataframes(spark):
    """DataFrameMixin works correctly."""
    df1 = spark.createDataFrame(pd.DataFrame({"A": [1]}))
    df2 = spark.createDataFrame(pd.DataFrame({"A": [2.0]}))
    df3 = spark.createDataFrame(pd.DataFrame({"A": [3]}))

    private_dfs, public_dfs, id_spaces = (
        _DataFrameBuilder()
        .with_id_space("id1")
        .with_id_space("df3")  # Make sure id spaces don't conflict with dataframes
        .with_private_dataframe("df1", df1, AddRowsWithID("A"))
        .with_private_dataframe("df2", df2, AddMaxRows(5))
        .with_public_dataframe("df3", df3)
        .build()
    )
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


def test_dataframes_invalid_ids(spark):
    """DataFrameMixin rejects invalid IDs."""
    df = spark.createDataFrame(pd.DataFrame({"A": [1]}))

    with pytest.raises(ValueError):
        _DataFrameBuilder().with_private_dataframe("1st", df, AddMaxRows(1))
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_private_dataframe("test-id", df, AddMaxRows(1))
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_public_dataframe("1st", df)
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_public_dataframe("test-id", df)
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_id_space("1st")
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_id_space("test-id")


def test_dataframes_name_conflicts(spark):
    """DataFrameMixin rejects duplicate IDs."""
    df = spark.createDataFrame(pd.DataFrame({"A": [1]}))

    with pytest.raises(ValueError):
        _DataFrameBuilder().with_private_dataframe(
            "t1", df, AddMaxRows(1)
        ).with_private_dataframe("t1", df, AddMaxRows(1))
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_public_dataframe("t1", df).with_private_dataframe(
            "t1", df, AddMaxRows(1)
        )
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_private_dataframe(
            "t1", df, AddMaxRows(1)
        ).with_public_dataframe("t1", df)
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_id_space("id1").with_id_space("id2").with_id_space(
            "id1"
        )


def test_dataframes_empty_column_name(spark):
    """DataFrameMixin rejects dataframes with empty column names."""
    df = spark.createDataFrame(pd.DataFrame({"": [1]}))
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_private_dataframe("t1", df, AddMaxRows(1))


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame({"A": [bytearray(b"x")]}),
        pd.DataFrame({"A": [{"x": 1}]}),
    ],
)
def test_dataframes_invalid_schemas(spark, dataframe):
    """DataFrameMixin rejects dataframes with unsupported schemas."""
    df = spark.createDataFrame(dataframe)
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_private_dataframe("t1", df, AddMaxRows(1))


class _ParameterBuilder(ParameterMixin, BaseBuilder):
    def build(self):
        return self._parameters


def test_parameters():
    """ParameterMixin works correctly."""
    parameters = (
        _ParameterBuilder()
        .with_parameter("a", 1)
        .with_parameter("b", "x")
        .with_parameter("c", {1: 2, 3: 4})
        .build()
    )
    assert parameters["a"] == 1
    assert parameters["b"] == "x"
    assert parameters["c"] == {1: 2, 3: 4}


def test_parameters_name_conflicts():
    """ParameterMixin rejects duplicate parameters."""
    builder = _ParameterBuilder().with_parameter("a", 1)
    with pytest.raises(ValueError):
        builder.with_parameter("a", 2)
    assert builder.build()["a"] == 1
