"""Tests for :mod:`tmlt.analytics._coerce_spark_schema`."""

# <placeholder: boilerplate>

import pandas as pd
from numpy import nan
from parameterized import parameterized
from pyspark.sql import types as st

from tmlt.analytics._coerce_spark_schema import (
    _fail_if_dataframe_contains_nulls_or_nans,
    coerce_spark_schema_or_fail,
)
from tmlt.core.utils.testing import PySparkTest


class TestNullsOrNans(PySparkTest):
    """Tests for _coerce_spark_schema._fail_if_dataframe_contains_nulls_or_nans."""

    def test_string_nan(self):
        """Test that the string 'NaN' is an acceptable value."""
        df = self.spark.createDataFrame(pd.DataFrame({"A": ["NaN"]}))
        # This should not throw an error
        _fail_if_dataframe_contains_nulls_or_nans(df)

    @parameterized.expand(
        [(pd.DataFrame({"A": [nan, 9.3]}),), (pd.DataFrame({"B": [100, nan]}),)]
    )
    def test_nan(self, pd_df):
        """Test that nans are not accepted."""
        df = self.spark.createDataFrame(pd_df)
        with self.assertRaisesRegex(
            ValueError, "This DataFrame contains a null or nan value"
        ):
            _fail_if_dataframe_contains_nulls_or_nans(df)

    @parameterized.expand(
        [
            (pd.DataFrame({"A": [9.3, None]}),),
            (pd.DataFrame({"B": [100, None]}),),
            (pd.DataFrame({"C": [None, "abcdef"]}),),
        ]
    )
    def test_null(self, pd_df):
        """Test that nulls are not accepted."""
        df = self.spark.createDataFrame(pd_df)
        with self.assertRaisesRegex(
            ValueError, "This DataFrame contains a null or nan value"
        ):
            _fail_if_dataframe_contains_nulls_or_nans(df)

    @parameterized.expand(
        [
            (pd.DataFrame({"A": ["X"], "B": [float("nan")]}),),
            (pd.DataFrame({"A": ["Y"], "B": [-float("nan")]}),),
        ]
    )
    def test_allow_nan(self, pd_df: pd.DataFrame) -> None:
        """Test that coerce_schema allows nans (when it ought to)"""
        df = self.spark.createDataFrame(pd_df)

        # check allowing nans
        result = coerce_spark_schema_or_fail(df, allow_nan_and_null=True)
        self.assert_frame_equal_with_sort(pd_df, result.toPandas())
        # and check not allowing them
        with self.assertRaisesRegex(
            ValueError, "This DataFrame contains a null or nan value"
        ):
            coerce_spark_schema_or_fail(df, allow_nan_and_null=False)

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": ["X", "Y", None]}),
                st.StructType([st.StructField("A", st.StringType(), nullable=True)]),
            ),
            (
                pd.DataFrame({"B": [1, None, 3]}),
                st.StructType([st.StructField("B", st.LongType(), nullable=True)]),
            ),
            (
                pd.DataFrame({"C": [None, 0.2, 0.3]}),
                st.StructType([st.StructField("C", st.DoubleType(), nullable=True)]),
            ),
        ]
    )
    def test_allow_null(self, pd_df: pd.DataFrame, schema: st.StructType) -> None:
        """Test that coerce_schema allows nulls (when it ought to)"""
        df = self.spark.createDataFrame(pd_df, schema=schema)
        # check allowing nulls
        result = coerce_spark_schema_or_fail(df, allow_nan_and_null=True)
        self.assert_frame_equal_with_sort(pd_df, result.toPandas())
        # check not allowing nulls
        with self.assertRaisesRegex(
            ValueError, "This DataFrame contains a null or nan value"
        ):
            coerce_spark_schema_or_fail(df, allow_nan_and_null=False)
