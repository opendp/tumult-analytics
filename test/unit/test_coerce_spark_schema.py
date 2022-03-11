"""Tests for :mod:`tmlt.analytics._coerce_spark_schema`."""

# <placeholder: boilerplate>

import pandas as pd
from numpy import nan
from parameterized import parameterized

from tmlt.analytics._coerce_spark_schema import (
    _fail_if_dataframe_contains_nulls_or_nans,
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
            ValueError,
            "Tumult Analytics does not yet handle DataFrames containing null or nan"
            " values",
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
            ValueError,
            "Tumult Analytics does not yet handle DataFrames containing null or nan"
            " values",
        ):
            _fail_if_dataframe_contains_nulls_or_nans(df)
