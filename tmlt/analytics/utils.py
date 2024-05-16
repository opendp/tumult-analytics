"""Utility functions."""
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import atexit
from textwrap import dedent

import pandas as pd
import tmlt.core
from pyspark.sql import SparkSession
from tmlt.core.utils import cleanup as core_cleanup

from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.protected_change import AddOneRow
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.session import Session


def cleanup():
    """Clean up the temporary table currently in use.

    If you call ``spark.stop()``, you should call this function first.
    """
    core_cleanup.cleanup()


def remove_all_temp_tables():
    """Remove all temporary tables created by Tumult Analytics.

    This will remove all temporary tables created by Tumult Analytics in
    the current Spark data warehouse, whether those tables were created
    by the current Tumult Analytics session or previous sessions.
    """
    core_cleanup.remove_all_temp_tables()


def check_installation():
    """Check to see if you have installed Tumult Analytics correctly.

    This function will:

    * create a new Spark session
    * create a Spark dataframe
    * create a :class:`~tmlt.analytics.session.Session` from that dataframe
    * perform a query on that dataframe

    If Tumult Analytics is correctly installed, this function should print
    a message and finish running within a few seconds.

    If Tumult Analytics has *not* been correctly installed, this function
    will raise an error.
    """
    try:
        # Printing version numbers to help people debug.
        print(f"Using Tumult Core {tmlt.core.__version__}")
        print(f"Using Tumult Analytics {tmlt.analytics.__version__}")

        try:
            print("Creating Spark session... ", end="")
            spark = SparkSession.builder.getOrCreate()
            print(f"Using PySpark {spark.version}")
            print(" OK")
        except RuntimeError as e:
            # If Spark is broken, the Core cleanup atexit hook will fail, which
            # produces some additional output the user doesn't need to see in
            # this case.
            atexit.unregister(
                core_cleanup._cleanup_temp  # pylint: disable=protected-access
            )
            if (
                e.args
                and isinstance(e.args[0], str)
                and e.args[0].startswith("Java gateway process exited before sending")
            ):
                raise AssertionError(
                    "Error setting up Spark session. This likely indicates that Java is"
                    " not installed, or is not available on your PATH."
                ) from e
            raise

        print("Creating Pandas dataframe... ", end="")
        # We use Pandas to create this dataframe,
        # just to check that Pandas is installed and we can access it
        pdf = pd.DataFrame([["a1", 1], ["a2", 2]], columns=["A", "B"])
        print(" OK")

        print("Converting to Spark dataframe... ", end="")
        sdf = spark.createDataFrame(pdf)
        print(" OK")

        print("Creating Tumult Analytics session... ", end="")
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(1),
            source_id="private_data",
            dataframe=sdf,
            protected_change=AddOneRow(),
        )
        print(" OK")

        print("Creating query...", end="")
        query = (
            QueryBuilder("private_data")
            .groupby(KeySet.from_dict({"A": ["a0", "a1", "a2"]}))
            .count(name="count")
        )
        print(" OK")

        print("Evaluating query...", end="")
        result = session.evaluate(query_expr=query, privacy_budget=PureDPBudget(1))
        print(" OK")

        print("Checking that output is as expected...", end="")
        if result.count() == 0:
            raise RuntimeError(
                """
                It looks like Tumult Analytics has not been configured properly.
                In most cases, this is because the Spark warehouse location has not been
                set correctly.

                For information on setting spark configuration, see our troubleshooting
                guide at
                https://docs.tmlt.dev/analytics/latest/howto-guides/troubleshooting.html"""
            )
        if (
            len(result.columns) != 2
            or not "A" in result.columns
            or not "count" in result.columns
        ):
            raise RuntimeError(
                "Expected output to have columns 'A' and 'count', but instead it had"
                f" these columns: {result.columns}"
            )
        if result.count() != 3:
            raise RuntimeError(
                f"Expected output to have 3 rows, but instead it had {result.count()}"
            )
        if (
            result.filter(result["A"] == "a0").count() != 1
            or result.filter(result["A"] == "a1").count() != 1
            or result.filter(result["A"] == "a2").count() != 1
        ):
            # result.toPandas() is used here so that the error message contains the
            # whole dataframe
            raise AssertionError(
                "Expected output to have 1 row where column A was 'a0', one row where"
                " column A was 'a1', and one row where column A was 'a2'. Instead, got"
                f" this result: {result.toPandas()}"
            )
        print(" OK")

        print(
            "Installation check complete. Tumult Analytics appears to be properly"
            " installed."
        )
    except Exception as e:  # pylint: disable=broad-except
        print(" FAILED\n")
        if not str(e).startswith("It looks like the analytics session"):
            raise RuntimeError(
                dedent(
                    """

                The installation test did not complete successfully. You may want to
                check:
                - your Java installation (try `java -version`)
                - your PySpark and Pandas installations (run `pip3 show pyspark pandas`)

                For more information, see the Tumult Analytics installation instructions
                at
                https://docs.tmlt.dev/analytics/latest/howto-guides/installation.html"""
                )
            ) from e
