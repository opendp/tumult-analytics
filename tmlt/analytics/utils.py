"""Utility functions for Analytics."""

import pandas as pd
from pyspark.sql import SparkSession

from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.session import Session
from tmlt.core.utils import cleanup as core_cleanup
from tmlt.core.utils import configuration


def cleanup():
    """Cleanup the temporary table currently in use.

    If you call ``spark.stop()``, you should call this function first.
    """
    core_cleanup.cleanup()


def remove_all_temp_tables():
    """Remove all temporary tables created by Analytics.

    This will remove all Analytics-created temporary tables in the current
    Spark data warehouse, whether those tables were created by the current
    Analytics session or previous Analytics sessions.
    """
    core_cleanup.remove_all_temp_tables()


def get_java_11_config():
    """Set Spark configuration for Java 11+ users."""
    return configuration.get_java11_config()


def check_installation():
    """Check to see if you have installed Analytics correctly.

    This function will:

    * create a new Spark session
    * create a Spark dataframe
    * create a :class:`~tmlt.analytics.session.Session` from that dataframe
    * perform a query on that dataframe

    If Analytics is correctly installed, this function should print a message
    and finish running within a few minutes.

    If Analytics has *not* been correctly installed, this function will raise
    an error.
    """
    try:
        print("Creating Spark session... ", end="")
        spark = SparkSession.builder.getOrCreate()
        print(" ok")

        print("Creating Pandas dataframe... ", end="")
        # We use Pandas to create this dataframe,
        # just to check that Pandas is installed and we can access it
        pdf = pd.DataFrame([["a1", 1], ["a2", 2]], columns=["A", "B"])
        print(" ok")

        print("Converting to Spark dataframe... ", end="")
        sdf = spark.createDataFrame(pdf)
        print(" ok")

        print("Creating Analytics session... ", end="")
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(1), source_id="private_data", dataframe=sdf
        )
        print(" ok")

        print("Creating query...", end="")
        query = (
            QueryBuilder("private_data")
            .groupby(KeySet.from_dict({"A": ["a0", "a1", "a2"]}))
            .count(name="count")
        )
        print(" ok")

        print("Evaluating query...", end="")
        result = session.evaluate(query_expr=query, privacy_budget=PureDPBudget(1))
        print(" ok")

        print("Checking that output is as expected...", end="")
        if (
            len(result.columns) != 2
            or not "A" in result.columns
            or not "count" in result.columns
        ):
            raise AssertionError(
                "Expected output to have columns 'A' and 'count', but instead it had"
                f" these columns: {result.columns}"
            )
        if result.count() != 3:
            raise AssertionError(
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
        print(" ok")

        print(
            "Installation check complete. Tumult Analytics appears to be properly"
            " installed."
        )
    except Exception as e:
        print(
            "\n\nThere was a problem running the installation checker. You may want to"
            " check:"
        )
        print("- your installed Java version (run `java -version`)")
        print("- your installed version of Pyspark (run `pip3 show pyspark`)")
        print("- your installed version of Pandas (run `pip3 show pandas`)")
        print(
            "- your installed version of Tumult Analytics "
            "(run `pip3 show tmlt.analytics`)"
        )
        print(
            "For more information, see the Tumult Analytics installation instructions\n"
            "at https://docs.tmlt.dev/analytics/latest/installation.html"
        )
        print("\nRe-raising original exception...")
        raise e
