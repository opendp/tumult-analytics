"""Benchmarking script for running groupby counts with no privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import time
from typing import Type

import pandas as pd
from benchmarking_utils import write_as_html
from pyspark.sql import SparkSession
from tabulate import tabulate

from tmlt.analytics import AddOneRow, KeySet, PureDPBudget, Query, QueryBuilder, Session
from tmlt.analytics.no_privacy_session import NoPrivacySession


def evaluate_runtime(
    query: Query,
    session: Session,
) -> float:
    """See how long it takes to run a query in a given session."""
    start = time.time()
    result = session.evaluate(query, session.remaining_privacy_budget)
    result.first()
    running_time = time.time() - start
    return round(running_time, 3)


def main() -> None:
    """Evaluate running time for groupby count queries without noise."""
    print("Benchmark query execution")
    spark = SparkSession.builder.getOrCreate()

    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "Type",
            "Number of groups",
            "Running time (s)",
        ],
    )

    for size in [100, 1000, 10000, 100000, 1000000]:
        keyset = KeySet.from_dict({"group": range(size)})
        dataset = spark.createDataFrame(pd.DataFrame({"group": range(size)}))

        query = QueryBuilder("private").groupby(keyset).count()

        session_class: Type[Session]
        for session_class in [Session, NoPrivacySession]:
            sess = session_class.from_dataframe(
                PureDPBudget(float("inf")), "private", dataset, AddOneRow()
            )

            # Make sure everything is "warmed up" for good comparisons.
            _ = evaluate_runtime(query, sess)
            running_time = evaluate_runtime(query, sess)
            row = {
                "Type": (
                    "NoPrivacySession"
                    if isinstance(sess, NoPrivacySession)
                    else "Session"
                ),
                "Number of groups": size,
                "Running time (s)": running_time,
            }
            print("Benchmark row:", row)
            benchmark_result = pd.concat(
                [benchmark_result, pd.Series(row).to_frame().T], ignore_index=True
            )
    spark.stop()
    print(tabulate(benchmark_result, headers="keys", showindex="never"))
    write_as_html(benchmark_result, "keyset_projection.html")


if __name__ == "__main__":
    main()
