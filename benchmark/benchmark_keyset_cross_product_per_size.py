"""Benchmarking script for taking the cross-product of large keysets."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import datetime
import time
from functools import reduce
from typing import List

import pandas as pd
from benchmarking_utils import write_as_html
from pyspark.sql import SparkSession

from tmlt.analytics.keyset import KeySet


def evaluate_runtime(keysets: List[KeySet]) -> float:
    """See how long it takes to perform product keyset."""
    start = time.time()
    product_keyset = reduce(lambda a, b: a * b, keysets)
    product_keyset_size = product_keyset.size()
    product_keyset.dataframe().write.format("noop").mode("overwrite").save()
    running_time = time.time() - start
    return round(running_time, 3), product_keyset_size


def add_benchmark_row(benchmark_result, keysets, size, hint):
    """Call evaluate and add a row to the benchmark result"""
    # Make sure everything is "warmed up" for good comparisons.
    evaluate_runtime(keysets)

    running_time, product_keyset_size = evaluate_runtime(keysets)
    row = {
        "Hint": hint,
        "Keyset Domain Size": size,
        "Product Keyset Size": product_keyset_size,
        "Running time (s)": running_time,
    }
    print("Benchmark row:", row)
    return pd.concat([benchmark_result, pd.Series(row).to_frame().T], ignore_index=True)

def main() -> None:
    """Evaluate running time for selecting subset of columns from product keysets."""
    print("Benchmark keyset cross-product with 2 factors and varying sizes")
    spark = SparkSession.builder.getOrCreate()
    keyset_ab = KeySet.from_dataframe(
        spark.createDataFrame(
            pd.DataFrame(
                [
                    ["abc", 123],
                    ["def", 123],
                ],
                columns=["A", "B"],
            ),
        ),
    )
    _DATE1 = datetime.date.fromisoformat("2022-01-01")
    keyset_cd = KeySet.from_dataframe(
        spark.createDataFrame(
            pd.DataFrame(
                [
                    ["abc", _DATE1],
                    ["def", _DATE1],
                ],
                columns=["C", "D"],
            ),
        ),
    )
    benchmark_result = pd.DataFrame(
        [], columns=["Keyset Domain Size", "Product Keyset Size", "Running time (s)"]
    )
    # Materialize all dataframes before benchmarking
    for keyset in keyset_ab, keyset_cd:
        keyset.dataframe().write.format("noop").mode("overwrite").save()
    benchmark_result = add_benchmark_row(
        benchmark_result, [keyset_ab, keyset_cd], 2, "tiny * tiny"
    )

    for size in [100, 400, 10000, 40000, 160000]:
        keyset_c = KeySet.from_dict({"C": list(range(size))})
        keyset_d = KeySet.from_dict({"D": list(range(size))})
        keysets = [keyset_ab, keyset_c]
        comparable_keysets = [keyset_c, keyset_d]

        # Materialize all dataframes before benchmarking
        for keyset in keyset_c, keyset_d:
            keyset.dataframe().write.format("noop").mode("overwrite").save()

        benchmark_result = add_benchmark_row(
            benchmark_result, keysets, size, "large * tiny"
        )
        benchmark_result = add_benchmark_row(
            benchmark_result, comparable_keysets, size, "large * large"
        )
    spark.stop()
    write_as_html(benchmark_result, "keyset_cross_product_per_size.html")


if __name__ == "__main__":
    main()
