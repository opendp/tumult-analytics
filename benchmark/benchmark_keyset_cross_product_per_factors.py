"""Benchmarking script for taking the cross-product of large keysets."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

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


def main() -> None:
    """Evaluate running time for selecting subset of columns from product keysets."""
    print("Benchmark keyset cross-product for fixed size and varying factors")
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

    benchmark_result = pd.DataFrame(
        [], columns=["Factor Size", "Product Keyset Size", "Running time (s)"]
    )

    keyset_c = KeySet.from_dict({"C": list(range(2000))})
    keyset_d = KeySet.from_dict({"D": list(range(-2000, 0))})
    keyset_e = KeySet.from_dict({"E": [str(n) for n in range(2000)]})
    all_keysets = [keyset_ab, keyset_c, keyset_d, keyset_e]

    for factors in range(2, len(all_keysets) + 1):
        keysets = all_keysets[:factors]

        # Materialize all dataframes before benchmarking
        for keyset in keysets:
            keyset.dataframe().write.format("noop").mode("overwrite").save()

        # Make sure everything is "warmed up" for good comparisons.
        evaluate_runtime(keysets)

        running_time, product_keyset_size = evaluate_runtime(keysets)
        row = {
            "Factor Size": len(keysets),
            "Product Keyset Size": product_keyset_size,
            "Running time (s)": running_time,
        }
        print("Benchmark row:", row)
        benchmark_result = pd.concat(
            [benchmark_result, pd.Series(row).to_frame().T], ignore_index=True
        )
    spark.stop()
    write_as_html(benchmark_result, "keyset_cross_product_per_factors.html")


if __name__ == "__main__":
    main()
