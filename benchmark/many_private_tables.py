"""Benchmarking script for per-query cost vs. the number of private tables.

Builds a Session with N small private tables, then answers a fixed number of
queries, each of which reads a single table (round-robin across tables). Ideally
the per-query time would not depend on N, since every query touches only one
table; this benchmark measures how far that is from true.

Two variants are measured:

* ``rows``: each table is protected with ``AddOneRow``, and each query is a
  ``groupby(KeySet).count()``.
* ``ids``: every table uses ``AddRowsWithID`` in a single shared ID space, and
  each query is ``enforce(MaxRowsPerID(...)).groupby(KeySet).count_distinct()``.

For each (variant, N) the script reports the Session build time, the median and
mean per-query time (both for ``Session.evaluate`` alone and including
collecting the result), and optionally the number of Python function calls per
query (via cProfile). It then fits ``per_query_time = intercept + slope * N``
for each variant, where the slope is the cost per table per query and the
intercept is the fixed per-query cost.

Run with ``--help`` for the available options; ``--full`` runs a bigger sweep.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import argparse
import cProfile
import gc
import json
import os
import platform
import pstats
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tmlt.core
from benchmarking_utils import write_as_html
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

import tmlt.analytics
from tmlt.analytics import (
    AddOneRow,
    AddRowsWithID,
    KeySet,
    MaxRowsPerID,
    PureDPBudget,
    Query,
    QueryBuilder,
    Session,
)

DEFAULT_TABLE_COUNTS = [1, 10, 100, 250, 500]
FULL_TABLE_COUNTS = [1, 10, 100, 250, 500, 1000, 2000]
VARIANTS = ["rows", "ids"]
ID_SPACE = "ids"
GROUP_VALUES = [f"g{i}" for i in range(5)]
SCHEMA = StructType(
    [
        StructField("id", IntegerType(), False),
        StructField("group", StringType(), False),
        StructField("value", IntegerType(), False),
    ]
)


def make_tables(spark: SparkSession, n_tables: int, n_rows: int) -> List[DataFrame]:
    """Create ``n_tables`` small, distinct Spark DataFrames."""
    tables = []
    for t in range(n_tables):
        rows = [
            (r // 2, GROUP_VALUES[(r + t) % len(GROUP_VALUES)], (r * 7 + t) % 100)
            for r in range(n_rows)
        ]
        tables.append(spark.createDataFrame(rows, schema=SCHEMA))
    return tables


def build_session(
    tables: List[DataFrame], total_budget: int, id_space: Optional[str] = None
) -> Session:
    """Build a Session with one private source per table.

    If ``id_space`` is given, every table uses ``AddRowsWithID`` in that ID space;
    otherwise every table uses ``AddOneRow``.
    """
    builder = Session.Builder().with_privacy_budget(PureDPBudget(total_budget))
    if id_space is not None:
        builder = builder.with_id_space(id_space)
    for t, df in enumerate(tables):
        protected_change = (
            AddRowsWithID(id_column="id", id_space=id_space)
            if id_space is not None
            else AddOneRow()
        )
        builder = builder.with_private_dataframe(f"t{t}", df, protected_change)
    return builder.build()


def make_query(variant: str, source_id: str, keyset: KeySet) -> Query:
    """Create the query for the given variant on one table."""
    if variant == "ids":
        return (
            QueryBuilder(source_id)
            .enforce(MaxRowsPerID(2))
            .groupby(keyset)
            .count_distinct(["id"])
        )
    return QueryBuilder(source_id).groupby(keyset).count()


def run_config(
    spark: SparkSession,
    variant: str,
    n_tables: int,
    args: argparse.Namespace,
    keyset: KeySet,
) -> Dict[str, Any]:
    """Build a Session with ``n_tables`` tables and time queries against it."""
    n_total = args.warmup + args.queries + args.profile_queries

    start = time.perf_counter()
    tables = make_tables(spark, n_tables, args.rows)
    data_s = time.perf_counter() - start

    start = time.perf_counter()
    session = build_session(
        tables, total_budget=n_total, id_space=ID_SPACE if variant == "ids" else None
    )
    build_s = time.perf_counter() - start

    # Each query gets an equal share (epsilon=1) of the total budget
    # (epsilon=n_total). Integer budgets keep the remaining budget rational.
    per_query_budget = PureDPBudget(1)
    eval_ms: List[float] = []
    total_ms: List[float] = []
    calls: List[int] = []

    def one_query(i: int) -> tuple[float, float]:
        query = make_query(variant, f"t{i % n_tables}", keyset)
        start = time.perf_counter()
        result = session.evaluate(query, per_query_budget)
        evaluated = time.perf_counter()
        result.collect()
        end = time.perf_counter()
        return (evaluated - start) * 1000, (end - start) * 1000

    i = 0
    for _ in range(args.warmup):
        one_query(i)
        i += 1
    for _ in range(args.queries):
        e, t = one_query(i)
        eval_ms.append(e)
        total_ms.append(t)
        i += 1
    for _ in range(args.profile_queries):
        prof = cProfile.Profile()
        prof.enable()
        one_query(i)
        prof.disable()
        calls.append(pstats.Stats(prof).total_calls)  # type: ignore[attr-defined]
        i += 1

    row: Dict[str, Any] = {
        "variant": variant,
        "tables": n_tables,
        "queries": args.queries,
        "data_s": round(data_s, 3),
        "build_s": round(build_s, 3),
        "median_eval_ms": round(statistics.median(eval_ms), 2),
        "mean_eval_ms": round(statistics.mean(eval_ms), 2),
        "median_total_ms": round(statistics.median(total_ms), 2),
        "mean_total_ms": round(statistics.mean(total_ms), 2),
        "min_total_ms": round(min(total_ms), 2),
        "max_total_ms": round(max(total_ms), 2),
        "calls_per_query": int(statistics.median(calls)) if calls else None,
        "raw_eval_ms": [round(x, 3) for x in eval_ms],
        "raw_total_ms": [round(x, 3) for x in total_ms],
    }
    gc.collect()
    spark.catalog.clearCache()
    return row


def fit(rows: List[Dict[str, Any]], column: str) -> Dict[str, Optional[float]]:
    """Least-squares fit of ``column = intercept + slope * tables``.

    The slope and intercept are None if there are fewer than two points.
    """
    points = [(r["tables"], r[column]) for r in rows if r[column] is not None]
    if len(points) < 2:
        return {"slope": None, "intercept": None}
    x, y = zip(*points)
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def fmt(value: Optional[float], spec: str) -> str:
    """Format a fitted value, or ``n/a`` if there is none."""
    return "n/a" if value is None else format(value, spec)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tables",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help=f"Comma-separated table counts (default: {DEFAULT_TABLE_COUNTS}).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Bigger sweep: tables={FULL_TABLE_COUNTS} (overridden by --tables).",
    )
    parser.add_argument(
        "--variants",
        type=lambda s: s.split(","),
        default=VARIANTS,
        help="Comma-separated variants to run: rows, ids (default: both).",
    )
    parser.add_argument("--queries", type=int, default=50, help="Timed queries.")
    parser.add_argument(
        "--warmup", type=int, default=2, help="Untimed queries per Session."
    )
    parser.add_argument(
        "--profile-queries",
        type=int,
        default=2,
        help="Extra queries run under cProfile to count Python calls (0 to skip).",
    )
    parser.add_argument("--rows", type=int, default=100, help="Rows per table.")
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=1,
        help="spark.sql.shuffle.partitions (small data, so keep it small).",
    )
    parser.add_argument(
        "--json-out", default=None, help="Also write full results as JSON here."
    )
    args = parser.parse_args()
    if args.tables is None:
        args.tables = FULL_TABLE_COUNTS if args.full else DEFAULT_TABLE_COUNTS
    for v in args.variants:
        if v not in VARIANTS:
            parser.error(f"Unknown variant {v!r}; expected one of {VARIANTS}")
    if args.queries < 1:
        parser.error("--queries must be at least 1")
    if args.warmup < 0 or args.profile_queries < 0:
        parser.error("--warmup and --profile-queries must be non-negative")
    return args


def main() -> None:
    """Evaluate per-query running time as a function of the number of tables."""
    args = parse_args()
    print("Benchmark per-query cost vs. number of private tables")
    print(f"Configuration: {vars(args)}")
    # Make Spark's Python workers use this interpreter (and so the same tmlt
    # packages) even when this venv is not first on the PATH.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "4g")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    keyset = KeySet.from_dict({"group": GROUP_VALUES})

    # Warm up Spark and the Python code paths before any timed run.
    warmup_args = argparse.Namespace(**{**vars(args), "queries": 3, "rows": 10})
    for variant in args.variants:
        run_config(spark, variant, 1, warmup_args, keyset)

    rows: List[Dict[str, Any]] = []
    for variant in args.variants:
        for n_tables in args.tables:
            row = run_config(spark, variant, n_tables, args, keyset)
            print(
                "Benchmark row:",
                {k: v for k, v in row.items() if not k.startswith("raw_")},
                flush=True,
            )
            rows.append(row)

    fits = {}
    for variant in args.variants:
        variant_rows = [r for r in rows if r["variant"] == variant]
        fits[variant] = {
            column: fit(variant_rows, column)
            for column in ["median_total_ms", "median_eval_ms", "calls_per_query"]
        }
        total = fits[variant]["median_total_ms"]
        evaluate = fits[variant]["median_eval_ms"]
        calls = fits[variant]["calls_per_query"]
        print(
            f"Fit [{variant}]: per-query total = {fmt(total['intercept'], '.1f')} ms"
            f" + {fmt(total['slope'], '.4f')} ms/table * N;"
            f" evaluate only = {fmt(evaluate['intercept'], '.1f')} ms"
            f" + {fmt(evaluate['slope'], '.4f')} ms/table * N;"
            f" calls/query = {fmt(calls['intercept'], '.0f')}"
            f" + {fmt(calls['slope'], '.1f')} * N"
        )

    benchmark_result = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("raw_")} for r in rows]
    )
    print(benchmark_result.to_string(index=False))
    spark_version = spark.version
    spark.stop()
    write_as_html(benchmark_result, "many_private_tables.html")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": vars(args),
                    "versions": {
                        "python": platform.python_version(),
                        "tmlt.analytics": tmlt.analytics.__version__,
                        "tmlt.analytics_path": tmlt.analytics.__file__,
                        "tmlt.core": tmlt.core.__version__,
                        "tmlt.core_path": tmlt.core.__file__,
                        "pyspark": spark_version,
                    },
                    "rows": rows,
                    "fits": fits,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
