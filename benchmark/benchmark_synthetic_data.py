"""Benchmarking script for generating synthetic data."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import random
import time
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
from benchmarking_utils import write_as_html
from pyspark.sql import SparkSession

from tmlt.analytics import ApproxDPBudget, AddOneRow, KeySet, Session 
from tmlt.synthetics import ClampingBounds, Count, FixedMarginals, Sum
from tmlt.analytics.synthetics._toolkit import SyntheticDataToolkit

def create_chain_marginals(
    num_attributes: int, 
    chain_size: int,
    include_first_n: Optional[int] = None,
) -> Tuple[List[Count], List[Sum]]:
    """Creates a list of marginals forming a chain pattern.
    
    For chain_size=3, creates marginals like: (A1,A2,A3), (A2,A3,A4), (A3,A4,A5)...
    For chain_size=1, creates marginals like: (A1), (A2), (A3)...

    If include_first_n is specified, the first n columns are included in all
    measurements after creating the chain marginals.

    For include_first_n=1, chain_size=2: (A1,A2), (A1,A2,A3), (A1,A3,A4), ...

    Args:
        num_attributes: Number of attributes in total
        chain_size: Size of each chain marginal
        include_first_n: Number of first columns to include in all measurements (None or >= 1)
    """
    count_marginals = []
    sum_marginals = []
    
    # Generate column names
    columns = [f"A{i+1}" for i in range(num_attributes)]

    # Validate parameters
    if include_first_n is not None:
        if include_first_n < 1:
            raise ValueError("include_first_n must be at least 1")
        if include_first_n > num_attributes:
            raise ValueError(f"include_first_n ({include_first_n}) cannot be greater than num_attributes ({num_attributes})")

    # Create chain marginals
    for i in range(num_attributes - chain_size + 1):
        groupby_columns = columns[i:i + chain_size]
        
        # Add first n columns to each marginal if specified
        if include_first_n is not None:
            first_n_cols = columns[:include_first_n]
            # Avoid duplicating columns that are already in the chain
            extra_cols = [col for col in first_n_cols if col not in groupby_columns]
            groupby_columns = extra_cols + groupby_columns
        count_marginals.append(Count(groupby_columns))
        sum_marginals.append(Sum(groupby_columns, "X"))
    
    return count_marginals, sum_marginals


def evaluate_runtime(
    session: Session,
    keyset: KeySet,
    count_marginals: List[Count],
    sum_marginals: List[Sum],
    split_columns: Optional[List[str]],
    clamping_bounds: ClampingBounds,
    privacy_budget: ApproxDPBudget,
) -> Dict[str, float]:
    """See how long it takes to generate synthetic data."""
    measurement_strategies = [FixedMarginals(marginals=count_marginals + sum_marginals)]
    
    toolkit = SyntheticDataToolkit(
        session=session,
        source_id="private",
        keyset=keyset,
    )
    start = time.time()
    toolkit.measure.evaluate_measurement_strategies(
        measurement_strategies=measurement_strategies,
        privacy_budget=privacy_budget,
        clamping_bounds=clamping_bounds,
    )
    measurement_time = time.time() - start
    print(f"Measurement time: {measurement_time} seconds")
    start = time.time()
    toolkit.fit.fit_model(
        split_columns=split_columns,
        model_iterations=1,
    )
    model_fitting_time = time.time() - start
    print(f"Model fitting time: {model_fitting_time} seconds")
    start = time.time()
    toolkit.generate.generate_from_model()
    toolkit.synthetic_data.count() # Force computation
    categorical_data_generation_time = time.time() - start
    print(f"Categorical data generation time: {categorical_data_generation_time} seconds")
    start = time.time()
    toolkit.generate.with_column_from_least_squares(
        measure_column="X",
        new_column="X",
        split_columns=split_columns,
    )
    toolkit.synthetic_data.count() # Force computation
    numeric_data_generation_time = time.time() - start
    print(f"Numeric data generation time: {numeric_data_generation_time} seconds")
    return {
        "measurement_time": measurement_time,
        "model_fitting_time": model_fitting_time,
        "categorical_data_generation_time": categorical_data_generation_time,
        "numeric_data_generation_time": numeric_data_generation_time,
    }


def main() -> None:
    """Evaluate running time for synthetic data generation with different parameters."""
    print("Benchmark synthetic data generation")
    spark = (
        SparkSession.builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    warnings.filterwarnings("ignore")
    
    test_cases = [
        {
            "name": f"baseline (in memory)",
            "size": 100_000,
            "num_attributes": 10,
            "attribute_domain": 100,
            "chain_size": 2,
            "include_first_n": None,
            "split_first_m": None,
        }
    ] + [
        {
            "name": f"larger dataset (in memory)",
            "size": size,
            "num_attributes": 10,
            "attribute_domain": 100,
            "chain_size": 2,
            "include_first_n": None,
            "split_first_m": None,
        }
        for size in [1_000_000]
    ] + [
        {
            "name": f"more attributes (in memory)",
            "size": 100_000,
            "num_attributes": num_attributes,
            "attribute_domain": 100,
            "chain_size": 2,
            "include_first_n": None,
            "split_first_m": None,
        }
        for num_attributes in [20]
    ] + [
        {
            "name": f"larger domains (in memory)",
            "size": 100_000,
            "num_attributes": 10,
            "attribute_domain": domain,
            "chain_size": 2,
            "include_first_n": None,
            "split_first_m": None,
        }
        for domain in [200]
    ] + [
        {
            "name": f"larger chains (in memory)",
            "size": 100_000,
            "num_attributes": 10,
            "attribute_domain": 10,
            "chain_size": chain_size,
            "include_first_n": None,
            "split_first_m": None,
        }
        for chain_size in [3]
    ] + [
        {
            "name": f"first n columns included in all measurements (in memory)",
            "size": 100_000,
            "num_attributes": 10,
            "attribute_domain": 10,
            "chain_size": 2,
            "include_first_n": include_first_n,
            "split_first_m": None,
        }
        for include_first_n in [1]
    ] + [
        {
            "name": f"first n columns included in all measurements (splitting on first n columns)",
            "size": 100_000,
            "num_attributes": 10,
            "attribute_domain": 10,
            "chain_size": 2,
            "include_first_n": include_first_n,
            "split_first_m": include_first_n,
        }
        for include_first_n in [1]
    ]

    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "Test Case",
            "Size",
            "Num Attributes",
            "Attribute Domain",
            "Chain Size",
            "Include First N",
            "Split First M",
            "Measurement time (s)",
            "Model fitting time (s)",
            "Categorical data generation time (s)",
            "Numeric data generation time (s)",
        ],
    )

    for case in test_cases:
        print(f"Running test case: {case['name']}")

        # Create column names and domains
        columns = [f"A{i+1}" for i in range(case["num_attributes"])]
        domain = list(range(case["attribute_domain"]))
        
        # Create test data and session
        df_dict = {col: [random.randint(0, case["attribute_domain"] - 1) for _ in range(case["size"])] for col in columns}
        df_dict["X"] = [random.random() for _ in range(case["size"])]
        df = spark.createDataFrame(pd.DataFrame(df_dict))
        
        # Create keyset with specified domain for each attribute
        keyset_dict = {col: domain for col in columns}
        keyset = KeySet.from_dict(keyset_dict)
        
        session = Session.from_dataframe(
            privacy_budget=ApproxDPBudget(epsilon=float("inf"), delta=1e-6),
            protected_change=AddOneRow(),
            source_id="private",
            dataframe=df,
        )

        # Create marginals and clamping bounds
        count_marginals, sum_marginals = create_chain_marginals(
            case["num_attributes"],
            case["chain_size"],
            case["include_first_n"],
        )
        
        clamping_bounds = ClampingBounds(
            bounds_per_column={
                "X": (0, 1)
            }
        )

        # measurement
        running_time = evaluate_runtime(
            session=session,
            keyset=keyset,
            count_marginals=count_marginals,
            sum_marginals=sum_marginals,
            split_columns=columns[:case["split_first_m"]] if case["split_first_m"] is not None else None,
            clamping_bounds=clamping_bounds,
            privacy_budget=ApproxDPBudget(epsilon=1.0, delta=1e-6),
        )

        row = {
            "Test Case": case["name"],
            "Size": case["size"],
            "Num Attributes": case["num_attributes"],
            "Attribute Domain": case["attribute_domain"],
            "Chain Size": case["chain_size"],
            "Include First N": case["include_first_n"],
            "Split First M": case["split_first_m"],
            **running_time,
        }
        print("Benchmark row:", row)
        benchmark_result = pd.concat(
            [benchmark_result, pd.Series(row).to_frame().T], ignore_index=True
        )

    spark.stop()
    write_as_html(benchmark_result, "synthetic_data_generation.html")


if __name__ == "__main__":
    main()
