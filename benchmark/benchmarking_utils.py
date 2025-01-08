"""Common utility functions for benchmarking scripts."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from pathlib import Path

import pandas as pd


def write_as_html(df: pd.DataFrame, fname: str) -> None:
    """Writes out DataFrame as an html file."""
    output_dir = Path(__file__).parent.parent / "benchmark_output"
    output_dir.mkdir(exist_ok=True)
    with open(str(output_dir / fname), "a", encoding="utf-8") as f:
        df.to_html(f)
