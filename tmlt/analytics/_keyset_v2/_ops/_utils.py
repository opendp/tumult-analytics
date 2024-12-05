"""Shared utilities for KeySetOps."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from typing import Iterable

from tmlt.analytics import ColumnType

KEYSET_COLUMN_TYPES = [ColumnType.INTEGER, ColumnType.DATE, ColumnType.VARCHAR]
"""Column types that are allowed in KeySets."""


def validate_column_names(columns: Iterable[str]):
    """Ensure that the given collection of column names are all valid."""
    for col in columns:
        if not isinstance(col, str):
            raise ValueError(
                f"Column names must be strings, not {type(col).__qualname__}."
            )
        if len(col) == 0:
            raise ValueError("Empty column names are not allowed.")
