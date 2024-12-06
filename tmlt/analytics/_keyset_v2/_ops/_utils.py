"""Shared utilities for KeySetOps."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from typing import Iterable, Mapping

from tmlt.analytics import ColumnDescriptor, ColumnType

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


def validate_schema(column_descriptors: Mapping[str, ColumnDescriptor]):
    """Ensure that the given column schema is valid."""
    validate_column_names(column_descriptors.keys())
    for col, desc in column_descriptors.items():
        if desc.column_type not in KEYSET_COLUMN_TYPES:
            raise ValueError(
                f"Column '{col}' has type {desc.column_type.name}, but "
                "only allowed types in KeySets are: "
                f"{', '.join(t.name for t in KEYSET_COLUMN_TYPES)}"
            )
