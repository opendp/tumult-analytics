"""Contains classes for specifying schemas and constraints for tables."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Union

from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema


@dataclass
class Table(ABC):
    """Metadata for a public or private table."""

    source_id: str
    """The source id, or unique identifier, for the table."""
    schema: Schema
    """The analytics schema for the table. Describes the column types."""
    id_space: Optional[str] = None
    """The identifier space for the table."""


@dataclass
class PublicTable(Table):
    """Metadata for a public table.

    Public tables contain information that is generally not sensitive and does not
    require any special privacy protections.
    """

    def __post_init__(self):
        """Check inputs to constructor."""
        if self.schema.grouping_column is not None:
            raise ValueError("Public tables cannot have a grouping_column")


@dataclass
class PrivateTable(Table):
    """Metadata for a private table.

    Private tables contain sensitive information, such as PII, whose privacy has to be
    protected.
    """


class Catalog:
    """Specifies schemas and constraints on public and private tables."""

    def __init__(self):
        """Constructor."""
        self._tables = {}

    def _add_table(self, table: Table):
        """Adds table to catalog.

        Args:
            table: The table, public or private.
        """
        if table.source_id in self._tables:
            raise ValueError(f"{table.source_id} already exists in catalog.")
        self._tables[table.source_id] = table

    def add_private_table(
        self,
        source_id: str,
        col_types: Mapping[str, Union[ColumnDescriptor, ColumnType]],
        grouping_column: Optional[str] = None,
        id_column: Optional[str] = None,
        id_space: Optional[str] = None,
    ):
        """Adds a private table to catalog. There may only be a single private table.

        Args:
            source_id: The source id, or unique identifier, for the private table.
            col_types: Mapping from column names to types for private table.
            grouping_column: Name of the column (if any) that must be grouped by in any
                groupby aggregations that use this table.
            id_column: Name of the ID column for this table (if any).
            id_space: Name of the identifier space for this table (if any).

        Raises:
            ValueError: If there is already a private table.
        """
        self._add_table(
            PrivateTable(
                source_id=source_id,
                schema=Schema(
                    col_types,
                    grouping_column=grouping_column,
                    id_column=id_column,
                    id_space=id_space,
                ),
            )
        )

    def add_public_table(
        self,
        source_id: str,
        col_types: Mapping[str, Union[ColumnDescriptor, ColumnType]],
    ):
        """Adds public table to catalog.

        Args:
            source_id: The source id, or unique identifier, for the public table.
            col_types: Mapping from column names to types for the public table.

        Raises:
            ValueError: If there is already a private table.
        """
        self._add_table(PublicTable(source_id=source_id, schema=Schema(col_types)))

    @property
    def tables(self) -> Dict[str, Table]:
        """Returns the catalog as a dictionary of tables."""
        return self._tables
