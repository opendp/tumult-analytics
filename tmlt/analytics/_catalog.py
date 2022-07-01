"""Contains classes for specifying schemas and constraints for tables."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

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

    stability: int
    """The maximum number of rows a single individual can modify if they are modified.

    This can become higher than 1 if there have been transformations, such as joins,
    which allow single individuals to affect many rows in the table.
    """


@dataclass
class PrivateView(Table):
    """Metadata for a view on a private table.

    PrivateViews are similar to database views, which are extensions of the original
    tables. Information in a private view is private and needs to be protected.
    """

    stability: int
    """The maximum number of rows a single individual can modify if they are modified.

    This can become higher than 1 if there have been transformations, such as joins,
    which allow single individuals to affect many rows in the table.
    """


class Catalog:
    """Specifies schemas and constraints on public and private tables."""

    def __init__(self):
        """Constructor."""
        self._tables = {}
        self._private_source_id = None

    def _add_table(self, table: Table):
        """Adds table to catalog.

        Args:
            table: The table, public or private.
        """
        if table.source_id in self._tables:
            raise ValueError(f"{table.source_id} already exists in catalog.")
        self._tables[table.source_id] = table

    def remove_table(self, source_id: str):
        """Removes a table from the catalog.

        Args:
            source_id: The name of the table.
        """
        try:
            del self._tables[source_id]
        except KeyError:
            pass

    def add_private_source(
        self,
        source_id: str,
        col_types: Mapping[str, Union[ColumnDescriptor, ColumnType]],
        stability: int,
        grouping_column: Optional[str] = None,
    ):
        """Adds a private table to catalog. There may only be a single private table.

        Args:
            source_id: The source id, or unique identifier, for the private table.
            col_types: Mapping from column names to types for private table.
            stability: The maximum number of rows that could be added or removed in
                the table if a single individual is added or removed.
            grouping_column: Name of the column (if any) that must be grouped by in any
                groupby aggregations that use this table.

        Raises:
            RuntimeError: If there is already a private table.
        """
        if self._private_source_id is not None:
            raise RuntimeError("Cannot have more than one private source")
        self._private_source_id = source_id
        self._add_table(
            PrivateTable(
                source_id=source_id,
                schema=Schema(col_types, grouping_column=grouping_column),
                stability=stability,
            )
        )

    def add_private_view(
        self,
        source_id: str,
        col_types: Mapping[str, Union[ColumnDescriptor, ColumnType]],
        stability: int,
        grouping_column: Optional[str] = None,
    ):
        """Adds view table to catalog.

        Args:
            source_id: The source id, or unique identifier, for the view table.
            col_types: Mapping from column names to types for private view.
            stability: The maximum number of rows that could be added or removed in
                the table if a single individual is added or removed.
            grouping_column: Name of the column (if any) that must be grouped by in any
                groupby aggregations that use this table.
        """
        self._add_table(
            PrivateView(
                source_id=source_id,
                schema=Schema(col_types, grouping_column=grouping_column),
                stability=stability,
            )
        )

    def add_public_source(
        self,
        source_id: str,
        col_types: Mapping[str, Union[ColumnDescriptor, ColumnType]],
    ):
        """Adds public table to catalog.

        Args:
            source_id: The source id, or unique identifier, for the public table.
            col_types: Mapping from column names to types for the public table.
        """
        self._add_table(PublicTable(source_id=source_id, schema=Schema(col_types)))

    @property
    def private_table(self) -> Optional[PrivateTable]:
        """Returns the primary private table, if it exists."""
        private_table = self.tables[self._private_source_id]
        if not isinstance(private_table, PrivateTable):
            raise AssertionError(
                "private_table does not have type PrivateTable. This is "
                "probably a bug; please let us know about it so we can "
                "fix it!"
            )
        return private_table

    @property
    def tables(self) -> Dict[str, Table]:
        """Returns the catalog as a dictionary of tables."""
        return self._tables
