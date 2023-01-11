"""Objects for representing tables."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from dataclasses import dataclass


class Identifier:
    """Base class for tables, which are each an Identifier type."""


@dataclass(frozen=True)
class NamedTable(Identifier):
    """Identify named tables. In most cases, these are user-provided."""

    name: str
    """The name of the table."""

    def __str__(self):
        """String representation of the NamedTable."""
        return f"NamedTable({self.name})"


@dataclass(frozen=True)
class TableCollection(Identifier):
    """Identify a collection of tables."""

    name: str
    """The name of the table."""

    def __str__(self):
        """Returns the string representation of the NamedTable."""
        return f"TableCollection({self.name})"


# It is essential that every TemporaryTable is equal only to itself.
class TemporaryTable(Identifier):
    """Identify temporary tables."""

    def __str__(self):
        """Returns the hashed string representation of the NamedTable."""
        return f"TemporaryTable({hash(self)})"

    def __repr__(self):
        """Returns the hashed object representation in string format."""
        return f"TemporaryTable({hash(self)})"
