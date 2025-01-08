"""Objects for representing tables."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

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


# It is essential that every TemporaryTable is equal only to itself and copies of
# itself.
@dataclass(init=False, frozen=True)
class TemporaryTable(Identifier):
    """Identify temporary tables."""

    _id: int

    def __init__(self):
        """Creates a new TemporaryTable with an automatically-generated id."""
        object.__setattr__(self, "_id", id(self))

    def __str__(self):
        """Returns the hashed string representation of the NamedTable."""
        return f"TemporaryTable({self._id})"

    def __repr__(self):
        """Returns the hashed object representation in string format."""
        return f"TemporaryTable({self._id})"

    def __copy__(self):
        """Creates a copy of this TemporaryTable with the same id."""
        other = TemporaryTable()
        object.__setattr__(other, "_id", self._id)
