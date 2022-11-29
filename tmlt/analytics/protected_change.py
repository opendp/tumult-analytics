"""Types for programmatically specifying what changes in input tables are protected."""

# Copyright Tumult Labs 2022
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from dataclasses import dataclass
from typing import Union

from typeguard import check_type


class ProtectedChange(ABC):
    """A description of the largest change in a dataset that is protected under DP.

    A :class:`ProtectedChange` describes, for a particular table, the largest
    change that can be made to that table while still being indistinguishable
    under Tumult Analytics' DP guarantee. The appropriate protected change to
    use is one corresponding to the largest possible change to the table when
    adding or removing a unit of protection, e.g. a person. For more
    information, see the :ref:`privacy promise topic guide
    <privacy-promise#unit-of-protection>`.
    """


@dataclass(frozen=True)
class AddMaxRows(ProtectedChange):
    """Protect the addition or removal of any set of ``max_rows`` rows.

    This ProtectedChange is a generalization of the standard "add/remove one
    row" DP guarantee, hiding the addition or removal of any set of at most
    ``max_rows`` rows from a table.
    """

    max_rows: int
    """The maximum number of rows that may be added or removed."""

    def __post_init__(self):
        """Validate attributes."""
        check_type("max_rows", self.max_rows, int)
        if self.max_rows < 1:
            raise ValueError("max_rows must be positive")


@dataclass(frozen=True)
class AddOneRow(AddMaxRows):
    """A shorthand for the common case of :class:`AddMaxRows` with ``max_rows = 1``."""

    max_rows = 1

    def __init__(self):
        """@nodoc."""
        super(AddOneRow, self).__init__(max_rows=1)


@dataclass(frozen=True)
class AddMaxRowsInMaxGroups(ProtectedChange):
    """Protect the addition or removal of rows across a finite number of groups.

    :class:`AddMaxRowsInMaxGroups` provides a similar guarantee to
    :class:`AddMaxRows`, but it uses some additional information to apply less
    noise in some cases. That information is about *groups*: collections of rows
    which share the same value in a particular column. That column would
    typically be some kind of categorical value, for example a state where a
    person lives or has lived. Instead of specifying a maximum total number of
    rows that may be added or removed, :class:`AddMaxRowsInMaxGroups` limits the
    number of rows that may be added or removed in any particular group, as well
    as the maximum total number of groups that may be affected. If these limits
    are meant to correspond to the maximum contribution of a specific entity to
    the dataset, that must be enforced *before* the data is passed to Tumult
    Analytics.

    :class:`AddMaxRowsInMaxGroups` is intended for advanced use cases, and its
    use should be considered carefully. Note that it only provides improved
    accuracy when used with zCDP -- with pure DP, it is equivalent to using
    :class:`AddMaxRows` with the same total number of rows to be added/removed.
    """

    grouping_column: str
    """The name of the column specifying the group."""
    max_groups: int
    """The maximum number of groups that may differ."""
    max_rows_per_group: Union[int, float]
    """The maximum number of rows which may be added to or removed from each group."""

    def __post_init__(self):
        """Validate attributes."""
        check_type("column", self.grouping_column, str)
        check_type("max_groups", self.max_groups, int)
        check_type("max_rows_per_group", self.max_rows_per_group, Union[int, float])
        if self.max_groups < 1:
            raise ValueError("max_groups must be positive")
        if self.max_rows_per_group < 1:
            raise ValueError("max_rows_per_group must be positive")
