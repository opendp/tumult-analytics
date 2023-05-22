"""Types for programmatically specifying what changes in input tables are protected."""

# Copyright Tumult Labs 2023
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
        super().__init__(max_rows=1)


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

    The most common case where :class:`AddMaxRowsInMaxGroups` is useful is for
    dealing with datasets that have already undergone some type of preprocessing
    before being turned over to an analyst. Where possible, it is preferred to
    do such processing inside of Tumult Analytics instead, as it allows
    specifying a simpler protected change (e.g. :class:`AddRowsWithID`)
    and relying on Analytics' privacy tracking to handle the complex parts
    of the analysis.
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


@dataclass(frozen=True)
class AddRowsWithID(ProtectedChange):
    """Protect the addition or removal of rows with a specific identifier.

    Instead of limiting the number of rows that may be added or removed,
    :class:`AddRowsWithID` hides the addition or removal of *all rows*
    with the same value in the specified column.

    The id column *must* be a string, integer (or long), or date; it cannot
    be a float or a timestamp.
    """

    id_column: str
    """The name of the column containing the identifier."""

    id_space: str = "default_id_space"
    """The identifier space of the rows that may be added or removed. If not specified,
    a default will be assigned when using this protected change with
    :class:`Session.from_dataframe()<tmlt.analytics.session.Session.from_dataframe>`."""

    def __post_init__(self):
        """Validate attributes."""
        check_type("id_space", self.id_space, str)
        if self.id_space == "":
            raise ValueError("identifier must be non-empty")
        check_type("id_column", self.id_column, str)
        if self.id_column == "":
            raise ValueError("id_column must be non-empty")
