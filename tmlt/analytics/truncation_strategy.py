"""Defines strategies for performing truncation in private joins."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from abc import ABC
from dataclasses import dataclass

from typeguard import check_type


class TruncationStrategy:
    """Strategies for performing truncation in private joins."""

    class Type(ABC):
        """Type of TruncationStrategy variants."""

    @dataclass(frozen=True)
    class DropExcess(Type):
        """Drop records with matching join keys above a threshold.

        This truncation strategy drops records such that no more than ``max_records``
        records have the same join key. Which records are kept is deterministic and does
        not depend on the order in which they appear in the private data. For example,
        using the ``DropExcess(1)`` strategy while joining on columns A and B in the
        below table:

        === === =====
         A   B   Val
        === === =====
         a   b    1
         a   c    2
         a   b    3
         b   a    4
        === === =====

        causes it to be treated as one of the below tables:

        === === =====
         A   B   Val
        === === =====
         a   b    1
         a   c    2
         b   a    4
        === === =====

        === === =====
         A   B   Val
        === === =====
         a   b    3
         a   c    2
         b   a    4
        === === =====

        This is generally the preferred truncation strategy, even when the
        :class:`~TruncationStrategy.DropNonUnique` strategy could also be used,
        because it results in fewer dropped rows.
        """

        max_records: int
        """Maximum number of records to keep."""

        def __post_init__(self):
            """Check arguments to constructor."""
            check_type("max_records", self.max_records, int)
            if self.max_records < 1:
                raise ValueError("At least one record must be kept for each join key.")

    @dataclass(frozen=True)
    class DropNonUnique(Type):
        """Drop all records with non-unique join keys.

        This truncation strategy drops all records which share join keys with another
        record in the dataset. It is similar to the ``DropExcess(1)`` strategy, but
        doesn't keep *any* of the records with duplicate join keys. For example, using
        the ``DropNonUnique`` strategy while joining on columns A and B in the below
        table:

        === === =====
         A   B   Val
        === === =====
         a   b    1
         a   c    2
         a   b    3
         b   a    4
        === === =====

        causes it to be treated as:

        === === =====
         A   B   Val
        === === =====
         a   c    2
         b   a    4
        === === =====

        This truncation strategy results in less noise than ``DropExcess(1)``. However,
        it also drops more rows in datasets where many records have non-unique join
        keys. In most cases, DropExcess is the preferred strategy.
        """
