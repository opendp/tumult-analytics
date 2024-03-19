"""Defines strategies for performing truncation in private joins."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from abc import ABC
from dataclasses import dataclass

from typeguard import check_type


class TruncationStrategy:
    """Strategies for performing truncation in private joins.

    These are used to determine the sensitivity of a private join between two tables
    having :class:`~tmlt.analytics.protected_change.AddMaxRows` as a protected change.
    The formula for the sensitivity of the table resulting from a private join is:

    :math:`sensitivity=(T_{left}*S_{right}*M_{left}) + (T_{right}*S_{left}*M_{right})`

    where:

      - :math:`T_{left}` and :math:`T_{right}` are the truncation thresholds for the
        left and right truncation strategies, respectively. This value is 1 for
        ``DropNonUnique``.
      - :math:`S_{left}` and :math:`S_{right}` are the stability of the left and right
        truncation strategies, respectively. This value is 2 for ``DropExcess`` and 1
        for ``DropNonUnique``.
      - :math:`M_{left}` and :math:`M_{right}` are the ``max_rows`` parameters of the
        ``AddMaxRows`` protected changes of the left and right tables, respectively.

    """

    class Type(ABC):
        """Type of TruncationStrategy variants."""

    @dataclass(frozen=True)
    class DropExcess(Type):
        """Drop rows with matching join keys above a threshold.

        This truncation strategy drops rows such that no more than ``max_rows``
        rows have the same join key. Which rows are kept is deterministic and does
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

        max_rows: int
        """Maximum number of rows to keep."""

        def __post_init__(self):
            """Check arguments to constructor."""
            check_type("max_rows", self.max_rows, int)
            if self.max_rows < 1:
                raise ValueError("At least one row must be kept for each join key.")

    @dataclass(frozen=True)
    class DropNonUnique(Type):
        """Drop all rows with non-unique join keys.

        This truncation strategy drops all rows which share join keys with another
        row in the dataset. It is similar to the ``DropExcess(1)`` strategy, but
        doesn't keep *any* of the rows with duplicate join keys. For example, using
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
        it also drops more rows in datasets where many rows have non-unique join
        keys. In most cases, DropExcess is the preferred strategy.
        """
